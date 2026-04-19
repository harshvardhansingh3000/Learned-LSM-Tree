#include "sstable/sstable_reader.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace lsm {

// ─── Low-Level Read Helpers ──────────────────────────────

bool SSTableReader::read_bytes(void* data, size_t size, std::ifstream& in) {
    in.read(static_cast<char*>(data), size);
    return in.good() && (static_cast<size_t>(in.gcount()) == size);
}

bool SSTableReader::read_uint32(uint32_t& value, std::ifstream& in) {
    return read_bytes(&value, sizeof(uint32_t), in);
}

bool SSTableReader::read_uint64(uint64_t& value, std::ifstream& in) {
    return read_bytes(&value, sizeof(uint64_t), in);
}

// Reads a length-prefixed string: [length (4 bytes)] [string data]
bool SSTableReader::read_string(std::string& str, std::ifstream& in) {
    uint32_t len;
    if (!read_uint32(len, in)) return false;
    if (len > 10 * 1024 * 1024) return false;  // Sanity check: max 10 MB
    str.resize(len);
    if (len > 0) {
        if (!read_bytes(str.data(), len, in)) return false;
    }
    return true;
}

// ─── Constructor ──────────────────────────────────────────
// Opens the SSTable file and reads footer + index into memory.
//
// After this, the reader is ready for lookups.
// Data blocks are NOT loaded — they're read on-demand during get().

SSTableReader::SSTableReader(const std::string& file_path)
    : file_path_(file_path)
{
    read_footer();
    read_index();
}

// ─── Move Constructor ─────────────────────────────────────
// Transfers ownership of all data from 'other' to 'this'.
// After the move, 'other' is in a valid but empty state.

SSTableReader::SSTableReader(SSTableReader&& other) noexcept
    : file_path_(std::move(other.file_path_))
    , footer_(std::move(other.footer_))
    , index_(std::move(other.index_))
{
}

SSTableReader& SSTableReader::operator=(SSTableReader&& other) noexcept {
    if (this != &other) {
        file_path_ = std::move(other.file_path_);
        footer_ = std::move(other.footer_);
        index_ = std::move(other.index_);
    }
    return *this;
}

// ─── Read Footer ──────────────────────────────────────────
// The footer is at the END of the file. We read it by:
//   1. Seeking to the end to get file size
//   2. Reading backwards to find the magic number
//   3. Then reading the footer fields
//
// Footer layout (written by SSTableWriter):
//   [index_offset (8)] [index_size (4)] [entry_count (4)]
//   [min_key_len (4)] [min_key] [max_key_len (4)] [max_key]
//   [magic (4)]
//
// We read from the position stored in index_offset to reconstruct.
// But since the footer has variable-length strings (min/max keys),
// we read it sequentially from where the index block ends.

void SSTableReader::read_footer() {
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open SSTable file: " + file_path_);
    }

    // Get file size
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if (file_size < 4) {
        throw std::runtime_error("SSTable file too small: " + file_path_);
    }

    // Read magic number (last 4 bytes)
    in.seekg(-4, std::ios::end);
    uint32_t magic;
    if (!read_uint32(magic, in) || magic != SSTABLE_MAGIC) {
        throw std::runtime_error("Invalid SSTable magic number: " + file_path_);
    }

    // Now we need to find where the footer starts.
    // Strategy: read the file from the end, working backwards.
    // The footer structure before magic is:
    //   [index_offset(8)] [index_size(4)] [entry_count(4)]
    //   [min_key_len(4)] [min_key(var)] [max_key_len(4)] [max_key(var)]
    //   [magic(4)]
    //
    // We'll scan backwards to find index_offset, then read forward.
    // Simpler approach: we know index_offset is the first field of the footer.
    // After the index block ends, the footer begins.
    // Let's read the index_offset by trying positions.
    //
    // Actually, the cleanest approach: read from just after all data+index blocks.
    // We'll try reading index_offset from various positions.
    // The footer starts right after the index block.
    // Since we don't know the footer size (variable due to min/max keys),
    // let's read the first 8 bytes of the footer (index_offset) by
    // scanning backwards from the magic number.

    // We need to find where the footer starts. Let's read backwards:
    // Before magic: max_key (variable), before that: max_key_len (4),
    // before that: min_key (variable), before that: min_key_len (4),
    // before that: entry_count (4), before that: index_size (4),
    // before that: index_offset (8).
    //
    // Since we can't easily read variable-length strings backwards,
    // let's use a different strategy: try reading index_offset from
    // a reasonable position and validate.

    // Better approach: read the entire file tail (last ~1KB) and parse forward
    // from different offsets until we find a valid footer.
    // But simplest: since we wrote the footer sequentially, let's just
    // try to find it by reading from multiple candidate positions.

    // Simplest reliable approach: binary search for the footer start.
    // The index_offset value tells us where the index starts.
    // The footer starts at (index_offset + index_size).
    // So if we can read index_offset, we can validate everything.

    // Let's try reading from the end minus a reasonable max footer size.
    // Max footer = 8 + 4 + 4 + 4 + max_key_len + 4 + max_key_len + 4
    // With reasonable key sizes (< 1KB), footer < 2KB.

    // Read last 4KB of file (more than enough for any footer)
    size_t tail_size = std::min(static_cast<size_t>(file_size), static_cast<size_t>(4096));
    std::vector<uint8_t> tail(tail_size);
    in.seekg(-static_cast<std::streamoff>(tail_size), std::ios::end);
    in.read(reinterpret_cast<char*>(tail.data()), tail_size);

    // Parse the tail to find the footer.
    // We know the magic is at the very end (last 4 bytes of tail).
    // Work backwards from there.
    // Before magic: max_key data, before that: max_key_len
    // But we can't easily parse backwards with variable strings.

    // Best approach: try each possible footer start position.
    // The footer's index_offset field should point to a valid position in the file.
    // We try reading index_offset from each position and validate.

    bool found = false;
    for (size_t try_offset = 24; try_offset < tail_size; try_offset++) {
        size_t pos = tail_size - try_offset;

        // Try reading index_offset from this position
        if (pos + 8 > tail_size) continue;
        uint64_t try_index_offset;
        std::memcpy(&try_index_offset, tail.data() + pos, sizeof(uint64_t));

        // Validate: index_offset should be less than file_size
        if (try_index_offset >= static_cast<uint64_t>(file_size)) continue;

        // Try reading the rest of the footer from this position
        size_t fpos = pos + 8;

        if (fpos + 4 > tail_size) continue;
        uint32_t try_index_size;
        std::memcpy(&try_index_size, tail.data() + fpos, sizeof(uint32_t));
        fpos += 4;

        // Validate: index_offset + index_size should equal the footer start
        uint64_t expected_footer_start = try_index_offset + try_index_size;
        uint64_t actual_footer_start = static_cast<uint64_t>(file_size) - try_offset;
        if (expected_footer_start != actual_footer_start) continue;

        // This looks valid! Read the rest of the footer.
        if (fpos + 4 > tail_size) continue;
        uint32_t try_entry_count;
        std::memcpy(&try_entry_count, tail.data() + fpos, sizeof(uint32_t));
        fpos += 4;

        // Read min_key
        if (fpos + 4 > tail_size) continue;
        uint32_t min_key_len;
        std::memcpy(&min_key_len, tail.data() + fpos, sizeof(uint32_t));
        fpos += 4;
        if (min_key_len > 10000 || fpos + min_key_len > tail_size) continue;
        std::string try_min_key(reinterpret_cast<char*>(tail.data() + fpos), min_key_len);
        fpos += min_key_len;

        // Read max_key
        if (fpos + 4 > tail_size) continue;
        uint32_t max_key_len;
        std::memcpy(&max_key_len, tail.data() + fpos, sizeof(uint32_t));
        fpos += 4;
        if (max_key_len > 10000 || fpos + max_key_len > tail_size) continue;
        std::string try_max_key(reinterpret_cast<char*>(tail.data() + fpos), max_key_len);
        fpos += max_key_len;

        // Read magic
        if (fpos + 4 > tail_size) continue;
        uint32_t try_magic;
        std::memcpy(&try_magic, tail.data() + fpos, sizeof(uint32_t));
        fpos += 4;

        if (try_magic != SSTABLE_MAGIC) continue;

        // Verify fpos is at the end of tail
        if (fpos != tail_size) continue;

        // All checks passed! We found the footer.
        footer_.index_offset = try_index_offset;
        footer_.index_size = try_index_size;
        footer_.entry_count = try_entry_count;
        footer_.min_key = try_min_key;
        footer_.max_key = try_max_key;
        footer_.magic = try_magic;
        found = true;
        break;
    }

    if (!found) {
        throw std::runtime_error("Cannot parse SSTable footer: " + file_path_);
    }
}

// ─── Read Index ───────────────────────────────────────────
// Reads the index block from the file using the offset/size from the footer.
// Each index entry: [first_key (string)] [block_offset (8)] [block_size (4)]

void SSTableReader::read_index() {
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open SSTable file: " + file_path_);
    }

    // Seek to the index block
    in.seekg(footer_.index_offset);

    // Read index entries until we've consumed index_size bytes
    uint64_t bytes_read = 0;
    while (bytes_read < footer_.index_size) {
        IndexEntry entry;

        if (!read_string(entry.first_key, in)) break;
        if (!read_uint64(entry.block_offset, in)) break;
        if (!read_uint32(entry.block_size, in)) break;

        bytes_read = static_cast<uint64_t>(in.tellg()) - footer_.index_offset;
        index_.push_back(std::move(entry));
    }
}

// ─── Find Block for Key ──────────────────────────────────
// Binary search on fence pointers to find which block might contain the key.
//
// Fence pointers: ["apple", "dog", "fox", "zoo"]
// Looking for "elk":
//   - "elk" >= "dog" and "elk" < "fox"
//   - So "elk" is in block 1 (the "dog" block)
//
// Returns the block index, or -1 if the key is out of range.

int SSTableReader::find_block_for_key(const Key& key) const {
    if (index_.empty()) return -1;

    // Binary search: find the last block whose first_key <= key
    int lo = 0;
    int hi = static_cast<int>(index_.size()) - 1;
    int result = -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (index_[mid].first_key <= key) {
            result = mid;  // This block might contain the key
            lo = mid + 1;  // But a later block might also work
        } else {
            hi = mid - 1;
        }
    }

    return result;
}

// ─── Read Data Block ──────────────────────────────────────
// Reads a data block from disk and parses all entries in it.
//
// Each entry in the block:
//   [key_len (4)] [key] [type (1)] [seq_num (8)] [val_len (4)] [value]

std::vector<Entry> SSTableReader::read_data_block(uint64_t offset, uint32_t size) const {
    std::vector<Entry> entries;

    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) return entries;

    // Seek to the block
    in.seekg(offset);

    // Read the entire block into memory
    std::vector<uint8_t> block(size);
    in.read(reinterpret_cast<char*>(block.data()), size);
    if (!in.good() && !in.eof()) return entries;

    // Parse entries from the block
    size_t pos = 0;
    while (pos < size) {
        // Key length
        if (pos + 4 > size) break;
        uint32_t key_len;
        std::memcpy(&key_len, block.data() + pos, sizeof(uint32_t));
        pos += 4;

        // Key
        if (pos + key_len > size) break;
        std::string key(reinterpret_cast<char*>(block.data() + pos), key_len);
        pos += key_len;

        // Type
        if (pos + 1 > size) break;
        uint8_t type_byte = block[pos];
        pos += 1;

        // Sequence number
        if (pos + 8 > size) break;
        uint64_t seq;
        std::memcpy(&seq, block.data() + pos, sizeof(uint64_t));
        pos += 8;

        // Value length
        if (pos + 4 > size) break;
        uint32_t val_len;
        std::memcpy(&val_len, block.data() + pos, sizeof(uint32_t));
        pos += 4;

        // Value
        if (pos + val_len > size) break;
        std::string value(reinterpret_cast<char*>(block.data() + pos), val_len);
        pos += val_len;

        EntryType type = static_cast<EntryType>(type_byte);
        entries.emplace_back(std::move(key), std::move(value), type, seq);
    }

    return entries;
}

// ─── Get (Point Lookup) ──────────────────────────────────
// Searches for a key in this SSTable.
//
// Algorithm:
//   1. Quick range check: is key within [min_key, max_key]?
//   2. Binary search fence pointers to find the right block
//   3. Read that block from disk
//   4. Scan entries in the block for the key
//   5. Return the first match (newest version due to sort order)

GetResult SSTableReader::get(const Key& key) const {
    // Step 1: Range check
    if (!may_contain(key)) {
        return GetResult::NotFound();
    }

    // Step 2: Find the block
    int block_idx = find_block_for_key(key);
    if (block_idx < 0) {
        return GetResult::NotFound();
    }

    // Step 3: Read the block
    const IndexEntry& idx = index_[block_idx];
    auto entries = read_data_block(idx.block_offset, idx.block_size);

    // Step 4: Scan for the key
    for (const auto& entry : entries) {
        if (entry.key == key) {
            // Found! Check if it's a tombstone
            if (entry.type == EntryType::DELETE) {
                return GetResult::Deleted();
            }
            return GetResult::Found(entry.value);
        }
        // Since entries are sorted, if we've passed the key, it's not here
        if (entry.key > key) {
            break;
        }
    }

    return GetResult::NotFound();
}

// ─── May Contain ──────────────────────────────────────────
// Quick check: is the key within this SSTable's key range?

bool SSTableReader::may_contain(const Key& key) const {
    return key >= footer_.min_key && key <= footer_.max_key;
}

// ─── Get All Entries ──────────────────────────────────────
// Reads every data block and returns all entries in sorted order.
// Used during compaction.

std::vector<Entry> SSTableReader::get_all_entries() const {
    std::vector<Entry> all_entries;
    all_entries.reserve(footer_.entry_count);

    for (const auto& idx : index_) {
        auto block_entries = read_data_block(idx.block_offset, idx.block_size);
        all_entries.insert(all_entries.end(),
                          std::make_move_iterator(block_entries.begin()),
                          std::make_move_iterator(block_entries.end()));
    }

    return all_entries;
}

} // namespace lsm
