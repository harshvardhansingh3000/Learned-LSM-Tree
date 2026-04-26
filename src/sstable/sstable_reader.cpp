#include "sstable/sstable_reader.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace lsm {

// ─── Low-Level Read Helpers ──────────────────────────────

bool SSTableReader::read_bytes(void* data, size_t size, std::ifstream& in) {
    in.read(static_cast<char*>(data), size);
    return static_cast<size_t>(in.gcount()) == size;
}

bool SSTableReader::read_uint32(uint32_t& value, std::ifstream& in) {
    return read_bytes(&value, sizeof(uint32_t), in);
}

bool SSTableReader::read_uint64(uint64_t& value, std::ifstream& in) {
    return read_bytes(&value, sizeof(uint64_t), in);
}

bool SSTableReader::read_string(std::string& str, std::ifstream& in) {
    uint32_t len;
    if (!read_uint32(len, in)) return false;
    if (len > 10 * 1024 * 1024) return false;
    str.resize(len);
    if (len > 0) {
        if (!read_bytes(str.data(), len, in)) return false;
    }
    return true;
}

// ─── Constructor ──────────────────────────────────────────

SSTableReader::SSTableReader(const std::string& file_path)
    : file_path_(file_path)
{
    read_footer();
    read_index();
    read_bloom_filter();
}

// ─── Move Constructor ─────────────────────────────────────

SSTableReader::SSTableReader(SSTableReader&& other) noexcept
    : file_path_(std::move(other.file_path_))
    , footer_(std::move(other.footer_))
    , index_(std::move(other.index_))
    , bloom_(std::move(other.bloom_))
{
}

SSTableReader& SSTableReader::operator=(SSTableReader&& other) noexcept {
    if (this != &other) {
        file_path_ = std::move(other.file_path_);
        footer_ = std::move(other.footer_);
        index_ = std::move(other.index_);
        bloom_ = std::move(other.bloom_);
    }
    return *this;
}

// ─── Read Footer ──────────────────────────────────────────
// New approach (inspired by go-lsm):
//   1. Read last 8 bytes: [bloom_offset (4)] [magic (4)]
//   2. Validate magic
//   3. Read bloom filter from bloom_offset (handled separately)
//   4. The footer is between the bloom filter block and bloom_offset/magic
//      But we don't know where the footer starts...
//      Actually, we DO: the footer starts right after the bloom filter block.
//      And the footer contains index_offset as its first field.
//      We can find the footer by scanning from bloom_offset backwards.
//
// Simpler: the file layout is:
//   [data blocks][index block][bloom block][footer][bloom_offset(4)][magic(4)]
//
// The footer's first field is index_offset(8), and we can validate it.
// The footer starts right after the bloom block ends.
// We know bloom_offset, and the bloom block is: [k(4)][size(4)][data(size)]
// So bloom_block_end = bloom_offset + 4 + 4 + bloom_data_size
// Footer starts at bloom_block_end.
//
// But we don't know bloom_data_size without reading the bloom block first.
// So: read bloom block header → compute footer start → read footer.

void SSTableReader::read_footer() {
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open SSTable file: " + file_path_);
    }

    // Get file size
    in.seekg(0, std::ios::end);
    auto file_size = in.tellg();
    if (file_size < 8) {
        throw std::runtime_error("SSTable file too small: " + file_path_);
    }

    // Step 1: Read last 8 bytes — bloom_offset(4) + magic(4)
    in.seekg(-8, std::ios::end);
    uint32_t bloom_offset_val;
    if (!read_uint32(bloom_offset_val, in)) {
        throw std::runtime_error("Cannot read bloom_offset: " + file_path_);
    }
    uint32_t magic;
    if (!read_uint32(magic, in) || magic != SSTABLE_MAGIC) {
        throw std::runtime_error("Invalid SSTable magic number: " + file_path_);
    }

    footer_.bloom_offset = bloom_offset_val;
    footer_.magic = magic;

    // Step 2: Read bloom block header to find where footer starts
    // Bloom block format: [k (4)] [num_bits (4)] [data_size (4)] [data...]
    in.seekg(bloom_offset_val);
    uint32_t bloom_k;
    if (!read_uint32(bloom_k, in)) {
        throw std::runtime_error("Cannot read bloom k: " + file_path_);
    }
    uint32_t bloom_num_bits;
    if (!read_uint32(bloom_num_bits, in)) {
        throw std::runtime_error("Cannot read bloom num_bits: " + file_path_);
    }
    uint32_t bloom_data_size;
    if (!read_uint32(bloom_data_size, in)) {
        throw std::runtime_error("Cannot read bloom data size: " + file_path_);
    }

    // Skip bloom data to get to footer
    uint64_t footer_start = static_cast<uint64_t>(bloom_offset_val) + 4 + 4 + 4 + bloom_data_size;
    in.seekg(footer_start);

    // Step 3: Read footer fields sequentially
    if (!read_uint64(footer_.index_offset, in)) {
        throw std::runtime_error("Cannot read index_offset: " + file_path_);
    }
    if (!read_uint32(footer_.index_size, in)) {
        throw std::runtime_error("Cannot read index_size: " + file_path_);
    }
    if (!read_uint32(footer_.entry_count, in)) {
        throw std::runtime_error("Cannot read entry_count: " + file_path_);
    }
    if (!read_string(footer_.min_key, in)) {
        throw std::runtime_error("Cannot read min_key: " + file_path_);
    }
    if (!read_string(footer_.max_key, in)) {
        throw std::runtime_error("Cannot read max_key: " + file_path_);
    }
}

// ─── Read Bloom Filter ────────────────────────────────────
// Read the bloom filter from the exact position stored in footer_.bloom_offset.
// Format: [k (4)] [data_size (4)] [data (data_size)]
// No scanning, no guessing — we know exactly where it is!

void SSTableReader::read_bloom_filter() {
    if (footer_.bloom_offset == 0) return;

    try {
        std::ifstream in(file_path_, std::ios::binary);
        if (!in.is_open()) return;

        // Bloom block format: [k (4)] [num_bits (4)] [data_size (4)] [data...]
        in.seekg(footer_.bloom_offset);

        uint32_t k;
        if (!read_uint32(k, in)) return;
        if (k < 1 || k > 30) return;

        uint32_t num_bits;
        if (!read_uint32(num_bits, in)) return;

        uint32_t data_size;
        if (!read_uint32(data_size, in)) return;
        if (data_size == 0 || data_size > 10 * 1024 * 1024) return;

        std::vector<uint8_t> bloom_data(data_size);
        if (!read_bytes(bloom_data.data(), data_size, in)) return;

        bloom_ = std::make_unique<BloomFilter>(bloom_data, k, num_bits);
    } catch (...) {
        bloom_.reset();
    }
}

// ─── Read Index ───────────────────────────────────────────

void SSTableReader::read_index() {
    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open SSTable file: " + file_path_);
    }

    in.seekg(footer_.index_offset);

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

int SSTableReader::find_block_for_key(const Key& key) const {
    if (index_.empty()) return -1;

    int lo = 0;
    int hi = static_cast<int>(index_.size()) - 1;
    int result = -1;

    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (index_[mid].first_key <= key) {
            result = mid;
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }

    return result;
}

// ─── Read Data Block ──────────────────────────────────────

std::vector<Entry> SSTableReader::read_data_block(uint64_t offset, uint32_t size) const {
    std::vector<Entry> entries;

    std::ifstream in(file_path_, std::ios::binary);
    if (!in.is_open()) return entries;

    in.seekg(offset);

    std::vector<uint8_t> block(size);
    in.read(reinterpret_cast<char*>(block.data()), size);
    if (!in.good() && !in.eof()) return entries;

    size_t pos = 0;
    while (pos < size) {
        if (pos + 4 > size) break;
        uint32_t key_len;
        std::memcpy(&key_len, block.data() + pos, sizeof(uint32_t));
        pos += 4;

        if (pos + key_len > size) break;
        std::string key(reinterpret_cast<char*>(block.data() + pos), key_len);
        pos += key_len;

        if (pos + 1 > size) break;
        uint8_t type_byte = block[pos];
        pos += 1;

        if (pos + 8 > size) break;
        uint64_t seq;
        std::memcpy(&seq, block.data() + pos, sizeof(uint64_t));
        pos += 8;

        if (pos + 4 > size) break;
        uint32_t val_len;
        std::memcpy(&val_len, block.data() + pos, sizeof(uint32_t));
        pos += 4;

        if (pos + val_len > size) break;
        std::string value(reinterpret_cast<char*>(block.data() + pos), val_len);
        pos += val_len;

        EntryType type = static_cast<EntryType>(type_byte);
        entries.emplace_back(std::move(key), std::move(value), type, seq);
    }

    return entries;
}

// ─── Get (Point Lookup) ──────────────────────────────────

GetResult SSTableReader::get(const Key& key) const {
    if (!may_contain(key)) {
        return GetResult::NotFound();
    }

    int block_idx = find_block_for_key(key);
    if (block_idx < 0) {
        return GetResult::NotFound();
    }

    const IndexEntry& idx = index_[block_idx];
    auto entries = read_data_block(idx.block_offset, idx.block_size);

    for (const auto& entry : entries) {
        if (entry.key == key) {
            if (entry.type == EntryType::DELETE) {
                return GetResult::Deleted();
            }
            return GetResult::Found(entry.value);
        }
        if (entry.key > key) {
            break;
        }
    }

    return GetResult::NotFound();
}

// ─── May Contain ──────────────────────────────────────────
// Uses Bloom filter (if available) + key range check.

bool SSTableReader::may_contain(const Key& key) const {
    // Range check first
    if (key < footer_.min_key || key > footer_.max_key) {
        return false;
    }

    // Bloom filter check
    if (bloom_ && !bloom_->may_contain(key)) {
        return false;
    }

    return true;
}

// ─── Get All Entries ──────────────────────────────────────

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
