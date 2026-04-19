#include "sstable/sstable_writer.h"
#include <filesystem>
#include <stdexcept>
#include <cstring>

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
// Stores the file path and block size. File is opened in write_entries().

SSTableWriter::SSTableWriter(const std::string& file_path, size_t block_size)
    : file_path_(file_path)
    , block_size_(block_size)
{
}

// ─── Low-Level Write Helpers ──────────────────────────────

void SSTableWriter::write_bytes(const void* data, size_t size) {
    out_.write(static_cast<const char*>(data), size);
}

void SSTableWriter::write_uint32(uint32_t value) {
    write_bytes(&value, sizeof(uint32_t));
}

void SSTableWriter::write_uint64(uint64_t value) {
    write_bytes(&value, sizeof(uint64_t));
}

// Writes a string as: [length (4 bytes)] [string data]
void SSTableWriter::write_string(const std::string& str) {
    uint32_t len = static_cast<uint32_t>(str.size());
    write_uint32(len);
    if (len > 0) {
        write_bytes(str.data(), len);
    }
}

// ─── Write Entry to Buffer ───────────────────────────────
// Serializes one Entry into raw bytes and appends to a buffer.
//
// Entry format in a data block:
// ┌──────────┬──────┬──────┬──────────┬──────────┬───────┐
// │ key_len  │ key  │ type │ seq_num  │ val_len  │ value │
// │ (4 bytes)│(var) │(1 b) │ (8 bytes)│ (4 bytes)│ (var) │
// └──────────┴──────┴──────┴──────────┴──────────┴───────┘
//
// Example: Entry("cat", "meow", PUT, seq=42)
//   [0x03,0,0,0] [c,a,t] [0x00] [42,0,0,0,0,0,0,0] [0x04,0,0,0] [m,e,o,w]
//   = 4 + 3 + 1 + 8 + 4 + 4 = 24 bytes

size_t SSTableWriter::write_entry_to_buffer(std::vector<uint8_t>& buffer, const Entry& entry) {
    size_t start_size = buffer.size();

    // Key length (4 bytes)
    uint32_t key_len = static_cast<uint32_t>(entry.key.size());
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&key_len),
                  reinterpret_cast<const uint8_t*>(&key_len) + sizeof(uint32_t));

    // Key data
    buffer.insert(buffer.end(), entry.key.begin(), entry.key.end());

    // Entry type (1 byte)
    uint8_t type = static_cast<uint8_t>(entry.type);
    buffer.push_back(type);

    // Sequence number (8 bytes)
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&entry.sequence_number),
                  reinterpret_cast<const uint8_t*>(&entry.sequence_number) + sizeof(uint64_t));

    // Value length (4 bytes)
    uint32_t val_len = static_cast<uint32_t>(entry.value.size());
    buffer.insert(buffer.end(),
                  reinterpret_cast<const uint8_t*>(&val_len),
                  reinterpret_cast<const uint8_t*>(&val_len) + sizeof(uint32_t));

    // Value data
    if (!entry.value.empty()) {
        buffer.insert(buffer.end(), entry.value.begin(), entry.value.end());
    }

    return buffer.size() - start_size;
}

// ─── Write Entries ────────────────────────────────────────
// The main function that creates a complete SSTable file.
//
// Algorithm:
//   1. Create parent directories
//   2. Open the output file
//   3. For each entry:
//      a. Serialize it into the current data block buffer
//      b. If the buffer exceeds block_size, flush it as a data block
//      c. Record the fence pointer (first key) and offset
//   4. Flush any remaining entries as the last data block
//   5. Write the index block (all fence pointers)
//   6. Write the footer (metadata)
//   7. Close the file
//
// Visual:
//   entries: [a, b, c, d, e, f, g, h, i, j]
//
//   Block 0: [a, b, c]  → fence_ptr = "a", offset = 0
//   Block 1: [d, e, f]  → fence_ptr = "d", offset = 350
//   Block 2: [g, h, i, j] → fence_ptr = "g", offset = 700
//
//   Index: [("a", 0, 350), ("d", 350, 350), ("g", 700, 400)]
//   Footer: {index_offset=1100, index_size=..., count=10, min="a", max="j"}

bool SSTableWriter::write_entries(const std::vector<Entry>& entries) {
    if (entries.empty()) {
        return false;
    }

    // Step 1: Create parent directories
    std::filesystem::path dir = std::filesystem::path(file_path_).parent_path();
    if (!dir.empty()) {
        std::filesystem::create_directories(dir);
    }

    // Step 2: Open the output file
    out_.open(file_path_, std::ios::binary | std::ios::trunc);
    if (!out_.is_open()) {
        return false;
    }

    // Track index entries (fence pointers)
    std::vector<IndexEntry> index_entries;

    // Current data block buffer
    std::vector<uint8_t> block_buffer;
    std::string block_first_key;       // Fence pointer for current block
    uint64_t block_start_offset = 0;   // Where current block starts in file
    uint64_t current_offset = 0;       // Current write position in file

    // Step 3: Process each entry
    for (size_t i = 0; i < entries.size(); i++) {
        const Entry& entry = entries[i];

        // If this is the first entry in a new block, record its key as fence pointer
        if (block_buffer.empty()) {
            block_first_key = entry.key;
            block_start_offset = current_offset;
        }

        // Serialize the entry into the block buffer
        write_entry_to_buffer(block_buffer, entry);

        // Check if the block is full (exceeds target block size)
        // OR if this is the last entry (must flush remaining data)
        bool is_last_entry = (i == entries.size() - 1);
        bool block_full = (block_buffer.size() >= block_size_);

        if (block_full || is_last_entry) {
            // Step 3b: Flush the block to file
            write_bytes(block_buffer.data(), block_buffer.size());

            // Record the index entry (fence pointer)
            IndexEntry idx;
            idx.first_key = block_first_key;
            idx.block_offset = block_start_offset;
            idx.block_size = static_cast<uint32_t>(block_buffer.size());
            index_entries.push_back(idx);

            // Update current offset
            current_offset += block_buffer.size();

            // Reset block buffer for next block
            block_buffer.clear();
        }
    }

    // Step 5: Write the index block
    uint64_t index_offset = current_offset;

    for (const auto& idx : index_entries) {
        // Write each index entry: first_key (string) + offset (8) + size (4)
        write_string(idx.first_key);
        write_uint64(idx.block_offset);
        write_uint32(idx.block_size);
    }

    // Calculate index size
    uint64_t after_index = static_cast<uint64_t>(out_.tellp());
    uint32_t index_size = static_cast<uint32_t>(after_index - index_offset);

    // Step 6: Write the footer
    // index_offset (8 bytes)
    write_uint64(index_offset);

    // index_size (4 bytes)
    write_uint32(index_size);

    // entry_count (4 bytes)
    write_uint32(static_cast<uint32_t>(entries.size()));

    // min_key (string: length + data)
    write_string(entries.front().key);

    // max_key (string: length + data)
    // Find the actual max key (last unique key in sorted order)
    write_string(entries.back().key);

    // magic number (4 bytes)
    write_uint32(SSTABLE_MAGIC);

    // Step 7: Close the file
    out_.flush();
    out_.close();

    return true;
}

} // namespace lsm
