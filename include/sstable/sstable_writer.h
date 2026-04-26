#pragma once

#include <string>
#include <fstream>
#include <vector>
#include "common/types.h"
#include "common/config.h"
#include "sstable/sstable_format.h"
#include "bloom/bloom_filter.h"

namespace lsm {

// ─── SSTable Writer ───────────────────────────────────────
//
// Writes a sorted list of entries to an SSTable file on disk.
//
// Usage (called during MemTable flush):
//   1. Create an SSTableWriter with a file path
//   2. Call write_entries() with sorted entries from the MemTable
//   3. The writer creates the complete SSTable file:
//      - Data blocks (sorted entries, split into 4 KB chunks)
//      - Index block (fence pointers for each data block)
//      - Footer (metadata: index offset, entry count, min/max keys)
//
// The writer is a one-shot object: create it, write once, done.
// Each flush creates a new SSTable file with a unique name.

class SSTableWriter {
public:
    // Create a writer that will write to the given file path.
    // block_size = target size for each data block (default 4 KB).
    explicit SSTableWriter(const std::string& file_path,
                           size_t block_size = 4 * 1024);

    // Write all entries to the SSTable file.
    // Entries MUST be sorted (by key ascending, then seq descending).
    // This creates the complete file: data blocks + index + footer.
    // Returns true on success, false on failure.
    bool write_entries(const std::vector<Entry>& entries);

    // Get the file path
    const std::string& file_path() const { return file_path_; }

    // ── No Copy ───────────────────────────────────────────
    SSTableWriter(const SSTableWriter&) = delete;
    SSTableWriter& operator=(const SSTableWriter&) = delete;

private:
    // Write a single entry to the current data block buffer
    // Returns the number of bytes written
    size_t write_entry_to_buffer(std::vector<uint8_t>& buffer, const Entry& entry);

    // Write raw bytes to the output file
    void write_bytes(const void* data, size_t size);

    // Write a uint32_t to the output file
    void write_uint32(uint32_t value);

    // Write a uint64_t to the output file
    void write_uint64(uint64_t value);

    // Write a string (length-prefixed) to the output file
    void write_string(const std::string& str);

    std::string file_path_;
    size_t block_size_;
    std::ofstream out_;
};

} // namespace lsm
