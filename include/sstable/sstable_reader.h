#pragma once

#include <string>
#include <fstream>
#include <vector>
#include "common/types.h"
#include "sstable/sstable_format.h"
#include "bloom/bloom_filter.h"
#include <memory>

namespace lsm {

// ─── SSTable Reader ───────────────────────────────────────
//
// Reads an SSTable file from disk and supports point lookups.
//
// How a lookup works:
//   1. Open the file and read the footer (metadata at the end)
//   2. Read the index block (fence pointers for each data block)
//   3. Binary search the fence pointers to find which block might contain the key
//   4. Read that one data block and scan for the key
//
// This means a point lookup requires only 2 disk reads:
//   - One for the index (or cached in memory)
//   - One for the data block
//
// The reader also supports:
//   - Getting all entries (for compaction — merging SSTables)
//   - Checking min/max key range (to skip irrelevant SSTables)

class SSTableReader {
public:
    // Open an SSTable file for reading.
    // Reads the footer and index block into memory immediately.
    // Throws if the file is invalid or corrupted.
    explicit SSTableReader(const std::string& file_path);

    // ── Point Lookup ──────────────────────────────────────

    // Search for a key in this SSTable.
    // Uses fence pointers to find the right block, then scans within it.
    // Returns Found(value), Deleted(), or NotFound().
    GetResult get(const Key& key) const;

    // ── Range Check ───────────────────────────────────────

    // Could this SSTable possibly contain the given key?
    // Quick check using min/max keys — avoids reading any blocks.
    bool may_contain(const Key& key) const;

    // ── Full Scan ─────────────────────────────────────────

    // Read ALL entries from this SSTable (for compaction).
    // Returns entries in sorted order.
    std::vector<Entry> get_all_entries() const;

    // ── Metadata ──────────────────────────────────────────

    const std::string& file_path() const { return file_path_; }
    uint32_t entry_count() const { return footer_.entry_count; }
    const std::string& min_key() const { return footer_.min_key; }
    const std::string& max_key() const { return footer_.max_key; }

    // ── No Copy ───────────────────────────────────────────
    SSTableReader(const SSTableReader&) = delete;
    SSTableReader& operator=(const SSTableReader&) = delete;

    // Allow move (for storing in vectors)
    SSTableReader(SSTableReader&& other) noexcept;
    SSTableReader& operator=(SSTableReader&& other) noexcept;

private:
    // Read the footer from the end of the file
    void read_footer();

    // Read the index block using info from the footer
    void read_index();

    // Read a specific data block and parse its entries
    std::vector<Entry> read_data_block(uint64_t offset, uint32_t size) const;

    // Find which data block might contain the key (binary search on fence pointers)
    int find_block_for_key(const Key& key) const;

    // Low-level read helpers
    static bool read_bytes(void* data, size_t size, std::ifstream& in);
    static bool read_uint32(uint32_t& value, std::ifstream& in);
    static bool read_uint64(uint64_t& value, std::ifstream& in);
    static bool read_string(std::string& str, std::ifstream& in);

    // Read the Bloom filter from the file using bloom_offset from footer
    void read_bloom_filter();

    std::string file_path_;
    Footer footer_;                      // Metadata (read once on open)
    std::vector<IndexEntry> index_;      // Fence pointers (read once on open)
    std::unique_ptr<BloomFilter> bloom_; // Bloom filter (read once on open)
};

} // namespace lsm
