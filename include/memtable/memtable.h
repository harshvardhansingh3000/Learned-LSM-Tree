#pragma once

#include "memtable/skip_list.h"
#include "common/types.h"
#include "common/config.h"
#include <vector>
#include <atomic>

namespace lsm {

// ─── MemTable ─────────────────────────────────────────────
//
// The MemTable is the write buffer of the LSM-tree.
// All writes (Put/Delete) go here first before being flushed to disk.
//
// It wraps a Skip List and adds:
//   1. Automatic sequence number assignment (monotonically increasing)
//   2. Size threshold checking (is it time to flush?)
//   3. A cleaner API: Put(key, value), Get(key), Delete(key)
//
// Lifecycle:
//   1. Created empty
//   2. Receives Put/Delete operations
//   3. When size exceeds threshold (1 MB), it becomes "immutable"
//   4. The immutable MemTable is flushed to disk as an SSTable
//   5. A new empty MemTable takes its place
//
// The MemTable does NOT handle flushing itself — that's the LSM-tree's job.
// The MemTable just says "I'm full" via should_flush().

class MemTable {
public:
    // Create a MemTable with the given configuration
    explicit MemTable(const Config& config = default_config());

    // ── Write Operations ──────────────────────────────────

    // Insert a key-value pair.
    // Automatically assigns the next sequence number.
    // Returns the assigned sequence number.
    uint64_t put(const Key& key, const Value& value);

    // Insert an entry with a specific sequence number (used by LSMTree for global ordering).
    void insert_entry(const Entry& entry);

    // Mark a key as deleted (writes a tombstone).
    // Automatically assigns the next sequence number.
    // Returns the assigned sequence number.
    uint64_t remove(const Key& key);

    // ── Read Operations ───────────────────────────────────

    // Look up a key. Returns Found(value), Deleted(), or NotFound().
    GetResult get(const Key& key) const;

    // ── Flush Support ─────────────────────────────────────

    // Should this MemTable be flushed to disk?
    // Returns true when memory usage exceeds the configured threshold.
    bool should_flush() const;

    // Get all entries in sorted order (for flushing to SSTable).
    // Entries are sorted by key ascending, then sequence number descending.
    std::vector<Entry> get_all_entries() const;

    // ── Statistics ─────────────────────────────────────────

    // Number of entries (including tombstones)
    size_t count() const;

    // Approximate memory usage in bytes
    size_t memory_usage() const;

    // Is the MemTable empty?
    bool empty() const;

    // Current sequence number (for debugging/testing)
    uint64_t current_sequence_number() const { return sequence_number_; }

    // ── No Copy ───────────────────────────────────────────
    MemTable(const MemTable&) = delete;
    MemTable& operator=(const MemTable&) = delete;

private:
    SkipList skip_list_;           // The underlying sorted data structure
    uint64_t sequence_number_;     // Monotonically increasing counter
    size_t max_size_;              // Flush threshold in bytes
};

} // namespace lsm
