#pragma once

#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include "common/types.h"
#include "common/config.h"
#include "sstable/sstable_reader.h"
#include "bloom/bloom_filter.h"

namespace lsm {

// ─── Level ────────────────────────────────────────────────
//
// A Level represents one tier in the LSM-tree hierarchy.
// Each level contains a collection of SSTables.
//
// Level structure:
//   Level 0: ~1 MB   — direct flush target, SSTables MAY overlap
//   Level 1: ~10 MB  — non-overlapping, sorted SSTables
//   Level 2: ~100 MB — non-overlapping, sorted SSTables
//   Level 3: ~1 GB   — non-overlapping, sorted SSTables
//
// Level 0 is special:
//   - SSTables are added directly from MemTable flushes
//   - They CAN have overlapping key ranges (because each flush is independent)
//   - Must search ALL L0 SSTables (newest first) during reads
//
// Levels 1+:
//   - SSTables are NON-OVERLAPPING (each key exists in exactly one SSTable)
//   - Created by compaction (merging SSTables from the level above)
//   - Can use binary search on SSTable key ranges for fast lookup
//
// Compaction trigger:
//   When a level's total size exceeds its capacity, we pick an SSTable
//   and merge it into the next level.

class Level {
public:
    // Create a level with the given number and capacity
    // level_num: 0, 1, 2, 3, ...
    // max_size_bytes: maximum total size of all SSTables in this level
    Level(int level_num, size_t max_size_bytes, const std::string& base_dir);

    // ── SSTable Management ────────────────────────────────

    // Add an SSTable to this level (used after flush or compaction)
    void add_sstable(const std::string& sstable_path);

    // Remove an SSTable from this level (used during compaction)
    void remove_sstable(const std::string& sstable_path);

    // ── Lookup ────────────────────────────────────────────

    // Search for a key in this level.
    // For L0: searches all SSTables (newest first)
    // For L1+: binary search on key ranges to find the right SSTable
    GetResult get(const Key& key) const;

    // ── Compaction ────────────────────────────────────────

    // Is this level full? (total size exceeds capacity)
    bool needs_compaction() const;

    // Get all entries from all SSTables in this level (for compaction)
    std::vector<Entry> get_all_entries() const;

    // Get the SSTables whose key ranges overlap with [min_key, max_key]
    std::vector<std::string> get_overlapping_sstables(
        const Key& min_key, const Key& max_key) const;

    // ── Metadata ──────────────────────────────────────────

    int level_num() const { return level_num_; }
    size_t max_size_bytes() const { return max_size_bytes_; }
    size_t current_size_bytes() const;
    size_t sstable_count() const { return sstables_.size(); }
    bool empty() const { return sstables_.empty(); }

    // Get all SSTable file paths
    std::vector<std::string> sstable_paths() const;

    // ── File Naming ───────────────────────────────────────

    // Generate a unique SSTable file path for this level
    std::string next_sstable_path();

private:
    int level_num_;                                    // 0, 1, 2, 3, ...
    size_t max_size_bytes_;                            // Capacity of this level
    std::string base_dir_;                             // e.g., "data/sstables"
    std::vector<std::unique_ptr<SSTableReader>> sstables_;  // SSTables in this level
    uint64_t next_sstable_id_;                         // For generating unique file names
};

} // namespace lsm
