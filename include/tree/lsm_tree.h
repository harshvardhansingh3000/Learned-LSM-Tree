#pragma once

#include <vector>
#include <string>
#include <memory>
#include "common/types.h"
#include "common/config.h"
#include "memtable/memtable.h"
#include "wal/wal.h"
#include "tree/level.h"
#include "sstable/sstable_writer.h"

namespace lsm {

// ─── LSM-Tree ─────────────────────────────────────────────
//
// The main orchestrator that ties everything together:
//   MemTable + WAL + SSTables + Levels + Compaction
//
// This is the class that users interact with. It provides:
//   - Put(key, value)  — write a key-value pair
//   - Get(key)         — read a value by key
//   - Delete(key)      — delete a key
//
// Internal flow:
//
//   PUT("name", "Alice")
//     │
//     ├── 1. Write to WAL (crash safety)
//     ├── 2. Write to MemTable (in-memory, fast)
//     ├── 3. If MemTable full → flush to L0 SSTable
//     └── 4. If L0 full → compact into L1 → if L1 full → compact into L2 → ...
//
//   GET("name")
//     │
//     ├── 1. Check MemTable (newest data)
//     ├── 2. Check L0 SSTables (newest to oldest)
//     ├── 3. Check L1 SSTables
//     ├── 4. Check L2 SSTables
//     └── 5. ... until found or all levels checked
//
// Compaction (leveled):
//   When a level is full, pick an SSTable from it and merge into the next level.
//   This keeps SSTables sorted and non-overlapping in L1+.
//
//   L0 → L1: merge all L0 SSTables into L1
//   L1 → L2: pick one L1 SSTable, merge with overlapping L2 SSTables
//   L2 → L3: same pattern

class LSMTree {
public:
    // Create/open an LSM-tree with the given configuration.
    // If data directory exists, recovers state from WAL and existing SSTables.
    explicit LSMTree(const Config& config = default_config());

    // Destructor: flushes MemTable if non-empty, closes WAL
    ~LSMTree();

    // ── Write Operations ──────────────────────────────────

    // Insert or update a key-value pair.
    void put(const Key& key, const Value& value);

    // Delete a key (writes a tombstone).
    void remove(const Key& key);

    // ── Read Operations ───────────────────────────────────

    // Look up a key. Returns the value if found, empty string if not.
    // Searches: MemTable → L0 → L1 → L2 → ...
    GetResult get(const Key& key);

    // Range scan: returns all key-value pairs where start_key <= key <= end_key.
    // Results are sorted by key. Only returns the newest version of each key.
    // Tombstones (deleted keys) are excluded from results.
    std::vector<std::pair<Key, Value>> scan(const Key& start_key, const Key& end_key);

    // ── Maintenance ───────────────────────────────────────

    // Force flush the current MemTable to disk (even if not full).
    void flush();

    // Run compaction on all levels that need it.
    void compact();

    // ── Statistics ─────────────────────────────────────────

    // Total number of entries across all levels (approximate)
    size_t total_entries() const;

    // Number of levels currently in use
    size_t num_levels() const { return levels_.size(); }

    // Number of SSTables across all levels
    size_t total_sstables() const;

    // ── No Copy ───────────────────────────────────────────
    LSMTree(const LSMTree&) = delete;
    LSMTree& operator=(const LSMTree&) = delete;

private:
    // Flush the current MemTable to an L0 SSTable
    void flush_memtable();

    // Compact a specific level into the next level
    void compact_level(int level_num);

    // Ensure we have enough levels (creates new ones as needed)
    void ensure_level(int level_num);

    // Merge entries from multiple sources into sorted, deduplicated entries
    // Keeps only the newest version of each key
    std::vector<Entry> merge_entries(const std::vector<Entry>& a,
                                     const std::vector<Entry>& b);

    Config config_;
    std::unique_ptr<MemTable> memtable_;     // Current active MemTable
    std::unique_ptr<WAL> wal_;               // Write-ahead log
    std::vector<std::unique_ptr<Level>> levels_;  // L0, L1, L2, ...
    uint64_t sequence_number_;               // Global sequence counter
};

} // namespace lsm
