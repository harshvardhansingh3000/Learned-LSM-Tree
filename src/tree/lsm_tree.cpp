#include "tree/lsm_tree.h"
#include <algorithm>
#include <filesystem>
#include <iostream>

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
// Creates or opens an LSM-tree.
//
// Steps:
//   1. Store configuration
//   2. Create data directories
//   3. Create WAL
//   4. Create MemTable
//   5. Create Level 0 (always exists)
//   6. Recover from WAL if it has data (crash recovery)

LSMTree::LSMTree(const Config& config)
    : config_(config)
    , sequence_number_(0)
{
    // Create data directories
    std::filesystem::create_directories(config_.data_dir);
    std::filesystem::create_directories(config_.wal_dir);
    std::filesystem::create_directories(config_.sstable_dir);

    // Create WAL
    std::string wal_path = config_.wal_dir + "/wal.log";
    wal_ = std::make_unique<WAL>(wal_path);

    // Create MemTable
    memtable_ = std::make_unique<MemTable>(config_);

    // Create Level 0 (always exists)
    ensure_level(0);

    // Recover from WAL (replay any entries from a previous crash)
    auto wal_entries = wal_->replay();
    for (const auto& entry : wal_entries) {
        memtable_->put(entry.key, entry.value);
        if (entry.sequence_number > sequence_number_) {
            sequence_number_ = entry.sequence_number;
        }
    }
}

// ─── Destructor ───────────────────────────────────────────
// Flushes the MemTable if it has data, ensuring nothing is lost.

LSMTree::~LSMTree() {
    if (memtable_ && !memtable_->empty()) {
        try {
            flush_memtable();
        } catch (...) {
            // Don't throw from destructor — data is in WAL anyway
        }
    }
}

// ─── Put ──────────────────────────────────────────────────
// Write a key-value pair to the database.
//
// Flow:
//   1. Write to WAL (crash safety — if we crash after this, data is recoverable)
//   2. Write to MemTable (fast, in-memory)
//   3. If MemTable is full → flush to L0 SSTable
//   4. If any level needs compaction → compact

void LSMTree::put(const Key& key, const Value& value) {
    // Step 1: WAL first (durability)
    wal_->append_put(key, value);

    // Step 2: MemTable — insert with GLOBAL sequence number
    sequence_number_++;
    memtable_->insert_entry(Entry(key, value, EntryType::PUT, sequence_number_));

    // Step 3: Check if MemTable needs flushing
    if (memtable_->should_flush()) {
        flush_memtable();

        // Step 4: Check if any level needs compaction
        compact();
    }
}

// ─── Remove (Delete) ─────────────────────────────────────
// Delete a key by writing a tombstone.
// Same flow as put() but with a DELETE entry.

void LSMTree::remove(const Key& key) {
    // Step 1: WAL first
    wal_->append_delete(key);

    // Step 2: MemTable (tombstone with GLOBAL sequence number)
    sequence_number_++;
    memtable_->insert_entry(Entry(key, "", EntryType::DELETE, sequence_number_));

    // Step 3: Check if MemTable needs flushing
    if (memtable_->should_flush()) {
        flush_memtable();
        compact();
    }
}

// ─── Get ──────────────────────────────────────────────────
// Look up a key in the database.
//
// Search order (newest data first):
//   1. MemTable (most recent writes)
//   2. Level 0 (newest SSTables, may overlap)
//   3. Level 1 (non-overlapping SSTables)
//   4. Level 2
//   5. ... until found or all levels checked
//
// If we find a tombstone (DELETE), we stop and return "not found"
// because the key was intentionally deleted.

GetResult LSMTree::get(const Key& key) {
    // Step 1: Check MemTable
    GetResult result = memtable_->get(key);
    if (result.found) {
        if (result.is_deleted) {
            // Tombstone found — key was deleted
            return GetResult::NotFound();
        }
        return result;
    }

    // Step 2-N: Check each level
    for (const auto& level : levels_) {
        result = level->get(key);
        if (result.found) {
            if (result.is_deleted) {
                // Tombstone found in SSTable — key was deleted
                return GetResult::NotFound();
            }
            return result;
        }
    }

    // Not found anywhere
    return GetResult::NotFound();
}

// ─── Scan (Range Query) ───────────────────────────────────
// Returns all key-value pairs where start_key <= key <= end_key.
//
// Algorithm:
//   1. Collect ALL entries from MemTable + all levels
//   2. Sort and deduplicate (keep newest version of each key)
//   3. Filter to the requested range [start_key, end_key]
//   4. Exclude tombstones (deleted keys)
//
// This is not the most efficient implementation (reads everything),
// but it's correct. A production system would use merge iterators.

std::vector<std::pair<Key, Value>> LSMTree::scan(const Key& start_key, const Key& end_key) {
    // Step 1: Collect all entries
    std::vector<Entry> all_entries;

    // From MemTable
    auto mem_entries = memtable_->get_all_entries();
    all_entries.insert(all_entries.end(), mem_entries.begin(), mem_entries.end());

    // From all levels
    for (const auto& level : levels_) {
        auto level_entries = level->get_all_entries();
        all_entries.insert(all_entries.end(),
                          std::make_move_iterator(level_entries.begin()),
                          std::make_move_iterator(level_entries.end()));
    }

    // Step 2: Sort and deduplicate
    std::sort(all_entries.begin(), all_entries.end());

    // Step 3 & 4: Filter range and exclude tombstones
    std::vector<std::pair<Key, Value>> results;
    std::string last_key;

    for (const auto& entry : all_entries) {
        // Skip duplicates (we already have the newest version)
        if (entry.key == last_key) continue;
        last_key = entry.key;

        // Filter to range
        if (entry.key < start_key) continue;
        if (entry.key > end_key) break;  // Past the range, done (entries are sorted)

        // Exclude tombstones
        if (entry.type == EntryType::DELETE) continue;

        results.emplace_back(entry.key, entry.value);
    }

    return results;
}

// ─── Flush (Public) ───────────────────────────────────────
// Force flush the current MemTable to disk.

void LSMTree::flush() {
    if (!memtable_->empty()) {
        flush_memtable();
    }
}

// ─── Flush MemTable (Private) ─────────────────────────────
// Writes the current MemTable to a new SSTable in Level 0.
//
// Steps:
//   1. Get all sorted entries from MemTable
//   2. Write them to a new SSTable file in L0
//   3. Add the SSTable to Level 0
//   4. Clear the WAL (data is now safely in the SSTable)
//   5. Create a new empty MemTable
//
// Visual:
//   MemTable (8800 entries, 1 MB)
//       │
//       ▼
//   SSTableWriter → "data/sstables/L0_001.sst"
//       │
//       ▼
//   Level 0: [L0_001.sst]
//       │
//       ▼
//   WAL cleared, new empty MemTable created

void LSMTree::flush_memtable() {
    // Step 1: Get sorted entries
    auto entries = memtable_->get_all_entries();
    if (entries.empty()) return;

    // Step 2: Write to SSTable
    ensure_level(0);
    std::string sst_path = levels_[0]->next_sstable_path();
    SSTableWriter writer(sst_path);
    if (!writer.write_entries(entries)) {
        throw std::runtime_error("Failed to write SSTable: " + sst_path);
    }

    // Step 3: Add to Level 0
    levels_[0]->add_sstable(sst_path);

    // Step 4: Clear WAL
    wal_->clear();

    // Step 5: New MemTable
    memtable_ = std::make_unique<MemTable>(config_);
}

// ─── Compact (Public) ─────────────────────────────────────
// Checks all levels and compacts any that are full.
// Compaction cascades: if compacting L0→L1 makes L1 full, then L1→L2, etc.

void LSMTree::compact() {
    for (size_t i = 0; i < levels_.size(); i++) {
        if (levels_[i]->needs_compaction()) {
            compact_level(static_cast<int>(i));
        }
    }
}

// ─── Compact Level (Private) ──────────────────────────────
// Merges SSTables from level_num into level_num + 1.
//
// Algorithm:
//   1. Get all entries from the source level
//   2. Get all entries from overlapping SSTables in the target level
//   3. Merge them (keep newest version of each key, drop old duplicates)
//   4. Write merged entries as new SSTables in the target level
//   5. Remove old SSTables from both levels
//
// For L0 → L1:
//   We merge ALL L0 SSTables because they can overlap.
//
// For L1+ → L(n+1):
//   We could pick just one SSTable, but for simplicity we merge all
//   overlapping SSTables.

void LSMTree::compact_level(int level_num) {
    int target_level = level_num + 1;
    ensure_level(target_level);

    // Step 1: Get all entries from source level
    auto source_entries = levels_[level_num]->get_all_entries();
    if (source_entries.empty()) return;

    // Find the key range of source entries
    std::string min_key = source_entries.front().key;
    std::string max_key = source_entries.front().key;
    for (const auto& e : source_entries) {
        if (e.key < min_key) min_key = e.key;
        if (e.key > max_key) max_key = e.key;
    }

    // Step 2: Get overlapping entries from target level
    auto overlapping_paths = levels_[target_level]->get_overlapping_sstables(min_key, max_key);
    std::vector<Entry> target_entries;
    for (const auto& path : overlapping_paths) {
        SSTableReader reader(path);
        auto entries = reader.get_all_entries();
        target_entries.insert(target_entries.end(),
                             std::make_move_iterator(entries.begin()),
                             std::make_move_iterator(entries.end()));
    }

    // Step 3: Merge entries
    auto merged = merge_entries(source_entries, target_entries);

    // Step 4: Write merged entries as new SSTable(s) in target level
    if (!merged.empty()) {
        std::string new_sst_path = levels_[target_level]->next_sstable_path();
        SSTableWriter writer(new_sst_path);
        if (writer.write_entries(merged)) {
            levels_[target_level]->add_sstable(new_sst_path);
        }
    }

    // Step 5: Remove old SSTables
    // Remove overlapping SSTables from target level
    for (const auto& path : overlapping_paths) {
        levels_[target_level]->remove_sstable(path);
    }

    // Remove all SSTables from source level
    auto source_paths = levels_[level_num]->sstable_paths();
    for (const auto& path : source_paths) {
        levels_[level_num]->remove_sstable(path);
    }
}

// ─── Merge Entries ────────────────────────────────────────
// Merges two sorted lists of entries into one sorted, deduplicated list.
//
// Rules:
//   1. Sort by key ascending, then by sequence number descending
//   2. For duplicate keys, keep only the NEWEST version (highest seq)
//   3. If the newest version is a tombstone at the deepest level,
//      we could drop it — but for safety, we keep tombstones for now
//
// Example:
//   a: [("apple", seq=1), ("cat", seq=3)]
//   b: [("banana", seq=2), ("cat", seq=4)]
//
//   Combined & sorted: [("apple",1), ("banana",2), ("cat",4), ("cat",3)]
//   After dedup:       [("apple",1), ("banana",2), ("cat",4)]
//   "cat" seq=3 is dropped because seq=4 is newer

std::vector<Entry> LSMTree::merge_entries(const std::vector<Entry>& a,
                                           const std::vector<Entry>& b) {
    // Combine all entries
    std::vector<Entry> all;
    all.reserve(a.size() + b.size());
    all.insert(all.end(), a.begin(), a.end());
    all.insert(all.end(), b.begin(), b.end());

    // Sort: key ascending, then sequence number descending (newest first)
    std::sort(all.begin(), all.end());

    // Deduplicate: keep only the first (newest) entry for each key
    std::vector<Entry> result;
    result.reserve(all.size());

    for (size_t i = 0; i < all.size(); i++) {
        // Skip if this key is the same as the previous entry
        // (the previous entry has a higher sequence number = newer)
        if (i > 0 && all[i].key == all[i - 1].key) {
            continue;  // Drop older version
        }
        result.push_back(std::move(all[i]));
    }

    return result;
}

// ─── Ensure Level ─────────────────────────────────────────
// Creates levels up to and including the given level number.
// Level capacity follows the geometric progression:
//   L0: memtable_size
//   L1: memtable_size * T
//   L2: memtable_size * T^2
//   etc.

void LSMTree::ensure_level(int level_num) {
    while (static_cast<int>(levels_.size()) <= level_num) {
        int num = static_cast<int>(levels_.size());
        size_t capacity = config_.memtable_size_bytes;
        for (int i = 0; i < num; i++) {
            capacity *= config_.size_ratio;
        }
        levels_.push_back(std::make_unique<Level>(num, capacity, config_.sstable_dir));
    }
}

// ─── Statistics ───────────────────────────────────────────

size_t LSMTree::total_entries() const {
    size_t total = memtable_->count();
    for (const auto& level : levels_) {
        // Approximate: sum of all SSTable entry counts
        auto paths = level->sstable_paths();
        for (const auto& path : paths) {
            try {
                SSTableReader reader(path);
                total += reader.entry_count();
            } catch (...) {
                // Skip unreadable SSTables
            }
        }
    }
    return total;
}

size_t LSMTree::total_sstables() const {
    size_t total = 0;
    for (const auto& level : levels_) {
        total += level->sstable_count();
    }
    return total;
}

} // namespace lsm
