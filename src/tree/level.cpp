#include "tree/level.h"
#include <filesystem>
#include <algorithm>
#include <stdexcept>

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
// Creates a level with the given number and capacity.
//
// Level 0: max_size = memtable_size (1 MB)
// Level 1: max_size = memtable_size * T (10 MB)
// Level 2: max_size = memtable_size * T^2 (100 MB)
// etc.

Level::Level(int level_num, size_t max_size_bytes, const std::string& base_dir)
    : level_num_(level_num)
    , max_size_bytes_(max_size_bytes)
    , base_dir_(base_dir)
    , next_sstable_id_(0)
{
    // Create the directory for this level's SSTables
    std::filesystem::create_directories(base_dir_);
}

// ─── Add SSTable ──────────────────────────────────────────
// Opens an SSTable file and adds it to this level.
// New SSTables are added at the END (newest last).
// For L0, this means the last SSTable is the newest (most recent flush).

void Level::add_sstable(const std::string& sstable_path) {
    auto reader = std::make_unique<SSTableReader>(sstable_path);
    sstables_.push_back(std::move(reader));
}

// ─── Remove SSTable ───────────────────────────────────────
// Removes an SSTable from this level (by file path).
// Used during compaction when an SSTable is merged into the next level.
// Also deletes the file from disk.

void Level::remove_sstable(const std::string& sstable_path) {
    // Find and remove the SSTable from our list
    auto it = std::remove_if(sstables_.begin(), sstables_.end(),
        [&sstable_path](const std::unique_ptr<SSTableReader>& reader) {
            return reader->file_path() == sstable_path;
        });

    sstables_.erase(it, sstables_.end());

    // Delete the file from disk
    std::filesystem::remove(sstable_path);
}

// ─── Get (Lookup) ─────────────────────────────────────────
// Searches for a key in this level.
//
// Level 0 behavior:
//   Search ALL SSTables from NEWEST to OLDEST (reverse order).
//   Because L0 SSTables can overlap, we must check all of them.
//   The first match we find is the most recent version.
//
// Level 1+ behavior:
//   SSTables are non-overlapping, so at most ONE SSTable can contain the key.
//   We check each SSTable's key range (min_key, max_key) to find the right one.
//   This is fast because we skip SSTables whose range doesn't include our key.

GetResult Level::get(const Key& key) const {
    if (level_num_ == 0) {
        // L0: search newest to oldest (reverse order)
        for (int i = static_cast<int>(sstables_.size()) - 1; i >= 0; i--) {
            // Quick range check first
            if (!sstables_[i]->may_contain(key)) {
                continue;
            }

            GetResult result = sstables_[i]->get(key);
            if (result.found) {
                return result;  // Found (either value or tombstone)
            }
        }
    } else {
        // L1+: SSTables are non-overlapping, find the right one
        for (const auto& sst : sstables_) {
            if (!sst->may_contain(key)) {
                continue;  // Key is outside this SSTable's range
            }

            GetResult result = sst->get(key);
            if (result.found) {
                return result;
            }
            // Since SSTables are non-overlapping, if the key is in range
            // but not found, it doesn't exist in this level
            break;
        }
    }

    return GetResult::NotFound();
}

// ─── Needs Compaction ─────────────────────────────────────
// Returns true if this level's total size exceeds its capacity.
//
// For L0, we also trigger compaction if there are too many SSTables
// (even if total size is small), because too many L0 SSTables
// slow down reads (we must check all of them).

bool Level::needs_compaction() const {
    if (level_num_ == 0) {
        // L0: compact if more than 4 SSTables OR size exceeds capacity
        return sstables_.size() >= 4 || current_size_bytes() > max_size_bytes_;
    }
    return current_size_bytes() > max_size_bytes_;
}

// ─── Current Size ─────────────────────────────────────────
// Total size of all SSTables in this level.
// We estimate based on entry count * average entry size.

size_t Level::current_size_bytes() const {
    size_t total = 0;
    for (const auto& sst : sstables_) {
        // Estimate: each entry is roughly 116 bytes (16 key + 100 value)
        // Plus overhead for SSTable format
        total += sst->entry_count() * 130;
    }
    return total;
}

// ─── Get All Entries ──────────────────────────────────────
// Returns all entries from all SSTables in this level.
// Used during compaction to merge everything.

std::vector<Entry> Level::get_all_entries() const {
    std::vector<Entry> all_entries;
    for (const auto& sst : sstables_) {
        auto entries = sst->get_all_entries();
        all_entries.insert(all_entries.end(),
                          std::make_move_iterator(entries.begin()),
                          std::make_move_iterator(entries.end()));
    }
    return all_entries;
}

// ─── Get Overlapping SSTables ─────────────────────────────
// Finds SSTables whose key ranges overlap with [min_key, max_key].
// Used during compaction: when merging an SSTable from L(n) into L(n+1),
// we need to find which L(n+1) SSTables overlap with it.
//
// Two ranges [a,b] and [c,d] overlap if: a <= d AND c <= b

std::vector<std::string> Level::get_overlapping_sstables(
    const Key& min_key, const Key& max_key) const {

    std::vector<std::string> overlapping;
    for (const auto& sst : sstables_) {
        // Check if ranges overlap
        if (sst->min_key() <= max_key && min_key <= sst->max_key()) {
            overlapping.push_back(sst->file_path());
        }
    }
    return overlapping;
}

// ─── SSTable Paths ────────────────────────────────────────
// Returns file paths of all SSTables in this level.

std::vector<std::string> Level::sstable_paths() const {
    std::vector<std::string> paths;
    for (const auto& sst : sstables_) {
        paths.push_back(sst->file_path());
    }
    return paths;
}

// ─── Next SSTable Path ────────────────────────────────────
// Generates a unique file path for a new SSTable in this level.
// Format: "data/sstables/L{level}_{id}.sst"
// Example: "data/sstables/L0_001.sst", "data/sstables/L1_002.sst"

std::string Level::next_sstable_path() {
    next_sstable_id_++;
    std::string filename = "L" + std::to_string(level_num_) + "_"
                         + std::to_string(next_sstable_id_) + ".sst";
    return base_dir_ + "/" + filename;
}

} // namespace lsm
