#include "memtable/memtable.h"

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
// Creates an empty MemTable with a Skip List configured from the Config.
//
// sequence_number_ starts at 0 and increments with each write.
// max_size_ is the flush threshold (default 1 MB from Monkey paper).

MemTable::MemTable(const Config& config)
    : skip_list_(config.skip_list_max_level, config.skip_list_probability)
    , sequence_number_(0)
    , max_size_(config.memtable_size_bytes)
{
}

// ─── Put ──────────────────────────────────────────────────
// Inserts a key-value pair into the MemTable.
//
// Each put gets a unique, increasing sequence number.
// If the same key is put multiple times, all versions are stored.
// The newest version (highest sequence number) wins during reads.
//
// Example:
//   put("name", "Alice")  → seq=1
//   put("age", "25")      → seq=2
//   put("name", "Bob")    → seq=3  (newer version of "name")
//
//   get("name") → returns "Bob" (seq=3 > seq=1)

uint64_t MemTable::put(const Key& key, const Value& value) {
    uint64_t seq = ++sequence_number_;
    Entry entry(key, value, EntryType::PUT, seq);
    skip_list_.insert(entry);
    return seq;
}

void MemTable::insert_entry(const Entry& entry) {
    skip_list_.insert(entry);
    if (entry.sequence_number > sequence_number_) {
        sequence_number_ = entry.sequence_number;
    }
}

// ─── Remove (Delete) ─────────────────────────────────────
// Marks a key as deleted by inserting a tombstone.
//
// A tombstone is an Entry with type=DELETE and an empty value.
// It gets a sequence number like any other write.
//
// Example:
//   put("name", "Alice")  → seq=1
//   remove("name")        → seq=2 (tombstone)
//
//   get("name") → finds tombstone (seq=2) → returns Deleted()
//   The old value "Alice" (seq=1) is effectively hidden.

uint64_t MemTable::remove(const Key& key) {
    uint64_t seq = ++sequence_number_;
    Entry entry(key, "", EntryType::DELETE, seq);
    skip_list_.insert(entry);
    return seq;
}

// ─── Get ──────────────────────────────────────────────────
// Looks up the most recent value for a key.
//
// Delegates to the Skip List's get() method.
// Returns:
//   - Found(value) if the key exists with a PUT entry
//   - Deleted() if the most recent entry is a tombstone
//   - NotFound() if the key was never written

GetResult MemTable::get(const Key& key) const {
    return skip_list_.get(key);
}

// ─── Should Flush ─────────────────────────────────────────
// Returns true when the MemTable's memory usage exceeds the threshold.
//
// The LSM-tree checks this after each write:
//   if (memtable.should_flush()) {
//       flush_to_sstable(memtable);
//       memtable = new MemTable();
//   }

bool MemTable::should_flush() const {
    return skip_list_.memory_usage() >= max_size_;
}

// ─── Get All Entries ─────────────────────────────────────
// Returns all entries in sorted order for flushing to an SSTable.
// Sorted by key ascending, then sequence number descending (newest first).

std::vector<Entry> MemTable::get_all_entries() const {
    return skip_list_.get_all_entries();
}

// ─── Statistics ───────────────────────────────────────────

size_t MemTable::count() const {
    return skip_list_.count();
}

size_t MemTable::memory_usage() const {
    return skip_list_.memory_usage();
}

bool MemTable::empty() const {
    return skip_list_.empty();
}

} // namespace lsm
