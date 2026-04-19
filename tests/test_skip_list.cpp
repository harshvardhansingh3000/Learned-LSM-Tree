// ─── Skip List Tests ──────────────────────────────────────
// Tests for the Skip List data structure.
// We verify: insert, search, ordering, duplicates, and iteration.

#include <gtest/gtest.h>
#include "memtable/skip_list.h"
#include <string>

using namespace lsm;

// ─── Test: Empty Skip List ────────────────────────────────
// A freshly created skip list should be empty.
TEST(SkipListTest, EmptyList) {
    SkipList sl;
    EXPECT_TRUE(sl.empty());
    EXPECT_EQ(sl.count(), 0u);
    EXPECT_EQ(sl.memory_usage(), 0u);
}

// ─── Test: Single Insert and Get ──────────────────────────
// Insert one entry, then retrieve it.
TEST(SkipListTest, SingleInsertAndGet) {
    SkipList sl;
    Entry e("hello", "world", EntryType::PUT, 1);
    sl.insert(e);

    EXPECT_FALSE(sl.empty());
    EXPECT_EQ(sl.count(), 1u);

    GetResult result = sl.get("hello");
    EXPECT_TRUE(result.found);
    EXPECT_FALSE(result.is_deleted);
    EXPECT_EQ(result.value, "world");
}

// ─── Test: Get Non-Existent Key ───────────────────────────
// Searching for a key that doesn't exist should return NotFound.
TEST(SkipListTest, GetNonExistent) {
    SkipList sl;
    Entry e("hello", "world", EntryType::PUT, 1);
    sl.insert(e);

    GetResult result = sl.get("goodbye");
    EXPECT_FALSE(result.found);
}

// ─── Test: Multiple Inserts ───────────────────────────────
// Insert several entries and verify all can be retrieved.
TEST(SkipListTest, MultipleInserts) {
    SkipList sl;

    sl.insert(Entry("banana", "yellow", EntryType::PUT, 1));
    sl.insert(Entry("apple", "red", EntryType::PUT, 2));
    sl.insert(Entry("cherry", "red", EntryType::PUT, 3));
    sl.insert(Entry("date", "brown", EntryType::PUT, 4));

    EXPECT_EQ(sl.count(), 4u);

    EXPECT_EQ(sl.get("apple").value, "red");
    EXPECT_EQ(sl.get("banana").value, "yellow");
    EXPECT_EQ(sl.get("cherry").value, "red");
    EXPECT_EQ(sl.get("date").value, "brown");
}

// ─── Test: Duplicate Key (Newer Wins) ─────────────────────
// Inserting the same key twice with different sequence numbers.
// The newer entry (higher seq) should be returned by get().
TEST(SkipListTest, DuplicateKeyNewerWins) {
    SkipList sl;

    sl.insert(Entry("name", "Alice", EntryType::PUT, 1));  // Older
    sl.insert(Entry("name", "Bob", EntryType::PUT, 5));    // Newer

    GetResult result = sl.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Bob");  // Newer version wins
}

// ─── Test: Tombstone (Delete Marker) ──────────────────────
// A DELETE entry should cause get() to return Deleted().
TEST(SkipListTest, Tombstone) {
    SkipList sl;

    sl.insert(Entry("name", "Alice", EntryType::PUT, 1));
    sl.insert(Entry("name", "", EntryType::DELETE, 2));  // Tombstone

    GetResult result = sl.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_TRUE(result.is_deleted);  // Tombstone found
}

// ─── Test: Put After Delete ───────────────────────────────
// If we put a key, delete it, then put it again, the latest put should win.
TEST(SkipListTest, PutAfterDelete) {
    SkipList sl;

    sl.insert(Entry("name", "Alice", EntryType::PUT, 1));
    sl.insert(Entry("name", "", EntryType::DELETE, 2));
    sl.insert(Entry("name", "Charlie", EntryType::PUT, 3));

    GetResult result = sl.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_FALSE(result.is_deleted);
    EXPECT_EQ(result.value, "Charlie");
}

// ─── Test: Sorted Iteration ──────────────────────────────
// get_all_entries() should return entries sorted by key.
TEST(SkipListTest, SortedIteration) {
    SkipList sl;

    // Insert in random order
    sl.insert(Entry("delta", "4", EntryType::PUT, 1));
    sl.insert(Entry("alpha", "1", EntryType::PUT, 2));
    sl.insert(Entry("charlie", "3", EntryType::PUT, 3));
    sl.insert(Entry("bravo", "2", EntryType::PUT, 4));

    auto entries = sl.get_all_entries();
    ASSERT_EQ(entries.size(), 4u);

    // Should be in alphabetical order
    EXPECT_EQ(entries[0].key, "alpha");
    EXPECT_EQ(entries[1].key, "bravo");
    EXPECT_EQ(entries[2].key, "charlie");
    EXPECT_EQ(entries[3].key, "delta");
}

// ─── Test: Large Insert (1000 keys) ──────────────────────
// Insert 1000 keys and verify all can be retrieved.
TEST(SkipListTest, LargeInsert) {
    SkipList sl;

    // Insert 1000 keys: "key_0000" to "key_0999"
    for (int i = 0; i < 1000; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        std::string value = "value_" + num;
        sl.insert(Entry(key, value, EntryType::PUT, i + 1));
    }

    EXPECT_EQ(sl.count(), 1000u);

    // Verify all keys can be found
    for (int i = 0; i < 1000; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        std::string expected_value = "value_" + num;
        GetResult result = sl.get(key);
        EXPECT_TRUE(result.found) << "Key not found: " << key;
        EXPECT_EQ(result.value, expected_value) << "Wrong value for key: " << key;
    }
}

// ─── Test: Memory Usage Tracking ──────────────────────────
// Memory usage should increase with each insert.
TEST(SkipListTest, MemoryUsageTracking) {
    SkipList sl;

    size_t initial_memory = sl.memory_usage();
    EXPECT_EQ(initial_memory, 0u);

    sl.insert(Entry("key1", "value1", EntryType::PUT, 1));
    size_t after_one = sl.memory_usage();
    EXPECT_GT(after_one, 0u);

    sl.insert(Entry("key2", "value2", EntryType::PUT, 2));
    size_t after_two = sl.memory_usage();
    EXPECT_GT(after_two, after_one);
}

// ─── Test: Iteration Order with Duplicates ────────────────
// When the same key has multiple versions, they should appear
// in order: key ascending, then sequence number descending.
TEST(SkipListTest, IterationOrderWithDuplicates) {
    SkipList sl;

    sl.insert(Entry("b", "v1", EntryType::PUT, 1));
    sl.insert(Entry("a", "v2", EntryType::PUT, 2));
    sl.insert(Entry("b", "v3", EntryType::PUT, 3));  // Newer version of "b"

    auto entries = sl.get_all_entries();
    ASSERT_EQ(entries.size(), 3u);

    // "a" comes first (alphabetical)
    EXPECT_EQ(entries[0].key, "a");

    // Then both "b" entries, newest first (seq=3 before seq=1)
    EXPECT_EQ(entries[1].key, "b");
    EXPECT_EQ(entries[1].sequence_number, 3u);
    EXPECT_EQ(entries[2].key, "b");
    EXPECT_EQ(entries[2].sequence_number, 1u);
}
