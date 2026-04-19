// ─── MemTable Tests ───────────────────────────────────────
// Tests for the MemTable wrapper around the Skip List.
// We verify: put, get, delete, sequence numbers, flush threshold, and iteration.

#include <gtest/gtest.h>
#include "memtable/memtable.h"
#include <string>

using namespace lsm;

// ─── Test: Empty MemTable ─────────────────────────────────
TEST(MemTableTest, EmptyMemTable) {
    MemTable mt;
    EXPECT_TRUE(mt.empty());
    EXPECT_EQ(mt.count(), 0u);
    EXPECT_EQ(mt.current_sequence_number(), 0u);
}

// ─── Test: Put and Get ────────────────────────────────────
TEST(MemTableTest, PutAndGet) {
    MemTable mt;
    mt.put("name", "Alice");

    GetResult result = mt.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Alice");
    EXPECT_FALSE(result.is_deleted);
}

// ─── Test: Sequence Numbers Auto-Increment ────────────────
TEST(MemTableTest, SequenceNumbers) {
    MemTable mt;

    uint64_t seq1 = mt.put("a", "1");
    uint64_t seq2 = mt.put("b", "2");
    uint64_t seq3 = mt.put("c", "3");

    EXPECT_EQ(seq1, 1u);
    EXPECT_EQ(seq2, 2u);
    EXPECT_EQ(seq3, 3u);
    EXPECT_EQ(mt.current_sequence_number(), 3u);
}

// ─── Test: Update Overwrites ──────────────────────────────
TEST(MemTableTest, UpdateOverwrites) {
    MemTable mt;

    mt.put("name", "Alice");
    mt.put("name", "Bob");

    GetResult result = mt.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Bob");  // Latest value wins
}

// ─── Test: Delete (Tombstone) ─────────────────────────────
TEST(MemTableTest, Delete) {
    MemTable mt;

    mt.put("name", "Alice");
    mt.remove("name");

    GetResult result = mt.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_TRUE(result.is_deleted);
}

// ─── Test: Put After Delete ───────────────────────────────
TEST(MemTableTest, PutAfterDelete) {
    MemTable mt;

    mt.put("name", "Alice");
    mt.remove("name");
    mt.put("name", "Charlie");

    GetResult result = mt.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_FALSE(result.is_deleted);
    EXPECT_EQ(result.value, "Charlie");
}

// ─── Test: Get Non-Existent ───────────────────────────────
TEST(MemTableTest, GetNonExistent) {
    MemTable mt;
    mt.put("name", "Alice");

    GetResult result = mt.get("age");
    EXPECT_FALSE(result.found);
}

// ─── Test: Should Flush ───────────────────────────────────
// With a tiny threshold, the MemTable should signal flush quickly.
TEST(MemTableTest, ShouldFlush) {
    Config config;
    config.memtable_size_bytes = 500;  // Tiny threshold: 500 bytes
    MemTable mt(config);

    EXPECT_FALSE(mt.should_flush());

    // Insert entries until we exceed 500 bytes
    for (int i = 0; i < 100; i++) {
        mt.put("key_" + std::to_string(i), "value_" + std::to_string(i));
        if (mt.should_flush()) break;
    }

    EXPECT_TRUE(mt.should_flush());
}

// ─── Test: Get All Entries Sorted ─────────────────────────
TEST(MemTableTest, GetAllEntriesSorted) {
    MemTable mt;

    mt.put("cherry", "red");
    mt.put("apple", "green");
    mt.put("banana", "yellow");

    auto entries = mt.get_all_entries();
    ASSERT_EQ(entries.size(), 3u);

    EXPECT_EQ(entries[0].key, "apple");
    EXPECT_EQ(entries[1].key, "banana");
    EXPECT_EQ(entries[2].key, "cherry");
}

// ─── Test: Multiple Operations ────────────────────────────
// A realistic sequence of puts, deletes, and gets.
TEST(MemTableTest, MultipleOperations) {
    MemTable mt;

    // Insert several keys
    mt.put("user:1", "Alice");
    mt.put("user:2", "Bob");
    mt.put("user:3", "Charlie");

    // Verify all exist
    EXPECT_EQ(mt.get("user:1").value, "Alice");
    EXPECT_EQ(mt.get("user:2").value, "Bob");
    EXPECT_EQ(mt.get("user:3").value, "Charlie");

    // Delete one
    mt.remove("user:2");
    EXPECT_TRUE(mt.get("user:2").is_deleted);

    // Update one
    mt.put("user:1", "Alice Updated");
    EXPECT_EQ(mt.get("user:1").value, "Alice Updated");

    // Others unchanged
    EXPECT_EQ(mt.get("user:3").value, "Charlie");

    // Non-existent still not found
    EXPECT_FALSE(mt.get("user:99").found);
}

// ─── Test: Delete Sequence Number ─────────────────────────
// Remove should also increment the sequence number.
TEST(MemTableTest, DeleteSequenceNumber) {
    MemTable mt;

    uint64_t seq1 = mt.put("a", "1");
    uint64_t seq2 = mt.remove("a");

    EXPECT_EQ(seq1, 1u);
    EXPECT_EQ(seq2, 2u);  // Delete also gets a sequence number
}
