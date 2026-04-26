// ─── SSTable Tests ────────────────────────────────────────
// Tests for SSTable writer and reader.
// We verify: write, read, point lookup, range check, and round-trip.

#include <gtest/gtest.h>
#include "sstable/sstable_writer.h"
#include "sstable/sstable_reader.h"
#include "memtable/memtable.h"
#include <filesystem>

using namespace lsm;

// Helper: clean up test files after each test
class SSTableTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove_all("test_data");
    }
};

// Helper: create sorted entries
static std::vector<Entry> make_entries(int count) {
    std::vector<Entry> entries;
    for (int i = 0; i < count; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        std::string value = "value_" + num;
        entries.emplace_back(key, value, EntryType::PUT, i + 1);
    }
    return entries;
}

// ─── Test: Write and Read Back ────────────────────────────
// Write entries to an SSTable, then read them all back.
TEST_F(SSTableTest, WriteAndReadAll) {
    std::string path = "test_data/sstable_basic.sst";
    auto entries = make_entries(10);

    // Write
    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));
    EXPECT_TRUE(std::filesystem::exists(path));

    // Read all
    SSTableReader reader(path);
    EXPECT_EQ(reader.entry_count(), 10u);
    EXPECT_EQ(reader.min_key(), "key_0000");
    EXPECT_EQ(reader.max_key(), "key_0009");

    auto read_entries = reader.get_all_entries();
    ASSERT_EQ(read_entries.size(), 10u);

    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(read_entries[i].key, entries[i].key);
        EXPECT_EQ(read_entries[i].value, entries[i].value);
        EXPECT_EQ(read_entries[i].type, entries[i].type);
        EXPECT_EQ(read_entries[i].sequence_number, entries[i].sequence_number);
    }
}

// ─── Test: Point Lookup ───────────────────────────────────
// Look up specific keys in the SSTable.
TEST_F(SSTableTest, PointLookup) {
    std::string path = "test_data/sstable_lookup.sst";
    auto entries = make_entries(100);

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);

    // Look up existing keys
    auto result = reader.get("key_0000");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "value_0");

    result = reader.get("key_0050");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "value_50");

    result = reader.get("key_0099");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "value_99");

    // Look up non-existent key
    result = reader.get("key_9999");
    EXPECT_FALSE(result.found);

    result = reader.get("nonexistent");
    EXPECT_FALSE(result.found);
}

// ─── Test: Tombstone Lookup ───────────────────────────────
// SSTable should correctly handle DELETE entries.
TEST_F(SSTableTest, TombstoneLookup) {
    std::string path = "test_data/sstable_tombstone.sst";

    std::vector<Entry> entries;
    entries.emplace_back("alive", "value", EntryType::PUT, 1);
    entries.emplace_back("dead", "", EntryType::DELETE, 2);
    entries.emplace_back("zombie", "brains", EntryType::PUT, 3);

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);

    auto result = reader.get("alive");
    EXPECT_TRUE(result.found);
    EXPECT_FALSE(result.is_deleted);
    EXPECT_EQ(result.value, "value");

    result = reader.get("dead");
    EXPECT_TRUE(result.found);
    EXPECT_TRUE(result.is_deleted);

    result = reader.get("zombie");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "brains");
}

// ─── Test: May Contain (Range Check) ──────────────────────
TEST_F(SSTableTest, MayContain) {
    std::string path = "test_data/sstable_range.sst";

    std::vector<Entry> entries;
    entries.emplace_back("cat", "meow", EntryType::PUT, 1);
    entries.emplace_back("dog", "woof", EntryType::PUT, 2);
    entries.emplace_back("fox", "ring", EntryType::PUT, 3);

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);

    EXPECT_TRUE(reader.may_contain("cat"));   // min key
    EXPECT_TRUE(reader.may_contain("dog"));   // middle
    EXPECT_TRUE(reader.may_contain("fox"));   // max key
    // Keys in range but NOT in SSTable — Bloom filter correctly rejects them
    EXPECT_FALSE(reader.may_contain("elk"));  // not in SSTable, bloom says no

    EXPECT_FALSE(reader.may_contain("ant"));  // before min
    EXPECT_FALSE(reader.may_contain("zoo"));  // after max
}

// ─── Test: Large SSTable (1000 entries) ───────────────────
// Tests multiple data blocks and fence pointer binary search.
TEST_F(SSTableTest, LargeSSTable) {
    std::string path = "test_data/sstable_large.sst";
    auto entries = make_entries(1000);

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);
    EXPECT_EQ(reader.entry_count(), 1000u);

    // Verify every key can be found
    for (int i = 0; i < 1000; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        std::string expected_value = "value_" + num;

        auto result = reader.get(key);
        EXPECT_TRUE(result.found) << "Key not found: " << key;
        EXPECT_EQ(result.value, expected_value) << "Wrong value for: " << key;
    }
}

// ─── Test: MemTable Flush to SSTable ──────────────────────
// Simulates the actual flush process: MemTable → SSTable.
TEST_F(SSTableTest, MemTableFlush) {
    std::string path = "test_data/sstable_flush.sst";

    // Fill a MemTable
    MemTable mt;
    mt.put("banana", "yellow");
    mt.put("apple", "red");
    mt.put("cherry", "red");
    mt.put("date", "brown");

    // Flush: get sorted entries → write SSTable
    auto entries = mt.get_all_entries();
    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    // Read back and verify
    SSTableReader reader(path);
    EXPECT_EQ(reader.entry_count(), 4u);
    EXPECT_EQ(reader.min_key(), "apple");
    EXPECT_EQ(reader.max_key(), "date");

    EXPECT_EQ(reader.get("apple").value, "red");
    EXPECT_EQ(reader.get("banana").value, "yellow");
    EXPECT_EQ(reader.get("cherry").value, "red");
    EXPECT_EQ(reader.get("date").value, "brown");

    // Non-existent
    EXPECT_FALSE(reader.get("elderberry").found);
}

// ─── Test: Empty Entries ──────────────────────────────────
// Writing empty entries should fail gracefully.
TEST_F(SSTableTest, EmptyEntries) {
    std::string path = "test_data/sstable_empty.sst";
    std::vector<Entry> entries;

    SSTableWriter writer(path);
    EXPECT_FALSE(writer.write_entries(entries));
}

// ─── Test: Single Entry ───────────────────────────────────
TEST_F(SSTableTest, SingleEntry) {
    std::string path = "test_data/sstable_single.sst";

    std::vector<Entry> entries;
    entries.emplace_back("only_key", "only_value", EntryType::PUT, 1);

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);
    EXPECT_EQ(reader.entry_count(), 1u);
    EXPECT_EQ(reader.min_key(), "only_key");
    EXPECT_EQ(reader.max_key(), "only_key");

    auto result = reader.get("only_key");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "only_value");
}

// ─── Test: Duplicate Keys (Multiple Versions) ─────────────
// SSTable should store all versions and return the first match (newest).
TEST_F(SSTableTest, DuplicateKeys) {
    std::string path = "test_data/sstable_dup.sst";

    // Entries sorted: key ascending, seq descending (newest first)
    std::vector<Entry> entries;
    entries.emplace_back("name", "Charlie", EntryType::PUT, 3);  // Newest
    entries.emplace_back("name", "Bob", EntryType::PUT, 2);
    entries.emplace_back("name", "Alice", EntryType::PUT, 1);    // Oldest

    SSTableWriter writer(path);
    ASSERT_TRUE(writer.write_entries(entries));

    SSTableReader reader(path);

    // Should return the first match (newest = "Charlie")
    auto result = reader.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Charlie");
}
