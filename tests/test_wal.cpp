// ─── WAL Tests ────────────────────────────────────────────
// Tests for the Write-Ahead Log.
// We verify: write, replay, corruption detection, clear, and crash recovery.

#include <gtest/gtest.h>
#include "wal/wal.h"
#include <filesystem>
#include <fstream>

using namespace lsm;

// Helper: create a unique WAL path for each test (avoids conflicts)
static std::string test_wal_path(const std::string& name) {
    return "test_data/wal_" + name + ".log";
}

// Helper: clean up test files after each test
class WALTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Clean up test data directory
        std::filesystem::remove_all("test_data");
    }
};

// ─── Test: Create WAL ─────────────────────────────────────
// Creating a WAL should create the file and directories.
TEST_F(WALTest, CreateWAL) {
    std::string path = test_wal_path("create");
    WAL wal(path);

    EXPECT_EQ(wal.file_path(), path);
    EXPECT_EQ(wal.entry_count(), 0u);
    EXPECT_TRUE(std::filesystem::exists(path));
}

// ─── Test: Append and Replay PUT ──────────────────────────
// Write a PUT entry, then replay it.
TEST_F(WALTest, AppendAndReplayPut) {
    std::string path = test_wal_path("put");

    {
        WAL wal(path);
        wal.append_put("name", "Alice");
    }  // WAL destructor closes the file

    // Replay
    WAL wal2(path);
    auto entries = wal2.replay();

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].key, "name");
    EXPECT_EQ(entries[0].value, "Alice");
    EXPECT_EQ(entries[0].type, EntryType::PUT);
}

// ─── Test: Append and Replay DELETE ───────────────────────
// Write a DELETE entry, then replay it.
TEST_F(WALTest, AppendAndReplayDelete) {
    std::string path = test_wal_path("delete");

    {
        WAL wal(path);
        wal.append_delete("name");
    }

    WAL wal2(path);
    auto entries = wal2.replay();

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].key, "name");
    EXPECT_EQ(entries[0].value, "");
    EXPECT_EQ(entries[0].type, EntryType::DELETE);
}

// ─── Test: Multiple Entries ───────────────────────────────
// Write several entries, replay all of them.
TEST_F(WALTest, MultipleEntries) {
    std::string path = test_wal_path("multi");

    {
        WAL wal(path);
        wal.append_put("user:1", "Alice");
        wal.append_put("user:2", "Bob");
        wal.append_delete("user:1");
        wal.append_put("user:3", "Charlie");

        EXPECT_EQ(wal.entry_count(), 4u);
    }

    WAL wal2(path);
    auto entries = wal2.replay();

    ASSERT_EQ(entries.size(), 4u);

    EXPECT_EQ(entries[0].key, "user:1");
    EXPECT_EQ(entries[0].value, "Alice");
    EXPECT_EQ(entries[0].type, EntryType::PUT);

    EXPECT_EQ(entries[1].key, "user:2");
    EXPECT_EQ(entries[1].value, "Bob");

    EXPECT_EQ(entries[2].key, "user:1");
    EXPECT_EQ(entries[2].type, EntryType::DELETE);

    EXPECT_EQ(entries[3].key, "user:3");
    EXPECT_EQ(entries[3].value, "Charlie");
}

// ─── Test: Sequence Numbers During Replay ─────────────────
// Replayed entries should get sequential sequence numbers.
TEST_F(WALTest, ReplaySequenceNumbers) {
    std::string path = test_wal_path("seq");

    {
        WAL wal(path);
        wal.append_put("a", "1");
        wal.append_put("b", "2");
        wal.append_put("c", "3");
    }

    WAL wal2(path);
    auto entries = wal2.replay();

    ASSERT_EQ(entries.size(), 3u);
    EXPECT_EQ(entries[0].sequence_number, 1u);
    EXPECT_EQ(entries[1].sequence_number, 2u);
    EXPECT_EQ(entries[2].sequence_number, 3u);
}

// ─── Test: Clear WAL ──────────────────────────────────────
// After clear(), replay should return nothing.
TEST_F(WALTest, Clear) {
    std::string path = test_wal_path("clear");

    WAL wal(path);
    wal.append_put("name", "Alice");
    wal.append_put("age", "25");
    EXPECT_EQ(wal.entry_count(), 2u);

    wal.clear();
    EXPECT_EQ(wal.entry_count(), 0u);

    auto entries = wal.replay();
    EXPECT_EQ(entries.size(), 0u);
}

// ─── Test: Write After Clear ──────────────────────────────
// After clear(), we should be able to write new entries.
TEST_F(WALTest, WriteAfterClear) {
    std::string path = test_wal_path("write_after_clear");

    WAL wal(path);
    wal.append_put("old_key", "old_value");
    wal.clear();

    wal.append_put("new_key", "new_value");

    auto entries = wal.replay();
    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].key, "new_key");
    EXPECT_EQ(entries[0].value, "new_value");
}

// ─── Test: Corruption Detection ───────────────────────────
// If we corrupt the WAL file, replay should stop at the corrupted entry.
TEST_F(WALTest, CorruptionDetection) {
    std::string path = test_wal_path("corrupt");

    // Write 3 valid entries
    {
        WAL wal(path);
        wal.append_put("key1", "value1");
        wal.append_put("key2", "value2");
        wal.append_put("key3", "value3");
    }

    // Corrupt the file: flip a byte somewhere in the middle
    {
        std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
        // Seek to somewhere in the second entry and flip a byte
        file.seekp(30, std::ios::beg);
        char byte;
        file.read(&byte, 1);
        byte = ~byte;  // Flip all bits
        file.seekp(30, std::ios::beg);
        file.write(&byte, 1);
    }

    // Replay should recover at least the first entry
    WAL wal2(path);
    auto entries = wal2.replay();

    // We should get at least 1 entry (the first one before corruption)
    EXPECT_GE(entries.size(), 1u);
    EXPECT_EQ(entries[0].key, "key1");
    EXPECT_EQ(entries[0].value, "value1");

    // But we should NOT get all 3 (corruption stops replay)
    EXPECT_LT(entries.size(), 3u);
}

// ─── Test: Empty WAL Replay ───────────────────────────────
// Replaying an empty WAL should return nothing.
TEST_F(WALTest, EmptyReplay) {
    std::string path = test_wal_path("empty");

    WAL wal(path);
    auto entries = wal.replay();
    EXPECT_EQ(entries.size(), 0u);
}

// ─── Test: Replay Non-Existent File ───────────────────────
// Replaying a WAL that doesn't exist should return nothing (not crash).
TEST_F(WALTest, ReplayNonExistent) {
    WAL wal("test_data/nonexistent.log");
    // Clear the file first so it's empty
    wal.clear();
    auto entries = wal.replay();
    EXPECT_EQ(entries.size(), 0u);
}

// ─── Test: Large Values ───────────────────────────────────
// WAL should handle large keys and values correctly.
TEST_F(WALTest, LargeValues) {
    std::string path = test_wal_path("large");

    std::string big_key(1000, 'K');     // 1000 bytes of 'K'
    std::string big_value(10000, 'V');  // 10000 bytes of 'V'

    {
        WAL wal(path);
        wal.append_put(big_key, big_value);
    }

    WAL wal2(path);
    auto entries = wal2.replay();

    ASSERT_EQ(entries.size(), 1u);
    EXPECT_EQ(entries[0].key, big_key);
    EXPECT_EQ(entries[0].value, big_value);
}
