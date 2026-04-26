// ─── LSM-Tree Integration Tests ───────────────────────────
// Tests for the complete LSM-tree: put, get, delete, flush, compaction.
// These are end-to-end tests that exercise the full write and read paths.

#include <gtest/gtest.h>
#include "tree/lsm_tree.h"
#include <filesystem>
#include <string>

using namespace lsm;

// Helper: create a test config with small sizes to trigger flush/compaction quickly
static Config test_config(const std::string& name) {
    Config config;
    config.data_dir = "test_data/lsm_" + name;
    config.wal_dir = config.data_dir + "/wal";
    config.sstable_dir = config.data_dir + "/sstables";
    config.memtable_size_bytes = 4 * 1024;  // 4 KB (tiny, triggers flush quickly)
    config.size_ratio = 4;                   // 4x ratio (smaller for testing)
    return config;
}

// Helper: clean up test files
class LSMTreeTest : public ::testing::Test {
protected:
    void TearDown() override {
        std::filesystem::remove_all("test_data");
    }
};

// ─── Test: Basic Put and Get ──────────────────────────────
TEST_F(LSMTreeTest, BasicPutAndGet) {
    auto config = test_config("basic");
    LSMTree db(config);

    db.put("name", "Alice");
    db.put("age", "25");

    auto r1 = db.get("name");
    EXPECT_TRUE(r1.found);
    EXPECT_EQ(r1.value, "Alice");

    auto r2 = db.get("age");
    EXPECT_TRUE(r2.found);
    EXPECT_EQ(r2.value, "25");
}

// ─── Test: Get Non-Existent Key ───────────────────────────
TEST_F(LSMTreeTest, GetNonExistent) {
    auto config = test_config("nonexist");
    LSMTree db(config);

    db.put("name", "Alice");

    auto result = db.get("missing");
    EXPECT_FALSE(result.found);
}

// ─── Test: Update Key ─────────────────────────────────────
TEST_F(LSMTreeTest, UpdateKey) {
    auto config = test_config("update");
    LSMTree db(config);

    db.put("name", "Alice");
    db.put("name", "Bob");

    auto result = db.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Bob");
}

// ─── Test: Delete Key ─────────────────────────────────────
TEST_F(LSMTreeTest, DeleteKey) {
    auto config = test_config("delete");
    LSMTree db(config);

    db.put("name", "Alice");
    db.remove("name");

    auto result = db.get("name");
    EXPECT_FALSE(result.found);  // Deleted keys return not found
}

// ─── Test: Put After Delete ───────────────────────────────
TEST_F(LSMTreeTest, PutAfterDelete) {
    auto config = test_config("put_after_del");
    LSMTree db(config);

    db.put("name", "Alice");
    db.remove("name");
    db.put("name", "Charlie");

    auto result = db.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Charlie");
}

// ─── Test: Flush to SSTable ───────────────────────────────
// Write enough data to trigger a MemTable flush.
TEST_F(LSMTreeTest, FlushToSSTable) {
    auto config = test_config("flush");
    LSMTree db(config);

    // Write enough to trigger flush (4 KB MemTable)
    for (int i = 0; i < 100; i++) {
        db.put("key_" + std::to_string(i), "value_" + std::to_string(i));
    }

    // Should have at least 1 SSTable now
    EXPECT_GE(db.total_sstables(), 1u);

    // All keys should still be readable
    for (int i = 0; i < 100; i++) {
        auto result = db.get("key_" + std::to_string(i));
        EXPECT_TRUE(result.found) << "Key not found: key_" << i;
        EXPECT_EQ(result.value, "value_" + std::to_string(i));
    }
}

// ─── Test: Read After Flush ───────────────────────────────
// Data should be readable from SSTables after flush.
TEST_F(LSMTreeTest, ReadAfterFlush) {
    auto config = test_config("read_flush");
    LSMTree db(config);

    db.put("before_flush", "value1");

    // Force flush
    db.flush();

    // Should still be readable from SSTable
    auto result = db.get("before_flush");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "value1");

    // Write more after flush
    db.put("after_flush", "value2");

    auto r2 = db.get("after_flush");
    EXPECT_TRUE(r2.found);
    EXPECT_EQ(r2.value, "value2");
}

// ─── Test: Multiple Flushes ───────────────────────────────
// Multiple flushes should create multiple SSTables in L0.
TEST_F(LSMTreeTest, MultipleFlushes) {
    auto config = test_config("multi_flush");
    LSMTree db(config);

    // Write and flush 3 times
    for (int batch = 0; batch < 3; batch++) {
        for (int i = 0; i < 50; i++) {
            std::string key = "batch" + std::to_string(batch) + "_key" + std::to_string(i);
            db.put(key, "value_" + std::to_string(i));
        }
        db.flush();
    }

    // All keys from all batches should be readable
    for (int batch = 0; batch < 3; batch++) {
        for (int i = 0; i < 50; i++) {
            std::string key = "batch" + std::to_string(batch) + "_key" + std::to_string(i);
            auto result = db.get(key);
            EXPECT_TRUE(result.found) << "Key not found: " << key;
        }
    }
}

// ─── Test: Compaction ─────────────────────────────────────
// Write enough data to trigger compaction from L0 to L1.
TEST_F(LSMTreeTest, Compaction) {
    auto config = test_config("compact");
    config.memtable_size_bytes = 1024;  // 1 KB — very small for fast compaction
    LSMTree db(config);

    // Write enough to trigger multiple flushes and compaction
    for (int i = 0; i < 200; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        db.put(key, "value_" + num);
    }

    // All keys should still be readable after compaction
    for (int i = 0; i < 200; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        auto result = db.get(key);
        EXPECT_TRUE(result.found) << "Key not found after compaction: " << key;
        EXPECT_EQ(result.value, "value_" + num);
    }
}

// ─── Test: Delete Across Flush ────────────────────────────
// Delete a key, flush, then verify it's still deleted.
TEST_F(LSMTreeTest, DeleteAcrossFlush) {
    auto config = test_config("del_flush");
    LSMTree db(config);

    db.put("name", "Alice");
    db.flush();  // "name" is now in SSTable

    db.remove("name");  // Tombstone in MemTable
    db.flush();          // Tombstone is now in SSTable

    auto result = db.get("name");
    EXPECT_FALSE(result.found);  // Should be deleted
}

// ─── Test: Update Across Flush ────────────────────────────
// Update a key after flush, verify newest value is returned.
TEST_F(LSMTreeTest, UpdateAcrossFlush) {
    auto config = test_config("update_flush");
    LSMTree db(config);

    db.put("name", "Alice");
    db.flush();  // "Alice" in SSTable

    db.put("name", "Bob");  // "Bob" in MemTable (newer)

    auto result = db.get("name");
    EXPECT_TRUE(result.found);
    EXPECT_EQ(result.value, "Bob");  // MemTable version wins
}

// ─── Test: Large Scale (1000 keys) ───────────────────────
TEST_F(LSMTreeTest, LargeScale) {
    auto config = test_config("large");
    config.memtable_size_bytes = 2 * 1024;  // 2 KB
    LSMTree db(config);

    // Write 1000 keys
    for (int i = 0; i < 1000; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        db.put(key, "value_" + num);
    }

    // Read all 1000 keys
    for (int i = 0; i < 1000; i++) {
        std::string num = std::to_string(i);
        std::string key = "key_" + std::string(4 - num.length(), '0') + num;
        auto result = db.get(key);
        EXPECT_TRUE(result.found) << "Key not found: " << key;
        EXPECT_EQ(result.value, "value_" + num) << "Wrong value for: " << key;
    }
}

// ─── Test: Scan (Range Query) ─────────────────────────────
TEST_F(LSMTreeTest, Scan) {
    auto config = test_config("scan");
    LSMTree db(config);

    db.put("apple", "red");
    db.put("banana", "yellow");
    db.put("cherry", "red");
    db.put("date", "brown");
    db.put("elderberry", "purple");

    // Scan a range
    auto results = db.scan("banana", "date");
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].first, "banana");
    EXPECT_EQ(results[0].second, "yellow");
    EXPECT_EQ(results[1].first, "cherry");
    EXPECT_EQ(results[2].first, "date");

    // Scan all
    auto all = db.scan("a", "z");
    EXPECT_EQ(all.size(), 5u);

    // Scan empty range
    auto empty = db.scan("x", "z");
    EXPECT_EQ(empty.size(), 0u);
}

// ─── Test: Scan Excludes Deleted Keys ─────────────────────
TEST_F(LSMTreeTest, ScanExcludesDeleted) {
    auto config = test_config("scan_del");
    LSMTree db(config);

    db.put("a", "1");
    db.put("b", "2");
    db.put("c", "3");
    db.remove("b");  // Delete "b"

    auto results = db.scan("a", "c");
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].first, "a");
    EXPECT_EQ(results[1].first, "c");
    // "b" should NOT appear (it's deleted)
}

// ─── Test: Scan After Flush ───────────────────────────────
TEST_F(LSMTreeTest, ScanAfterFlush) {
    auto config = test_config("scan_flush");
    LSMTree db(config);

    db.put("x", "1");
    db.put("y", "2");
    db.flush();
    db.put("z", "3");  // In MemTable

    auto results = db.scan("x", "z");
    ASSERT_EQ(results.size(), 3u);
    EXPECT_EQ(results[0].first, "x");
    EXPECT_EQ(results[1].first, "y");
    EXPECT_EQ(results[2].first, "z");
}

// ─── Test: Statistics ─────────────────────────────────────
TEST_F(LSMTreeTest, Statistics) {
    auto config = test_config("stats");
    LSMTree db(config);

    EXPECT_EQ(db.num_levels(), 1u);  // At least L0
    EXPECT_EQ(db.total_sstables(), 0u);

    for (int i = 0; i < 50; i++) {
        db.put("key_" + std::to_string(i), "value_" + std::to_string(i));
    }
    db.flush();

    EXPECT_GE(db.total_sstables(), 1u);
}
