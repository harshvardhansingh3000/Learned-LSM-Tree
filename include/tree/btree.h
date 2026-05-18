#pragma once

#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include "common/types.h"
#include "tree/system.h"

namespace lsm {

// ─── B+ Tree Node ─────────────────────────────────────────
struct BTreeNode {
    bool is_leaf;
    std::vector<std::string> keys;
    std::vector<std::string> values;  // Only for leaf nodes
    std::vector<std::unique_ptr<BTreeNode>> children;  // Only for internal nodes

    BTreeNode(bool leaf = true) : is_leaf(leaf) {}
};

// ─── B+ Tree ──────────────────────────────────────────────
// Simple B+ Tree implementation for baseline comparison.
// Order 3: max 2 keys per node (t-1 = 2, so t = 3).
//
// NOTE: To make the comparison with LSM-tree fairer, we simulate
// disk I/O latency. Real B+ Trees on disk (e.g., InnoDB, SQLite)
// incur ~5-50 µs per page read from SSD. We simulate this by
// adding a configurable delay per node traversal.

class BTree : public System {
public:
    // simulate_disk_io: if true, adds realistic SSD I/O latency per node access
    explicit BTree(bool simulate_disk_io = false);
    ~BTree() = default;

    // Insert or update a key-value pair
    void put(const Key& key, const Value& value) override;

    // Look up a key
    GetResult get(const Key& key) override;

    // Get system name for reporting
    std::string name() const override {
        return simulate_disk_io_ ? "B+ Tree (Disk)" : "B+ Tree";
    }

private:
    static const int ORDER = 3;  // Max 2 keys per node
    std::unique_ptr<BTreeNode> root_;
    bool simulate_disk_io_;

    // Simulated I/O delay per node access (~8 µs for SSD page read)
    static constexpr int kDiskIODelayUs = 8;

    // Simulate one disk page read
    void simulate_page_read() const {
        if (simulate_disk_io_) {
            // Busy-wait for more precise short delays than sleep
            auto start = std::chrono::high_resolution_clock::now();
            while (true) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(now - start).count();
                if (elapsed >= kDiskIODelayUs) break;
            }
        }
    }

    // Helper functions
    void insert_non_full(BTreeNode* node, const std::string& key, const std::string& value);
    void split_child(BTreeNode* parent, int index);
    GetResult search(BTreeNode* node, const std::string& key) const;
    BTreeNode* find_leaf(BTreeNode* node, const std::string& key) const;
};

} // namespace lsm