#pragma once

#include <vector>
#include <string>
#include <memory>
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
// Simple B+ Tree implementation for baseline comparison
// Order 3: max 2 keys per node (t-1 = 2, so t = 3)

class BTree : public System {
public:
    BTree();
    ~BTree() = default;

    // Insert or update a key-value pair
    void put(const Key& key, const Value& value) override;

    // Look up a key
    GetResult get(const Key& key) override;

    // Get system name for reporting
    std::string name() const override { return "B+ Tree"; }

private:
    static const int ORDER = 3;  // Max 2 keys per node
    std::unique_ptr<BTreeNode> root_;

    // Helper functions
    void insert_non_full(BTreeNode* node, const std::string& key, const std::string& value);
    void split_child(BTreeNode* parent, int index);
    GetResult search(BTreeNode* node, const std::string& key) const;
    BTreeNode* find_leaf(BTreeNode* node, const std::string& key) const;
};

} // namespace lsm