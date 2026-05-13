#include "tree/btree.h"
#include <algorithm>

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
BTree::BTree() {
    root_ = std::make_unique<BTreeNode>(true);  // Start with leaf root
}

// ─── Put (Insert/Update) ───────────────────────────────────
void BTree::put(const Key& key, const Value& value) {
    std::string k = key;
    std::string v = value;

    BTreeNode* root = root_.get();

    // If root is full, create new root
    if (root->keys.size() == ORDER - 1) {
        auto new_root = std::make_unique<BTreeNode>(false);
        new_root->children.push_back(std::move(root_));
        root_.swap(new_root);
        root = root_.get();

        split_child(root, 0);
        insert_non_full(root, k, v);
    } else {
        insert_non_full(root, k, v);
    }
}

// ─── Insert Non-Full ───────────────────────────────────────
void BTree::insert_non_full(BTreeNode* node, const std::string& key, const std::string& value) {
    if (node->is_leaf) {
        // Insert into leaf
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        size_t pos = it - node->keys.begin();
        node->keys.insert(it, key);
        node->values.insert(node->values.begin() + pos, value);
    } else {
        // Find child to insert into
        auto it = std::upper_bound(node->keys.begin(), node->keys.end(), key);
        size_t child_idx = it - node->keys.begin();

        BTreeNode* child = node->children[child_idx].get();

        // If child is full, split it
        if (child->keys.size() == ORDER - 1) {
            split_child(node, child_idx);
            // After split, decide which child to insert into
            if (key > node->keys[child_idx]) {
                child_idx++;
            }
            child = node->children[child_idx].get();
        }

        insert_non_full(child, key, value);
    }
}

// ─── Split Child ───────────────────────────────────────────
void BTree::split_child(BTreeNode* parent, int index) {
    BTreeNode* child = parent->children[index].get();
    auto new_child = std::make_unique<BTreeNode>(child->is_leaf);

    // Move half the keys/values to new child
    size_t mid = (ORDER - 1) / 2;
    std::string mid_key = child->keys[mid];

    // Move keys after mid to new child
    new_child->keys.assign(child->keys.begin() + mid + 1, child->keys.end());
    child->keys.resize(mid);

    if (child->is_leaf) {
        // Move values for leaf
        new_child->values.assign(child->values.begin() + mid + 1, child->values.end());
        child->values.resize(mid);
    } else {
        // Move children for internal
        new_child->children.assign(std::make_move_iterator(child->children.begin() + mid + 1),
                                   std::make_move_iterator(child->children.end()));
        child->children.resize(mid + 1);
    }

    // Insert new child into parent
    parent->children.insert(parent->children.begin() + index + 1, std::move(new_child));

    // Insert mid key into parent
    parent->keys.insert(parent->keys.begin() + index, mid_key);
}

// ─── Get (Search) ──────────────────────────────────────────
GetResult BTree::get(const Key& key) {
    std::string k = key;
    return search(root_.get(), k);
}

// ─── Search Helper ─────────────────────────────────────────
GetResult BTree::search(BTreeNode* node, const std::string& key) const {
    if (!node) {
        return GetResult::NotFound();
    }

    if (node->is_leaf) {
        // Search in leaf
        auto it = std::lower_bound(node->keys.begin(), node->keys.end(), key);
        if (it != node->keys.end() && *it == key) {
            size_t pos = it - node->keys.begin();
            return GetResult::Found(node->values[pos]);
        }
        return GetResult::NotFound();
    } else {
        // Find child to search
        auto it = std::upper_bound(node->keys.begin(), node->keys.end(), key);
        size_t child_idx = it - node->keys.begin();
        return search(node->children[child_idx].get(), key);
    }
}

} // namespace lsm