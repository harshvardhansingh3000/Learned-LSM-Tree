#ifndef BPTREE_H
#define BPTREE_H

#include <string>
#include <vector>
#include <optional>
#include <memory>
#include <algorithm>

namespace bptree {

static constexpr size_t ORDER = 4;  // B+ Tree order (M)

class BPlusTree {
public:
    BPlusTree();
    ~BPlusTree();

    // Core operations
    bool put(const std::string& key, const std::string& value);
    std::optional<std::string> get(const std::string& key) const;
    size_t size() const { return num_keys_; }
    void clear();

private:
    struct Node {
        bool is_leaf;
        std::vector<std::string> keys;
        std::vector<std::string> values;  // For leaf nodes
        std::vector<std::shared_ptr<Node>> children;
        std::shared_ptr<Node> next_leaf;  // Linked list of leaves

        Node(bool leaf = false) : is_leaf(leaf), next_leaf(nullptr) {}
    };

    std::shared_ptr<Node> root_;
    size_t num_keys_;

    // Helper functions
    std::shared_ptr<Node> search_node(const std::string& key) const;
    void insert_into_leaf(std::shared_ptr<Node> leaf, 
                         const std::string& key, 
                         const std::string& value);
    void insert_into_parent(std::shared_ptr<Node> old_node,
                           const std::string& key,
                           std::shared_ptr<Node> new_node);
    std::shared_ptr<Node> split_leaf(std::shared_ptr<Node> leaf);
    std::shared_ptr<Node> split_internal(std::shared_ptr<Node> internal);
};

}  // namespace bptree

#endif  // BPTREE_H
