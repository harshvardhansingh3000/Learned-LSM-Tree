#pragma once

#include <vector>
#include <random>
#include <functional>
#include <memory>
#include "common/types.h"
#include "common/config.h"

namespace lsm {

// ─── Skip List ────────────────────────────────────────────
//
// A Skip List is a probabilistic data structure that allows O(log n)
// search, insert, and delete operations. It's essentially a linked list
// with multiple "express lanes" stacked on top.
//
// Visual example with 4 levels (searching for key "dog"):
//
// Level 3:  HEAD ──────────────────────────────────────→ NIL
// Level 2:  HEAD ────────────→ "fox" ──────────────────→ NIL
// Level 1:  HEAD ──→ "cat" ──→ "fox" ──→ "pig" ───────→ NIL
// Level 0:  HEAD → "ant" → "cat" → "dog" → "fox" → "pig" → "zoo" → NIL
//
// To find "dog":
//   1. Start at HEAD, Level 3 → next is NIL → go down
//   2. Level 2 → next is "fox" > "dog" → go down
//   3. Level 1 → next is "cat" < "dog" → move right to "cat"
//      → next is "fox" > "dog" → go down
//   4. Level 0 → next is "dog" = "dog" → FOUND!
//
// Only 4 comparisons instead of scanning all 6 nodes!
//
// Why Skip List for MemTable?
//   - O(log n) insert/search/delete (same as balanced BST)
//   - Simpler to implement than Red-Black trees or AVL trees
//   - Easy to iterate in sorted order (just walk Level 0)
//   - Lock-free versions exist for concurrency (future optimization)
//   - Used by LevelDB, RocksDB, and many real databases
//

// ─── Node ─────────────────────────────────────────────────
// Each node in the skip list holds an Entry and an array of forward pointers.
// forward[i] points to the next node at level i.
// A node with height h has forward pointers for levels 0, 1, ..., h-1.

struct SkipListNode {
    Entry entry;
    
    // forward[i] = pointer to the next node at level i
    // A node at height 3 has forward[0], forward[1], forward[2]
    std::vector<SkipListNode*> forward;

    // Create a node with the given entry and height
    SkipListNode(Entry e, int height)
        : entry(std::move(e)), forward(height, nullptr) {}

    // Create a sentinel node (HEAD) with no entry
    explicit SkipListNode(int height)
        : entry(), forward(height, nullptr) {}
};

// ─── Skip List Class ─────────────────────────────────────

class SkipList {
public:
    // Constructor: creates an empty skip list with the given config
    explicit SkipList(int max_level = 12, double probability = 0.5);

    // Destructor: frees all nodes
    ~SkipList();

    // ── Core Operations ───────────────────────────────────

    // Insert or update an entry.
    // If a key already exists with a lower sequence number, the new entry wins.
    void insert(const Entry& entry);

    // Search for a key. Returns the most recent entry for that key, if found.
    // "Most recent" = highest sequence number.
    GetResult get(const Key& key) const;

    // ── Iteration (for flushing to SSTable) ───────────────

    // Returns all entries in sorted order (by key, then by sequence number desc).
    // This is used when flushing the MemTable to an SSTable.
    std::vector<Entry> get_all_entries() const;

    // ── Size Tracking ─────────────────────────────────────

    // Number of entries in the skip list
    size_t count() const { return count_; }

    // Approximate memory usage in bytes
    size_t memory_usage() const { return memory_usage_; }

    // Is the skip list empty?
    bool empty() const { return count_ == 0; }

    // ── No Copy (skip lists manage raw pointers) ──────────
    SkipList(const SkipList&) = delete;
    SkipList& operator=(const SkipList&) = delete;

private:
    // Generate a random level for a new node (coin flipping)
    int random_level();

    // The sentinel head node (doesn't hold real data)
    SkipListNode* head_;

    // Current maximum level in use (starts at 0, grows as nodes are added)
    int current_level_;

    // Configuration
    int max_level_;
    double probability_;

    // Statistics
    size_t count_;
    size_t memory_usage_;

    // Random number generator for level generation
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> dist_;
};

} // namespace lsm
