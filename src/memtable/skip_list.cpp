#include "memtable/skip_list.h"
#include <cassert>
#include <chrono>

namespace lsm {

// ─── Constructor ──────────────────────────────────────────
// Creates an empty skip list with a HEAD sentinel node.
//
// The HEAD node has max_level forward pointers, all set to nullptr.
// Think of it as the leftmost column in our "express lane" diagram:
//
// Level 3:  HEAD → nullptr
// Level 2:  HEAD → nullptr
// Level 1:  HEAD → nullptr
// Level 0:  HEAD → nullptr
//
// current_level_ starts at 0 (only level 0 exists initially).

SkipList::SkipList(int max_level, double probability)
    : head_(new SkipListNode(max_level))   // Create HEAD with max_level pointers
    , current_level_(0)                     // No levels in use yet
    , max_level_(max_level)
    , probability_(probability)
    , count_(0)
    , memory_usage_(0)
    , rng_(std::chrono::steady_clock::now().time_since_epoch().count())  // Seed with current time
    , dist_(0.0, 1.0)                      // Uniform distribution [0.0, 1.0)
{
}

// ─── Destructor ───────────────────────────────────────────
// Walk through level 0 (which contains ALL nodes) and delete each one.
//
// Level 0: HEAD → A → B → C → D → nullptr
//          ↑      ↑   ↑   ↑   ↑
//        delete  delete delete delete delete
//
// We must delete every node we created with 'new' to avoid memory leaks.

SkipList::~SkipList() {
    SkipListNode* current = head_;
    while (current != nullptr) {
        SkipListNode* next = current->forward[0];  // Save next before deleting
        delete current;
        current = next;
    }
}

// ─── Random Level Generator ──────────────────────────────
// Simulates coin flipping to determine a new node's height.
//
// Example sequence of coin flips:
//   Flip 1: 0.32 < 0.5 → heads! level = 2
//   Flip 2: 0.71 > 0.5 → tails! stop.
//   Result: height = 2
//
// Distribution of heights:
//   Height 1: 50%    (every other node)
//   Height 2: 25%    (1 in 4 nodes)
//   Height 3: 12.5%  (1 in 8 nodes)
//   Height 4: 6.25%  (1 in 16 nodes)
//   ...

int SkipList::random_level() {
    int level = 1;
    while (dist_(rng_) < probability_ && level < max_level_) {
        level++;
    }
    return level;
}

// ─── Insert ──────────────────────────────────────────────
// Inserts a new entry into the skip list.
//
// Algorithm:
// 1. Find the position where the entry should go at each level
//    (track "update" pointers — the last node before the insertion point)
// 2. Generate a random height for the new node
// 3. If the new height exceeds current_level_, update HEAD's pointers
// 4. Create the new node and splice it into the list at each level
//
// Visual example — inserting "dog" with height 2:
//
// Before:
//   Level 1: HEAD ──→ "cat" ──→ "fox" ──→ NIL
//   Level 0: HEAD → "ant" → "cat" → "fox" → "zoo" → NIL
//
//   update[1] = "cat"  (last node < "dog" at level 1)
//   update[0] = "cat"  (last node < "dog" at level 0)
//
// After:
//   Level 1: HEAD ──→ "cat" ──→ "dog" ──→ "fox" ──→ NIL
//   Level 0: HEAD → "ant" → "cat" → "dog" → "fox" → "zoo" → NIL

void SkipList::insert(const Entry& entry) {
    // Step 1: Find the insertion position at each level.
    // update[i] will hold the last node whose key is less than entry.key at level i.
    // These are the nodes whose forward pointers need to be updated.
    std::vector<SkipListNode*> update(max_level_, head_);

    SkipListNode* current = head_;

    // Start from the highest level and work down.
    // At each level, move right as far as possible while staying less than the target.
    for (int i = current_level_ - 1; i >= 0; i--) {
        while (current->forward[i] != nullptr &&
               current->forward[i]->entry < entry) {
            current = current->forward[i];  // Move right
        }
        update[i] = current;  // Record the last node before insertion point
    }

    // Step 2: Generate a random height for the new node.
    int new_level = random_level();

    // Step 3: If the new node is taller than any existing node,
    // we need to update HEAD's pointers for the new levels.
    if (new_level > current_level_) {
        for (int i = current_level_; i < new_level; i++) {
            update[i] = head_;  // HEAD is the predecessor at new levels
        }
        current_level_ = new_level;
    }

    // Step 4: Create the new node and splice it in at each level.
    SkipListNode* new_node = new SkipListNode(entry, new_level);

    for (int i = 0; i < new_level; i++) {
        // new_node points to what update[i] used to point to
        new_node->forward[i] = update[i]->forward[i];
        // update[i] now points to new_node
        update[i]->forward[i] = new_node;
    }

    // Step 5: Update statistics.
    count_++;
    memory_usage_ += entry.size() + sizeof(SkipListNode) + new_level * sizeof(SkipListNode*);
}

// ─── Get (Search) ────────────────────────────────────────
// Searches for the most recent entry with the given key.
//
// Algorithm:
// 1. Start at HEAD, highest level
// 2. At each level, move right while the next node's key is less than target
// 3. When we can't move right, go down one level
// 4. At level 0, check if the next node matches our key
//
// Visual example — searching for "fox":
//
//   Level 2: HEAD ──────────→ "dog" ──────────→ NIL
//                              ↓ (dog < fox, move right... next is NIL, go down)
//   Level 1: HEAD ──→ "cat" ──→ "dog" ──→ "fox" ──→ NIL
//                                          ^ FOUND at level 1!
//
// Because entries with the same key are sorted by sequence number (newest first),
// the first match we find is always the most recent version.

GetResult SkipList::get(const Key& key) const {
    SkipListNode* current = head_;

    // Navigate through levels from top to bottom
    for (int i = current_level_ - 1; i >= 0; i--) {
        while (current->forward[i] != nullptr &&
               current->forward[i]->entry.key < key) {
            current = current->forward[i];  // Move right
        }
    }

    // current is now the last node with key < target.
    // current->forward[0] should be the first node with key >= target.
    current = current->forward[0];

    // Check if we found the key
    if (current != nullptr && current->entry.key == key) {
        // Found! Check if it's a tombstone (deletion marker)
        if (current->entry.type == EntryType::DELETE) {
            return GetResult::Deleted();
        }
        return GetResult::Found(current->entry.value);
    }

    // Key not found in this skip list
    return GetResult::NotFound();
}

// ─── Get All Entries ─────────────────────────────────────
// Returns all entries in sorted order by walking level 0.
//
// Level 0 contains every node in sorted order:
//   HEAD → "ant" → "cat" → "dog" → "fox" → "zoo" → NIL
//
// We simply walk from HEAD to NIL, collecting each entry.
// This is used when flushing the MemTable to an SSTable.

std::vector<Entry> SkipList::get_all_entries() const {
    std::vector<Entry> entries;
    entries.reserve(count_);  // Pre-allocate for efficiency

    SkipListNode* current = head_->forward[0];  // Skip HEAD sentinel
    while (current != nullptr) {
        entries.push_back(current->entry);
        current = current->forward[0];  // Move to next on level 0
    }

    return entries;
}

} // namespace lsm
