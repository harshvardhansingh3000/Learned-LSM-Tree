# 📘 Learned LSM-Tree Database — Technical Report

> **Project**: A high-performance LSM-tree based key-value store in C++ with ML-enhanced Bloom filters  
> **Based on**: *"Learned LSM-trees: Two Approaches Using Learned Bloom Filters"* — Fidalgo & Ye, Harvard, 2025  
> **Language**: C++17  
> **Authors**: [Your Name]  
> **Date**: April 2026

---

## Table of Contents

1. [Introduction & Motivation](#1-introduction--motivation)
2. [Background: What is an LSM-Tree?](#2-background-what-is-an-lsm-tree)
3. [System Architecture](#3-system-architecture)
4. [Component 1: Common Types (`types.h`)](#4-component-1-common-types)
5. [Component 2: Configuration (`config.h`)](#5-component-2-configuration)
6. [Component 3: Skip List (`skip_list.h/.cpp`)](#6-component-3-skip-list)
7. [Component 4: MemTable (`memtable.h/.cpp`)](#7-component-4-memtable)
8. [Component 5: Write-Ahead Log (WAL)](#8-component-5-write-ahead-log)
9. [Component 6: SSTable (Sorted String Table)](#9-component-6-sstable)
10. [Component 7: Bloom Filters](#10-component-7-bloom-filters)
11. [Component 8: Leveled Compaction](#11-component-8-leveled-compaction)
12. [Component 9: ML Classifier (Paper Approach 1)](#12-component-9-ml-classifier)
13. [Component 10: Learned Bloom Filter (Paper Approach 2)](#13-component-10-learned-bloom-filter)
14. [Performance Analysis & Results](#14-performance-analysis--results)
15. [References](#15-references)

---

## 1. Introduction & Motivation

### 1.1 What Problem Are We Solving?

Modern applications generate massive amounts of data that needs to be stored and retrieved efficiently. Key-value stores are the backbone of many systems — from social media feeds to financial transactions. The fundamental challenge is:

> **How do we build a storage system that handles both fast writes AND fast reads?**

Traditional databases (like those using B-trees) optimize for reads but suffer on writes because every write requires updating the on-disk data structure in-place, which involves random disk I/O — the slowest operation a computer can do.

### 1.2 The LSM-Tree Solution

Log-Structured Merge Trees (LSM-trees) flip this trade-off:
- **Writes go to memory first** (extremely fast — nanoseconds)
- **Memory is periodically flushed to disk** as sorted, immutable files (sequential I/O — fast)
- **Reads search memory first, then disk** (may need to check multiple files)

This makes writes 10-100× faster than B-trees, at the cost of slightly slower reads.

### 1.3 The Research Paper's Contribution

The paper by Fidalgo & Ye (Harvard, 2025) proposes using **machine learning** to make LSM-tree reads faster:

1. **Classifier-Augmented Lookup**: An ML model predicts which disk levels to skip during reads, reducing latency by up to 2.28×
2. **Learned Bloom Filters**: Replace traditional Bloom filters with compact ML models, reducing memory usage by 70-80%

### 1.4 Our Implementation

We implement a complete LSM-tree database in C++ with both ML enhancements, producing a system that:
- Handles standard database operations (Put, Get, Delete, Scan)
- Provides crash recovery via Write-Ahead Logging
- Implements leveled compaction with Monkey-optimized Bloom filters
- Integrates two ML-based optimizations from the paper
- Includes comprehensive benchmarking and performance comparison

---

## 2. Background: What is an LSM-Tree?

### 2.1 The Core Idea

An LSM-tree organizes data in two regions:

```
┌─────────────────────────────────────────────┐
│                  MEMORY                      │
│  ┌─────────────────────────────────────┐    │
│  │           MemTable                   │    │
│  │  (sorted in-memory buffer)           │    │
│  │  Holds recent writes                 │    │
│  │  Capacity: 1 MB                      │    │
│  └──────────────┬──────────────────────┘    │
│                  │ flush when full            │
├──────────────────┼──────────────────────────┤
│                  ▼        DISK               │
│  ┌─────────────────────────────────────┐    │
│  │  Level 0  (newest data, ~1 MB)      │    │
│  ├─────────────────────────────────────┤    │
│  │  Level 1  (10× bigger, ~10 MB)      │    │
│  ├─────────────────────────────────────┤    │
│  │  Level 2  (10× bigger, ~100 MB)     │    │
│  ├─────────────────────────────────────┤    │
│  │  Level 3  (10× bigger, ~1 GB)       │    │
│  └─────────────────────────────────────┘    │
│  Each level contains sorted, immutable       │
│  files called SSTables                       │
└─────────────────────────────────────────────┘
```

### 2.2 Write Path

```
Client: PUT("user:123", "John Doe")
    │
    ▼
┌──────────┐     ┌──────────┐
│   WAL    │ ←── │  Write   │  Step 1: Log to WAL (crash safety)
│ (append) │     │ Request  │
└──────────┘     └────┬─────┘
                      │
                      ▼
               ┌──────────┐
               │ MemTable │     Step 2: Insert into MemTable (fast, in-memory)
               │(Skip List)│
               └─────┬────┘
                     │ if size > 1 MB
                     ▼
               ┌──────────┐
               │  Flush   │     Step 3: Write sorted data to disk as SSTable
               │ to disk  │
               └─────┬────┘
                     │ if level full
                     ▼
               ┌──────────┐
               │Compaction │    Step 4: Merge SSTables to maintain sorted order
               └──────────┘
```

### 2.3 Read Path

```
Client: GET("user:123")
    │
    ▼
┌──────────┐
│ MemTable │ ──→ Found? Return immediately (fastest path)
└────┬─────┘
     │ Not found
     ▼
┌──────────┐
│ Level 0  │ ──→ Check Bloom filter → if "maybe yes" → search SSTable
└────┬─────┘
     │ Not found
     ▼
┌──────────┐
│ Level 1  │ ──→ Check Bloom filter → if "maybe yes" → search SSTable
└────┬─────┘
     │ Not found
     ▼
   ... (continue through all levels)
     │
     ▼
  NOT FOUND (key doesn't exist)
```

### 2.4 The Read Amplification Problem

The deeper the tree, the more levels we must check for each read. With 4 levels, a single GET might:
1. Check MemTable (fast, in memory)
2. Check L0 Bloom filter + possibly read SSTable from disk
3. Check L1 Bloom filter + possibly read SSTable from disk
4. Check L2 Bloom filter + possibly read SSTable from disk
5. Check L3 Bloom filter + possibly read SSTable from disk

This is called **read amplification** — one logical read becomes multiple physical reads. The paper's ML techniques aim to reduce this.

### 2.5 Key Terminology

| Term | Definition |
|------|-----------|
| **MemTable** | In-memory sorted buffer for recent writes |
| **SSTable** | Sorted String Table — immutable sorted file on disk |
| **WAL** | Write-Ahead Log — append-only log for crash recovery |
| **Bloom Filter** | Probabilistic data structure: "Is key possibly in this set?" |
| **Compaction** | Background process that merges SSTables to maintain sorted order |
| **Tombstone** | Special marker indicating a key has been deleted |
| **Fence Pointer** | Index entry pointing to the first key of each data block in an SSTable |
| **Level** | A tier in the LSM-tree hierarchy; each level is T× larger than the previous |
| **Size Ratio (T)** | Multiplier between level sizes (T=10 in our implementation) |
| **Read Amplification** | Number of disk reads per logical read operation |
| **Write Amplification** | Number of disk writes per logical write operation |

---

## 3. System Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Interface                        │
│              CLI: put/get/delete/scan/stats                   │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│                     LSM-Tree Engine                           │
│                                                              │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────────────┐  │
│  │   WAL    │  │  MemTable │  │    Read Path Router      │  │
│  │ (crash   │→ │ (SkipList)│  │                          │  │
│  │ recovery)│  └─────┬─────┘  │  Mode 1: Traditional     │  │
│  └──────────┘        │flush   │  Mode 2: ML Classifier   │  │
│                ┌─────▼─────┐  │  Mode 3: Learned BF      │  │
│                │  Level 0  │◄─┤                          │  │
│                │ (SSTables)│  │  Each mode changes how    │  │
│                └─────┬─────┘  │  we check Bloom filters   │  │
│                ┌─────▼─────┐  │  during the read path.    │  │
│                │  Level 1  │◄─┤                          │  │
│                │+ Bloom/ML │  └──────────────────────────┘  │
│                └─────┬─────┘                                 │
│                ┌─────▼─────┐                                 │
│                │  Level 2  │                                 │
│                │+ Bloom/ML │                                 │
│                └─────┬─────┘                                 │
│                ┌─────▼─────┐                                 │
│                │  Level N  │                                 │
│                │+ Bloom/ML │                                 │
│                └───────────┘                                 │
│                                                              │
├─────────────────────────────────────────────────────────────┤
│                Compaction Engine (Background)                 │
│         Merges SSTables when levels get too full             │
├─────────────────────────────────────────────────────────────┤
│              Benchmark & Metrics Collector                    │
│        Measures latency, throughput, memory, FPR/FNR         │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 File Organization

```
LSM/
├── include/           ← Header files (declarations — WHAT things do)
│   ├── common/        ← Shared types and configuration
│   ├── memtable/      ← MemTable and Skip List
│   ├── wal/           ← Write-Ahead Log
│   ├── sstable/       ← SSTable reader/writer
│   ├── bloom/         ← Bloom filters (traditional + learned)
│   ├── compaction/    ← Compaction engine
│   ├── tree/          ← LSM-tree orchestrator
│   ├── ml/            ← Machine learning components
│   └── cli/           ← Command-line interface
├── src/               ← Source files (implementations — HOW things work)
├── tests/             ← Unit tests (verification — DO things work correctly?)
├── bench/             ← Benchmarks (performance — HOW FAST do things work?)
├── docs/              ← Documentation (this report!)
└── data/              ← Runtime data (WAL files, SSTables, ML models)
```

### 3.3 Technology Choices

| Choice | What | Why |
|--------|------|-----|
| **C++17** | Programming language | High performance, memory control, used by real databases (RocksDB, LevelDB) |
| **CMake** | Build system | Industry standard for C++ projects, cross-platform |
| **Google Test** | Testing framework | Most popular C++ testing framework, easy to use |
| **Skip List** | MemTable data structure | O(log n) operations, simple to implement, used by LevelDB/RocksDB |
| **MurmurHash3** | Hash function for Bloom filters | Fast, excellent distribution, non-cryptographic |
| **Custom GBT** | ML model (Gradient Boosted Trees) | Lightweight, no Python dependency, matches paper's approach |

---

## 4. Component 1: Common Types (`types.h`)

### 4.1 Purpose
Defines the fundamental data types used throughout the entire database. Think of it as the "vocabulary" — every other component speaks this language.

### 4.2 Key and Value Types

```cpp
using Key = std::string;
using Value = std::string;
```

Both keys and values are byte sequences represented as strings. The paper uses:
- **Keys**: 16 bytes (matching RocksDB's default benchmark configuration)
- **Values**: 100 bytes (matching RocksDB's default benchmark configuration)

We use `std::string` for flexibility — it can hold any byte sequence of any length.

**Why `using` aliases?** If we later want to change the type (e.g., to fixed-size arrays for performance), we only change it in one place.

### 4.3 Entry Type: PUT vs DELETE

```cpp
enum class EntryType : uint8_t {
    PUT = 0,     // Normal write — store this key-value pair
    DELETE = 1   // Tombstone — mark this key as deleted
};
```

**Why tombstones instead of actual deletion?**

In an LSM-tree, data exists across multiple levels. If we just removed a key from the MemTable, older versions might still exist in SSTables on disk. A read would find the old version and incorrectly return it.

Instead, we write a **tombstone** — a special entry that says "this key is deleted." When the read path encounters a tombstone, it knows to stop searching and return "not found."

```
Example: Delete "user:123"

Before delete:
  MemTable: (empty)
  Level 1 SSTable: user:123 → "John Doe"

If we just did nothing in MemTable:
  GET("user:123") → searches MemTable (miss) → searches Level 1 → finds "John Doe" ❌

With tombstone:
  MemTable: user:123 → TOMBSTONE
  Level 1 SSTable: user:123 → "John Doe"

  GET("user:123") → searches MemTable → finds TOMBSTONE → returns "not found" ✓
```

Tombstones are eventually cleaned up during compaction (when the tombstone and the old value end up in the same level, both are discarded).

### 4.4 Entry Struct

```cpp
struct Entry {
    Key key;                    // e.g., "user:123"
    Value value;                // e.g., "John Doe" (empty for DELETE)
    EntryType type;             // PUT or DELETE
    uint64_t sequence_number;   // Monotonically increasing counter
};
```

**Sequence numbers** are crucial for correctness. Every write operation gets a unique, increasing sequence number:

```
Operation 1: PUT("a", "hello")   → seq=1
Operation 2: PUT("b", "world")   → seq=2
Operation 3: PUT("a", "updated") → seq=3  ← newer version of "a"
Operation 4: DELETE("b")         → seq=4  ← tombstone for "b"
```

When we encounter multiple entries for the same key, the one with the **highest sequence number wins** (it's the most recent).

### 4.5 Entry Ordering

```cpp
bool operator<(const Entry& other) const {
    if (key != other.key) return key < other.key;        // Primary: alphabetical by key
    return sequence_number > other.sequence_number;       // Secondary: newest first
}
```

This ordering is critical for the Skip List and SSTables:
- Keys are sorted **alphabetically** (enables binary search and range scans)
- For the same key, entries are sorted by **sequence number descending** (newest first)

This means when we iterate through sorted entries, for any key, we always see the newest version first.

### 4.6 GetResult

```cpp
struct GetResult {
    bool found;        // Did we find any entry for this key?
    Value value;       // The value (meaningful only if found && !is_deleted)
    bool is_deleted;   // Did we find a tombstone?
};
```

Three possible outcomes:
| Scenario | found | is_deleted | Meaning |
|----------|-------|------------|---------|
| `GetResult::Found("John")` | true | false | Key exists, value is "John" |
| `GetResult::Deleted()` | true | true | Key was explicitly deleted (tombstone) |
| `GetResult::NotFound()` | false | false | Key never existed at this level |

The distinction between Deleted and NotFound matters:
- **Deleted** → Stop searching deeper levels (the key was intentionally removed)
- **NotFound** → Keep searching deeper levels (the key might exist there)

---

## 5. Component 2: Configuration (`config.h`)

### 5.1 Purpose
Centralizes all tunable parameters in one place. Based on the **Monkey paper** (Dayan & Idreos, 2018) recommendations for optimal LSM-tree performance.

### 5.2 The Monkey Paper

**"Monkey: Optimal Navigable Key-Value Store"** (SIGMOD 2018) is one of the most influential papers in LSM-tree research. Its key contributions:

1. **Mathematical model** for LSM-tree read/write costs
2. **Optimal Bloom filter allocation**: deeper levels should get more bits per key
3. **Configuration recommendations**: 1 MB MemTable, 10× size ratio, leveled compaction

We adopt Monkey's configuration as our baseline because:
- It's theoretically optimal for the traditional approach
- The paper we're implementing (Fidalgo & Ye) uses the same configuration
- It provides a strong baseline to compare our ML enhancements against

### 5.3 Parameter Details

| Parameter | Value | Formula/Rationale |
|-----------|-------|-------------------|
| `memtable_size_bytes` | 1 MB | Monkey recommendation. Balances write latency (smaller = more frequent flushes) vs read performance (larger = fewer levels) |
| `size_ratio` | 10 | Each level is 10× larger. L0=1MB, L1=10MB, L2=100MB, L3=1GB. Standard choice balancing compaction cost and query performance |
| `bloom_bits_per_key` | 10 | Gives ~0.82% FPR. Formula: FPR ≈ (1/2)^(m/n × ln2) where m/n = bits per key |
| `sstable_block_size` | 4 KB | Matches OS page size for efficient I/O. Each block holds ~35 entries |
| `skip_list_max_level` | 12 | Supports 2^12 = 4096 entries efficiently. Our MemTable holds ~8800 entries |
| `skip_list_probability` | 0.5 | Optimal for balanced search speed vs memory overhead |

### 5.4 Level Size Calculation

With size ratio T=10 and MemTable size M=1MB:

```
Level 0:  M          = 1 MB      (direct flush from MemTable)
Level 1:  M × T      = 10 MB
Level 2:  M × T²     = 100 MB
Level 3:  M × T³     = 1,000 MB  = 1 GB
Level 4:  M × T⁴     = 10,000 MB = 10 GB
```

Total capacity with 4 levels: ~11.1 GB

### 5.5 Bloom Filter FPR by Level (Monkey Allocation)

Instead of uniform 10 bits/key everywhere, Monkey allocates:

```
Level 1: FPR = 1/T¹ = 10%     (fewer bits, cheap to check)
Level 2: FPR = 1/T² = 1%      (moderate bits)
Level 3: FPR = 1/T³ = 0.1%    (more bits, expensive to check)
Level 4: FPR = 1/T⁴ = 0.01%   (most bits, very expensive to check)
```

This is optimal because deeper levels contain more data and are more expensive to search, so we invest more memory to avoid false positives there.

---

## 6. Component 3: Skip List (`skip_list.h` / `skip_list.cpp`)

### 6.1 What is a Skip List?

A **Skip List** is a probabilistic data structure invented by William Pugh in 1990. It provides the same O(log n) performance as balanced binary search trees (Red-Black trees, AVL trees) but is much simpler to implement.

### 6.2 Why Skip List for MemTable?

| Data Structure | Insert | Search | Delete | Iterate | Complexity |
|---------------|--------|--------|--------|---------|------------|
| Sorted Array | O(n) | O(log n) | O(n) | O(n) | Simple |
| Hash Table | O(1) | O(1) | O(1) | O(n log n)* | No sorted iteration |
| Red-Black Tree | O(log n) | O(log n) | O(log n) | O(n) | Complex (rotations) |
| **Skip List** | **O(log n)** | **O(log n)** | **O(log n)** | **O(n)** | **Simple** |

*Hash tables can't iterate in sorted order without sorting first.

Skip Lists are used by **LevelDB**, **RocksDB**, **Redis**, and **MemSQL** for their in-memory sorted structures.

### 6.3 How a Skip List Works

#### The Concept: Express Lanes

Imagine a sorted linked list of train stations:

```
Level 0: A → B → C → D → E → F → G → H → I → J → NIL
         (local train — stops everywhere)
```

To find station H, you must visit A, B, C, D, E, F, G, H — **8 stops**.

Now add express lanes:

```
Level 3: A ─────────────────────────────────────→ NIL
Level 2: A ──────────→ D ──────────→ H ────────→ NIL
Level 1: A ────→ C ────→ E ────→ G ────→ I ────→ NIL
Level 0: A → B → C → D → E → F → G → H → I → J → NIL
```

To find H:
1. Start at A, Level 3 → next is NIL (too far) → go down
2. Level 2 → next is D (< H) → move to D → next is H → **FOUND!**

Only **3 steps** instead of 8!

#### The Probabilistic Part

When inserting a new node, we randomly decide its height by flipping a coin:
- Flip heads? Go up one level. Flip again.
- Flip tails? Stop.

```
Probability of height 1: 50%     (1 in 2 nodes)
Probability of height 2: 25%     (1 in 4 nodes)
Probability of height 3: 12.5%   (1 in 8 nodes)
Probability of height 4: 6.25%   (1 in 16 nodes)
```

This naturally creates the express lane structure without any complex rebalancing!

#### Insert Algorithm

To insert key "F" with height 2:

```
Step 1: Find where "F" should go at each level (track "update" pointers)

Level 2: HEAD ──────→ D ──────→ H ──→ NIL
                      ^ update[2] = D (last node < F at level 2)

Level 1: HEAD ──→ C ──→ E ──→ G ──→ NIL
                        ^ update[1] = E (last node < F at level 1)

Level 0: HEAD → B → C → D → E → G → H → NIL
                              ^ update[0] = E (last node < F at level 0)

Step 2: Create new node "F" with height 2

Step 3: Splice it in at levels 0 and 1:
  F.forward[0] = update[0].forward[0]  (F→G)
  update[0].forward[0] = F             (E→F)
  F.forward[1] = update[1].forward[1]  (F→G)
  update[1].forward[1] = F             (E→F)

Result:
Level 2: HEAD ──────→ D ──────→ H ──→ NIL
Level 1: HEAD ──→ C ──→ E ──→ F ──→ G ──→ NIL
Level 0: HEAD → B → C → D → E → F → G → H → NIL
```

#### Search Algorithm

To find key "G":

```
current = HEAD, level = highest level

Level 2: HEAD → next is D (< G) → move to D
         D → next is H (> G) → go down

Level 1: D → next is E (< G) → move to E
         E → next is F (< G) → move to F
         F → next is G (= G) → FOUND!

Total comparisons: 5 (vs 7 for linear scan)
```

#### Complexity Analysis

| Operation | Average Case | Worst Case |
|-----------|-------------|------------|
| Insert | O(log n) | O(n) — extremely unlikely |
| Search | O(log n) | O(n) — extremely unlikely |
| Delete | O(log n) | O(n) — extremely unlikely |
| Iterate all | O(n) | O(n) |
| Space | O(n) | O(n log n) — extremely unlikely |

The worst case requires all coin flips to go the same way — probability decreases exponentially.

### 6.4 Skip List Header (`skip_list.h`) — Line by Line

#### SkipListNode Structure

```cpp
struct SkipListNode {
    Entry entry;
    std::vector<SkipListNode*> forward;
```

Each node contains:
- `entry` — The key-value data stored at this node
- `forward` — Array of pointers to the next node at each level

A node with height 3 has `forward[0]`, `forward[1]`, `forward[2]`:

```
         ┌──────────────┐
Level 2: │ forward[2] ──────→ next node at level 2 (or nullptr)
         ├──────────────┤
Level 1: │ forward[1] ──────→ next node at level 1 (or nullptr)
         ├──────────────┤
Level 0: │ forward[0] ──────→ next node at level 0 (or nullptr)
         ├──────────────┤
         │ entry:        │
         │  key = "dog"  │
         │  val = "woof" │
         │  seq = 42     │
         └──────────────┘
```

```cpp
    SkipListNode(Entry e, int height)
        : entry(std::move(e)), forward(height, nullptr) {}
```
Constructor for data nodes. Takes an Entry and a height. Creates `height` forward pointers, all initialized to `nullptr`. Uses `std::move` to efficiently transfer the Entry without copying.

```cpp
    explicit SkipListNode(int height)
        : entry(), forward(height, nullptr) {}
```
Constructor for the HEAD sentinel node. No real data, just forward pointers. `explicit` prevents accidental implicit conversion from `int` to `SkipListNode`.

#### SkipList Class — Public Interface

```cpp
class SkipList {
public:
    explicit SkipList(int max_level = 12, double probability = 0.5);
```
**Constructor**: Creates an empty skip list.
- `max_level = 12` — Maximum height any node can have (supports ~4096 entries efficiently)
- `probability = 0.5` — Coin flip probability for level promotion
- `explicit` — Prevents `SkipList sl = 12;` (accidental implicit conversion)

```cpp
    ~SkipList();
```
**Destructor**: Walks through level 0 and `delete`s every node. Critical because we allocate nodes with `new` — without this, we'd have memory leaks.

```cpp
    void insert(const Entry& entry);
```
**Insert**: Adds a new entry to the skip list. If a key already exists with a lower sequence number, the new entry is placed before it (due to our ordering: same key, higher seq first). The `const Entry&` means we take a reference (no copy) to a read-only Entry.

```cpp
    GetResult get(const Key& key) const;
```
**Search**: Finds the most recent entry for a key. Returns `GetResult::Found(value)`, `GetResult::Deleted()`, or `GetResult::NotFound()`. The `const` at the end means this method doesn't modify the skip list.

```cpp
    std::vector<Entry> get_all_entries() const;
```
**Iterate**: Walks level 0 from left to right, collecting all entries in sorted order. Used when flushing the MemTable to an SSTable on disk.

```cpp
    size_t count() const { return count_; }
    size_t memory_usage() const { return memory_usage_; }
    bool empty() const { return count_ == 0; }
```
**Size tracking**: `count_` tracks number of entries, `memory_usage_` tracks approximate bytes used. These are defined inline in the header (simple one-liners).

```cpp
    SkipList(const SkipList&) = delete;
    SkipList& operator=(const SkipList&) = delete;
```
**Disable copying**: The skip list manages raw pointers (`new`/`delete`). If we allowed copying, two SkipList objects would point to the same nodes — when one is destroyed, the other would have dangling pointers (use-after-free bug). `= delete` makes the compiler reject any attempt to copy.

#### SkipList Class — Private Members

```cpp
private:
    int random_level();
```
**Random level generator**: Flips coins to determine a new node's height.
```
level = 1
while (random(0,1) < 0.5 AND level < max_level):
    level++
return level
```

```cpp
    SkipListNode* head_;
```
**HEAD sentinel**: The entry point for all operations. Has `max_level` forward pointers, all initially `nullptr`. Doesn't hold real data.

```cpp
    int current_level_;
```
**Current height**: The highest level currently in use. Starts at 0, grows as tall nodes are inserted. Avoids scanning empty upper levels during search.

```cpp
    int max_level_;
    double probability_;
```
**Configuration**: Stored from constructor parameters.

```cpp
    size_t count_;
    size_t memory_usage_;
```
**Statistics**: Track number of entries and approximate memory usage.

```cpp
    mutable std::mt19937 rng_;
    mutable std::uniform_real_distribution<double> dist_;
```
**Random number generator**:
- `std::mt19937` — Mersenne Twister engine. A high-quality pseudo-random number generator. The name comes from its period: 2^19937 - 1 (an astronomically large number).
- `std::uniform_real_distribution<double>` — Converts the engine's output to uniform doubles in [0.0, 1.0).
- `mutable` — These can be modified even in `const` methods. Why? Generating a random number changes the RNG's internal state, but it doesn't change the skip list's logical content. Without `mutable`, we couldn't call `random_level()` from `const` methods.

---

*[Components 7-14 will be added as we implement them]*

---

## 15. References

1. Fidalgo, N. & Ye, P. (2025). *Learned LSM-trees: Two Approaches Using Learned Bloom Filters*. arXiv:2508.00882.
2. Dayan, N. & Idreos, S. (2018). *Monkey: Optimal Navigable Key-Value Store*. SIGMOD.
3. Kraska, T. et al. (2018). *The Case for Learned Index Structures*. SIGMOD.
4. Mitzenmacher, M. (2018). *A Model for Learned Bloom Filters and Optimizing by Sandwiching*. PODC.
5. Bloom, B. H. (1970). *Space/Time Trade-offs in Hash Coding with Allowable Errors*. CACM.
6. O'Neil, P. et al. (1996). *The Log-Structured Merge-Tree (LSM-Tree)*. Acta Informatica.
7. Pugh, W. (1990). *Skip Lists: A Probabilistic Alternative to Balanced Trees*. CACM.

---

*This report is a living document — updated as each component is implemented and tested.*  
*Last updated: April 18, 2026*
