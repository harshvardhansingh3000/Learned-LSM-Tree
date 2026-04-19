#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace lsm {

// ─── LSM-Tree Configuration ──────────────────────────────
// These constants define the behavior of our LSM-tree.
// They follow the Monkey paper's recommendations for optimal performance.

struct Config {
    // ── MemTable ──────────────────────────────────────────
    // Maximum size of the in-memory MemTable before it's flushed to disk.
    // Monkey paper recommends 1 MB.
    // When MemTable exceeds this, it becomes immutable and is written as an SSTable.
    size_t memtable_size_bytes = 1 * 1024 * 1024;  // 1 MB

    // ── Level Size Ratio ──────────────────────────────────
    // Each level is T times larger than the previous one.
    // Level 0: ~1 MB, Level 1: ~10 MB, Level 2: ~100 MB, Level 3: ~1 GB, ...
    // T=10 is the standard choice balancing compaction cost and query performance.
    size_t size_ratio = 10;

    // ── Bloom Filter ──────────────────────────────────────
    // Base bits per key for Bloom filters.
    // More bits = lower false positive rate = less wasted disk reads.
    // 10 bits/key gives ~1% FPR (standard choice).
    // With Monkey allocation, deeper levels get more bits.
    size_t bloom_bits_per_key = 10;

    // ── SSTable ───────────────────────────────────────────
    // Size of each data block within an SSTable.
    // Fence pointers index at block granularity.
    // 4 KB matches typical OS page size for efficient I/O.
    size_t sstable_block_size = 4 * 1024;  // 4 KB

    // ── Data Directory ────────────────────────────────────
    // Where all persistent data is stored (SSTables, WAL, models).
    std::string data_dir = "data";

    // ── WAL ───────────────────────────────────────────────
    // Write-ahead log directory (inside data_dir).
    std::string wal_dir = "data/wal";

    // ── SSTables ──────────────────────────────────────────
    // SSTable storage directory (inside data_dir).
    std::string sstable_dir = "data/sstables";

    // ── ML Models ─────────────────────────────────────────
    // Trained model storage directory (inside data_dir).
    std::string model_dir = "data/models";

    // ── Skip List ─────────────────────────────────────────
    // Maximum height of the skip list.
    // With probability p=0.5, expected height is log2(n).
    // 12 levels supports up to ~4096 entries efficiently.
    // For 1 MB MemTable with 116-byte entries, that's ~8800 entries.
    int skip_list_max_level = 12;

    // Probability of promoting a node to the next level.
    // p=0.5 gives the best balance of search speed vs memory.
    // Higher p = taller lists = faster search but more memory.
    double skip_list_probability = 0.5;
};

// Global default configuration
// We use a function to avoid static initialization order issues.
inline Config default_config() {
    return Config{};
}

} // namespace lsm
