#pragma once

#include <cstdint>
#include <string>

namespace lsm {

// ─── SSTable File Format ──────────────────────────────────
//
// An SSTable (Sorted String Table) is an IMMUTABLE file on disk containing
// sorted key-value pairs. Once written, it is never modified — only read or deleted.
//
// File Layout (updated with Bloom filter):
// ┌─────────────────────────────────────────────────────────┐
// │                    Data Block 0                          │
// │  ┌──────┬──────┬─────┬───────┬─────┬───────┐           │
// │  │keylen│key   │type │seqnum │vlen │value  │ (entry 1) │
// │  │keylen│key   │type │seqnum │vlen │value  │ (entry 2) │
// │  └──────┴──────┴─────┴───────┴─────┴───────┘           │
// ├─────────────────────────────────────────────────────────┤
// │                    Data Block N                          │
// ├─────────────────────────────────────────────────────────┤
// │                    Index Block                           │
// │  For each data block:                                   │
// │  [first_key (string)] [block_offset (8)] [block_size(4)]│
// ├─────────────────────────────────────────────────────────┤
// │                    Bloom Filter Block                    │
// │  [num_hash_functions (4)] [data_size (4)] [data...]     │
// ├─────────────────────────────────────────────────────────┤
// │                    Footer                                │
// │  [index_offset(8)] [index_size(4)] [entry_count(4)]     │
// │  [min_key_len(4)] [min_key] [max_key_len(4)] [max_key]  │
// ├─────────────────────────────────────────────────────────┤
// │  [bloom_offset (4 bytes)]  ← offset where bloom starts  │
// │  [magic_number (4 bytes)]  ← always 0x53535401           │
// └─────────────────────────────────────────────────────────┘
//
// Reading strategy:
//   1. Read last 8 bytes: bloom_offset(4) + magic(4)
//   2. Validate magic number
//   3. Read bloom filter from bloom_offset
//   4. Parse footer (between index block end and bloom_offset)
//   5. Read index block from index_offset
//   6. Point lookups: bloom filter → fence pointers → data block

// Magic number to identify SSTable files
constexpr uint32_t SSTABLE_MAGIC = 0x53535401;  // "SST\x01"

// ─── Index Entry ──────────────────────────────────────────
struct IndexEntry {
    std::string first_key;   // First key in this data block (fence pointer)
    uint64_t block_offset;   // Byte offset of this data block in the file
    uint32_t block_size;     // Size of this data block in bytes
};

// ─── Footer ──────────────────────────────────────────────
struct Footer {
    uint64_t index_offset;   // Where the index block starts
    uint32_t index_size;     // Size of the index block in bytes
    uint32_t entry_count;    // Total number of key-value entries
    std::string min_key;     // Smallest key in this SSTable
    std::string max_key;     // Largest key in this SSTable
    uint32_t bloom_offset;   // Where the bloom filter block starts (read from end of file)
    uint32_t magic;          // Magic number for validation
};

} // namespace lsm
