#pragma once

#include <cstdint>
#include <string>

namespace lsm {

// ─── SSTable File Format ──────────────────────────────────
//
// An SSTable (Sorted String Table) is an IMMUTABLE file on disk containing
// sorted key-value pairs. Once written, it is never modified — only read or deleted.
//
// Why immutable?
//   - No need for locks during reads (safe for concurrent access)
//   - No fragmentation (data is written sequentially)
//   - Simple crash recovery (file is either complete or not)
//
// File Layout:
// ┌─────────────────────────────────────────────────────────┐
// │                    Data Block 0                          │
// │  ┌──────┬──────┬─────┬───────┬─────┬───────┐           │
// │  │keylen│key   │type │seqnum │vlen │value  │ (entry 1) │
// │  │keylen│key   │type │seqnum │vlen │value  │ (entry 2) │
// │  │ ...  │      │     │       │     │       │           │
// │  └──────┴──────┴─────┴───────┴─────┴───────┘           │
// ├─────────────────────────────────────────────────────────┤
// │                    Data Block 1                          │
// │  (same format as above)                                 │
// ├─────────────────────────────────────────────────────────┤
// │                    ...                                   │
// ├─────────────────────────────────────────────────────────┤
// │                    Data Block N                          │
// ├─────────────────────────────────────────────────────────┤
// │                    Index Block                           │
// │  For each data block:                                   │
// │  ┌──────────────┬──────────────┬──────────────┐        │
// │  │ first_key    │ block_offset │ block_size   │        │
// │  │ (fence ptr)  │ (8 bytes)    │ (4 bytes)    │        │
// │  └──────────────┴──────────────┴──────────────┘        │
// ├─────────────────────────────────────────────────────────┤
// │                    Footer (fixed size)                   │
// │  ┌──────────────┬──────────────┬──────────────┐        │
// │  │ index_offset │ index_size   │ entry_count  │        │
// │  │ (8 bytes)    │ (4 bytes)    │ (4 bytes)    │        │
// │  ├──────────────┼──────────────┤              │        │
// │  │ min_key_len  │ min_key      │              │        │
// │  │ max_key_len  │ max_key      │              │        │
// │  ├──────────────┼──────────────┤              │        │
// │  │ magic_number │              │              │        │
// │  │ (4 bytes)    │              │              │        │
// │  └──────────────┴──────────────┴──────────────┘        │
// └─────────────────────────────────────────────────────────┘
//
// Reading an SSTable:
//   1. Read the footer (last bytes of file) to find index_offset
//   2. Read the index block to get fence pointers
//   3. Binary search fence pointers to find the right data block
//   4. Read that data block and scan for the key
//
// This is much faster than scanning the entire file!

// Magic number to identify SSTable files (helps detect corruption)
constexpr uint32_t SSTABLE_MAGIC = 0x53535401;  // "SST\x01"

// ─── Index Entry ──────────────────────────────────────────
// One entry in the index block. Points to a data block.
// The "fence pointer" is the first key in that data block.
struct IndexEntry {
    std::string first_key;   // First key in this data block (fence pointer)
    uint64_t block_offset;   // Byte offset of this data block in the file
    uint32_t block_size;     // Size of this data block in bytes
};

// ─── Footer ──────────────────────────────────────────────
// Fixed metadata at the end of the file.
// We read this first to know where everything else is.
struct Footer {
    uint64_t index_offset;   // Where the index block starts
    uint32_t index_size;     // Size of the index block in bytes
    uint32_t entry_count;    // Total number of key-value entries
    std::string min_key;     // Smallest key in this SSTable
    std::string max_key;     // Largest key in this SSTable
    uint32_t magic;          // Magic number for validation
};

} // namespace lsm
