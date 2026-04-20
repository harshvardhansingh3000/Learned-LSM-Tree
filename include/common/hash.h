#pragma once

#include <cstdint>
#include <string>

namespace lsm {

// ─── MurmurHash3 ─────────────────────────────────────────
//
// A fast, non-cryptographic hash function used for Bloom filters.
//
// Why MurmurHash3?
//   - Very fast (processes 4 bytes per cycle)
//   - Excellent distribution (uniform spread across output range)
//   - Low collision rate
//   - Used by LevelDB, RocksDB, Cassandra, and many databases
//   - NOT cryptographic (don't use for passwords/security)
//
// We use it to generate multiple hash values for Bloom filters.
// By using different seeds, one hash function can produce k independent hashes:
//   hash_0 = murmurhash3(key, seed=0)
//   hash_1 = murmurhash3(key, seed=1)
//   ...
//   hash_k = murmurhash3(key, seed=k)
//
// Actually, we use a more efficient technique called "double hashing":
//   hash_i = hash1 + i * hash2
// This gives us k hash values from just 2 hash computations.

// Compute MurmurHash3 (32-bit) of the given data with the given seed.
uint32_t murmurhash3(const void* key, size_t len, uint32_t seed);

// Convenience: hash a string
inline uint32_t murmurhash3(const std::string& key, uint32_t seed = 0) {
    return murmurhash3(key.data(), key.size(), seed);
}

} // namespace lsm
