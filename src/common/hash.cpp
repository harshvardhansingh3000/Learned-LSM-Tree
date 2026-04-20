#include "common/hash.h"

namespace lsm {

// ─── MurmurHash3 (32-bit) Implementation ─────────────────
//
// This is the standard MurmurHash3_x86_32 algorithm by Austin Appleby.
// Public domain — no license restrictions.
//
// The algorithm processes the input in 4-byte chunks:
//   1. For each 4-byte block: multiply, rotate, XOR into hash
//   2. Handle remaining 1-3 bytes (tail)
//   3. Final mixing (avalanche) to ensure all bits are well-distributed
//
// The magic constants (c1, c2, etc.) were chosen by the author
// through extensive testing to maximize distribution quality.

// Helper: rotate left
static inline uint32_t rotl32(uint32_t x, int8_t r) {
    return (x << r) | (x >> (32 - r));
}

// Helper: final mix — ensures all bits of the hash are affected by all input bits
static inline uint32_t fmix32(uint32_t h) {
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;
    return h;
}

uint32_t murmurhash3(const void* key, size_t len, uint32_t seed) {
    const uint8_t* data = static_cast<const uint8_t*>(key);
    const int nblocks = static_cast<int>(len / 4);

    uint32_t h1 = seed;

    // Magic constants
    const uint32_t c1 = 0xcc9e2d51;
    const uint32_t c2 = 0x1b873593;

    // ── Body: process 4-byte blocks ──────────────────────
    const uint32_t* blocks = reinterpret_cast<const uint32_t*>(data + nblocks * 4);

    for (int i = -nblocks; i; i++) {
        uint32_t k1 = blocks[i];

        k1 *= c1;
        k1 = rotl32(k1, 15);
        k1 *= c2;

        h1 ^= k1;
        h1 = rotl32(h1, 13);
        h1 = h1 * 5 + 0xe6546b64;
    }

    // ── Tail: handle remaining 1-3 bytes ─────────────────
    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;

    switch (len & 3) {  // len % 4
        case 3: k1 ^= static_cast<uint32_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k1 ^= static_cast<uint32_t>(tail[1]) << 8;  [[fallthrough]];
        case 1: k1 ^= static_cast<uint32_t>(tail[0]);
                k1 *= c1;
                k1 = rotl32(k1, 15);
                k1 *= c2;
                h1 ^= k1;
    }

    // ── Finalization: mix the hash to ensure avalanche ───
    h1 ^= static_cast<uint32_t>(len);
    h1 = fmix32(h1);

    return h1;
}

} // namespace lsm
