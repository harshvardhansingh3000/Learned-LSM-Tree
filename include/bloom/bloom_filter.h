#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include "common/types.h"

namespace lsm {

// ─── Bloom Filter ─────────────────────────────────────────
//
// A Bloom filter is a space-efficient probabilistic data structure that
// answers the question: "Is this key POSSIBLY in the set?"
//
// Two possible answers:
//   - "Definitely NOT in the set" → 100% accurate, skip this SSTable
//   - "MAYBE in the set"         → might be wrong (false positive)
//
// It NEVER says "no" when the key IS in the set (zero false negatives).
//
// How it works:
//   1. Create a bit array of m bits, all set to 0
//   2. To ADD a key: compute k hash values → set those k bits to 1
//   3. To CHECK a key: compute k hash values → if ALL k bits are 1, "maybe yes"
//                                             → if ANY bit is 0, "definitely no"
//
// Visual example (m=10 bits, k=3 hash functions):
//
//   Empty:     [0][0][0][0][0][0][0][0][0][0]
//
//   Add "cat": hash1=2, hash2=5, hash3=8
//              [0][0][1][0][0][1][0][0][1][0]
//
//   Add "dog": hash1=1, hash2=5, hash3=7
//              [0][1][1][0][0][1][0][1][1][0]
//
//   Check "cat": bits 2,5,8 → all 1 → "MAYBE yes" ✓ (correct)
//   Check "fox": bits 1,3,8 → bit 3 is 0 → "DEFINITELY no" ✓ (correct)
//   Check "pig": bits 1,2,5 → all 1 → "MAYBE yes" ✗ (false positive!)
//
// False positive rate formula:
//   FPR ≈ (1 - e^(-kn/m))^k
//   where n = number of keys, m = number of bits, k = number of hash functions
//
// Optimal k (minimizes FPR):
//   k = (m/n) * ln(2) ≈ 0.693 * (m/n)
//
// With 10 bits per key (m/n = 10):
//   k = 10 * 0.693 ≈ 7 hash functions
//   FPR ≈ 0.82%

class BloomFilter {
public:
    // Create a Bloom filter for the expected number of keys with given bits per key.
    // bits_per_key = 10 gives ~0.82% FPR (standard choice).
    BloomFilter(size_t num_keys, size_t bits_per_key = 10);

    // Create from serialized data (for loading from SSTable)
    // num_bits is the ORIGINAL num_bits_ used during add() — must match!
    BloomFilter(const std::vector<uint8_t>& data, size_t num_hash_functions, size_t num_bits);

    // ── Operations ────────────────────────────────────────

    // Add a key to the Bloom filter (set k bits to 1)
    void add(const Key& key);

    // Check if a key MIGHT be in the set
    // Returns true = "maybe yes" (could be false positive)
    // Returns false = "definitely no" (always correct)
    bool may_contain(const Key& key) const;

    // ── Serialization (for saving/loading with SSTables) ──

    // Get the raw bit array (for writing to disk)
    const std::vector<uint8_t>& data() const { return bits_; }

    // Get the number of hash functions
    size_t num_hash_functions() const { return num_hash_functions_; }

    // ── Statistics ─────────────────────────────────────────

    // Size of the bit array in bits
    size_t size_bits() const { return num_bits_; }

    // Size of the bit array in bytes
    size_t size_bytes() const { return bits_.size(); }

    // Number of bits set to 1 (for measuring fill ratio)
    size_t count_ones() const;

    // Estimated false positive rate based on current fill
    double estimated_fpr() const;

private:
    std::vector<uint8_t> bits_;       // The bit array (packed into bytes)
    size_t num_bits_;                  // Total number of bits
    size_t num_hash_functions_;        // k — number of hash functions

    // Set bit at position pos to 1
    void set_bit(size_t pos);

    // Check if bit at position pos is 1
    bool get_bit(size_t pos) const;
};

} // namespace lsm
