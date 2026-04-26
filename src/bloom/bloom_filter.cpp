#include "bloom/bloom_filter.h"
#include "common/hash.h"
#include <cmath>
#include <algorithm>

namespace lsm {

// ─── Constructor (Create New) ─────────────────────────────
// Creates a fresh Bloom filter sized for the expected number of keys.
//
// Steps:
//   1. Calculate total bits: m = num_keys * bits_per_key
//   2. Calculate optimal number of hash functions: k = (m/n) * ln(2)
//   3. Allocate the bit array (all zeros)
//
// Example: 1000 keys, 10 bits/key
//   m = 10,000 bits = 1,250 bytes
//   k = 10 * 0.693 = 6.93 → 7 hash functions
//   FPR ≈ 0.82%

BloomFilter::BloomFilter(size_t num_keys, size_t bits_per_key) {
    // Minimum 1 key to avoid division by zero
    num_keys = std::max(num_keys, static_cast<size_t>(1));

    // Total number of bits
    num_bits_ = num_keys * bits_per_key;

    // Minimum 64 bits (8 bytes) to avoid degenerate cases
    num_bits_ = std::max(num_bits_, static_cast<size_t>(64));

    // Optimal number of hash functions: k = (m/n) * ln(2)
    // Clamped to [1, 30] to avoid extremes
    num_hash_functions_ = static_cast<size_t>(
        std::round(static_cast<double>(bits_per_key) * 0.6931471805599453)  // ln(2)
    );
    num_hash_functions_ = std::max(num_hash_functions_, static_cast<size_t>(1));
    num_hash_functions_ = std::min(num_hash_functions_, static_cast<size_t>(30));

    // Allocate bit array (rounded up to whole bytes, all zeros)
    bits_.resize((num_bits_ + 7) / 8, 0);
}

// ─── Constructor (Load from Disk) ─────────────────────────
// Reconstructs a Bloom filter from serialized data.
// Used when loading an SSTable that has a Bloom filter.

BloomFilter::BloomFilter(const std::vector<uint8_t>& data, size_t num_hash_functions, size_t num_bits)
    : bits_(data)
    , num_bits_(num_bits)
    , num_hash_functions_(num_hash_functions)
{
}

// ─── Set Bit ──────────────────────────────────────────────
// Sets the bit at position 'pos' to 1.
//
// Bit packing: 8 bits per byte.
//   Byte index = pos / 8
//   Bit within byte = pos % 8
//
// Example: set_bit(13)
//   Byte 1 (13/8 = 1), Bit 5 (13%8 = 5)
//   bits_[1] |= (1 << 5)  →  bits_[1] |= 0b00100000
//
// The |= operator is bitwise OR-assign:
//   If the bit was 0, it becomes 1.
//   If the bit was already 1, it stays 1.

void BloomFilter::set_bit(size_t pos) {
    bits_[pos / 8] |= (1 << (pos % 8));
}

// ─── Get Bit ──────────────────────────────────────────────
// Checks if the bit at position 'pos' is 1.
//
// Example: get_bit(13)
//   Byte 1 (13/8 = 1), Bit 5 (13%8 = 5)
//   (bits_[1] >> 5) & 1  →  shift right by 5, check lowest bit

bool BloomFilter::get_bit(size_t pos) const {
    return (bits_[pos / 8] >> (pos % 8)) & 1;
}

// ─── Add ──────────────────────────────────────────────────
// Adds a key to the Bloom filter by setting k bits to 1.
//
// Uses double hashing to generate k hash positions from just 2 hashes:
//   hash1 = murmurhash3(key, seed=0)
//   hash2 = murmurhash3(key, seed=1)
//   position_i = (hash1 + i * hash2) % num_bits_
//
// Example: add("cat") with k=3, m=10
//   hash1 = murmurhash3("cat", 0) = 0x7A3B2C1D
//   hash2 = murmurhash3("cat", 1) = 0x12345678
//   pos_0 = (hash1 + 0*hash2) % 10 = 3
//   pos_1 = (hash1 + 1*hash2) % 10 = 7
//   pos_2 = (hash1 + 2*hash2) % 10 = 1
//   Set bits 3, 7, 1 to 1.

void BloomFilter::add(const Key& key) {
    uint32_t hash1 = murmurhash3(key, 0);
    uint32_t hash2 = murmurhash3(key, 1);

    for (size_t i = 0; i < num_hash_functions_; i++) {
        size_t pos = (hash1 + i * hash2) % num_bits_;
        set_bit(pos);
    }
}

// ─── May Contain ──────────────────────────────────────────
// Checks if a key MIGHT be in the set.
//
// Computes the same k hash positions as add(), then checks if ALL are 1.
//   - If ANY bit is 0 → "definitely not in set" (return false)
//   - If ALL bits are 1 → "maybe in set" (return true — could be false positive)
//
// This is the function that saves disk reads in the LSM-tree:
//   if (!bloom_filter.may_contain(key)) {
//       // Skip this SSTable entirely — key is definitely not here
//   }

bool BloomFilter::may_contain(const Key& key) const {
    uint32_t hash1 = murmurhash3(key, 0);
    uint32_t hash2 = murmurhash3(key, 1);

    for (size_t i = 0; i < num_hash_functions_; i++) {
        size_t pos = (hash1 + i * hash2) % num_bits_;
        if (!get_bit(pos)) {
            return false;  // Definitely not in set
        }
    }

    return true;  // Maybe in set (could be false positive)
}

// ─── Count Ones ───────────────────────────────────────────
// Counts the number of bits set to 1 in the bit array.
// Uses the "popcount" technique: count bits in each byte.

size_t BloomFilter::count_ones() const {
    size_t count = 0;
    for (uint8_t byte : bits_) {
        // Brian Kernighan's bit counting algorithm
        uint8_t b = byte;
        while (b) {
            count++;
            b &= b - 1;  // Clear the lowest set bit
        }
    }
    return count;
}

// ─── Estimated FPR ────────────────────────────────────────
// Estimates the current false positive rate based on the fill ratio.
//
// Formula: FPR ≈ (ones / total_bits) ^ k
// This is an approximation of the theoretical formula:
//   FPR ≈ (1 - e^(-kn/m))^k

double BloomFilter::estimated_fpr() const {
    if (num_bits_ == 0) return 1.0;
    double fill_ratio = static_cast<double>(count_ones()) / static_cast<double>(num_bits_);
    return std::pow(fill_ratio, static_cast<double>(num_hash_functions_));
}

} // namespace lsm
