// ─── Bloom Filter Tests ───────────────────────────────────
// Tests for the Bloom filter and MurmurHash3.
// We verify: zero false negatives, reasonable FPR, serialization, and hash quality.

#include <gtest/gtest.h>
#include "bloom/bloom_filter.h"
#include "common/hash.h"
#include <set>
#include <string>

using namespace lsm;

// ─── MurmurHash3 Tests ───────────────────────────────────

// Hash should be deterministic (same input → same output)
TEST(HashTest, Deterministic) {
    uint32_t h1 = murmurhash3("hello", 0);
    uint32_t h2 = murmurhash3("hello", 0);
    EXPECT_EQ(h1, h2);
}

// Different inputs should (almost always) produce different hashes
TEST(HashTest, DifferentInputs) {
    uint32_t h1 = murmurhash3("cat", 0);
    uint32_t h2 = murmurhash3("dog", 0);
    EXPECT_NE(h1, h2);
}

// Different seeds should produce different hashes for the same input
TEST(HashTest, DifferentSeeds) {
    uint32_t h1 = murmurhash3("hello", 0);
    uint32_t h2 = murmurhash3("hello", 1);
    EXPECT_NE(h1, h2);
}

// Hash should handle empty strings
TEST(HashTest, EmptyString) {
    uint32_t h = murmurhash3("", 0);
    // Should not crash, and should produce a valid hash
    (void)h;  // Just verify it doesn't crash
}

// ─── Bloom Filter Tests ──────────────────────────────────

// Zero false negatives: every added key must be found
TEST(BloomFilterTest, ZeroFalseNegatives) {
    BloomFilter bf(1000, 10);

    // Add 1000 keys
    for (int i = 0; i < 1000; i++) {
        bf.add("key_" + std::to_string(i));
    }

    // Every added key MUST be found (zero false negatives)
    for (int i = 0; i < 1000; i++) {
        EXPECT_TRUE(bf.may_contain("key_" + std::to_string(i)))
            << "False negative for key_" << i;
    }
}

// Non-existent keys should mostly return false
TEST(BloomFilterTest, FalsePositiveRate) {
    BloomFilter bf(1000, 10);

    // Add 1000 keys with prefix "in_"
    for (int i = 0; i < 1000; i++) {
        bf.add("in_" + std::to_string(i));
    }

    // Check 10000 keys that were NOT added (prefix "out_")
    int false_positives = 0;
    int total_checks = 10000;
    for (int i = 0; i < total_checks; i++) {
        if (bf.may_contain("out_" + std::to_string(i))) {
            false_positives++;
        }
    }

    double fpr = static_cast<double>(false_positives) / total_checks;

    // With 10 bits/key, theoretical FPR ≈ 0.82%
    // Allow up to 3% to account for randomness
    EXPECT_LT(fpr, 0.03) << "FPR too high: " << (fpr * 100) << "%";

    // Should have SOME false positives (it's probabilistic)
    // With 10000 checks and ~1% FPR, we expect ~100 false positives
    // But it could be 0 in rare cases, so we don't assert > 0
}

// Empty Bloom filter should return false for everything
TEST(BloomFilterTest, EmptyFilter) {
    BloomFilter bf(100, 10);

    EXPECT_FALSE(bf.may_contain("anything"));
    EXPECT_FALSE(bf.may_contain("hello"));
    EXPECT_FALSE(bf.may_contain(""));
}

// Single key
TEST(BloomFilterTest, SingleKey) {
    BloomFilter bf(1, 10);
    bf.add("only_key");

    EXPECT_TRUE(bf.may_contain("only_key"));
    // Other keys should mostly return false
    EXPECT_FALSE(bf.may_contain("other_key"));
}

// Serialization: save and reload
TEST(BloomFilterTest, Serialization) {
    BloomFilter bf1(100, 10);

    // Add some keys
    for (int i = 0; i < 100; i++) {
        bf1.add("key_" + std::to_string(i));
    }

    // Serialize
    auto data = bf1.data();
    size_t k = bf1.num_hash_functions();

    // Reconstruct from serialized data
    BloomFilter bf2(data, k, bf1.size_bits());

    // All keys should still be found
    for (int i = 0; i < 100; i++) {
        EXPECT_TRUE(bf2.may_contain("key_" + std::to_string(i)))
            << "Lost key after serialization: key_" << i;
    }

    // Same size
    EXPECT_EQ(bf1.size_bytes(), bf2.size_bytes());
}

// Size tracking
TEST(BloomFilterTest, SizeTracking) {
    BloomFilter bf(1000, 10);

    // 1000 keys × 10 bits = 10000 bits = 1250 bytes
    EXPECT_EQ(bf.size_bits(), 10000u);
    EXPECT_EQ(bf.size_bytes(), 1250u);
    EXPECT_EQ(bf.num_hash_functions(), 7u);  // 10 * ln(2) ≈ 7
}

// Count ones should increase as we add keys
TEST(BloomFilterTest, CountOnes) {
    BloomFilter bf(100, 10);

    size_t initial = bf.count_ones();
    EXPECT_EQ(initial, 0u);

    bf.add("key1");
    size_t after_one = bf.count_ones();
    EXPECT_GT(after_one, 0u);

    bf.add("key2");
    size_t after_two = bf.count_ones();
    EXPECT_GE(after_two, after_one);  // >= because some bits might overlap
}

// Large scale test: 10000 keys
TEST(BloomFilterTest, LargeScale) {
    BloomFilter bf(10000, 10);

    for (int i = 0; i < 10000; i++) {
        bf.add("key_" + std::to_string(i));
    }

    // Zero false negatives
    for (int i = 0; i < 10000; i++) {
        EXPECT_TRUE(bf.may_contain("key_" + std::to_string(i)));
    }

    // Check FPR
    int false_positives = 0;
    for (int i = 10000; i < 20000; i++) {
        if (bf.may_contain("key_" + std::to_string(i))) {
            false_positives++;
        }
    }
    double fpr = static_cast<double>(false_positives) / 10000;
    EXPECT_LT(fpr, 0.03);
}

// Different bits_per_key should give different FPR
TEST(BloomFilterTest, DifferentBitsPerKey) {
    // 5 bits/key → higher FPR
    BloomFilter bf_low(1000, 5);
    // 20 bits/key → lower FPR
    BloomFilter bf_high(1000, 20);

    for (int i = 0; i < 1000; i++) {
        std::string key = "key_" + std::to_string(i);
        bf_low.add(key);
        bf_high.add(key);
    }

    // Count false positives for non-existent keys
    int fp_low = 0, fp_high = 0;
    for (int i = 1000; i < 11000; i++) {
        std::string key = "key_" + std::to_string(i);
        if (bf_low.may_contain(key)) fp_low++;
        if (bf_high.may_contain(key)) fp_high++;
    }

    // Higher bits/key should have fewer false positives
    EXPECT_LT(fp_high, fp_low);
}
