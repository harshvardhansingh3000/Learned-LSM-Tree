#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include "common/types.h"
#include "common/hash.h"

namespace lsm {

// ─── Feature Engineering ──────────────────────────────────
//
// Converts a key string into a numeric feature vector for the ML classifier.
//
// From the paper (Fidalgo & Ye, 2025):
//   "We engineer features from keys using logarithmic transforms,
//    trigonometric functions, digit-based statistics, modulo operations,
//    and binary encodings."
//
// The paper uses 45 features. We implement a subset that captures
// the key patterns needed for level prediction:
//
// 1. Hash-based features (capture key distribution)
// 2. Length-based features
// 3. Character statistics
// 4. Modulo features (capture clustering)
// 5. Mathematical transforms (log, trig)

class FeatureExtractor {
public:
    // Number of features produced
    static constexpr size_t NUM_FEATURES = 20;

    // Extract features from a key string.
    // Returns a vector of NUM_FEATURES doubles.
    static std::vector<double> extract(const Key& key);

private:
    // Convert key to a numeric value using hash
    static double key_to_numeric(const Key& key);
};

// ─── Implementation (inline for header-only) ──────────────

inline std::vector<double> FeatureExtractor::extract(const Key& key) {
    std::vector<double> features;
    features.reserve(NUM_FEATURES);

    double k = key_to_numeric(key);

    // 1. Raw hash value (normalized to [0, 1])
    features.push_back(k / static_cast<double>(UINT32_MAX));

    // 2. Key length
    features.push_back(static_cast<double>(key.size()));

    // 3. Power features: k^2, k^3 (capture non-linear patterns)
    double k_norm = k / static_cast<double>(UINT32_MAX);
    features.push_back(k_norm * k_norm);           // k^2
    features.push_back(k_norm * k_norm * k_norm);  // k^3

    // 4. Logarithmic features (capture exponential relationships)
    features.push_back(std::log1p(k));              // log(1+k)
    features.push_back(std::log1p(key.size()));     // log(1+len)

    // 5. Trigonometric features (capture periodic patterns)
    features.push_back(std::sin(k_norm * 3.14159));
    features.push_back(std::cos(k_norm * 3.14159));

    // 6. Digit/character statistics
    int digit_sum = 0;
    int digit_count = 0;
    int alpha_count = 0;
    for (char c : key) {
        if (c >= '0' && c <= '9') {
            digit_sum += (c - '0');
            digit_count++;
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            alpha_count++;
        }
    }
    features.push_back(static_cast<double>(digit_sum));
    features.push_back(static_cast<double>(digit_count));
    features.push_back(static_cast<double>(alpha_count));

    // 7. First and last character values
    features.push_back(key.empty() ? 0.0 : static_cast<double>(key.front()));
    features.push_back(key.empty() ? 0.0 : static_cast<double>(key.back()));

    // 8. Modulo features (capture clustering/partitioning)
    uint32_t hash_val = murmurhash3(key, 0);
    features.push_back(static_cast<double>(hash_val % 7));
    features.push_back(static_cast<double>(hash_val % 13));
    features.push_back(static_cast<double>(hash_val % 97));

    // 9. Bit-level features
    features.push_back(static_cast<double>(__builtin_popcount(hash_val)));  // bit count
    features.push_back(static_cast<double>(hash_val >> 24));  // high byte

    // 10. Second hash (different seed — independent feature)
    uint32_t hash2 = murmurhash3(key, 42);
    features.push_back(static_cast<double>(hash2) / static_cast<double>(UINT32_MAX));

    return features;
}

inline double FeatureExtractor::key_to_numeric(const Key& key) {
    return static_cast<double>(murmurhash3(key, 0));
}

} // namespace lsm
