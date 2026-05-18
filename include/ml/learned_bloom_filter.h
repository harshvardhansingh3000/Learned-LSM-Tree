#pragma once

#include <vector>
#include <string>
#include <memory>
#include "common/types.h"
#include "bloom/bloom_filter.h"
#include "ml/classifier.h"
#include "ml/feature_eng.h"

namespace lsm {

// ─── Learned Bloom Filter ─────────────────────────────────
//
// Second approach from the paper (Fidalgo & Ye, 2025):
//
// Replaces traditional Bloom filters with a compact hybrid structure:
//   1. A lightweight ML classifier predicts membership
//   2. A small backup Bloom filter stores the classifier's false negatives
//
// Lookup algorithm:
//   if classifier(key) == POSITIVE:
//       → key is probably in the set, proceed to SSTable search
//   else:  // classifier says NEGATIVE
//       if backup_bloom.may_contain(key):
//           → classifier was wrong (false negative), search SSTable anyway
//       else:
//           → key is definitely NOT in the set, skip this level
//
// Benefits:
//   - Memory: Traditional BF uses 10 bits/key. Learned BF uses model (small)
//     + backup BF (only for false negatives, much smaller).
//   - Correctness: Zero false negatives guaranteed (backup catches them all).
//   - Latency: Approximately same as traditional (model inference ≈ hash computation).
//
// Memory comparison (from paper):
//   Traditional: ~68 KB per level (at 10 bits/key for 5000 keys)
//   Learned:     ~12 KB per level (model + tiny backup for ~1% false negatives)
//   Savings:     ~80% reduction

class LearnedBloomFilter {
public:
    LearnedBloomFilter() = default;
    
    // Load the learned filter for a specific level
    // model_path: path to the per-level classifier JSON
    // The backup bloom filter will be populated during add() calls
    bool load_model(const std::string& model_path);
    
    // Add a key that is known to be in the set (builds backup filter)
    // Must be called for ALL keys in the level's SSTables after model is loaded.
    // The model predicts each key; if it misses (false negative), the key
    // goes into the backup Bloom filter.
    void add(const Key& key);
    
    // Initialize the backup bloom filter with expected false negative count
    void init_backup(size_t expected_keys, double expected_fn_rate = 0.05);
    
    // Check if a key MIGHT be in the set.
    // Returns true = "maybe yes" (should search the SSTable)
    // Returns false = "definitely no" (safe to skip)
    // Guarantees: ZERO false negatives (backup catches them all)
    bool may_contain(const Key& key) const;
    
    // Is the model loaded?
    bool is_loaded() const { return classifier_ != nullptr && classifier_->is_loaded(); }
    
    // Statistics
    size_t model_size_bytes() const { return model_size_bytes_; }
    size_t backup_size_bytes() const { return backup_bloom_ ? backup_bloom_->size_bytes() : 0; }
    size_t total_size_bytes() const { return model_size_bytes_ + backup_size_bytes(); }
    size_t false_negatives_caught() const { return false_negatives_caught_; }
    size_t total_keys_added() const { return total_keys_added_; }
    std::string classifier_type() const { 
        return classifier_ ? classifier_->type_name() : "none"; 
    }
    
private:
    std::unique_ptr<BinaryClassifier> classifier_;
    std::unique_ptr<BloomFilter> backup_bloom_;
    size_t model_size_bytes_ = 0;
    size_t false_negatives_caught_ = 0;
    size_t total_keys_added_ = 0;
    
    // Threshold for classifier prediction
    // Lower = more conservative (fewer false negatives, bigger backup)
    static constexpr double kClassifierThreshold = 0.5;
};

} // namespace lsm
