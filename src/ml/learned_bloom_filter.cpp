#include "ml/learned_bloom_filter.h"
#include <filesystem>
#include <iostream>

namespace lsm {

// ─── Load Model ───────────────────────────────────────────
bool LearnedBloomFilter::load_model(const std::string& model_path) {
    if (!std::filesystem::exists(model_path)) {
        return false;
    }
    
    // Use the classifier factory to auto-detect model type (GBT/RF/MLP)
    classifier_ = create_classifier_from_json(model_path);
    if (classifier_ && classifier_->is_loaded()) {
        // Estimate model size from file
        model_size_bytes_ = std::filesystem::file_size(model_path);
        return true;
    }
    
    return false;
}

// ─── Init Backup ──────────────────────────────────────────
void LearnedBloomFilter::init_backup(size_t expected_keys, double expected_fn_rate) {
    // The backup Bloom filter only needs to store false negatives.
    // If the model has ~5% FN rate, we only need space for 5% of keys.
    size_t expected_fn_keys = static_cast<size_t>(expected_keys * expected_fn_rate);
    if (expected_fn_keys < 10) expected_fn_keys = 10;  // Minimum
    
    // Use 10 bits per key in the backup (standard)
    backup_bloom_ = std::make_unique<BloomFilter>(expected_fn_keys, 10);
}

// ─── Add ──────────────────────────────────────────────────
// Called for every key known to be in the level's SSTables.
// The classifier predicts whether the key is in the set.
// If the classifier says NO (false negative), we add it to the backup.

void LearnedBloomFilter::add(const Key& key) {
    total_keys_added_++;
    
    if (!classifier_ || !classifier_->is_loaded()) {
        // No model loaded — add everything to backup (degenerate case)
        if (backup_bloom_) {
            backup_bloom_->add(key);
        }
        return;
    }
    
    // Extract features and classify
    auto features = FeatureExtractor::extract(key);
    double prob = classifier_->predict_proba(features);
    
    if (prob < kClassifierThreshold) {
        // Classifier says "not in set" — but we KNOW this key IS in the set.
        // This is a false negative! Add to backup Bloom filter.
        false_negatives_caught_++;
        if (backup_bloom_) {
            backup_bloom_->add(key);
        }
    }
    // If classifier says "in set" (prob >= threshold), no backup needed.
    // The classifier will correctly identify this key at query time.
}

// ─── May Contain ──────────────────────────────────────────
// Algorithm 2 from the paper (Learned Bloom Filter GET):
//
//   if classifier(key) == POSITIVE → return true (maybe in set)
//   else if backup_bloom.may_contain(key) → return true (false neg caught!)
//   else → return false (definitely not in set)
//
// This guarantees ZERO false negatives:
//   - If the key IS in the set and classifier says YES → true ✓
//   - If the key IS in the set and classifier says NO → backup says YES → true ✓
//   - If the key is NOT in the set → classifier may say YES (FP) or NO
//     and backup may say YES (FP from backup BF) or NO

bool LearnedBloomFilter::may_contain(const Key& key) const {
    if (!classifier_ || !classifier_->is_loaded()) {
        // No model — fall back to always returning true (safe but no benefit)
        return true;
    }
    
    // Step 1: Ask the classifier
    auto features = FeatureExtractor::extract(key);
    double prob = classifier_->predict_proba(features);
    
    if (prob >= kClassifierThreshold) {
        // Classifier says "probably in set"
        return true;
    }
    
    // Step 2: Classifier says "probably not in set"
    // Check backup Bloom filter for false negatives
    if (backup_bloom_) {
        return backup_bloom_->may_contain(key);
    }
    
    // No backup filter — be safe, return true
    return true;
}

} // namespace lsm
