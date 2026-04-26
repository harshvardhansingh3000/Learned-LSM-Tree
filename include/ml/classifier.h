#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <memory>
#include "ml/feature_eng.h"

namespace lsm {

// ─── Decision Tree Node ───────────────────────────────────
// A single node in a decision tree. Either an internal node (splits on a feature)
// or a leaf node (returns a prediction value).

struct TreeNode {
    int feature_index;     // Which feature to split on (-2 = leaf node)
    double threshold;      // Split threshold (go left if feature <= threshold)
    int left_child;        // Index of left child node
    int right_child;       // Index of right child node
    double value;          // Leaf value (prediction)
    
    bool is_leaf() const { return feature_index == -2; }
};

// ─── Decision Tree ────────────────────────────────────────
// A single decision tree in the GBT ensemble.

class DecisionTree {
public:
    std::vector<TreeNode> nodes;
    
    // Predict: walk the tree from root to leaf
    double predict(const std::vector<double>& features) const {
        int node_idx = 0;
        while (!nodes[node_idx].is_leaf()) {
            const auto& node = nodes[node_idx];
            if (features[node.feature_index] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        return nodes[node_idx].value;
    }
};

// ─── GBT Classifier ──────────────────────────────────────
// Gradient Boosted Trees binary classifier.
// Loaded from JSON exported by Python sklearn.
//
// For binary classification:
//   prediction = sigmoid(init_prediction + learning_rate * sum(tree_predictions))
//   if prediction > 0.5 → class 1 (key IS at this level)
//   else → class 0 (key is NOT at this level)

class GBTClassifier {
public:
    GBTClassifier() : learning_rate_(0.1), init_prediction_(0.0) {}
    
    // Load model from JSON file exported by train_classifier.py
    bool load_from_json(const std::string& filepath);
    
    // Predict probability that the key belongs to this level
    // Returns value in [0, 1]
    double predict_proba(const std::vector<double>& features) const;
    
    // Predict class: true = key is at this level, false = not
    bool predict(const std::vector<double>& features, double threshold = 0.5) const;
    
    // Is the model loaded?
    bool is_loaded() const { return !trees_.empty(); }
    
    size_t num_trees() const { return trees_.size(); }

private:
    std::vector<DecisionTree> trees_;
    double learning_rate_;
    double init_prediction_;
    
    // Simple JSON value extraction helpers
    static double parse_double(const std::string& json, const std::string& key);
    static std::vector<double> parse_double_array(const std::string& json, const std::string& key);
    static std::vector<int> parse_int_array(const std::string& json, const std::string& key);
    
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

// ─── Level Classifier ─────────────────────────────────────
// Per-level binary classifier that predicts whether a key exists at a specific level.
// Used in the classifier-augmented GET path.

class LevelClassifier {
public:
    // Load per-level classifiers from JSON files
    bool load(const std::string& model_dir, int num_levels);
    
    // Predict which levels to check for a given key
    // Returns a vector of booleans: true = check this level, false = skip
    std::vector<bool> predict_levels(const Key& key) const;
    
    // Is the classifier loaded and ready?
    bool is_loaded() const { return !classifiers_.empty(); }
    
    // Statistics
    size_t num_levels() const { return classifiers_.size(); }

private:
    std::vector<std::unique_ptr<GBTClassifier>> classifiers_;
};

} // namespace lsm
