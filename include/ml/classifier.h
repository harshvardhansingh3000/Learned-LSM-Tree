#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <memory>
#include <algorithm>
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
// A single decision tree used in GBT and RF ensembles.

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

// ─── Random Forest Decision Tree ──────────────────────────
// A decision tree from a Random Forest. Leaf nodes store per-class
// sample counts rather than a single value. We need multiple values
// per node to do majority voting.

struct RFTreeNode {
    int feature_index;     // Which feature to split on (-2 = leaf node)
    double threshold;
    int left_child;
    int right_child;
    std::vector<double> class_values;  // Per-class sample counts at this node
    
    bool is_leaf() const { return feature_index == -2; }
};

class RFDecisionTree {
public:
    std::vector<RFTreeNode> nodes;
    int n_classes = 2;
    
    // Predict: walk the tree and return the class with the most samples
    int predict_class(const std::vector<double>& features) const {
        int node_idx = 0;
        while (!nodes[node_idx].is_leaf()) {
            const auto& node = nodes[node_idx];
            if (features[node.feature_index] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        // Return the class with the highest count
        const auto& vals = nodes[node_idx].class_values;
        if (vals.empty()) return 0;
        return static_cast<int>(std::max_element(vals.begin(), vals.end()) - vals.begin());
    }
    
    // Return the probability of class 1 (for binary classification)
    double predict_proba_class1(const std::vector<double>& features) const {
        int node_idx = 0;
        while (!nodes[node_idx].is_leaf()) {
            const auto& node = nodes[node_idx];
            if (features[node.feature_index] <= node.threshold) {
                node_idx = node.left_child;
            } else {
                node_idx = node.right_child;
            }
        }
        const auto& vals = nodes[node_idx].class_values;
        if (vals.size() < 2) return 0.0;
        double total = 0;
        for (double v : vals) total += v;
        if (total == 0) return 0.0;
        return vals[1] / total;  // probability of class 1
    }
};

// ─── Abstract Base Classifier ─────────────────────────────
// Common interface for all binary classifiers (GBT, RF, MLP).
// Each per-level classifier answers: "is this key at this level?"

class BinaryClassifier {
public:
    virtual ~BinaryClassifier() = default;
    
    // Load model from JSON file exported by train_classifier.py
    virtual bool load_from_json(const std::string& filepath) = 0;
    
    // Predict probability that the key belongs to this level [0, 1]
    virtual double predict_proba(const std::vector<double>& features) const = 0;
    
    // Predict class: true = key is at this level, false = not
    virtual bool predict(const std::vector<double>& features, double threshold = 0.5) const {
        return predict_proba(features) >= threshold;
    }
    
    // Is the model loaded?
    virtual bool is_loaded() const = 0;
    
    // Name of this classifier type (for logging)
    virtual std::string type_name() const = 0;
};

// ─── GBT Classifier ──────────────────────────────────────
// Gradient Boosted Trees binary classifier.
// Loaded from JSON exported by Python sklearn.
//
// For binary classification:
//   prediction = sigmoid(init_prediction + learning_rate * sum(tree_predictions))
//   if prediction > 0.5 → class 1 (key IS at this level)
//   else → class 0 (key is NOT at this level)

class GBTClassifier : public BinaryClassifier {
public:
    GBTClassifier() : learning_rate_(0.1), init_prediction_(0.0) {}
    
    bool load_from_json(const std::string& filepath) override;
    double predict_proba(const std::vector<double>& features) const override;
    bool predict(const std::vector<double>& features, double threshold = 0.5) const override;
    bool is_loaded() const override { return !trees_.empty(); }
    std::string type_name() const override { return "GBT"; }
    
    size_t num_trees() const { return trees_.size(); }

private:
    std::vector<DecisionTree> trees_;
    double learning_rate_;
    double init_prediction_;
    
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
};

// ─── Random Forest Classifier ─────────────────────────────
// Random Forest binary classifier.
// Each tree votes independently; majority vote / averaged probability
// determines the final prediction.

class RFClassifier : public BinaryClassifier {
public:
    RFClassifier() = default;
    
    bool load_from_json(const std::string& filepath) override;
    double predict_proba(const std::vector<double>& features) const override;
    bool is_loaded() const override { return !trees_.empty(); }
    std::string type_name() const override { return "RF"; }
    
    size_t num_trees() const { return trees_.size(); }

private:
    std::vector<RFDecisionTree> trees_;
    int n_classes_ = 2;
};

// ─── MLP Layer ────────────────────────────────────────────
// A single fully-connected layer: output = activation(weights * input + bias)

struct MLPLayer {
    std::vector<std::vector<double>> weights;  // [input_size][output_size]
    std::vector<double> biases;                // [output_size]
    int input_size = 0;
    int output_size = 0;
};

// ─── MLP Classifier ──────────────────────────────────────
// Multi-Layer Perceptron (basic neural network) binary classifier.
// Loaded from JSON with layer weights, biases, and scaler params.
//
// Forward pass:
//   1. Normalize input features using scaler (mean, scale)
//   2. For each hidden layer: output = relu(W * x + b)
//   3. Output layer: softmax or sigmoid

class MLPClassifier : public BinaryClassifier {
public:
    MLPClassifier() = default;
    
    bool load_from_json(const std::string& filepath) override;
    double predict_proba(const std::vector<double>& features) const override;
    bool is_loaded() const override { return !layers_.empty(); }
    std::string type_name() const override { return "MLP"; }

private:
    std::vector<MLPLayer> layers_;
    std::vector<double> scaler_mean_;
    std::vector<double> scaler_scale_;
    std::string activation_ = "relu";
    int n_classes_ = 2;
    
    // Activation functions
    static std::vector<double> relu(const std::vector<double>& x);
    static std::vector<double> softmax(const std::vector<double>& x);
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    
    // Scale features using stored scaler parameters
    std::vector<double> scale_features(const std::vector<double>& features) const;
    
    // Forward pass through the network
    std::vector<double> forward(const std::vector<double>& input) const;
};

// ─── JSON Parsing Helpers (shared) ────────────────────────
// Minimal hand-written JSON parsing to avoid library dependencies.

namespace json_helpers {
    std::string find_json_value(const std::string& json, const std::string& key);
    double parse_double(const std::string& json, const std::string& key);
    std::string parse_string(const std::string& json, const std::string& key);
    std::vector<double> parse_double_array(const std::string& json, const std::string& key);
    std::vector<int> parse_int_array(const std::string& json, const std::string& key);
    // Parse a 2D array like [[1,2],[3,4]]
    std::vector<std::vector<double>> parse_2d_double_array(const std::string& json, const std::string& key);
    // Read entire file into string
    std::string read_file_contents(const std::string& filepath);
    // Detect model type from JSON ("gbt", "rf", or "mlp")
    std::string detect_model_type(const std::string& json);
}

// ─── Classifier Factory ──────────────────────────────────
// Creates the right classifier type based on the model_type field in the JSON.

std::unique_ptr<BinaryClassifier> create_classifier_from_json(const std::string& filepath);

// ─── Level Classifier ─────────────────────────────────────
// Per-level binary classifier that predicts whether a key exists at a specific level.
// Used in the classifier-augmented GET path.
// Now supports GBT, RF, or MLP per level — automatically detected from JSON.

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
    
    // Get the type name of the classifier at a given level
    std::string classifier_type(size_t level) const {
        if (level < classifiers_.size() && classifiers_[level]) {
            return classifiers_[level]->type_name();
        }
        return "none";
    }

private:
    std::vector<std::unique_ptr<BinaryClassifier>> classifiers_;
};

} // namespace lsm
