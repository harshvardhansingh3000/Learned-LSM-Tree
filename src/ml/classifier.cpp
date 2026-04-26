#include "ml/classifier.h"
#include <iostream>
#include <algorithm>
#include <filesystem>

namespace lsm {

// ─── Simple JSON Parsing Helpers ──────────────────────────
// We use minimal hand-written parsing instead of a JSON library
// to avoid adding dependencies. These work for the specific JSON
// format exported by our Python training script.

// Find a key in JSON and extract the value after it
static std::string find_json_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n')) pos++;
    return json.substr(pos);
}

double GBTClassifier::parse_double(const std::string& json, const std::string& key) {
    std::string val = find_json_value(json, key);
    if (val.empty()) return 0.0;
    return std::stod(val);
}

std::vector<double> GBTClassifier::parse_double_array(const std::string& json, const std::string& key) {
    std::vector<double> result;
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '[') return result;
    
    size_t pos = 1; // skip '['
    while (pos < val.size() && val[pos] != ']') {
        // Skip whitespace and commas
        while (pos < val.size() && (val[pos] == ' ' || val[pos] == ',' || val[pos] == '\n')) pos++;
        if (pos >= val.size() || val[pos] == ']') break;
        
        // Parse number
        size_t end;
        double num = std::stod(val.substr(pos), &end);
        result.push_back(num);
        pos += end;
    }
    return result;
}

std::vector<int> GBTClassifier::parse_int_array(const std::string& json, const std::string& key) {
    std::vector<int> result;
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '[') return result;
    
    size_t pos = 1;
    while (pos < val.size() && val[pos] != ']') {
        while (pos < val.size() && (val[pos] == ' ' || val[pos] == ',' || val[pos] == '\n')) pos++;
        if (pos >= val.size() || val[pos] == ']') break;
        
        size_t end;
        int num = std::stoi(val.substr(pos), &end);
        result.push_back(num);
        pos += end;
    }
    return result;
}

// ─── Load Model from JSON ─────────────────────────────────
// Parses the JSON file and builds the decision tree ensemble.
//
// The JSON format (from sklearn export):
// {
//   "learning_rate": 0.1,
//   "n_estimators": 30,
//   "trees": [
//     [  // estimator 0
//       {  // tree for class 0 (binary: only 1 tree per estimator)
//         "feature": [2, -2, 5, -2, -2],
//         "threshold": [0.5, -2.0, 0.3, -2.0, -2.0],
//         "children_left": [1, -1, 3, -1, -1],
//         "children_right": [2, -1, 4, -1, -1],
//         "value": [0.0, -0.1, 0.0, 0.2, -0.3],
//         "n_nodes": 5
//       }
//     ],
//     ...
//   ]
// }

bool GBTClassifier::load_from_json(const std::string& filepath) {
    // Read entire file
    std::ifstream file(filepath);
    if (!file.is_open()) return false;
    
    std::string json((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    file.close();
    
    // Parse learning rate
    learning_rate_ = parse_double(json, "learning_rate");
    if (learning_rate_ == 0.0) learning_rate_ = 0.1;
    
    // Parse init predictions (for binary, it's the log-odds prior)
    auto init_preds = parse_double_array(json, "init_predictions");
    if (!init_preds.empty()) {
        // For binary classification, use log-odds of class 1
        if (init_preds.size() >= 2 && init_preds[1] > 0) {
            init_prediction_ = std::log(init_preds[1] / (1.0 - init_preds[1] + 1e-10));
        }
    }
    
    // Parse trees — find each tree block
    // We look for "feature" arrays within the "trees" section
    std::string trees_section = find_json_value(json, "trees");
    if (trees_section.empty()) return false;
    
    // Find all tree blocks by looking for "feature" keys
    size_t search_pos = 0;
    while (true) {
        size_t tree_start = trees_section.find("\"feature\":", search_pos);
        if (tree_start == std::string::npos) break;
        
        // Find the enclosing {} for this tree
        // Go backwards to find '{'
        size_t brace_start = tree_start;
        while (brace_start > 0 && trees_section[brace_start] != '{') brace_start--;
        
        // Find matching '}'
        int depth = 0;
        size_t brace_end = brace_start;
        for (size_t i = brace_start; i < trees_section.size(); i++) {
            if (trees_section[i] == '{') depth++;
            if (trees_section[i] == '}') {
                depth--;
                if (depth == 0) { brace_end = i + 1; break; }
            }
        }
        
        std::string tree_json = trees_section.substr(brace_start, brace_end - brace_start);
        
        // Parse tree arrays
        auto features = parse_int_array(tree_json, "feature");
        auto thresholds = parse_double_array(tree_json, "threshold");
        auto left_children = parse_int_array(tree_json, "children_left");
        auto right_children = parse_int_array(tree_json, "children_right");
        auto values = parse_double_array(tree_json, "value");
        
        if (features.empty()) {
            search_pos = brace_end;
            continue;
        }
        
        // Build decision tree
        DecisionTree tree;
        for (size_t i = 0; i < features.size(); i++) {
            TreeNode node;
            node.feature_index = features[i];
            node.threshold = (i < thresholds.size()) ? thresholds[i] : 0.0;
            node.left_child = (i < left_children.size()) ? left_children[i] : -1;
            node.right_child = (i < right_children.size()) ? right_children[i] : -1;
            node.value = (i < values.size()) ? values[i] : 0.0;
            tree.nodes.push_back(node);
        }
        
        trees_.push_back(std::move(tree));
        search_pos = brace_end;
    }
    
    return !trees_.empty();
}

// ─── Predict Probability ──────────────────────────────────
// For binary GBT:
//   raw_score = init_prediction + learning_rate * sum(tree.predict(features))
//   probability = sigmoid(raw_score)

double GBTClassifier::predict_proba(const std::vector<double>& features) const {
    if (trees_.empty()) return 0.5;
    
    double raw_score = init_prediction_;
    for (const auto& tree : trees_) {
        raw_score += learning_rate_ * tree.predict(features);
    }
    
    return sigmoid(raw_score);
}

// ─── Predict Class ────────────────────────────────────────

bool GBTClassifier::predict(const std::vector<double>& features, double threshold) const {
    return predict_proba(features) >= threshold;
}

// ─── Level Classifier ─────────────────────────────────────

bool LevelClassifier::load(const std::string& model_dir, int num_levels) {
    classifiers_.clear();
    
    for (int level = 0; level < num_levels; level++) {
        std::string path = model_dir + "/level_" + std::to_string(level) + "_classifier.json";
        
        if (!std::filesystem::exists(path)) {
            // No model for this level — skip
            classifiers_.push_back(nullptr);
            continue;
        }
        
        auto clf = std::make_unique<GBTClassifier>();
        if (clf->load_from_json(path)) {
            classifiers_.push_back(std::move(clf));
        } else {
            classifiers_.push_back(nullptr);
        }
    }
    
    return !classifiers_.empty();
}

std::vector<bool> LevelClassifier::predict_levels(const Key& key) const {
    auto features = FeatureExtractor::extract(key);
    std::vector<bool> should_check(classifiers_.size(), true);
    
    for (size_t i = 0; i < classifiers_.size(); i++) {
        if (classifiers_[i] && classifiers_[i]->is_loaded()) {
            // Classifier predicts: should we check this level?
            should_check[i] = classifiers_[i]->predict(features);
        }
        // If no classifier for this level, default to checking it (safe)
    }
    
    return should_check;
}

} // namespace lsm
