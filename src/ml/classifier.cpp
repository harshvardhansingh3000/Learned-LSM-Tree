#include "ml/classifier.h"
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <numeric>

namespace lsm {

// ═══════════════════════════════════════════════════════════
// JSON Parsing Helpers (shared across all classifier types)
// ═══════════════════════════════════════════════════════════

namespace json_helpers {

std::string find_json_value(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";
    pos += search.size();
    // Skip whitespace
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\n' || json[pos] == '\r' || json[pos] == '\t')) pos++;
    return json.substr(pos);
}

double parse_double(const std::string& json, const std::string& key) {
    std::string val = find_json_value(json, key);
    if (val.empty()) return 0.0;
    return std::stod(val);
}

std::string parse_string(const std::string& json, const std::string& key) {
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '"') return "";
    size_t end = val.find('"', 1);
    if (end == std::string::npos) return "";
    return val.substr(1, end - 1);
}

std::vector<double> parse_double_array(const std::string& json, const std::string& key) {
    std::vector<double> result;
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '[') return result;
    
    size_t pos = 1; // skip '['
    while (pos < val.size() && val[pos] != ']') {
        // Skip whitespace and commas
        while (pos < val.size() && (val[pos] == ' ' || val[pos] == ',' || val[pos] == '\n' || val[pos] == '\r' || val[pos] == '\t')) pos++;
        if (pos >= val.size() || val[pos] == ']') break;
        
        // Skip nested arrays (for 2D arrays, we just want flat arrays here)
        if (val[pos] == '[') { pos++; continue; }
        
        // Parse number
        size_t end;
        try {
            double num = std::stod(val.substr(pos), &end);
            result.push_back(num);
            pos += end;
        } catch (...) {
            pos++;
        }
    }
    return result;
}

std::vector<int> parse_int_array(const std::string& json, const std::string& key) {
    std::vector<int> result;
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '[') return result;
    
    size_t pos = 1;
    while (pos < val.size() && val[pos] != ']') {
        while (pos < val.size() && (val[pos] == ' ' || val[pos] == ',' || val[pos] == '\n' || val[pos] == '\r' || val[pos] == '\t')) pos++;
        if (pos >= val.size() || val[pos] == ']') break;
        
        size_t end;
        try {
            int num = std::stoi(val.substr(pos), &end);
            result.push_back(num);
            pos += end;
        } catch (...) {
            pos++;
        }
    }
    return result;
}

std::vector<std::vector<double>> parse_2d_double_array(const std::string& json, const std::string& key) {
    std::vector<std::vector<double>> result;
    std::string val = find_json_value(json, key);
    if (val.empty() || val[0] != '[') return result;
    
    // Find each inner array
    size_t pos = 1; // skip outer '['
    while (pos < val.size()) {
        // Find next inner '['
        size_t inner_start = val.find('[', pos);
        if (inner_start == std::string::npos) break;
        
        // Find matching ']'
        size_t inner_end = val.find(']', inner_start);
        if (inner_end == std::string::npos) break;
        
        // Parse the inner array
        std::string inner = val.substr(inner_start, inner_end - inner_start + 1);
        std::vector<double> row;
        size_t rpos = 1; // skip '['
        while (rpos < inner.size() && inner[rpos] != ']') {
            while (rpos < inner.size() && (inner[rpos] == ' ' || inner[rpos] == ',' || inner[rpos] == '\n')) rpos++;
            if (rpos >= inner.size() || inner[rpos] == ']') break;
            
            size_t end;
            try {
                double num = std::stod(inner.substr(rpos), &end);
                row.push_back(num);
                rpos += end;
            } catch (...) {
                rpos++;
            }
        }
        
        result.push_back(std::move(row));
        pos = inner_end + 1;
    }
    
    return result;
}

std::string read_file_contents(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return "";
    return std::string((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
}

std::string detect_model_type(const std::string& json) {
    std::string type = parse_string(json, "model_type");
    if (!type.empty()) return type;
    
    // Fallback: if there's "learning_rate", it's GBT; if "layers", it's MLP;
    // if there are trees without learning_rate, it's RF
    if (json.find("\"layers\"") != std::string::npos &&
        json.find("\"scaler_mean\"") != std::string::npos) {
        return "mlp";
    }
    if (json.find("\"learning_rate\"") != std::string::npos) {
        return "gbt";
    }
    return "gbt"; // default fallback
}

} // namespace json_helpers


// ═══════════════════════════════════════════════════════════
// GBT Classifier Implementation
// ═══════════════════════════════════════════════════════════

bool GBTClassifier::load_from_json(const std::string& filepath) {
    std::string json = json_helpers::read_file_contents(filepath);
    if (json.empty()) return false;
    
    // Parse learning rate
    learning_rate_ = json_helpers::parse_double(json, "learning_rate");
    if (learning_rate_ == 0.0) learning_rate_ = 0.1;
    
    // Parse init predictions (for binary, it's the log-odds prior)
    auto init_preds = json_helpers::parse_double_array(json, "init_predictions");
    if (!init_preds.empty()) {
        if (init_preds.size() >= 2 && init_preds[1] > 0) {
            init_prediction_ = std::log(init_preds[1] / (1.0 - init_preds[1] + 1e-10));
        }
    }
    
    // Parse trees — find each tree block
    std::string trees_section = json_helpers::find_json_value(json, "trees");
    if (trees_section.empty()) return false;
    
    // Find all tree blocks by looking for "feature" keys
    size_t search_pos = 0;
    while (true) {
        size_t tree_start = trees_section.find("\"feature\":", search_pos);
        if (tree_start == std::string::npos) break;
        
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
        
        auto features = json_helpers::parse_int_array(tree_json, "feature");
        auto thresholds = json_helpers::parse_double_array(tree_json, "threshold");
        auto left_children = json_helpers::parse_int_array(tree_json, "children_left");
        auto right_children = json_helpers::parse_int_array(tree_json, "children_right");
        auto values = json_helpers::parse_double_array(tree_json, "value");
        
        if (features.empty()) {
            search_pos = brace_end;
            continue;
        }
        
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

double GBTClassifier::predict_proba(const std::vector<double>& features) const {
    if (trees_.empty()) return 0.5;
    
    double raw_score = init_prediction_;
    for (const auto& tree : trees_) {
        raw_score += learning_rate_ * tree.predict(features);
    }
    
    return sigmoid(raw_score);
}

bool GBTClassifier::predict(const std::vector<double>& features, double threshold) const {
    return predict_proba(features) >= threshold;
}


// ═══════════════════════════════════════════════════════════
// Random Forest Classifier Implementation
// ═══════════════════════════════════════════════════════════

bool RFClassifier::load_from_json(const std::string& filepath) {
    std::string json = json_helpers::read_file_contents(filepath);
    if (json.empty()) return false;
    
    n_classes_ = static_cast<int>(json_helpers::parse_double(json, "n_classes"));
    if (n_classes_ < 2) n_classes_ = 2;
    
    // Parse trees — each tree is a flat object with feature/threshold/etc arrays
    std::string trees_section = json_helpers::find_json_value(json, "trees");
    if (trees_section.empty()) return false;
    
    // Find all tree blocks by looking for "feature" keys
    size_t search_pos = 0;
    while (true) {
        size_t tree_start = trees_section.find("\"feature\":", search_pos);
        if (tree_start == std::string::npos) break;
        
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
        
        auto features = json_helpers::parse_int_array(tree_json, "feature");
        auto thresholds = json_helpers::parse_double_array(tree_json, "threshold");
        auto left_children = json_helpers::parse_int_array(tree_json, "children_left");
        auto right_children = json_helpers::parse_int_array(tree_json, "children_right");
        auto values = json_helpers::parse_double_array(tree_json, "value");
        
        // Get per-tree n_classes
        int tree_n_classes = static_cast<int>(json_helpers::parse_double(tree_json, "n_classes"));
        if (tree_n_classes < 2) tree_n_classes = n_classes_;
        
        if (features.empty()) {
            search_pos = brace_end;
            continue;
        }
        
        // Build RF decision tree with per-class values
        RFDecisionTree tree;
        tree.n_classes = tree_n_classes;
        
        for (size_t i = 0; i < features.size(); i++) {
            RFTreeNode node;
            node.feature_index = features[i];
            node.threshold = (i < thresholds.size()) ? thresholds[i] : 0.0;
            node.left_child = (i < left_children.size()) ? left_children[i] : -1;
            node.right_child = (i < right_children.size()) ? right_children[i] : -1;
            
            // Extract per-class values for this node
            // The flattened value array has tree_n_classes entries per node
            // (from sklearn's tree.value which is [n_nodes, n_outputs, n_classes])
            // Since n_outputs=1, it's just [n_nodes * n_classes] flattened
            size_t val_start = i * tree_n_classes;
            for (int c = 0; c < tree_n_classes; c++) {
                size_t idx = val_start + c;
                if (idx < values.size()) {
                    node.class_values.push_back(values[idx]);
                } else {
                    node.class_values.push_back(0.0);
                }
            }
            
            tree.nodes.push_back(std::move(node));
        }
        
        trees_.push_back(std::move(tree));
        search_pos = brace_end;
    }
    
    return !trees_.empty();
}

double RFClassifier::predict_proba(const std::vector<double>& features) const {
    if (trees_.empty()) return 0.5;
    
    // Average the probability of class 1 across all trees
    double sum_proba = 0.0;
    for (const auto& tree : trees_) {
        sum_proba += tree.predict_proba_class1(features);
    }
    
    return sum_proba / static_cast<double>(trees_.size());
}


// ═══════════════════════════════════════════════════════════
// MLP Classifier Implementation
// ═══════════════════════════════════════════════════════════

std::vector<double> MLPClassifier::relu(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::max(0.0, x[i]);
    }
    return result;
}

std::vector<double> MLPClassifier::softmax(const std::vector<double>& x) {
    std::vector<double> result(x.size());
    double max_val = *std::max_element(x.begin(), x.end());
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = std::exp(x[i] - max_val);
        sum += result[i];
    }
    for (size_t i = 0; i < result.size(); i++) {
        result[i] /= sum;
    }
    return result;
}

std::vector<double> MLPClassifier::scale_features(const std::vector<double>& features) const {
    std::vector<double> scaled(features.size());
    for (size_t i = 0; i < features.size(); i++) {
        if (i < scaler_mean_.size() && i < scaler_scale_.size()) {
            double scale = scaler_scale_[i];
            if (scale == 0.0) scale = 1.0;
            scaled[i] = (features[i] - scaler_mean_[i]) / scale;
        } else {
            scaled[i] = features[i];
        }
    }
    return scaled;
}

std::vector<double> MLPClassifier::forward(const std::vector<double>& input) const {
    std::vector<double> current = input;
    
    for (size_t layer_idx = 0; layer_idx < layers_.size(); layer_idx++) {
        const auto& layer = layers_[layer_idx];
        std::vector<double> output(layer.output_size, 0.0);
        
        // Matrix multiply: output = weights^T * current + biases
        for (int j = 0; j < layer.output_size; j++) {
            double sum = layer.biases[j];
            for (int i = 0; i < layer.input_size && i < static_cast<int>(current.size()); i++) {
                if (i < static_cast<int>(layer.weights.size()) && 
                    j < static_cast<int>(layer.weights[i].size())) {
                    sum += layer.weights[i][j] * current[i];
                }
            }
            output[j] = sum;
        }
        
        // Apply activation function
        bool is_last_layer = (layer_idx == layers_.size() - 1);
        if (is_last_layer) {
            // Output layer: softmax for multi-class, sigmoid for binary
            if (output.size() > 1) {
                output = softmax(output);
            } else {
                output[0] = sigmoid(output[0]);
            }
        } else {
            // Hidden layers: ReLU activation
            if (activation_ == "relu") {
                output = relu(output);
            } else if (activation_ == "tanh") {
                for (auto& v : output) v = std::tanh(v);
            } else {
                // logistic/sigmoid
                for (auto& v : output) v = sigmoid(v);
            }
        }
        
        current = std::move(output);
    }
    
    return current;
}

bool MLPClassifier::load_from_json(const std::string& filepath) {
    std::string json = json_helpers::read_file_contents(filepath);
    if (json.empty()) return false;
    
    // Parse basic info
    n_classes_ = static_cast<int>(json_helpers::parse_double(json, "n_classes"));
    if (n_classes_ < 2) n_classes_ = 2;
    
    activation_ = json_helpers::parse_string(json, "activation");
    if (activation_.empty()) activation_ = "relu";
    
    // Parse scaler parameters
    scaler_mean_ = json_helpers::parse_double_array(json, "scaler_mean");
    scaler_scale_ = json_helpers::parse_double_array(json, "scaler_scale");
    
    // Parse layers — find each layer block in the "layers" array
    std::string layers_section = json_helpers::find_json_value(json, "layers");
    if (layers_section.empty()) return false;
    
    // Find each layer block by looking for "input_size" keys
    size_t search_pos = 0;
    while (true) {
        size_t layer_start = layers_section.find("\"input_size\":", search_pos);
        if (layer_start == std::string::npos) break;
        
        // Go backwards to find '{'
        size_t brace_start = layer_start;
        while (brace_start > 0 && layers_section[brace_start] != '{') brace_start--;
        
        // Find matching '}'
        int depth = 0;
        size_t brace_end = brace_start;
        for (size_t i = brace_start; i < layers_section.size(); i++) {
            if (layers_section[i] == '{') depth++;
            if (layers_section[i] == '}') {
                depth--;
                if (depth == 0) { brace_end = i + 1; break; }
            }
        }
        
        std::string layer_json = layers_section.substr(brace_start, brace_end - brace_start);
        
        MLPLayer layer;
        layer.input_size = static_cast<int>(json_helpers::parse_double(layer_json, "input_size"));
        layer.output_size = static_cast<int>(json_helpers::parse_double(layer_json, "output_size"));
        
        // Parse biases
        layer.biases = json_helpers::parse_double_array(layer_json, "biases");
        
        // Parse weights as 2D array
        layer.weights = json_helpers::parse_2d_double_array(layer_json, "weights");
        
        // Validate
        if (layer.input_size > 0 && layer.output_size > 0 && !layer.weights.empty()) {
            layers_.push_back(std::move(layer));
        }
        
        search_pos = brace_end;
    }
    
    return !layers_.empty();
}

double MLPClassifier::predict_proba(const std::vector<double>& features) const {
    if (layers_.empty()) return 0.5;
    
    // Step 1: Scale features
    std::vector<double> scaled = scale_features(features);
    
    // Step 2: Forward pass
    std::vector<double> output = forward(scaled);
    
    // Step 3: Return probability of class 1
    if (output.size() >= 2) {
        // Multi-class output — find index for class 1
        // For binary: output[0] = P(class 0), output[1] = P(class 1)
        return output[1];
    } else if (output.size() == 1) {
        return output[0];
    }
    
    return 0.5;
}


// ═══════════════════════════════════════════════════════════
// Classifier Factory
// ═══════════════════════════════════════════════════════════

std::unique_ptr<BinaryClassifier> create_classifier_from_json(const std::string& filepath) {
    std::string json = json_helpers::read_file_contents(filepath);
    if (json.empty()) return nullptr;
    
    std::string model_type = json_helpers::detect_model_type(json);
    
    std::unique_ptr<BinaryClassifier> clf;
    if (model_type == "mlp") {
        clf = std::make_unique<MLPClassifier>();
    } else if (model_type == "rf") {
        clf = std::make_unique<RFClassifier>();
    } else {
        // Default to GBT
        clf = std::make_unique<GBTClassifier>();
    }
    
    if (clf->load_from_json(filepath)) {
        return clf;
    }
    
    return nullptr;
}


// ═══════════════════════════════════════════════════════════
// Level Classifier — now auto-detects model type per level
// ═══════════════════════════════════════════════════════════

bool LevelClassifier::load(const std::string& model_dir, int num_levels) {
    classifiers_.clear();
    
    for (int level = 0; level < num_levels; level++) {
        std::string path = model_dir + "/level_" + std::to_string(level) + "_classifier.json";
        
        if (!std::filesystem::exists(path)) {
            // No model for this level — skip
            classifiers_.push_back(nullptr);
            continue;
        }
        
        // Use the factory to create the right classifier type
        auto clf = create_classifier_from_json(path);
        if (clf && clf->is_loaded()) {
            std::cout << "  Level " << level << " classifier: " 
                      << clf->type_name() << " (loaded from " << path << ")\n";
            classifiers_.push_back(std::move(clf));
        } else {
            std::cout << "  Level " << level << " classifier: failed to load from " << path << "\n";
            classifiers_.push_back(nullptr);
        }
    }
    
    return !classifiers_.empty();
}

std::vector<bool> LevelClassifier::predict_levels(const Key& key) const {
    auto features = FeatureExtractor::extract(key);
    std::vector<bool> should_check(classifiers_.size(), true);
    
    // Conservative threshold: lower value = more likely to check the level
    // This reduces false negatives (missing keys) at the cost of more Bloom filter probes
    // 0.3 threshold: if probability >= 30%, check the level (vs default 0.5)
    const double kThreshold = 0.3;
    
    for (size_t i = 0; i < classifiers_.size(); i++) {
        if (classifiers_[i] && classifiers_[i]->is_loaded()) {
            double prob = classifiers_[i]->predict_proba(features);
            should_check[i] = (prob >= kThreshold);
        }
        // If no classifier for this level, default to checking it (safe)
    }
    
    return should_check;
}

} // namespace lsm
