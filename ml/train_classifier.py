#!/usr/bin/env python3
"""
ML Classifier Training for LSM-Tree Level Prediction
=====================================================

This script trains THREE different classifiers and selects the best one:
  1. Gradient Boosted Trees (GBT)
  2. Random Forest (RF)
  3. Multi-Layer Perceptron (MLP) — a basic neural network

Each classifier predicts which level of the LSM-tree a key is likely
to reside in.

Based on: "Learned LSM-trees: Two Approaches Using Learned Bloom Filters"
          (Fidalgo & Ye, Harvard, 2025)

Usage:
    1. Run the LSM-tree database and load data:
       ./build/lsm_db
       > load 10000
       > flush
       > compact
       > quit

    2. Run this script to collect training data and train the model:
       python3 ml/train_classifier.py

    3. The best trained model is exported to data/models/
       which the C++ code loads for inference.
"""

import json
import os
import struct
import sys
import hashlib
import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")  # suppress convergence warnings for MLP

# ─── MurmurHash3 (must match C++ implementation) ──────────
# We need the SAME hash function as C++ to generate matching features.
# Using mmh3 library if available, otherwise a simple fallback.

try:
    import mmh3
    def murmurhash3(key: str, seed: int = 0) -> int:
        """MurmurHash3 32-bit, matching C++ implementation."""
        return mmh3.hash(key, seed, signed=False)
except ImportError:
    # Fallback: use hashlib (won't match C++ exactly, but works for demo)
    def murmurhash3(key: str, seed: int = 0) -> int:
        h = hashlib.md5((str(seed) + key).encode()).hexdigest()
        return int(h[:8], 16)
    print("Warning: mmh3 not installed. Install with: pip3 install mmh3")
    print("Using fallback hash (features won't match C++ exactly)")


# ─── Feature Extraction (must match C++ FeatureExtractor) ──
def extract_features(key: str) -> list:
    """Extract features from a key string. Must match C++ FeatureExtractor::extract()."""
    features = []
    
    k = float(murmurhash3(key, 0))
    UINT32_MAX = 4294967295.0
    
    # 1. Raw hash value (normalized)
    features.append(k / UINT32_MAX)
    
    # 2. Key length
    features.append(float(len(key)))
    
    # 3. Power features
    k_norm = k / UINT32_MAX
    features.append(k_norm * k_norm)
    features.append(k_norm * k_norm * k_norm)
    
    # 4. Logarithmic features
    features.append(math.log1p(k))
    features.append(math.log1p(len(key)))
    
    # 5. Trigonometric features
    features.append(math.sin(k_norm * math.pi))
    features.append(math.cos(k_norm * math.pi))
    
    # 6. Digit/character statistics
    digit_sum = sum(int(c) for c in key if c.isdigit())
    digit_count = sum(1 for c in key if c.isdigit())
    alpha_count = sum(1 for c in key if c.isalpha())
    features.append(float(digit_sum))
    features.append(float(digit_count))
    features.append(float(alpha_count))
    
    # 7. First and last character values
    features.append(float(ord(key[0])) if key else 0.0)
    features.append(float(ord(key[-1])) if key else 0.0)
    
    # 8. Modulo features
    hash_val = murmurhash3(key, 0)
    features.append(float(hash_val % 7))
    features.append(float(hash_val % 13))
    features.append(float(hash_val % 97))
    
    # 9. Bit-level features
    features.append(float(bin(hash_val).count('1')))  # popcount
    features.append(float(hash_val >> 24))  # high byte
    
    # 10. Second hash
    hash2 = murmurhash3(key, 42)
    features.append(float(hash2) / UINT32_MAX)
    
    return features


# ─── Generate Training Data ───────────────────────────────
def generate_training_data(num_keys=10000, num_levels=3):
    """
    Generate synthetic training data that simulates key-level distribution.
    
    In a real system, we'd read the actual SSTable metadata to know which
    keys are at which level. For training, we simulate the distribution:
    - Keys are distributed across levels based on when they were written
    - Older keys tend to be in deeper levels (after compaction)
    - Newer keys are in L0 or MemTable
    """
    X = []  # Feature vectors
    y = []  # Labels (level number)
    keys = []  # For reference
    
    for i in range(num_keys):
        # Generate key
        key = f"key_{i:06d}"
        
        # Simulate level assignment:
        # - First 60% of keys → Level 2 (oldest, deepest)
        # - Next 30% → Level 1
        # - Last 10% → Level 0 (newest)
        if i < num_keys * 0.6:
            level = min(2, num_levels - 1)
        elif i < num_keys * 0.9:
            level = min(1, num_levels - 1)
        else:
            level = 0
        
        features = extract_features(key)
        X.append(features)
        y.append(level)
        keys.append(key)
    
    # Also add some random non-existent keys (label = -1 means "not in any level")
    for i in range(num_keys // 2):
        key = f"nonexist_{i:06d}"
        features = extract_features(key)
        X.append(features)
        y.append(-1)  # Not in any level
        keys.append(key)
    
    return np.array(X), np.array(y), keys


# ─── Export GBT/RF Model to JSON ───────────────────────────
def export_tree_model_to_json(model, filepath, model_type="gbt"):
    """
    Export a trained GradientBoostingClassifier or RandomForestClassifier
    to JSON format that can be loaded by C++ for inference.
    
    The JSON contains the tree structure: for each tree, we store
    the feature index, threshold, left/right children, and leaf values.
    """
    model_data = {
        "model_type": model_type,  # "gbt" or "rf"
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist(),
    }

    if model_type == "gbt":
        model_data["n_estimators"] = model.n_estimators
        model_data["learning_rate"] = model.learning_rate
        model_data["n_classes"] = model.n_classes_
        model_data["trees"] = []

        for estimator_idx in range(model.n_estimators):
            estimator_trees = []
            n_trees_per_estimator = model.estimators_[estimator_idx].shape[0]
            for class_idx in range(n_trees_per_estimator):
                tree = model.estimators_[estimator_idx][class_idx].tree_
                tree_data = {
                    "feature": tree.feature.tolist(),
                    "threshold": tree.threshold.tolist(),
                    "children_left": tree.children_left.tolist(),
                    "children_right": tree.children_right.tolist(),
                    "value": tree.value.flatten().tolist(),
                    "n_nodes": tree.node_count
                }
                estimator_trees.append(tree_data)
            model_data["trees"].append(estimator_trees)

        # Store the initial prediction (prior)
        model_data["init_predictions"] = (
            model.init_.class_prior_.tolist()
            if hasattr(model.init_, 'class_prior_')
            else [0.0] * model.n_classes_
        )

    elif model_type == "rf":
        model_data["n_estimators"] = model.n_estimators
        model_data["n_classes"] = model.n_classes_
        model_data["trees"] = []

        for estimator in model.estimators_:
            tree = estimator.tree_
            tree_data = {
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "children_left": tree.children_left.tolist(),
                "children_right": tree.children_right.tolist(),
                # For RF, value[node][class] gives the count of samples per class
                # We store the full value array flattened
                "value": tree.value.flatten().tolist(),
                "n_nodes": tree.node_count,
                "n_classes": int(tree.n_classes[0]) if len(tree.n_classes) > 0 else 2,
                "n_outputs": tree.n_outputs,
            }
            model_data["trees"].append(tree_data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(model_data, f)
    
    print(f"  Model exported to {filepath}")
    print(f"    File size: {os.path.getsize(filepath) / 1024:.1f} KB")


# ─── Export MLP Model to JSON ──────────────────────────────
def export_mlp_model_to_json(model, scaler, filepath):
    """
    Export a trained MLPClassifier to JSON format
    that can be loaded by C++ for inference.
    
    The JSON contains:
    - Layer weights and biases
    - Activation function type
    - Scaler parameters (mean, scale) for feature normalization
    """
    model_data = {
        "model_type": "mlp",
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist(),
        "n_classes": len(model.classes_),
        "n_layers": len(model.coefs_),
        "activation": model.activation,
        "layers": [],
        # Scaler parameters for feature normalization
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    for i, (weights, biases) in enumerate(zip(model.coefs_, model.intercepts_)):
        layer_data = {
            "weights": weights.tolist(),  # shape: (n_input, n_output)
            "biases": biases.tolist(),    # shape: (n_output,)
            "input_size": weights.shape[0],
            "output_size": weights.shape[1],
        }
        model_data["layers"].append(layer_data)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(model_data, f)

    print(f"  Model exported to {filepath}")
    print(f"    File size: {os.path.getsize(filepath) / 1024:.1f} KB")


# ─── Print Metrics Table ──────────────────────────────────
def print_metrics_comparison(metrics_dict):
    """
    Pretty-print a comparison table of all classifier metrics.
    metrics_dict: { name: { "accuracy": ..., "precision": ..., ... } }
    """
    names = list(metrics_dict.keys())
    metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    metric_labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-Score (macro)"]

    # Header
    col_width = 24
    header = f"{'Metric':<{col_width}}"
    for name in names:
        header += f"| {name:^{col_width}} "
    print("=" * len(header))
    print("  CLASSIFIER COMPARISON TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for label, key in zip(metric_labels, metric_keys):
        row = f"  {label:<{col_width - 2}}"
        for name in names:
            val = metrics_dict[name][key]
            row += f"| {val:^{col_width}.4f} "
        print(row)

    print("-" * len(header))

    # Best selection
    best_name = max(metrics_dict, key=lambda n: metrics_dict[n]["accuracy"])
    best_acc = metrics_dict[best_name]["accuracy"]
    print(f"\n  ★ BEST CLASSIFIER: {best_name}  (Accuracy = {best_acc:.4f} = {best_acc*100:.1f}%)")
    print()

    return best_name


# ─── Main ─────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  LSM-Tree ML Classifier Training — Multi-Model Comparison")
    print("=" * 70)
    
    # Step 1: Generate training data
    print("\n[1/6] Generating training data...")
    X, y, keys = generate_training_data(num_keys=10000, num_levels=3)
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {len(X[0])}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Step 2: Split into train/test
    print("\n[2/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features for MLP (neural networks need normalized input)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ─── Step 3: Train ALL THREE Classifiers ───────────────
    print("\n[3/6] Training all three classifiers...\n")

    classifiers = {}
    metrics = {}

    # ── 3a) Gradient Boosted Trees ──────────────────────────
    print("  ── Training Gradient Boosted Trees (GBT) ──")
    gbt = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    gbt.fit(X_train, y_train)
    y_pred_gbt = gbt.predict(X_test)
    acc_gbt = accuracy_score(y_test, y_pred_gbt)
    classifiers["GBT"] = gbt
    metrics["GBT"] = {
        "accuracy": acc_gbt,
        "precision_macro": precision_score(y_test, y_pred_gbt, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_gbt, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred_gbt, average='macro', zero_division=0),
        "y_pred": y_pred_gbt,
    }
    print(f"     Accuracy: {acc_gbt:.4f} ({acc_gbt*100:.1f}%)")
    print(f"     Classification Report:")
    print(classification_report(y_test, y_pred_gbt, zero_division=0))

    # ── 3b) Random Forest ───────────────────────────────────
    print("  ── Training Random Forest (RF) ──")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    classifiers["RF"] = rf
    metrics["RF"] = {
        "accuracy": acc_rf,
        "precision_macro": precision_score(y_test, y_pred_rf, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_rf, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred_rf, average='macro', zero_division=0),
        "y_pred": y_pred_rf,
    }
    print(f"     Accuracy: {acc_rf:.4f} ({acc_rf*100:.1f}%)")
    print(f"     Classification Report:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))

    # ── 3c) Multi-Layer Perceptron (MLP) ────────────────────
    print("  ── Training Multi-Layer Perceptron (MLP) ──")
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),    # 2 hidden layers: 64 and 32 neurons
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
    )
    mlp.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp.predict(X_test_scaled)
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    classifiers["MLP"] = mlp
    metrics["MLP"] = {
        "accuracy": acc_mlp,
        "precision_macro": precision_score(y_test, y_pred_mlp, average='macro', zero_division=0),
        "recall_macro": recall_score(y_test, y_pred_mlp, average='macro', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred_mlp, average='macro', zero_division=0),
        "y_pred": y_pred_mlp,
    }
    print(f"     Accuracy: {acc_mlp:.4f} ({acc_mlp*100:.1f}%)")
    print(f"     Classification Report:")
    print(classification_report(y_test, y_pred_mlp, zero_division=0))

    # ─── Step 4: Compare and Select Best ───────────────────
    print("\n[4/6] Comparing classifiers...\n")
    # Remove y_pred from metrics for the comparison table
    metrics_for_table = {
        name: {k: v for k, v in m.items() if k != "y_pred"}
        for name, m in metrics.items()
    }
    best_name = print_metrics_comparison(metrics_for_table)

    # Also print confusion matrices
    print("  Confusion Matrices:")
    for name in ["GBT", "RF", "MLP"]:
        print(f"\n  ── {name} ──")
        cm = confusion_matrix(y_test, metrics[name]["y_pred"])
        print(f"  {cm}")

    # ─── Step 5: Export the winning multi-class model ──────
    print(f"\n[5/6] Exporting BEST multi-class model ({best_name})...")
    if best_name == "GBT":
        export_tree_model_to_json(gbt, "data/models/classifier.json", model_type="gbt")
    elif best_name == "RF":
        export_tree_model_to_json(rf, "data/models/classifier.json", model_type="rf")
    else:  # MLP
        export_mlp_model_to_json(mlp, scaler, "data/models/classifier.json")

    # ─── Step 6: Train per-level binary classifiers (best type) ──
    print(f"\n[6/6] Training per-level binary classifiers (using {best_name})...")

    per_level_metrics = {}
    for level in range(3):
        y_binary = (y == level).astype(int)
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )

        # Scale for MLP
        X_train_b_scaled = scaler.fit_transform(X_train_b)
        X_test_b_scaled = scaler.transform(X_test_b)

        # Train all 3 classifiers for this level
        level_clfs = {}
        level_metrics = {}

        # GBT
        clf_gbt = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42
        )
        clf_gbt.fit(X_train_b, y_train_b)
        acc = accuracy_score(y_test_b, clf_gbt.predict(X_test_b))
        level_clfs["GBT"] = clf_gbt
        level_metrics["GBT"] = acc

        # RF
        clf_rf = RandomForestClassifier(
            n_estimators=50, max_depth=4, random_state=42, n_jobs=-1
        )
        clf_rf.fit(X_train_b, y_train_b)
        acc = accuracy_score(y_test_b, clf_rf.predict(X_test_b))
        level_clfs["RF"] = clf_rf
        level_metrics["RF"] = acc

        # MLP
        clf_mlp = MLPClassifier(
            hidden_layer_sizes=(32, 16), activation='relu', solver='adam',
            max_iter=300, random_state=42, early_stopping=True
        )
        clf_mlp.fit(X_train_b_scaled, y_train_b)
        acc = accuracy_score(y_test_b, clf_mlp.predict(X_test_b_scaled))
        level_clfs["MLP"] = clf_mlp
        level_metrics["MLP"] = acc

        # Pick best for this level
        best_level_clf_name = max(level_metrics, key=level_metrics.get)
        best_acc = level_metrics[best_level_clf_name]

        per_level_metrics[level] = level_metrics

        print(f"\n  Level {level}:")
        print(f"    GBT accuracy:  {level_metrics['GBT']:.4f}")
        print(f"    RF  accuracy:  {level_metrics['RF']:.4f}")
        print(f"    MLP accuracy:  {level_metrics['MLP']:.4f}")
        print(f"    → Selected: {best_level_clf_name} (accuracy={best_acc:.4f})")

        # Export the best per-level classifier
        filepath = f"data/models/level_{level}_classifier.json"
        if best_level_clf_name == "GBT":
            export_tree_model_to_json(
                level_clfs["GBT"], filepath, model_type="gbt"
            )
        elif best_level_clf_name == "RF":
            export_tree_model_to_json(
                level_clfs["RF"], filepath, model_type="rf"
            )
        else:  # MLP
            # Refit scaler for this level's data
            export_mlp_model_to_json(level_clfs["MLP"], scaler, filepath)

    # ─── Save a metadata file with the selected models ─────
    selection_info = {
        "multiclass_best": best_name,
        "multiclass_accuracy": metrics[best_name]["accuracy"],
        "per_level_selections": {},
    }
    for level in range(3):
        lm = per_level_metrics[level]
        best_l = max(lm, key=lm.get)
        selection_info["per_level_selections"][f"level_{level}"] = {
            "selected": best_l,
            "accuracy": lm[best_l],
            "all_accuracies": lm,
        }

    os.makedirs("data/models", exist_ok=True)
    with open("data/models/model_selection.json", "w") as f:
        json.dump(selection_info, f, indent=2)

    # ─── Final Summary ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE — SUMMARY")
    print("=" * 70)
    print(f"\n  Multi-class classifier: {best_name} "
          f"(accuracy={metrics[best_name]['accuracy']:.4f})")
    for level in range(3):
        lm = per_level_metrics[level]
        best_l = max(lm, key=lm.get)
        print(f"  Level {level} binary classifier: {best_l} "
              f"(accuracy={lm[best_l]:.4f})")
    print(f"\n  Models saved to data/models/")
    print(f"  Selection info saved to data/models/model_selection.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
