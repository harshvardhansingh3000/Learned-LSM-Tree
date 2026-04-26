#!/usr/bin/env python3
"""
ML Classifier Training for LSM-Tree Level Prediction
=====================================================

This script trains a Gradient Boosted Trees classifier that predicts
which level of the LSM-tree a key is likely to reside in.

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

    3. The trained model is exported to data/models/classifier.json
       which the C++ code loads for inference.
"""

import json
import os
import struct
import sys
import hashlib
import math
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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


# ─── Export Model to JSON ──────────────────────────────────
def export_model_to_json(model, filepath):
    """
    Export a trained GradientBoostingClassifier to JSON format
    that can be loaded by C++ for inference.
    
    The JSON contains the tree structure: for each tree, we store
    the feature index, threshold, left/right children, and leaf values.
    """
    model_data = {
        "n_estimators": model.n_estimators,
        "learning_rate": model.learning_rate,
        "n_classes": model.n_classes_,
        "n_features": model.n_features_in_,
        "classes": model.classes_.tolist(),
        "trees": []
    }
    
    # For binary classification, there's one tree per estimator
    # For multi-class, there are n_classes trees per estimator
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
    
    # Also store the initial prediction (prior)
    # For GBT, the initial prediction is the log-odds of each class
    model_data["init_predictions"] = model.init_.class_prior_.tolist() if hasattr(model.init_, 'class_prior_') else [0.0] * model.n_classes_
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(model_data, f)
    
    print(f"Model exported to {filepath}")
    print(f"  File size: {os.path.getsize(filepath) / 1024:.1f} KB")


# ─── Main ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("LSM-Tree ML Classifier Training")
    print("=" * 60)
    
    # Step 1: Generate training data
    print("\n[1/4] Generating training data...")
    X, y, keys = generate_training_data(num_keys=10000, num_levels=3)
    print(f"  Total samples: {len(X)}")
    print(f"  Features per sample: {len(X[0])}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Step 2: Split into train/test
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 3: Train the model
    print("\n[3/4] Training Gradient Boosted Trees classifier...")
    model = GradientBoostingClassifier(
        n_estimators=50,       # 50 trees (paper uses 200, we use fewer for speed)
        max_depth=4,           # Max depth per tree (paper uses 6)
        learning_rate=0.1,     # Step size
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 4: Export model
    print("[4/4] Exporting model...")
    export_model_to_json(model, "data/models/classifier.json")
    
    # Also export a simpler format: per-level binary classifiers
    # For each level, train a binary classifier: "is key at this level?"
    print("\n[Bonus] Training per-level binary classifiers...")
    for level in range(3):
        y_binary = (y == level).astype(int)
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X, y_binary, test_size=0.2, random_state=42
        )
        
        clf = GradientBoostingClassifier(
            n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42
        )
        clf.fit(X_train_b, y_train_b)
        
        acc = accuracy_score(y_test_b, clf.predict(X_test_b))
        print(f"  Level {level} classifier: accuracy={acc:.4f}")
        
        # Export
        export_model_to_json(clf, f"data/models/level_{level}_classifier.json")
    
    print("\n" + "=" * 60)
    print("Training complete! Models saved to data/models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
