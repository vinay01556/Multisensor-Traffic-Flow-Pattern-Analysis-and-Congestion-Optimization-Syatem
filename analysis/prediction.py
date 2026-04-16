"""
IntelliTraffic Pro - Congestion Prediction Module

Uses a sliding window of recent traffic states to predict
the upcoming congestion level using KNN / Decision Tree.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import pickle

FEATURE_COLS = [
    "vehicle_count", "avg_speed", "std_speed",
    "avg_distance",  "occupancy_ratio", "flow_rate"
]


def create_prediction_dataset(features_df, lookahead=1):
    """
    Build (X, y) for prediction.
    X = current window features
    y = traffic state `lookahead` windows into the future
    """
    if "cluster_kmeans" not in features_df.columns:
        raise ValueError("Run clustering.py first to generate cluster labels.")

    X = features_df[FEATURE_COLS].iloc[:-lookahead].values
    y = features_df["cluster_kmeans"].iloc[lookahead:].values

    return X, y


def train_and_evaluate(X, y):
    """Train KNN and Decision Tree, return results."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    results = {}

    # ── KNN ──
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn_train = knn.predict(X_train)
    y_pred_knn_test = knn.predict(X_test)
    results["KNN"] = {
        "model": knn,
        "y_pred": y_pred_knn_test,
        "train_accuracy": float(np.mean(y_pred_knn_train == y_train)),
        "test_accuracy": float(np.mean(y_pred_knn_test == y_test)),
        "report": classification_report(y_test, y_pred_knn_test, output_dict=True)
    }
    print(f"\n  KNN Training Accuracy: {results['KNN']['train_accuracy']:.3f}")
    print(f"  KNN Test Accuracy:     {results['KNN']['test_accuracy']:.3f}")

    # ── Decision Tree ──
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt_train = dt.predict(X_train)
    y_pred_dt_test = dt.predict(X_test)
    results["DecisionTree"] = {
        "model": dt,
        "y_pred": y_pred_dt_test,
        "train_accuracy": float(np.mean(y_pred_dt_train == y_train)),
        "test_accuracy": float(np.mean(y_pred_dt_test == y_test)),
        "report": classification_report(y_test, y_pred_dt_test, output_dict=True)
    }
    print(f"  Decision Tree Training Accuracy: {results['DecisionTree']['train_accuracy']:.3f}")
    print(f"  Decision Tree Test Accuracy:     {results['DecisionTree']['test_accuracy']:.3f}")

    return results, X_test, y_test


def plot_confusion_matrices(results, y_test, save_path=None):
    """Plot side-by-side confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Free-Flow", "Moderate", "Congested"]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        disp = ConfusionMatrixDisplay(cm, display_labels=labels)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        ax.set_title(f"{name}  (Acc: {res['test_accuracy']:.1%})", fontsize=12)

    fig.suptitle("Congestion Prediction — Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix saved to {save_path}")
    plt.close()


def plot_feature_importance(dt_model, save_path=None):
    """Bar chart of Decision Tree feature importances."""
    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(FEATURE_COLS)))
    ax.barh(
        [FEATURE_COLS[i] for i in indices],
        importances[indices],
        color=colors
    )
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title("Feature Importance (Decision Tree)", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Feature importance saved to {save_path}")
    plt.close()


# ───── CLI ─────
if __name__ == "__main__":
    print("=" * 50)
    print(" Congestion Prediction Pipeline")
    print("=" * 50)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    feat_path = os.path.join(data_dir, "features_labeled.csv")

    if not os.path.exists(feat_path):
        print("Error: Run clustering.py first to produce labeled features.")
        exit(1)

    features_df = pd.read_csv(feat_path)
    print(f"\nLoaded {len(features_df)} labeled feature vectors")

    # Build prediction dataset
    X, y = create_prediction_dataset(features_df, lookahead=1)
    print(f"Prediction dataset: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # Train & evaluate
    results, X_test, y_test = train_and_evaluate(X, y)

    # Plots
    plot_confusion_matrices(results, y_test,
                            save_path=os.path.join(data_dir, "confusion_matrices.png"))
    plot_feature_importance(results["DecisionTree"]["model"],
                           save_path=os.path.join(data_dir, "feature_importance.png"))

    # Detailed report
    print("\n-- KNN Classification Report --")
    print(classification_report(y_test, results["KNN"]["y_pred"],
                                target_names=["Free-Flow", "Moderate", "Congested"]))

    print("-- Decision Tree Classification Report --")
    print(classification_report(y_test, results["DecisionTree"]["y_pred"],
                                target_names=["Free-Flow", "Moderate", "Congested"]))

    # Save models
    knn_model_path = os.path.join(data_dir, "knn_model.pkl")
    with open(knn_model_path, "wb") as f:
        pickle.dump(results["KNN"]["model"], f)
    print(f"\n  KNN model saved to {knn_model_path}")
