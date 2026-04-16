"""
CSE275 — Clustering Module

Applies K-Means and DBSCAN to traffic feature vectors to
identify traffic states:
  0 = Free-Flow
  1 = Moderate
  2 = Congested

Produces labeled datasets and visualisations.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os

# Features used for clustering
CLUSTER_FEATURES = [
    "vehicle_count",
    "avg_speed",
    "occupancy_ratio",
    "flow_rate",
    "avg_distance"
]

TRAFFIC_LABELS = {0: "Free-Flow", 1: "Moderate", 2: "Congested"}


def load_features(path=None):
    """Load feature vectors from CSV."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "features.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}. Run feature_engineering.py first.")
    return pd.read_csv(path)


def run_kmeans(features_df, n_clusters=3):
    """
    Run K-Means clustering on traffic features.
    Returns (labels, model, scaler).
    """
    X = features_df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # Sort clusters by avg_speed so 0 = highest speed (free-flow)
    cluster_speeds = []
    for c in range(n_clusters):
        mask = labels == c
        cluster_speeds.append((c, features_df.loc[mask, "avg_speed"].mean()))
    cluster_speeds.sort(key=lambda x: -x[1])  # Descending speed

    label_map = {old: new for new, (old, _) in enumerate(cluster_speeds)}
    labels = np.array([label_map[l] for l in labels])

    sil_score = silhouette_score(X_scaled, labels)
    print(f"  K-Means Silhouette Score: {sil_score:.3f}")

    return labels, model, scaler


def run_dbscan(features_df, eps=0.8, min_samples=5):
    """
    Run DBSCAN clustering on traffic features.
    Returns (labels, model, scaler).
    """
    X = features_df[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

    n_clusters = len(set(labels) - {-1})
    n_noise = (labels == -1).sum()
    print(f"  DBSCAN: {n_clusters} clusters, {n_noise} noise points")

    if n_clusters > 1:
        valid = labels != -1
        sil = silhouette_score(X_scaled[valid], labels[valid])
        print(f"  DBSCAN Silhouette Score: {sil:.3f}")

    return labels, model, scaler


def plot_clusters(features_df, labels, title="Traffic Clusters", save_path=None):
    """Scatter plot of speed vs. occupancy, colored by cluster."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Speed vs Occupancy
    scatter1 = axes[0].scatter(
        features_df["occupancy_ratio"],
        features_df["avg_speed"],
        c=labels, cmap="RdYlGn_r", alpha=0.7, edgecolors="k", linewidths=0.3
    )
    axes[0].set_xlabel("Occupancy Ratio", fontsize=11)
    axes[0].set_ylabel("Avg Speed (cm/s)", fontsize=11)
    axes[0].set_title("Speed vs Occupancy", fontsize=12)
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Plot 2: Flow Rate vs Vehicle Count
    scatter2 = axes[1].scatter(
        features_df["vehicle_count"],
        features_df["flow_rate"],
        c=labels, cmap="RdYlGn_r", alpha=0.7, edgecolors="k", linewidths=0.3
    )
    axes[1].set_xlabel("Vehicle Count", fontsize=11)
    axes[1].set_ylabel("Flow Rate (veh/min)", fontsize=11)
    axes[1].set_title("Flow Rate vs Count", fontsize=12)
    plt.colorbar(scatter2, ax=axes[1], label="Cluster")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    plt.close()


def plot_time_series(features_df, labels, save_path=None):
    """Time-series view of traffic state over windows."""
    fig, ax = plt.subplots(figsize=(12, 4))

    colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    for i, (_, row) in enumerate(features_df.iterrows()):
        lbl = labels[i]
        color = colors.get(lbl, "#95a5a6")
        ax.bar(i, 1, color=color, width=1.0)

    ax.set_xlabel("Window Index", fontsize=11)
    ax.set_yticks([])
    ax.set_title("Traffic State Over Time", fontsize=13, fontweight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="Free-Flow"),
        Patch(facecolor="#f39c12", label="Moderate"),
        Patch(facecolor="#e74c3c", label="Congested"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Timeline saved to {save_path}")
    plt.close()


# ───── CLI ─────
if __name__ == "__main__":
    print("=" * 50)
    print(" Traffic Clustering Pipeline")
    print("=" * 50)

    features_df = load_features()
    print(f"\nLoaded {len(features_df)} feature vectors")

    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # K-Means
    print("\n-- K-Means Clustering --")
    km_labels, _, _ = run_kmeans(features_df, n_clusters=3)
    features_df["cluster_kmeans"] = km_labels
    features_df["traffic_state"] = [TRAFFIC_LABELS.get(l, "Unknown") for l in km_labels]

    plot_clusters(features_df, km_labels,
                  title="K-Means Traffic Clusters",
                  save_path=os.path.join(output_dir, "kmeans_clusters.png"))
    plot_time_series(features_df, km_labels,
                     save_path=os.path.join(output_dir, "traffic_timeline.png"))

    # DBSCAN
    print("\n-- DBSCAN Clustering --")
    db_labels, _, _ = run_dbscan(features_df)
    features_df["cluster_dbscan"] = db_labels

    plot_clusters(features_df, db_labels,
                  title="DBSCAN Traffic Clusters",
                  save_path=os.path.join(output_dir, "dbscan_clusters.png"))

    # Save labeled data
    labeled_path = os.path.join(output_dir, "features_labeled.csv")
    features_df.to_csv(labeled_path, index=False)
    print(f"\nLabeled features saved to {labeled_path}")

    # Summary
    print("\n-- Cluster Summary (K-Means) --")
    summary = features_df.groupby("traffic_state")[CLUSTER_FEATURES].mean()
    print(summary.to_string())
