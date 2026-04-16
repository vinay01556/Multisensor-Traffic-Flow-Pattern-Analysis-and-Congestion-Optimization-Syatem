"""
IntelliTraffic Pro - Feature Engineering Module

Features per window:
  - vehicle_count      : Total IR detections
  - avg_speed          : Mean speed estimate (cm/s)
  - std_speed          : Speed standard deviation
  - avg_distance       : Mean ultrasonic distance (cm)
  - occupancy_ratio    : Fraction of samples with IR = 1
  - avg_inter_arrival  : Mean gap between consecutive detections (s)
  - flow_rate          : Vehicles per minute
"""

import pandas as pd
import numpy as np
import os
import sys

# Allow importing database module from backend/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


def load_data(csv_path=None):
    """Load sensor data from CSV or from the SQLite database."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fall back to database
        from database import TrafficDatabase
        db = TrafficDatabase()
        records = db.get_all()
        if not records:
            raise ValueError("No sensor data found. Run sensors first or provide a CSV.")
        df = pd.DataFrame(records)

    # Normalise column names
    rename_map = {
        "ir_detected": "ir",
        "distance_cm": "dist_cm",
        "speed_cm_s":  "speed_cm_s",
        "lane_id":     "lane",
        "device_ts":   "ts"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    required = ["ir", "dist_cm", "speed_cm_s", "lane"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    return df


def compute_inter_arrivals(ir_series, ts_series):
    """Compute mean inter-arrival time (seconds) between consecutive IR detections."""
    detection_times = ts_series[ir_series == 1].values
    if len(detection_times) < 2:
        return 0.0
    gaps = np.diff(detection_times) / 1000.0  # ms → s
    return float(np.mean(gaps))


def extract_features(df, window_size=30, step_size=15):
    """
    Slide a time window over the data and compute traffic features.

    Parameters
    ----------
    df          : DataFrame with columns [ir, dist_cm, speed_cm_s, lane, ts]
    window_size : Number of samples per window
    step_size   : Samples to advance between windows

    Returns
    -------
    DataFrame of feature vectors, one row per window.
    """
    features = []

    for start in range(0, len(df) - window_size + 1, step_size):
        window = df.iloc[start : start + window_size]

        vehicle_count     = int(window["ir"].sum())
        avg_speed         = float(window["speed_cm_s"].mean())
        std_speed         = float(window["speed_cm_s"].std())
        avg_distance      = float(window["dist_cm"].mean())
        occupancy_ratio   = float(vehicle_count / len(window))
        avg_inter_arrival = compute_inter_arrivals(window["ir"], window["ts"]) if "ts" in window.columns else 0.0
        window_duration_s = (window["ts"].iloc[-1] - window["ts"].iloc[0]) / 1000.0 if "ts" in window.columns else (window_size * 0.2)
        flow_rate         = (vehicle_count / window_duration_s) * 60.0 if window_duration_s > 0 else 0.0
        lane_id           = int(window["lane"].mode().iloc[0]) if not window["lane"].mode().empty else 1

        features.append({
            "window_start":     start,
            "lane":             lane_id,
            "vehicle_count":    vehicle_count,
            "avg_speed":        round(avg_speed, 2),
            "std_speed":        round(std_speed, 2),
            "avg_distance":     round(avg_distance, 2),
            "occupancy_ratio":  round(occupancy_ratio, 3),
            "avg_inter_arrival": round(avg_inter_arrival, 3),
            "flow_rate":        round(flow_rate, 2),
        })

    return pd.DataFrame(features)


def generate_synthetic_data(n_samples=500, seed=42):
    """
    Generate synthetic sensor data for development / testing
    when real hardware is not connected.
    """
    np.random.seed(seed)
    data = []
    ts = 0

    for i in range(n_samples):
        # Simulate traffic phases: free-flow, moderate, congested
        phase = (i // 150) % 3
        if phase == 0:  # Free flow
            ir = np.random.choice([0, 1], p=[0.6, 0.4])
            speed = np.random.normal(80, 15) if ir else 0
            dist = np.random.normal(200, 40)
        elif phase == 1:  # Moderate
            ir = np.random.choice([0, 1], p=[0.3, 0.7])
            speed = np.random.normal(40, 10) if ir else 0
            dist = np.random.normal(100, 25)
        else:  # Congested
            ir = np.random.choice([0, 1], p=[0.1, 0.9])
            speed = np.random.normal(10, 5) if ir else 0
            dist = np.random.normal(30, 10)

        ts += np.random.randint(150, 250)  # ~200 ms intervals
        speed = max(0, speed)
        dist = max(5, dist)

        data.append({
            "sample": i + 1,
            "ts": ts,
            "ir": ir,
            "dist_cm": round(dist, 1),
            "speed_cm_s": round(speed, 2),
            "lane": 1
        })

    return pd.DataFrame(data)


# ───── CLI Usage ─────
if __name__ == "__main__":
    print("=" * 50)
    print(" Feature Engineering Pipeline")
    print("=" * 50)

    # Use synthetic data for demo
    print("\nGenerating synthetic sensor data (500 samples)...")
    df = generate_synthetic_data(500)
    print(f"  Raw data shape: {df.shape}")

    # Save raw synthetic data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "traffic_data.csv")
    df.to_csv(raw_path, index=False)
    print(f"  Saved to {raw_path}")

    # Extract features
    print("\nExtracting windowed features...")
    features = extract_features(df, window_size=30, step_size=15)
    print(f"  Feature vectors: {features.shape}")
    print(f"\n{features.head(10).to_string(index=False)}")

    feat_path = os.path.join(data_dir, "features.csv")
    features.to_csv(feat_path, index=False)
    print(f"\n  Saved features to {feat_path}")
