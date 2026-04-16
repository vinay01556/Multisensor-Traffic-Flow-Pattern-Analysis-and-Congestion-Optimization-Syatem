"""
CSE275 — Trajectory Variance Optimization Module

Objective: Minimize the variance of vehicle travel times (trajectory variance)
across traffic windows by optimizing green-light durations.

Approach:
  - Model travel time as a function of green-light split ratios
  - Use scipy.optimize.minimize to find optimal signal timings
  - Compare before/after variance metrics
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
import os


def simulate_travel_times(features_df, green_ratios, cycle_time=90):
    """
    Simple traffic model: estimate travel time per window given signal parameters.
    
    Parameters
    ----------
    features_df   : DataFrame with traffic features
    green_ratios  : Array of green-time ratios per phase (must sum to 1.0)
    cycle_time    : Total signal cycle time in seconds

    Returns
    -------
    Array of estimated travel times per window (seconds).
    """
    n = len(features_df)
    travel_times = np.zeros(n)

    base_travel = 60.0  # Base travel time at free-flow (seconds)
    
    for i in range(n):
        occupancy = features_df.iloc[i]["occupancy_ratio"]
        avg_speed = features_df.iloc[i]["avg_speed"]
        flow_rate = features_df.iloc[i]["flow_rate"]

        # Effective green time for this window's lane
        lane = int(features_df.iloc[i].get("lane", 1)) - 1
        phase_idx = lane % len(green_ratios)
        green_time = green_ratios[phase_idx] * cycle_time

        # Delay model: Webster-inspired approximation
        # Higher green ratio → lower delay; higher occupancy → higher delay
        capacity_factor = max(0.1, green_time / cycle_time)
        delay = (cycle_time * (1 - capacity_factor) ** 2) / (2 * (1 - occupancy * capacity_factor))
        delay = max(0, min(delay, 120))  # Clamp to [0, 120]s

        # Travel time = base + delay + speed penalty
        speed_factor = max(0.1, avg_speed / 80.0)  # Normalised to free-flow speed
        travel_times[i] = base_travel / speed_factor + delay

    return travel_times


def objective_function(green_ratios, features_df, cycle_time=90):
    """
    Objective: Minimize trajectory variance (variance of travel times).
    Constraint: green_ratios must sum to 1.0
    """
    # Enforce sum-to-one via softmax-like normalisation
    ratios = np.abs(green_ratios)
    ratios = ratios / ratios.sum()

    travel_times = simulate_travel_times(features_df, ratios, cycle_time)
    variance = np.var(travel_times)

    return variance


def optimize_signal_timing(features_df, n_phases=2, cycle_time=90):
    """
    Find optimal green-light split ratios to minimize trajectory variance.
    
    Returns
    -------
    dict with keys: optimal_ratios, optimal_green_times, before_variance,
                    after_variance, variance_reduction, travel_times_before,
                    travel_times_after
    """
    # ── Default (equal) split ──
    default_ratios = np.ones(n_phases) / n_phases
    tt_before = simulate_travel_times(features_df, default_ratios, cycle_time)
    var_before = np.var(tt_before)

    print(f"  Default equal split: {default_ratios}")
    print(f"  Before — Mean TT: {np.mean(tt_before):.2f}s, Variance: {var_before:.2f}")

    # ── Optimise ──
    bounds = [(0.1, 0.9)] * n_phases  # Each phase gets 10-90%

    result = differential_evolution(
        objective_function,
        bounds=bounds,
        args=(features_df, cycle_time),
        seed=42,
        maxiter=200,
        tol=1e-6
    )

    optimal = np.abs(result.x)
    optimal = optimal / optimal.sum()  # Normalise

    tt_after = simulate_travel_times(features_df, optimal, cycle_time)
    var_after = np.var(tt_after)

    reduction = ((var_before - var_after) / var_before) * 100 if var_before > 0 else 0

    print(f"\n  Optimised ratios: {np.round(optimal, 3)}")
    print(f"  Optimised green times: {np.round(optimal * cycle_time, 1)}s")
    print(f"  After  — Mean TT: {np.mean(tt_after):.2f}s, Variance: {var_after:.2f}")
    print(f"  Variance reduction: {reduction:.1f}%")

    return {
        "optimal_ratios": optimal,
        "optimal_green_times": optimal * cycle_time,
        "before_variance": var_before,
        "after_variance": var_after,
        "variance_reduction_pct": reduction,
        "mean_tt_before": np.mean(tt_before),
        "mean_tt_after": np.mean(tt_after),
        "travel_times_before": tt_before,
        "travel_times_after": tt_after,
    }


def plot_optimization_results(opt_result, save_path=None):
    """Visualize before/after travel time distributions and variance."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    tt_before = opt_result["travel_times_before"]
    tt_after  = opt_result["travel_times_after"]

    # 1. Histogram comparison
    axes[0].hist(tt_before, bins=20, alpha=0.6, color="#e74c3c", label="Before", edgecolor="white")
    axes[0].hist(tt_after,  bins=20, alpha=0.6, color="#2ecc71", label="After",  edgecolor="white")
    axes[0].set_xlabel("Travel Time (s)", fontsize=11)
    axes[0].set_ylabel("Frequency", fontsize=11)
    axes[0].set_title("Travel Time Distribution", fontsize=12)
    axes[0].legend(fontsize=10)

    # 2. Time-series comparison
    axes[1].plot(tt_before, color="#e74c3c", alpha=0.7, label="Before", linewidth=1)
    axes[1].plot(tt_after,  color="#2ecc71", alpha=0.7, label="After",  linewidth=1)
    axes[1].set_xlabel("Window Index", fontsize=11)
    axes[1].set_ylabel("Travel Time (s)", fontsize=11)
    axes[1].set_title("Travel Time Over Windows", fontsize=12)
    axes[1].legend(fontsize=10)

    # 3. Variance bar chart
    bars = axes[2].bar(
        ["Before", "After"],
        [opt_result["before_variance"], opt_result["after_variance"]],
        color=["#e74c3c", "#2ecc71"], edgecolor="white", width=0.5
    )
    axes[2].set_ylabel("Variance", fontsize=11)
    axes[2].set_title(f"Variance Reduction: {opt_result['variance_reduction_pct']:.1f}%",
                      fontsize=12, fontweight="bold")

    fig.suptitle("Trajectory Variance Optimization Results", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Optimization plot saved to {save_path}")
    plt.close()


# ───── CLI ─────
if __name__ == "__main__":
    print("=" * 50)
    print(" Trajectory Variance Optimization")
    print("=" * 50)

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    feat_path = os.path.join(data_dir, "features_labeled.csv")

    if not os.path.exists(feat_path):
        feat_path = os.path.join(data_dir, "features.csv")
    if not os.path.exists(feat_path):
        print("Error: Run feature_engineering.py first.")
        exit(1)

    features_df = pd.read_csv(feat_path)
    print(f"\nLoaded {len(features_df)} feature vectors")

    # Optimise
    result = optimize_signal_timing(features_df, n_phases=2, cycle_time=90)

    # Plot
    plot_optimization_results(result,
                              save_path=os.path.join(data_dir, "optimization_results.png"))

    # Save summary
    summary = {
        "optimal_green_ratios": result["optimal_ratios"].tolist(),
        "optimal_green_times_sec": result["optimal_green_times"].tolist(),
        "variance_before": result["before_variance"],
        "variance_after": result["after_variance"],
        "reduction_pct": result["variance_reduction_pct"],
        "mean_travel_time_before_s": result["mean_tt_before"],
        "mean_travel_time_after_s": result["mean_tt_after"]
    }

    import json
    summary_path = os.path.join(data_dir, "optimization_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved to {summary_path}")
