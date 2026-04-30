"""Poster-friendly sentiment model comparison visualization."""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def load_metrics():
    metrics_path = BASE_DIR / "results" / "sentiment_model_metrics.json"
    if not metrics_path.exists():
        print(f"Error: metrics file not found at {metrics_path}")
        print("Run comparison/compare_sentiment_models.py first.")
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_poster_visualization():
    metrics = load_metrics()
    if not metrics:
        return

    model_rows = metrics.get("model_metrics", [])
    baseline_rows = metrics.get("trivial_baselines", [])
    best_model = metrics.get("best_model_by_mae", "")

    if not model_rows:
        print("No model metrics found.")
        return

    models = [row["Model"] for row in model_rows]
    mae = [row["MAE"] for row in model_rows]
    speeds = [row.get("Speed (items/sec)", 0.0) for row in model_rows]

    best_baseline = min(baseline_rows, key=lambda row: row["MAE"]) if baseline_rows else None
    best_baseline_mae = best_baseline["MAE"] if best_baseline else None

    colors = ["#4c78a8" if model == best_model else "#bab0ac" for model in models]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    scatter_sizes = [260 if model == best_model else 170 for model in models]
    ax.scatter(speeds, mae, s=scatter_sizes, c=colors, edgecolors="#333333", linewidth=1.2)
    max_error = max(mae)
    min_error = min(mae)
    for model, speed, error in zip(models, speeds, mae):
        offset = (7, -18) if error > max_error - 0.015 else (7, 7)
        ax.annotate(
            f"{model}\nMAE {error:.2f}",
            (speed, error),
            textcoords="offset points",
            xytext=offset,
            fontsize=10,
            fontweight="bold",
            annotation_clip=False,
        )
    if best_baseline_mae is not None:
        ax.axhline(
            best_baseline_mae,
            color="#e45756",
            linestyle="--",
            linewidth=2,
            label=f"{best_baseline['Model']} baseline: MAE {best_baseline_mae:.2f}",
        )
    ax.set_xscale("log")
    ax.set_title("Sentiment Model Comparison: Speed vs Error", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Speed (items/sec, log scale)")
    ax.set_ylabel("Mean Absolute Error on 1-5 Score")
    ax.set_xlim(min(speeds) * 0.75, max(speeds) * 1.75)
    ax.set_ylim(max(0, min_error - 0.04), max_error + 0.08)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    output_path = BASE_DIR / "results" / "visualizations" / "sentiment_model_comparison_poster.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved sentiment visualization to {output_path}")


if __name__ == "__main__":
    create_poster_visualization()
