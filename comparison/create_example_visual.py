"""Create a poster-friendly human benchmark agreement heatmap."""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_ORDER = ["Llama3", "Gemma", "roBERTa", "DistilroBERTa"]

TOPIC_LABELS = {
    "Course organization and structure": "Organization",
    "Pace": "Pace",
    "Workload": "Workload",
    "Student engagement and participation": "Engagement",
    "Clarity of explanations": "Clarity",
    "Effectiveness of assignments": "Assignments",
    "Classroom atmosphere": "Atmosphere",
    "Instructor's communication and availability": "Communication",
    "Inclusivity and sense of belonging": "Inclusivity",
    "Assessment": "Assessment",
    "Grading and feedback": "Grading / feedback",
    "Learning resources and materials": "Learning resources",
    "None of the above / Other": "Other",
}


def load_per_topic_f1() -> pd.DataFrame:
    metrics_path = BASE_DIR / "results" / "topic_model_per_topic_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run comparison/model_comparison.py first."
        )

    metrics = pd.read_csv(metrics_path)
    metrics = metrics[metrics["Model"].isin(MODEL_ORDER)].copy()
    metrics = metrics[metrics["Support"] > 0].copy()
    metrics["Topic Label"] = metrics["Topic"].map(TOPIC_LABELS).fillna(metrics["Topic"])
    metrics["F1 Percent"] = metrics["F1"] * 100

    heatmap = metrics.pivot(index="Topic Label", columns="Model", values="F1 Percent")
    heatmap = heatmap[[model for model in MODEL_ORDER if model in heatmap.columns]]

    topic_order = (
        metrics.groupby("Topic Label")["Support"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    return heatmap.loc[topic_order]


def load_model_summary() -> pd.DataFrame:
    summary_path = BASE_DIR / "results" / "topic_model_comparison.csv"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing {summary_path}. Run comparison/model_comparison.py first."
        )
    summary = pd.read_csv(summary_path)
    summary = summary[summary["Model"].isin(MODEL_ORDER)].copy()
    return summary.set_index("Model").loc[[model for model in MODEL_ORDER if model in summary["Model"].values]]


def create_example_visual() -> None:
    heatmap = load_per_topic_f1()
    summary = load_model_summary()

    matrix = heatmap.to_numpy(dtype=float)
    rows = heatmap.index.tolist()
    cols = heatmap.columns.tolist()
    col_labels = [f"{model}\n{summary.loc[model, 'micro_f1']:.1f}%" for model in cols]

    fig, ax = plt.subplots(figsize=(8.8, 5.6), facecolor="white")
    fig.subplots_adjust(left=0.26, right=0.91, top=0.82, bottom=0.15)

    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")

    ax.set_title(
        "Classification Agreement with Human Benchmark",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels, fontsize=9.5, fontweight="bold")
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(rows, fontsize=9.2)
    ax.tick_params(length=0)

    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            value = matrix[row_idx, col_idx]
            text_color = "white" if value >= 62 else "#1f2933"
            ax.text(
                col_idx,
                row_idx,
                f"{value:.0f}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold",
                color=text_color,
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Per-topic F1 (%)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    fig.text(
        0.26,
        0.06,
        "Cells show per-topic F1 (%) against the manual benchmark. Column labels include overall micro-F1.",
        fontsize=8.8,
        color="#555555",
    )
    fig.text(
        0.26,
        0.025,
        "Takeaway: Llama3 performs best overall, while agreement varies substantially by feedback category.",
        fontsize=8.8,
        color="#153b5f",
        fontweight="bold",
    )

    output_path = BASE_DIR / "results" / "visualizations" / "example_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, facecolor="white")
    plt.close()
    print(f"Saved benchmark heatmap to {output_path}")


if __name__ == "__main__":
    create_example_visual()
