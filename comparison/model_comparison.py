from pathlib import Path
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
MATCH_THRESHOLD = 0.80


def normalize_text(text: str) -> str:
    """Normalize text enough for stable comparison without changing meaning."""
    if not isinstance(text, str):
        return ""

    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\uff1b": ";",
        "\u037e": ";",
        "\n": " ",
        "\r": " ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.strip().lower().split())


def text_similarity(left: str, right: str) -> float:
    """Score exact, prefix, and full-text similarity for comments of unequal length."""
    left_norm = normalize_text(left)
    right_norm = normalize_text(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm or left_norm in right_norm or right_norm in left_norm:
        return 1.0

    shorter, longer = (left_norm, right_norm) if len(left_norm) <= len(right_norm) else (right_norm, left_norm)
    window = len(shorter)
    full_ratio = SequenceMatcher(None, left_norm, right_norm).ratio()
    prefix_ratio = SequenceMatcher(None, shorter, longer[:window]).ratio()
    suffix_ratio = SequenceMatcher(None, shorter, longer[-window:]).ratio()
    return max(full_ratio, prefix_ratio, suffix_ratio)


def get_topics(row: pd.Series, topics_list: List[str]) -> set[str]:
    """Extract assigned topics from a wide classification row."""
    topics = set()
    for topic in topics_list:
        value = row.get(topic, "")
        if pd.notna(value) and str(value).strip():
            topics.add(topic)
    return topics


def match_rows(
    benchmark: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[List[Tuple[pd.Series, pd.Series, float]], List[Dict]]:
    """One-to-one match benchmark rows to model rows, returning unmatched rows separately."""
    current_records = list(current.iterrows())
    used_current_indices: set[int] = set()
    matches: List[Tuple[pd.Series, pd.Series, float]] = []
    unmatched: List[Dict] = []

    for bench_idx, bench_row in benchmark.iterrows():
        best_idx = None
        best_row = None
        best_score = 0.0

        for current_idx, current_row in current_records:
            if current_idx in used_current_indices:
                continue
            score = text_similarity(bench_row["Feedback"], current_row["Feedback"])
            if score > best_score:
                best_idx = current_idx
                best_row = current_row
                best_score = score

        if best_row is not None and best_idx is not None and best_score >= threshold:
            used_current_indices.add(best_idx)
            matches.append((bench_row, best_row, best_score))
        else:
            unmatched.append(
                {
                    "benchmark_index": bench_idx,
                    "best_similarity": round(best_score, 4),
                    "benchmark_feedback": bench_row.get("Feedback", ""),
                    "best_model_feedback": "" if best_row is None else best_row.get("Feedback", ""),
                }
            )

    return matches, unmatched


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)


def compare(
    benchmark: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float = MATCH_THRESHOLD,
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """Compare model output against benchmark using standard multilabel metrics."""
    topics = [col for col in benchmark.columns if col != "Feedback"]
    matches, unmatched = match_rows(benchmark, current, threshold=threshold)

    topic_counts = {
        topic: {"tp": 0, "fp": 0, "fn": 0, "support": 0}
        for topic in topics
    }

    exact_matches = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    predicted_topics = 0
    true_topics = 0

    for bench_row, model_row, _similarity in matches:
        benchmark_topics = get_topics(bench_row, topics)
        model_topics = get_topics(model_row, topics)

        if benchmark_topics == model_topics:
            exact_matches += 1

        predicted_topics += len(model_topics)
        true_topics += len(benchmark_topics)

        for topic in topics:
            in_benchmark = topic in benchmark_topics
            in_model = topic in model_topics

            if in_benchmark:
                topic_counts[topic]["support"] += 1
            if in_benchmark and in_model:
                topic_counts[topic]["tp"] += 1
                true_positives += 1
            elif in_model and not in_benchmark:
                topic_counts[topic]["fp"] += 1
                false_positives += 1
            elif in_benchmark and not in_model:
                topic_counts[topic]["fn"] += 1
                false_negatives += 1

    micro_precision = safe_divide(true_positives, true_positives + false_positives)
    micro_recall = safe_divide(true_positives, true_positives + false_negatives)
    micro_f1 = f1_score(micro_precision, micro_recall)

    per_topic_rows = []
    for topic, counts in topic_counts.items():
        precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        per_topic_rows.append(
            {
                "Topic": topic,
                "Support": counts["support"],
                "True Positives": counts["tp"],
                "False Positives": counts["fp"],
                "False Negatives": counts["fn"],
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1": round(f1_score(precision, recall), 4),
            }
        )

    matched_count = len(matches)
    per_topic_df = pd.DataFrame(per_topic_rows)
    macro_f1 = per_topic_df.loc[per_topic_df["Support"] > 0, "F1"].mean() if not per_topic_df.empty else 0.0
    similarities = [similarity for *_rows, similarity in matches]

    metrics = {
        "matched_comments": matched_count,
        "benchmark_comments": len(benchmark),
        "model_comments": len(current),
        "unmatched_comments": len(unmatched),
        "match_coverage": round(safe_divide(matched_count, len(benchmark)) * 100, 2),
        "min_similarity": round(min(similarities), 4) if similarities else 0.0,
        "mean_similarity": round(float(np.mean(similarities)), 4) if similarities else 0.0,
        "exact_set_matches": exact_matches,
        "exact_set_match_rate": round(safe_divide(exact_matches, matched_count) * 100, 2),
        "micro_precision": round(micro_precision * 100, 2),
        "micro_recall": round(micro_recall * 100, 2),
        "micro_f1": round(micro_f1 * 100, 2),
        "macro_f1": round(float(macro_f1) * 100, 2) if not pd.isna(macro_f1) else 0.0,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "true_topic_assignments": true_topics,
        "predicted_topic_assignments": predicted_topics,
    }

    return metrics, per_topic_df, pd.DataFrame(unmatched)


def print_results(scores: Dict[str, Dict]) -> None:
    """Pretty print comparison results."""
    print("\n" + "=" * 80)
    print("TOPIC CLASSIFICATION COMPARISON")
    print("=" * 80)

    sorted_scores = sorted(scores.items(), key=lambda item: item[1]["micro_f1"], reverse=True)
    for model, metrics in sorted_scores:
        print(f"\n{model}:")
        print(f"  Matched comments:       {metrics['matched_comments']}/{metrics['benchmark_comments']} ({metrics['match_coverage']}%)")
        print(f"  Exact set match:        {metrics['exact_set_match_rate']}%")
        print(f"  Micro precision/recall: {metrics['micro_precision']}% / {metrics['micro_recall']}%")
        print(f"  Micro F1:               {metrics['micro_f1']}%")
        print(f"  Macro F1:               {metrics['macro_f1']}%")
        print(f"  TP / FP / FN:           {metrics['true_positives']} / {metrics['false_positives']} / {metrics['false_negatives']}")


def visualize_results(scores: Dict[str, Dict], per_topic_by_model: Dict[str, pd.DataFrame]) -> None:
    """Create a poster-friendly classification model comparison."""
    del per_topic_by_model  # Per-topic metrics are saved to CSV; the poster figure stays intentionally simple.

    sorted_models = sorted(scores.keys(), key=lambda model: scores[model]["micro_f1"], reverse=True)
    y = np.arange(len(sorted_models))
    f1_values = [scores[model]["micro_f1"] for model in sorted_models]
    precision_values = [scores[model]["micro_precision"] for model in sorted_models]
    recall_values = [scores[model]["micro_recall"] for model in sorted_models]

    colors = ["#4c78a8" if model == sorted_models[0] else "#bab0ac" for model in sorted_models]

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    bars = ax.barh(y, f1_values, color=colors, edgecolor="#333333", linewidth=0.8, height=0.55)
    ax.scatter(precision_values, y, color="#1f77b4", marker="o", s=95, label="Precision", zorder=3)
    ax.scatter(recall_values, y, color="#f58518", marker="D", s=85, label="Recall", zorder=3)

    for bar, f1_value in zip(bars, f1_values):
        ax.text(
            f1_value + 1.2,
            bar.get_y() + bar.get_height() / 2,
            f"{f1_value:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_title("Topic Classification Model Comparison", fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Micro-F1 (%)")
    ax.set_yticks(y)
    ax.set_yticklabels(sorted_models, fontsize=12)
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    output_dir = BASE_DIR / "results" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classification_model_comparison_poster.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nSaved visualization to: {output_path}")
    plt.close()


def main() -> None:
    baseline_path = BASE_DIR / "HUMAN_CATEGORIZED_OUTPUT.csv"
    benchmark = pd.read_csv(baseline_path)

    print(f"Comparing {len(benchmark)} benchmark feedbacks with threshold {MATCH_THRESHOLD}...\n")

    models = {
        "Gemma": "GEMMA_OUTPUT.csv",
        "Llama3": "LLAMA_OUTPUT.csv",
        "roBERTa": "roBERTa_OUTPUT.csv",
        "DistilroBERTa": "DISTILROBERTA_OUTPUT.csv",
    }

    scores: Dict[str, Dict] = {}
    per_topic_by_model: Dict[str, pd.DataFrame] = {}
    unmatched_frames = []

    for model, path in models.items():
        model_path = BASE_DIR / "results" / model / path
        if not model_path.exists():
            print(f"Missing {model}: {model_path}")
            continue

        print(f"Comparing {model}...")
        current = pd.read_csv(model_path)
        metrics, per_topic, unmatched = compare(benchmark, current)
        scores[model] = metrics
        per_topic.insert(0, "Model", model)
        per_topic_by_model[model] = per_topic
        if not unmatched.empty:
            unmatched.insert(0, "Model", model)
            unmatched_frames.append(unmatched)

    print_results(scores)

    results_dir = BASE_DIR / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"Model": model, **metrics} for model, metrics in scores.items()]
    ).to_csv(results_dir / "topic_model_comparison.csv", index=False)
    pd.concat(per_topic_by_model.values(), ignore_index=True).to_csv(
        results_dir / "topic_model_per_topic_metrics.csv",
        index=False,
    )
    if unmatched_frames:
        pd.concat(unmatched_frames, ignore_index=True).to_csv(
            results_dir / "topic_model_unmatched.csv",
            index=False,
        )

    visualize_results(scores, per_topic_by_model)


if __name__ == "__main__":
    main()
