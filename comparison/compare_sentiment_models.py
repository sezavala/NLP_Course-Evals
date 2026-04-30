"""
Sentiment model comparison.

This script evaluates saved sentiment predictions against the human baseline using
metrics that are less sensitive to the strongly positive class skew in the data.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import math

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

MODELS = {
    "Llama3": BASE_DIR / "results" / "Llama3" / "LLAMA_SENTIMENT.json",
    "Gemma": BASE_DIR / "results" / "Gemma" / "GEMMA_SENTIMENT.json",
    "roBERTa": BASE_DIR / "results" / "roBERTa" / "ROBERTA_SENTIMENT.json",
    "DistilroBERTa": BASE_DIR / "results" / "DistilroBERTa" / "DISTILROBERTA_SENTIMENT.json",
}

SENTIMENT_LABELS = ["negative", "neutral", "positive"]


def normalize_text(text: str) -> str:
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


def make_key(topic: str, feedback: str) -> Tuple[str, str]:
    return str(topic).strip(), normalize_text(feedback)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_divide(2 * precision * recall, precision + recall)


def label_macro_f1(reference_labels: List[str], predicted_labels: List[str]) -> float:
    f1_values = []
    for label in SENTIMENT_LABELS:
        tp = sum(1 for ref, pred in zip(reference_labels, predicted_labels) if ref == label and pred == label)
        fp = sum(1 for ref, pred in zip(reference_labels, predicted_labels) if ref != label and pred == label)
        fn = sum(1 for ref, pred in zip(reference_labels, predicted_labels) if ref == label and pred != label)
        support = sum(1 for ref in reference_labels if ref == label)
        if support == 0:
            continue
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1_values.append(f1_score(precision, recall))
    return float(np.mean(f1_values)) if f1_values else 0.0


def score_balanced_accuracy(reference_scores: List[int], predicted_scores: List[int], within_one: bool = False) -> float:
    recalls = []
    for score in sorted(set(reference_scores)):
        class_items = [(ref, pred) for ref, pred in zip(reference_scores, predicted_scores) if ref == score]
        if not class_items:
            continue
        if within_one:
            correct = sum(1 for ref, pred in class_items if abs(ref - pred) <= 1)
        else:
            correct = sum(1 for ref, pred in class_items if ref == pred)
        recalls.append(correct / len(class_items))
    return float(np.mean(recalls)) if recalls else 0.0


def to_int_score(value, default: int = 3) -> int:
    try:
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return default


def load_human_baseline() -> Dict:
    possible_paths = [
        BASE_DIR / "HUMAN_SENTIMENT_BASELINE.csv",
        BASE_DIR.parent / "HUMAN_SENTIMENT_BASELINE.csv",
    ]

    baseline_path = next((path for path in possible_paths if path.exists()), None)
    if baseline_path is None:
        print(f"Warning: human baseline not found. Tried: {possible_paths}")
        return {}

    df = pd.read_csv(baseline_path)
    baseline = {}
    duplicates = 0
    for _, row in df.iterrows():
        key = make_key(row.get("Topic", "Unknown"), row.get("Feedback", ""))
        if key in baseline:
            duplicates += 1
        baseline[key] = {
            "feedback": str(row.get("Feedback", "")),
            "topic": str(row.get("Topic", "Unknown")).strip(),
            "sentiment": str(row.get("Sentiment", "neutral")).strip().lower(),
            "score": to_int_score(row.get("Score", 3)),
            "reasoning": row.get("Reasoning", ""),
        }

    if duplicates:
        print(f"Warning: collapsed {duplicates} duplicate human-baseline keys.")
    return baseline


def load_model_results(model_name: str) -> Tuple[Dict, Dict]:
    json_path = MODELS.get(model_name)
    if not json_path or not json_path.exists():
        print(f"Warning: {model_name} results not found at {json_path}")
        return {}, {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    results = {}
    duplicates = 0

    for topic_data in data.get("topics", []):
        topic = topic_data.get("topic", "Unknown")
        for feedback_data in topic_data.get("feedback_with_sentiment", []):
            text = feedback_data.get("text", "")
            key = make_key(topic, text)
            if key in results:
                duplicates += 1
            results[key] = {
                "feedback": text,
                "topic": topic,
                "sentiment": str(feedback_data.get("sentiment", "neutral")).strip().lower(),
                "score": to_int_score(feedback_data.get("score", 3)),
                "reasoning": feedback_data.get("reasoning", ""),
                "time": float(feedback_data.get("processing_time", 0) or 0),
            }

    if duplicates:
        print(f"Warning: collapsed {duplicates} duplicate {model_name} prediction keys.")
    return results, metadata


def calculate_metrics(predictions: Dict, reference: Dict, model_name: str, model_type: str = "model") -> Dict:
    common_keys = sorted(set(predictions.keys()) & set(reference.keys()))
    reference_only = sorted(set(reference.keys()) - set(predictions.keys()))
    prediction_only = sorted(set(predictions.keys()) - set(reference.keys()))

    if not common_keys:
        return {
            "Model": model_name,
            "Type": model_type,
            "Common Items": 0,
            "Missing Baseline Items": len(reference_only),
            "Extra Prediction Items": len(prediction_only),
            "Exact Score %": 0.0,
            "Within 1 Score %": 0.0,
            "MAE": math.nan,
            "RMSE": math.nan,
            "Score Balanced Accuracy %": 0.0,
            "Within 1 Balanced Accuracy %": 0.0,
            "Sentiment Label Accuracy %": 0.0,
            "Sentiment Macro F1 %": 0.0,
        }

    reference_scores = [reference[key]["score"] for key in common_keys]
    predicted_scores = [predictions[key]["score"] for key in common_keys]
    reference_labels = [reference[key]["sentiment"] for key in common_keys]
    predicted_labels = [predictions[key]["sentiment"] for key in common_keys]

    errors = [abs(ref - pred) for ref, pred in zip(reference_scores, predicted_scores)]
    squared_errors = [(ref - pred) ** 2 for ref, pred in zip(reference_scores, predicted_scores)]

    exact = safe_divide(sum(error == 0 for error in errors), len(common_keys))
    within_one = safe_divide(sum(error <= 1 for error in errors), len(common_keys))
    label_accuracy = safe_divide(
        sum(ref == pred for ref, pred in zip(reference_labels, predicted_labels)),
        len(common_keys),
    )

    return {
        "Model": model_name,
        "Type": model_type,
        "Common Items": len(common_keys),
        "Missing Baseline Items": len(reference_only),
        "Extra Prediction Items": len(prediction_only),
        "Exact Score %": round(exact * 100, 2),
        "Within 1 Score %": round(within_one * 100, 2),
        "MAE": round(float(np.mean(errors)), 4),
        "RMSE": round(float(np.sqrt(np.mean(squared_errors))), 4),
        "Score Balanced Accuracy %": round(score_balanced_accuracy(reference_scores, predicted_scores) * 100, 2),
        "Within 1 Balanced Accuracy %": round(score_balanced_accuracy(reference_scores, predicted_scores, within_one=True) * 100, 2),
        "Sentiment Label Accuracy %": round(label_accuracy * 100, 2),
        "Sentiment Macro F1 %": round(label_macro_f1(reference_labels, predicted_labels) * 100, 2),
    }


def constant_score_baselines(reference: Dict) -> List[Dict]:
    baselines = []
    for score in range(1, 6):
        predictions = {
            key: {
                "score": score,
                "sentiment": "negative" if score <= 2 else "neutral" if score == 3 else "positive",
            }
            for key in reference
        }
        baselines.append(calculate_metrics(predictions, reference, f"Constant {score}", model_type="baseline"))
    return baselines


def metadata_to_speed(metadata: Dict) -> float:
    total_time = float(metadata.get("total_time", 0) or 0)
    num_feedbacks = float(metadata.get("num_feedbacks", 0) or 0)
    return num_feedbacks / total_time if total_time > 0 else 0.0


def score_label_inconsistencies(results: Dict) -> int:
    inconsistent = 0
    for item in results.values():
        sentiment = item["sentiment"]
        score = item["score"]
        if sentiment == "positive" and score < 4:
            inconsistent += 1
        elif sentiment == "negative" and score > 2:
            inconsistent += 1
        elif sentiment == "neutral" and score not in {2, 3}:
            inconsistent += 1
    return inconsistent


def pairwise_model_agreement(all_results: Dict[str, Dict]) -> List[Dict]:
    rows = []
    model_names = list(all_results.keys())
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i + 1 :]:
            results1 = all_results[model1]
            results2 = all_results[model2]
            common_keys = sorted(set(results1.keys()) & set(results2.keys()))
            if not common_keys:
                continue
            scores1 = [results1[key]["score"] for key in common_keys]
            scores2 = [results2[key]["score"] for key in common_keys]
            errors = [abs(a - b) for a, b in zip(scores1, scores2)]
            rows.append(
                {
                    "Model 1": model1,
                    "Model 2": model2,
                    "Common Items": len(common_keys),
                    "Exact Match %": round(safe_divide(sum(error == 0 for error in errors), len(errors)) * 100, 2),
                    "Within 1 Match %": round(safe_divide(sum(error <= 1 for error in errors), len(errors)) * 100, 2),
                    "Mean Absolute Difference": round(float(np.mean(errors)), 4),
                }
            )
    return rows


def print_report(model_rows: List[Dict], baseline_rows: List[Dict], metrics_dict: Dict[str, Dict]) -> None:
    print("=" * 80)
    print("SENTIMENT MODEL COMPARISON")
    print("=" * 80)

    print("\nModel metrics:")
    for row in sorted(model_rows, key=lambda item: (item["MAE"], -item["Exact Score %"])):
        speed = metrics_dict[row["Model"]]["speed"]
        print(
            f"  {row['Model']}: MAE={row['MAE']:.4f}, "
            f"Exact={row['Exact Score %']:.2f}%, "
            f"Within1={row['Within 1 Score %']:.2f}%, "
            f"BalancedExact={row['Score Balanced Accuracy %']:.2f}%, "
            f"Speed={speed:.2f} items/sec"
        )

    print("\nTrivial baselines:")
    for row in sorted(baseline_rows, key=lambda item: item["MAE"]):
        print(
            f"  {row['Model']}: MAE={row['MAE']:.4f}, "
            f"Exact={row['Exact Score %']:.2f}%, "
            f"Within1={row['Within 1 Score %']:.2f}%"
        )


def generate_comparison_report() -> None:
    human_baseline = load_human_baseline()
    print(f"Loaded human baseline: {len(human_baseline)} annotations")
    if not human_baseline:
        print("No human baseline available. Stopping.")
        return

    all_results: Dict[str, Dict] = {}
    all_metadata: Dict[str, Dict] = {}
    model_rows: List[Dict] = []
    metrics_dict: Dict[str, Dict] = {}

    for model_name in MODELS:
        results, metadata = load_model_results(model_name)
        if not results:
            continue

        all_results[model_name] = results
        all_metadata[model_name] = metadata
        row = calculate_metrics(results, human_baseline, model_name)
        row["Score/Label Inconsistencies"] = score_label_inconsistencies(results)
        speed = metadata_to_speed(metadata)
        row["Speed (items/sec)"] = round(speed, 4)
        row["Total Time (s)"] = round(float(metadata.get("total_time", 0) or 0), 4)
        row["Avg Time/Item (s)"] = round(float(metadata.get("avg_time_per_feedback", 0) or 0), 4)
        model_rows.append(row)
        metrics_dict[model_name] = {
            "speed": speed,
            "total_time": float(metadata.get("total_time", 0) or 0),
            "num_feedbacks": int(metadata.get("num_feedbacks", 0) or 0),
        }

    baseline_rows = constant_score_baselines(human_baseline)
    print_report(model_rows, baseline_rows, metrics_dict)

    best_model_row = min(model_rows, key=lambda row: (row["MAE"], -row["Exact Score %"])) if model_rows else None
    best_baseline_row = min(baseline_rows, key=lambda row: row["MAE"])
    best_speed_model = max(model_rows, key=lambda row: row["Speed (items/sec)"]) if model_rows else None

    pairwise = pairwise_model_agreement(all_results)

    output_dir = BASE_DIR / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_df = pd.DataFrame(model_rows + baseline_rows)
    comparison_df.to_csv(output_dir / "sentiment_model_comparison.csv", index=False)

    detailed_metrics = {
        "model_metrics": model_rows,
        "trivial_baselines": baseline_rows,
        "pairwise_agreement": pairwise,
        "best_model_by_mae": best_model_row["Model"] if best_model_row else None,
        "best_speed_model": best_speed_model["Model"] if best_speed_model else None,
        "best_trivial_baseline_by_mae": best_baseline_row["Model"],
        "best_model_beats_trivial_baseline_mae": bool(best_model_row and best_model_row["MAE"] < best_baseline_row["MAE"]),
        "baseline_size": len(human_baseline),
        "baseline_score_distribution": {
            str(score): sum(1 for item in human_baseline.values() if item["score"] == score)
            for score in range(1, 6)
        },
        "model_scores": {
            row["Model"]: {
                "mae": row["MAE"],
                "rmse": row["RMSE"],
                "exact_score_accuracy": row["Exact Score %"],
                "within_one_accuracy": row["Within 1 Score %"],
                "balanced_score_accuracy": row["Score Balanced Accuracy %"],
                "within_one_balanced_accuracy": row["Within 1 Balanced Accuracy %"],
                "sentiment_label_accuracy": row["Sentiment Label Accuracy %"],
                "sentiment_macro_f1": row["Sentiment Macro F1 %"],
                "speed": row.get("Speed (items/sec)", 0.0),
            }
            for row in model_rows
        },
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    metrics_json_path = output_dir / "sentiment_model_metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(detailed_metrics, f, indent=2)

    print("\nSaved reports:")
    print(f"  {output_dir / 'sentiment_model_comparison.csv'}")
    print(f"  {metrics_json_path}")


if __name__ == "__main__":
    generate_comparison_report()
