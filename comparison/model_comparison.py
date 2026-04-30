from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("'", "'").replace(""", '"').replace(""", '"')
    text = " ".join(text.split())
    return text


def get_topics(row, topics_list):
    """Extract assigned topics from a row."""
    return {topic for topic in topics_list if pd.notna(row[topic]) and row[topic] != ""}


def find_closest_match(feedback, current_df):
    """Find the closest matching feedback using fuzzy matching."""
    normalized_benchmark = normalize_text(feedback)
    best_match = None
    best_ratio = 0
    
    for idx, row in current_df.iterrows():
        normalized_current = normalize_text(row["Feedback"])
        ratio = SequenceMatcher(None, normalized_benchmark, normalized_current).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = row
    
    return best_match, best_ratio


def compare(benchmark, current):
    """Compare model output against benchmark."""
    topics = [col for col in benchmark.columns if col != "Feedback"]
    
    total_score = 0
    perfect_matches = 0
    total_comments = 0
    all_correct = 0
    all_missed = 0
    all_overclassified = 0
    
    for _, bench_row in benchmark.iterrows():
        feedback = bench_row["Feedback"]
        
        # Find closest match in model output
        model_row, similarity = find_closest_match(feedback, current)
        
        if model_row is None:
            continue
        
        total_comments += 1
        
        # Get topic sets
        benchmark_topics = get_topics(bench_row, topics)
        model_topics = get_topics(model_row, topics)
        
        # Calculate metrics
        if benchmark_topics == model_topics:
            total_score += 10
            perfect_matches += 1
        else:
            true_positives = benchmark_topics & model_topics
            all_correct += len(true_positives)
            total_score += len(true_positives) * 3
            
            false_negatives = benchmark_topics - model_topics
            all_missed += len(false_negatives)
            total_score -= len(false_negatives) * 2
            
            false_positives = model_topics - benchmark_topics
            all_overclassified += len(false_positives)
            total_score -= len(false_positives) * 1.5
    
    if total_comments == 0:
        return {"accuracy": 0, "perfect_matches": 0, "total_comments": 0}
    
    max_possible_score = total_comments * 10
    accuracy_percentage = (total_score / max_possible_score) * 100
    accuracy_percentage = max(0, min(100, accuracy_percentage))
    
    return {
        "accuracy": round(accuracy_percentage, 2),
        "raw_score": round(total_score, 2),
        "perfect_matches": perfect_matches,
        "perfect_match_rate": round((perfect_matches / total_comments) * 100, 2),
        "total_comments": total_comments,
        "correct_topics": all_correct,
        "missed_topics": all_missed,
        "overclassified_topics": all_overclassified,
    }


def print_results(scores):
    """Pretty print comparison results."""
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for model, metrics in sorted_scores:
        print(f"\n{model}:")
        print(f"  Accuracy:              {metrics['accuracy']}%")
        print(f"  Raw Score:             {metrics['raw_score']}")
        print(f"  Perfect Matches:       {metrics['perfect_matches']}/{metrics['total_comments']} ({metrics['perfect_match_rate']}%)")
        print(f"  Correct Topics:        {metrics['correct_topics']}")
        print(f"  Missed Topics:         {metrics['missed_topics']}")
        print(f"  Overclassified Topics: {metrics['overclassified_topics']}")


def visualize_results(scores):
    """Create visualization charts for model comparison."""
    models = list(scores.keys())
    accuracies = [scores[m]["accuracy"] for m in models]
    perfect_matches = [scores[m]["perfect_match_rate"] for m in models]
    correct = [scores[m]["correct_topics"] for m in models]
    missed = [scores[m]["missed_topics"] for m in models]
    overclassified = [scores[m]["overclassified_topics"] for m in models]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Model Comparison Analysis", fontsize=16, fontweight="bold")
    
    # 1. Accuracy comparison
    colors = ["#2ecc71" if acc > 20 else "#e74c3c" for acc in accuracies]
    axes[0, 0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor="black")
    axes[0, 0].set_ylabel("Accuracy (%)", fontweight="bold")
    axes[0, 0].set_title("Overall Accuracy", fontweight="bold")
    axes[0, 0].set_ylim([0, 100])
    for i, v in enumerate(accuracies):
        axes[0, 0].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
    
    # 2. Perfect match rate
    axes[0, 1].bar(models, perfect_matches, color="#3498db", alpha=0.7, edgecolor="black")
    axes[0, 1].set_ylabel("Perfect Match Rate (%)", fontweight="bold")
    axes[0, 1].set_title("Perfect Matches", fontweight="bold")
    axes[0, 1].set_ylim([0, 100])
    for i, v in enumerate(perfect_matches):
        axes[0, 1].text(i, v + 2, f"{v}%", ha="center", fontweight="bold")
    
    # 3. Topic classification metrics
    x = np.arange(len(models))
    width = 0.25
    axes[1, 0].bar(x - width, correct, width, label="Correct", color="#2ecc71", alpha=0.7, edgecolor="black")
    axes[1, 0].bar(x, missed, width, label="Missed", color="#e74c3c", alpha=0.7, edgecolor="black")
    axes[1, 0].bar(x + width, overclassified, width, label="Overclassified", color="#f39c12", alpha=0.7, edgecolor="black")
    axes[1, 0].set_ylabel("Count", fontweight="bold")
    axes[1, 0].set_title("Topic Classification Metrics", fontweight="bold")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(models)
    axes[1, 0].legend()
    
    # 4. Raw scores
    raw_scores = [scores[m]["raw_score"] for m in models]
    colors_score = ["#2ecc71" if s >= 0 else "#e74c3c" for s in raw_scores]
    axes[1, 1].bar(models, raw_scores, color=colors_score, alpha=0.7, edgecolor="black")
    axes[1, 1].set_ylabel("Raw Score", fontweight="bold")
    axes[1, 1].set_title("Raw Scores", fontweight="bold")
    axes[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
    for i, v in enumerate(raw_scores):
        axes[1, 1].text(i, v + (3 if v >= 0 else -5), f"{v}", ha="center", fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(BASE_DIR / "results" / "visualizations" / "model_comparison.png", dpi=300, bbox_inches="tight")
    print("\n✓ Visualization saved to: visualization/model_comparison.png")
    plt.show()


if __name__ == "__main__":
    baseline_path = BASE_DIR / "HUMAN_CATEGORIZED_OUTPUT.csv"
    benchmark = pd.read_csv(baseline_path)
    
    print(f"Comparing {len(benchmark)} benchmark feedbacks...\n")
    
    models = {
        "Gemma": "GEMMA_OUTPUT.csv",
        "Llama3": "LLAMA_OUTPUT.csv",
        "roBERTa": "roBERTa_OUTPUT.csv",
        "DistilroBERTa": "DISTILROBERTA_OUTPUT.csv"
    }

    scores = {}

    for model, path in models.items():
        model_path = BASE_DIR / "results" / model / path
        if not model_path.exists():
            print(f"✗ {model}: File not found")
            continue
        
        print(f"Comparing {model}...")
        current = pd.read_csv(model_path)
        scores[model] = compare(benchmark, current)

    print_results(scores)
    visualize_results(scores)