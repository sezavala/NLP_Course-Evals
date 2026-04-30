"""
Sentiment Model Comparison Script
Compares accuracy and performance across all sentiment models (Llama3, Gemma, roBERTa, DistilroBERTa)
"""

from pathlib import Path
import json
import pandas as pd
from typing import Dict, Tuple, List
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

# Model paths
MODELS = {
    "Llama3": BASE_DIR / "results" / "Llama3" / "LLAMA_SENTIMENT.json",
    "Gemma": BASE_DIR / "results" / "Gemma" / "GEMMA_SENTIMENT.json",
    "roBERTa": BASE_DIR / "results" / "roBERTa" / "ROBERTA_SENTIMENT.json",
    "DistilroBERTa": BASE_DIR / "results" / "DistilroBERTa" / "DISTILROBERTA_SENTIMENT.json"
}


def load_human_baseline() -> Dict:
    """Load human-annotated baseline from CSV."""
    # Try multiple possible locations
    possible_paths = [
        BASE_DIR.parent / "HUMAN_SENTIMENT_BASELINE.csv",
        BASE_DIR / "HUMAN_SENTIMENT_BASELINE.csv",
        Path("/Users/seclab/Desktop/NLP_Experiments/HUMAN_SENTIMENT_BASELINE.csv")
    ]
    
    baseline_path = None
    for path in possible_paths:
        if path.exists():
            baseline_path = path
            break
    
    if not baseline_path:
        print(f"Warning: Human baseline not found. Tried: {possible_paths}")
        return {}
    
    baseline = {}
    df = pd.read_csv(baseline_path)
    
    for _, row in df.iterrows():
        feedback_text = str(row.get("Feedback", "")).strip()
        topic = str(row.get("Topic", "Unknown")).strip()
        
        # Use text + topic as unique key
        key = (topic, feedback_text[:100])
        
        baseline[key] = {
            "sentiment": row.get("Sentiment", "neutral").lower(),
            "score": int(row.get("Score", 3)),
            "reasoning": row.get("Reasoning", "")
        }
    
    return baseline


def load_model_results(model_name: str) -> Tuple[Dict, Dict]:
    """Load sentiment results and metadata for a model."""
    json_path = MODELS.get(model_name)
    if not json_path or not json_path.exists():
        print(f"Warning: {model_name} results not found at {json_path}")
        return {}, {}
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    metadata = data.get("metadata", {})
    results = {}
    
    # Extract feedback -> (sentiment, score) mapping
    for topic_data in data.get("topics", []):
        topic = topic_data.get("topic", "Unknown")
        for feedback_data in topic_data.get("feedback_with_sentiment", []):
            text = feedback_data.get("text", "")
            key = (topic, text[:100])  # Use topic + first 100 chars as key
            
            results[key] = {
                "sentiment": feedback_data.get("sentiment", "neutral"),
                "score": feedback_data.get("score", 3),
                "reasoning": feedback_data.get("reasoning", ""),
                "time": feedback_data.get("processing_time", 0)
            }
    
    return results, metadata


def calculate_agreement(score1: int, score2: int, threshold: int = 1) -> bool:
    """Check if two scores agree within a threshold."""
    return abs(score1 - score2) <= threshold


def calculate_model_accuracy(model_results: Dict, reference_results: Dict) -> float:
    """Calculate accuracy as percentage of common items with close agreement."""
    common = set(model_results.keys()) & set(reference_results.keys())
    if not common:
        return 0.0
    
    agreement_count = sum(
        1 for key in common
        if calculate_agreement(model_results[key]["score"], reference_results[key]["score"], threshold=1)
    )
    return (agreement_count / len(common)) * 100


def generate_comparison_report() -> None:
    """Generate comprehensive comparison report."""
    
    print("=" * 80)
    print("SENTIMENT MODEL COMPARISON REPORT")
    print("=" * 80)
    
    # Load human baseline (ground truth)
    human_baseline = load_human_baseline()
    print(f"\n✓ Loaded human baseline: {len(human_baseline)} annotations")
    
    if not human_baseline:
        print("Warning: Human baseline is empty. Accuracy metrics may be incomplete.")
    
    # Load all model results
    all_results = {}
    all_metadata = {}
    
    for model_name in MODELS.keys():
        results, metadata = load_model_results(model_name)
        all_results[model_name] = results
        all_metadata[model_name] = metadata
        print(f"✓ Loaded {model_name}: {len(results)} predictions")
    
    # Ensure we have data
    if not all_results or not any(all_results.values()):
        print("Error: No model results found. Please run sentiment models first.")
        return
    
    model_names = [m for m in MODELS.keys() if all_results.get(m)]
    
    # Performance metrics by model
    print("\n" + "=" * 80)
    print("PERFORMANCE & SPEED METRICS")
    print("=" * 80)
    
    performance_data = []
    metrics_dict = {}
    
    for model_name, metadata in all_metadata.items():
        if not metadata:
            continue
        
        total_time = metadata.get("total_time", 0)
        num_feedbacks = metadata.get("num_feedbacks", 0)
        avg_time = metadata.get("avg_time_per_feedback", 0)
        
        metrics_dict[model_name] = {
            "total_time": total_time,
            "num_feedbacks": num_feedbacks,
            "avg_time": avg_time,
            "speed": num_feedbacks / total_time if total_time > 0 else 0
        }
        
        performance_data.append({
            "Model": model_name,
            "Total Feedbacks": num_feedbacks,
            "Total Time (s)": f"{total_time:.2f}",
            "Avg Time/Feedback (s)": f"{avg_time:.4f}",
            "Speed (items/sec)": f"{num_feedbacks / total_time:.2f}" if total_time > 0 else "N/A"
        })
        
        print(f"\n{model_name}:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Feedbacks processed: {num_feedbacks}")
        print(f"  Avg time per feedback: {avg_time:.4f}s")
        print(f"  Speed: {num_feedbacks / total_time:.2f} items/sec" if total_time > 0 else "N/A")
    
    # Model agreement & accuracy analysis
    print("\n" + "=" * 80)
    print("MODEL ACCURACY ANALYSIS (vs Human Baseline)")
    print("=" * 80)
    
    accuracy_data = []
    agreement_data = []
    
    # Calculate accuracy against human baseline
    if human_baseline:
        print(f"\nCalculating accuracy for {len(model_names)} models...\n")
        
        for model in model_names:
            model_results = all_results[model]
            
            # Find common items between model and baseline
            common_keys = set(model_results.keys()) & set(human_baseline.keys())
            
            if common_keys:
                # Calculate exact and close agreement
                exact_matches = sum(
                    1 for key in common_keys
                    if model_results[key]["score"] == human_baseline[key]["score"]
                )
                close_matches = sum(
                    1 for key in common_keys
                    if calculate_agreement(model_results[key]["score"], 
                                         human_baseline[key]["score"], threshold=1)
                )
                
                exact_accuracy = (exact_matches / len(common_keys)) * 100
                close_accuracy = (close_matches / len(common_keys)) * 100
                
                accuracy_data.append({
                    "Model": model,
                    "Exact Match %": f"{exact_accuracy:.1f}%",
                    "Close Match (±1) %": f"{close_accuracy:.1f}%",
                    "Common Items": len(common_keys)
                })
                
                print(f"{model}:")
                print(f"  Common items with baseline: {len(common_keys)}")
                print(f"  Exact score match: {exact_matches}/{len(common_keys)} ({exact_accuracy:.1f}%)")
                print(f"  Close match (±1): {close_matches}/{len(common_keys)} ({close_accuracy:.1f}%)\n")
            else:
                print(f"{model}: No common items with baseline")
                accuracy_data.append({
                    "Model": model,
                    "Exact Match %": "0.0%",
                    "Close Match (±1) %": "0.0%",
                    "Common Items": 0
                })
    else:
        print("No human baseline available. Skipping accuracy calculation.")
    
    # Pairwise agreement between models
    print("-" * 80)
    print("Pairwise Model Agreement:")
    print("-" * 80)
    
    if len(model_names) > 1:
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                results1 = all_results[model1]
                results2 = all_results[model2]
                
                # Find common feedback items
                common = set(results1.keys()) & set(results2.keys())
                
                if common:
                    exact_matches = sum(
                        1 for key in common
                        if results1[key]["score"] == results2[key]["score"]
                    )
                    close_matches = sum(
                        1 for key in common
                        if calculate_agreement(results1[key]["score"], results2[key]["score"], threshold=1)
                    )
                    
                    exact_pct = (exact_matches / len(common)) * 100 if common else 0
                    close_pct = (close_matches / len(common)) * 100 if common else 0
                    
                    agreement_data.append({
                        "Model 1": model1,
                        "Model 2": model2,
                        "Common Items": len(common),
                        "Exact Match %": f"{exact_pct:.1f}%",
                        "Close Match (±1) %": f"{close_pct:.1f}%"
                    })
                    
                    print(f"\n{model1} vs {model2}:")
                    print(f"  Common feedbacks: {len(common)}")
                    print(f"  Exact score match: {exact_matches}/{len(common)} ({exact_pct:.1f}%)")
                    print(f"  Close match (±1): {close_matches}/{len(common)} ({close_pct:.1f}%)")
    
    # Determine best model (balance of speed and accuracy vs human baseline)
    print("\n" + "=" * 80)
    print("BEST MODEL ANALYSIS")
    print("=" * 80)
    
    # Extract accuracy values for scoring
    model_accuracy_map = {}
    for acc_data in accuracy_data:
        model = acc_data["Model"]
        close_accuracy = float(acc_data["Close Match (±1) %"].rstrip("%"))
        model_accuracy_map[model] = close_accuracy
    
    if model_accuracy_map and metrics_dict:
        accuracies = list(model_accuracy_map.values())
        speeds = [metrics_dict[m]["speed"] for m in model_accuracy_map.keys()]
        
        min_speed = min(speeds) if speeds else 1
        max_speed = max(speeds) if speeds else 1
        min_accuracy = min(accuracies) if accuracies else 0
        max_accuracy = max(accuracies) if accuracies else 100
        
        model_scores = {}
        best_accuracy = None
        best_speed = None
        best_balanced = None
        best_balanced_score = 0
        
        for model in model_accuracy_map.keys():
            # Get metrics
            speed = metrics_dict[model]["speed"]
            accuracy = model_accuracy_map[model]
            
            # Normalize to 0-100 scale
            norm_speed = ((speed - min_speed) / (max_speed - min_speed) * 100) if max_speed > min_speed else 50
            norm_accuracy = ((accuracy - min_accuracy) / (max_accuracy - min_accuracy) * 100) if max_accuracy > min_accuracy else 50
            
            # Balanced score (60% accuracy, 40% speed)
            balanced_score = (0.6 * norm_accuracy) + (0.4 * norm_speed)
            model_scores[model] = {
                "accuracy": accuracy,
                "speed": speed,
                "norm_speed": norm_speed,
                "norm_accuracy": norm_accuracy,
                "balanced_score": balanced_score
            }
            
            if best_accuracy is None or accuracy > model_scores[best_accuracy]["accuracy"]:
                best_accuracy = model
            
            if best_speed is None or speed > model_scores[best_speed]["speed"]:
                best_speed = model
            
            if balanced_score > best_balanced_score:
                best_balanced_score = balanced_score
                best_balanced = model
        
        print(f"\nBest for Accuracy (vs Human Baseline): {best_accuracy} ({model_accuracy_map[best_accuracy]:.1f}%)")
        print(f"Best for Speed: {best_speed} ({model_scores[best_speed]['speed']:.2f} items/sec)")
        print(f"\n🏆 BEST OVERALL: {best_balanced}")
        print(f"   Balanced Score: {best_balanced_score:.1f}/100")
        print(f"   - Accuracy (vs Baseline): {model_scores[best_balanced]['accuracy']:.1f}%")
        print(f"   - Speed: {model_scores[best_balanced]['speed']:.2f} items/sec")
    else:
        print("Insufficient data for model scoring")
        model_scores = {}
        best_balanced = None
    
    # Save comparison reports
    print("\n" + "=" * 80)
    print("SAVING REPORTS")
    print("=" * 80)
    
    comparison_df = pd.DataFrame(performance_data)
    output_path = BASE_DIR / "results" / "sentiment_model_comparison.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    comparison_df.to_csv(output_path, index=False)
    print(f"\n✓ Comparison report saved to {output_path}")
    
    # Save detailed metrics JSON
    detailed_metrics = {
        "performance": performance_data,
        "accuracy_vs_baseline": accuracy_data,
        "pairwise_agreement": agreement_data,
        "model_scores": {k: {k2: v2 if not isinstance(v2, str) else v2 for k2, v2 in v.items()} 
                        for k, v in model_scores.items()} if model_scores else {},
        "best_model": best_balanced if best_balanced else None,
        "baseline_size": len(human_baseline),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    
    metrics_json_path = BASE_DIR / "results" / "sentiment_model_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(detailed_metrics, f, indent=2)
    print(f"✓ Detailed metrics saved to {metrics_json_path}")
    
    print("\n" + "=" * 80)
    print("Comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    generate_comparison_report()
