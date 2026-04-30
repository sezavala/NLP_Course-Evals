"""
Sentiment Model Comparison Visualization for Poster
Creates a comprehensive visualization comparing models
"""

from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]

def load_metrics():
    """Load metrics from JSON file."""
    metrics_path = BASE_DIR / "results" / "sentiment_model_metrics.json"
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Please run compare_sentiment_models.py first")
        return None
    
    with open(metrics_path, "r") as f:
        return json.load(f)


def create_poster_visualization():
    """Create a clean, minimal poster visualization."""
    
    metrics = load_metrics()
    if not metrics:
        return
    
    # Extract data
    model_scores = metrics.get("model_scores", {})
    best_model = metrics.get("best_model", "")
    accuracy_vs_baseline = metrics.get("accuracy_vs_baseline", [])
    
    if not model_scores:
        print("No model scores found in metrics")
        return
    
    models = list(model_scores.keys())
    
    # Extract accuracies from accuracy_vs_baseline data
    accuracies = []
    for acc_item in accuracy_vs_baseline:
        if acc_item["Model"] in models:
            close_acc = float(acc_item["Close Match (±1) %"].rstrip("%"))
            accuracies.append(close_acc)
    
    if len(accuracies) != len(models):
        accuracies = [model_scores[m].get("accuracy", 0) for m in models]
    
    speeds = [model_scores[m]["speed"] for m in models]
    
    # Simple color scheme
    colors = ['#808080', '#A9A9A9', '#D3D3D3', '#404040']
    best_color = '#1f77b4'
    
    # Single clean plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, model in enumerate(models):
        if model == best_model:
            color = best_color
            size = 300
            marker = 'D'
            zorder = 5
            alpha = 1.0
        else:
            color = colors[i]
            size = 150
            marker = 'o'
            zorder = 3
            alpha = 0.7
        
        ax.scatter(speeds[i], accuracies[i], s=size, c=color, marker=marker, 
                  edgecolors='black', linewidth=1.5, alpha=alpha, zorder=zorder)
        ax.text(speeds[i], accuracies[i] + 1.5, model, ha='center', fontsize=11, fontweight='bold')
    
    # Clean styling
    ax.set_xlabel('Speed (items/sec)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy vs Human Baseline (%)', fontsize=13, fontweight='bold')
    ax.set_title('Sentiment Model Performance', fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-5, max(speeds) + 10)
    ax.set_ylim(80, 95)
    ax.tick_params(labelsize=11)
    
    # Save figure
    output_path = BASE_DIR / "results" / "visualizations" / "sentiment_model_comparison_poster.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Poster visualization saved to {output_path}")
    
    plt.close()


if __name__ == "__main__":
    create_poster_visualization()
