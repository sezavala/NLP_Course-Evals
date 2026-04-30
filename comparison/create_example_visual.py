"""Create a worked-example poster visual for one course-evaluation comment."""

from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

BASE_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_START = "Professor Wu is a wonderful teacher"

CLASSIFICATION_FILES = {
    "Human": BASE_DIR / "HUMAN_CATEGORIZED_OUTPUT.csv",
    "Llama3": BASE_DIR / "results" / "Llama3" / "LLAMA_OUTPUT.csv",
    "Gemma": BASE_DIR / "results" / "Gemma" / "GEMMA_OUTPUT.csv",
    "roBERTa": BASE_DIR / "results" / "roBERTa" / "roBERTa_OUTPUT.csv",
    "DistilroBERTa": BASE_DIR / "results" / "DistilroBERTa" / "DISTILROBERTA_OUTPUT.csv",
}

SENTIMENT_FILES = {
    "Human": BASE_DIR / "HUMAN_SENTIMENT_BASELINE.csv",
    "Llama3": BASE_DIR / "results" / "Llama3" / "LLAMA_SENTIMENT.csv",
    "Gemma": BASE_DIR / "results" / "Gemma" / "GEMMA_SENTIMENT.csv",
    "roBERTa": BASE_DIR / "results" / "roBERTa" / "ROBERTA_SENTIMENT.csv",
    "DistilroBERTa": BASE_DIR / "results" / "DistilroBERTa" / "DISTILROBERTA_SENTIMENT.csv",
}

TOPIC_LABELS = {
    "Course organization and structure": "Course organization",
    "Pace": "Pace",
    "Workload": "Workload",
    "Student engagement and participation": "Student engagement",
    "Clarity of explanations": "Clarity",
    "Effectiveness of assignments": "Assignments",
    "Classroom atmosphere": "Atmosphere",
    "Instructor's communication and availability": "Communication",
    "Inclusivity and sense of belonging": "Inclusivity",
    "Assessment": "Assessment",
    "Grading and feedback": "Grading/feedback",
    "Learning resources and materials": "Learning materials",
    "None of the above / Other": "Other",
}


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


def find_example_row(df: pd.DataFrame) -> pd.Series:
    matches = df[df["Feedback"].astype(str).str.startswith(EXAMPLE_START)]
    if matches.empty:
        raise ValueError(f"Could not find example comment starting with: {EXAMPLE_START}")
    return matches.iloc[0]


def assigned_topics(row: pd.Series) -> set[str]:
    topics = set()
    for column, value in row.items():
        if column == "Feedback":
            continue
        if pd.notna(value) and str(value).strip():
            topics.add(column)
    return topics


def load_classification_assignments() -> tuple[str, dict[str, set[str]], list[str]]:
    assignments = {}
    all_topics = []
    comment_text = ""

    for label, path in CLASSIFICATION_FILES.items():
        df = pd.read_csv(path).fillna("")
        row = find_example_row(df)
        if not comment_text:
            comment_text = row["Feedback"]
        topics = assigned_topics(row)
        assignments[label] = topics
        for topic in topics:
            if topic not in all_topics:
                all_topics.append(topic)

    canonical_order = [
        topic
        for topic in TOPIC_LABELS
        if topic in all_topics
    ]
    return comment_text, assignments, canonical_order


def load_sentiment_scores(comment_text: str, topics: list[str]) -> dict[str, dict[str, float]]:
    scores = {label: {} for label in SENTIMENT_FILES}
    normalized_comment = normalize_text(comment_text)

    for label, path in SENTIMENT_FILES.items():
        df = pd.read_csv(path).fillna("")
        matches = df[df["Feedback"].map(normalize_text) == normalized_comment]
        for _, row in matches.iterrows():
            topic = row["Topic"]
            if topic in topics:
                scores[label][topic] = float(row["Score"])
    return scores


def shorten_comment(comment_text: str) -> str:
    """Use a readable poster excerpt instead of the full multi-sentence comment."""
    excerpt = (
        "Professor Wu is a wonderful teacher and is very organized and clear with his teaching. "
        "He provided resources, practice material, lecture notes, and review sessions. "
        "The pace did seem a bit fast and rushed at times."
    )
    if not normalize_text(comment_text).startswith(normalize_text(EXAMPLE_START)):
        excerpt = comment_text
    return "\n".join(textwrap.wrap(f'"{excerpt}"', width=115))


def create_example_visual() -> None:
    comment_text, assignments, topics = load_classification_assignments()
    scores = load_sentiment_scores(comment_text, topics)
    labels = list(CLASSIFICATION_FILES.keys())
    topic_labels = [TOPIC_LABELS.get(topic, topic) for topic in topics]

    class_matrix = np.array(
        [
            [1 if topic in assignments[label] else 0 for label in labels]
            for topic in topics
        ]
    )

    score_matrix = np.full((len(topics), len(labels)), np.nan)
    for row_idx, topic in enumerate(topics):
        for col_idx, label in enumerate(labels):
            if topic in scores[label]:
                score_matrix[row_idx, col_idx] = scores[label][topic]

    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[0.7, 3.0])
    text_ax = fig.add_subplot(grid[0, :])
    class_ax = fig.add_subplot(grid[1, 0])
    score_ax = fig.add_subplot(grid[1, 1])

    fig.suptitle("Worked Example: One Comment Through the Pipeline", fontsize=18, fontweight="bold")
    text_ax.axis("off")
    text_ax.text(
        0.01,
        0.5,
        shorten_comment(comment_text),
        ha="left",
        va="center",
        fontsize=11,
        color="#222222",
    )

    class_cmap = ListedColormap(["#f2f2f2", "#54a24b"])
    class_ax.imshow(class_matrix, aspect="auto", cmap=class_cmap, vmin=0, vmax=1)
    class_ax.set_title("Topic Assignment", fontweight="bold")
    class_ax.set_xticks(np.arange(len(labels)))
    class_ax.set_xticklabels(labels, rotation=20, ha="right")
    class_ax.set_yticks(np.arange(len(topic_labels)))
    class_ax.set_yticklabels(topic_labels)

    score_cmap = plt.cm.get_cmap("RdYlGn").copy()
    score_cmap.set_bad("#f2f2f2")
    masked_scores = np.ma.masked_invalid(score_matrix)
    image = score_ax.imshow(masked_scores, aspect="auto", cmap=score_cmap, vmin=1, vmax=5)
    score_ax.set_title("Sentiment Score by Topic", fontweight="bold")
    score_ax.set_xticks(np.arange(len(labels)))
    score_ax.set_xticklabels(labels, rotation=20, ha="right")
    score_ax.set_yticks(np.arange(len(topic_labels)))
    score_ax.set_yticklabels(topic_labels)
    for row_idx in range(score_matrix.shape[0]):
        for col_idx in range(score_matrix.shape[1]):
            value = score_matrix[row_idx, col_idx]
            if not np.isnan(value):
                score_ax.text(
                    col_idx,
                    row_idx,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
    cbar = fig.colorbar(image, ax=score_ax, fraction=0.046, pad=0.04)
    cbar.set_label("Score (1=negative, 5=positive)")

    output_path = BASE_DIR / "results" / "visualizations" / "example_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved example visualization to {output_path}")


if __name__ == "__main__":
    create_example_visual()
