import json
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]


def json_to_dataframe(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Convert classification JSON to a DataFrame with:
    - Column 1: feedback comment
    - Columns 2+: one per topic (checkmark if assigned)
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # Extract all unique topics
    all_topics = [item["topic"] for item in data_dict.get("topics", [])]

    # Build a dict: feedback_text -> set of assigned topics
    feedback_to_topics: dict[str, set] = {}

    for topic_item in data_dict.get("topics", []):
        topic = topic_item.get("topic", "")
        for feedback_item in topic_item.get("feedback", []):
            text = feedback_item.get("text", "")
            if text not in feedback_to_topics:
                feedback_to_topics[text] = set()
            feedback_to_topics[text].add(topic)

    # Build rows
    rows = []
    for feedback_text, assigned_topics in feedback_to_topics.items():
        row = {"Feedback": feedback_text}
        for topic in all_topics:
            row[topic] = "✓" if topic in assigned_topics else ""
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns: Feedback first, then topics
    cols = ["Feedback"] + [t for t in all_topics if t != "Feedback"]
    df = df[cols]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved classification to {output_path}")

    return df


def sentiment_json_to_dataframe(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Convert sentiment JSON to a DataFrame with:
    - Column 1: feedback comment
    - Column 2: topic
    - Column 3: sentiment (positive/negative/neutral)
    - Column 4: score (0-4)
    - Column 5: reasoning
    """
    with open(input_path, "r", encoding="utf-8") as f:
        data_dict = json.load(f)

    # Build rows
    rows = []
    for topic_item in data_dict.get("topics", []):
        topic = topic_item.get("topic", "")
        for feedback_item in topic_item.get("feedback_with_sentiment", []):
            text = feedback_item.get("text", "")
            sentiment = feedback_item.get("sentiment", "neutral")
            score = feedback_item.get("score", 2)
            reasoning = feedback_item.get("reasoning", "")
            
            rows.append({
                "Feedback": text,
                "Topic": topic,
                "Sentiment": sentiment,
                "Score": score,
                "Reasoning": reasoning
            })

    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Reorder columns
    cols = ["Feedback", "Topic", "Sentiment", "Score", "Reasoning"]
    df = df[cols]

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved sentiment analysis to {output_path}")

    return df