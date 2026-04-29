from pathlib import Path
from typing import List, Dict
import json
import requests
import time

from experiments.struct import TopicCategorization, Comment, AllTopics
from experiments.json_to_sheet import json_to_dataframe
from data import TOPIC_DEFS, TOPIC_KEYS, FEEDBACK_LIST

BASE_DIR = Path(__file__).resolve().parents[2]
OTHER_LABEL = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER_LABEL]

# Ollama endpoint
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"  # Updated from llama2:13b


def classify_with_llama(feedback: str) -> List[str]:
    """Use Llama 3 via Ollama for topic classification only."""
    
    topics_str = "\n".join(f"- {t}: {TOPIC_DEFS[t]}" for t in TOPIC_KEYS)
    
    prompt = f"""Classify this course feedback into ALL applicable topics if explicitly mentioned.

TOPICS:
{topics_str}
- {OTHER_LABEL}: Generic praise with no substance.

RULES:
1. Assign to ALL applicable topics (can be 1 or more).
2. Generic praise only (e.g., "Best professor ever", "Eric Wu is my goat") → "{OTHER_LABEL}" ONLY.
3. Return ONLY valid JSON, no markdown.

FEEDBACK: "{feedback}"

Return JSON (topics must be exact matches):
{{"topics": ["Topic1", "Topic2"]}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
            },
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
        output_text = result.get("response", "")
        
    except Exception as e:
        print(f"  ERROR: {e}")
        return [OTHER_LABEL]
    
    # Parse JSON
    try:
        json_start = output_text.find("{")
        json_end = output_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found")
        json_str = output_text[json_start:json_end]
        parsed = json.loads(json_str)
        print(f"  Parsed: {parsed}")
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return [OTHER_LABEL]

    topics = parsed.get("topics", [OTHER_LABEL])
    if not isinstance(topics, list):
        topics = [topics]
    
    # Validate topics (exact match)
    valid_topics = [t for t in topics if t in TOPICS]
    if not valid_topics:
        return [OTHER_LABEL]
    
    return valid_topics


def main() -> None:
    bucket: Dict[str, List[Comment]] = {t: [] for t in TOPICS}

    for idx, feedback in enumerate(FEEDBACK_LIST, 1):
        print(f"\n[{idx}] Classifying feedback...")
        
        topics = classify_with_llama(feedback)
        
        comment = Comment(text=feedback, sentiment=None)
        for topic in topics:
            bucket[topic].append(comment)
        

    final_output = AllTopics(
        topics=[
            TopicCategorization(topic=t, feedback=bucket[t])
            for t in TOPICS
        ]
    )

    output_path = BASE_DIR / "results" / "Llama3" / "LLAMA_OUTPUT.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output.model_dump_json(indent=2))

    json_to_dataframe(output_path, output_path.parent / "LLAMA_OUTPUT.csv")
    print(f"\n✓ Analysis complete. Saved to {output_path}.")


if __name__ == "__main__":
    main()