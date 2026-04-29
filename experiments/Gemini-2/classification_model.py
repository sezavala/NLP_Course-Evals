from pathlib import Path
from typing import List, Dict
import json
import requests
import time

from pydantic import BaseModel, Field

from experiments.struct import TopicCategorization, Comment, AllTopics
from experiments.json_to_sheet import json_to_dataframe
from data import TOPIC_KEYS, TOPIC_DEFS, FEEDBACK_LIST

BASE_DIR = Path(__file__).resolve().parents[2]

OTHER_LABEL = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER_LABEL]

# Local Ollama endpoint using Gemma
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:latest"

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


class CommentClassification(BaseModel):
    topics: List[str] = Field(
        description=f"List of topics this feedback belongs to. Must be from: {TOPIC_KEYS + [OTHER_LABEL]}"
    )


def classify_with_gemma(feedback: str, attempt: int = 1) -> List[str]:
    """Classify a single feedback comment into one or more topics using a local Gemma model via Ollama."""

    topics_str = "\n".join(f"- {t}: {TOPIC_DEFS[t]}" for t in TOPIC_KEYS)

    prompt = f"""You are classifying a course evaluation comment into one or more topics.

ALLOWED TOPICS:
{topics_str}
- {OTHER_LABEL}: Generic praise or comments that don't fit the above categories.

RULES:
1. Assign to ALL applicable topics (can be 1 or multiple).
2. If a comment is generic praise with no substance (e.g., "Best professor ever", "Eric Wu is my goat"), assign ONLY to "{OTHER_LABEL}". Do NOT assign to any other topic if "{OTHER_LABEL}" is selected.
3. Topic names must be exact matches from the list above.
4. Return ONLY valid JSON with this schema: {{"topics": ["Topic 1", "Topic 2"]}}
5. Do not include markdown, explanations, or extra text.

FEEDBACK:
"{feedback}"
""".strip()

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.2,
                "format": "json",
            },
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = payload.get("response", "")

        # Some Ollama responses may still include surrounding text; extract the JSON object defensively.
        json_start = raw_text.find("{")
        json_end = raw_text.rfind("}") + 1
        if json_start == -1 or json_end <= json_start:
            raise ValueError("No JSON object found in model response")
        json_text = raw_text[json_start:json_end]

        result = CommentClassification.model_validate_json(json_text)

        valid_topics = [t for t in result.topics if t in TOPICS]
        if not valid_topics:
            valid_topics = [OTHER_LABEL]

        # Enforce the fallback rule cleanly.
        if OTHER_LABEL in valid_topics and len(valid_topics) > 1:
            valid_topics = [OTHER_LABEL]

        return valid_topics

    except Exception as e:
        if attempt < MAX_RETRIES:
            backoff = INITIAL_BACKOFF ** attempt
            print(f"  Attempt {attempt} failed: {type(e).__name__}. Retrying in {backoff}s...")
            time.sleep(backoff)
            return classify_with_gemma(feedback, attempt + 1)
        else:
            print(f"  Failed after {MAX_RETRIES} attempts. Assigning to Other.")
            return [OTHER_LABEL]


def main() -> None:
    bucket: Dict[str, List[Comment]] = {t: [] for t in TOPICS}

    for idx, feedback in enumerate(FEEDBACK_LIST, 1):
        print(f"Processing {idx}/{len(FEEDBACK_LIST)}: {feedback[:50]}...")

        topics = classify_with_gemma(feedback)
        print(f"  → Topics: {topics}")

        comment = Comment(text=feedback, sentiment=None)
        for topic in topics:
            bucket[topic].append(comment)

    final_output = AllTopics(
        topics=[
            TopicCategorization(topic=t, feedback=bucket[t])
            for t in TOPICS
        ]
    )

    output_path = BASE_DIR / "results" / "Gemma" / "GEMMA_OUTPUT.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output.model_dump_json(indent=2))

    json_to_dataframe(output_path, output_path.parent / "GEMMA_OUTPUT.csv")

    print(f"\nAnalysis complete. Results saved to {output_path}.")


if __name__ == "__main__":
    main()
