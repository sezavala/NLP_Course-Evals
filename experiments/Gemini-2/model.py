from pathlib import Path
from typing import List, Dict
import os
import time

from google import genai
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from experiments.struct import TopicCategorization, Comment, AllTopics
from experiments.json_to_sheet import json_to_dataframe
from data import TOPIC_KEYS, TOPIC_DEFS, FEEDBACK_LIST

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
BASE_DIR = Path(__file__).resolve().parents[2]

OTHER_LABEL = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER_LABEL]

# Retry settings
MAX_RETRIES = 5
INITIAL_BACKOFF = 2


class CommentClassification(BaseModel):
    topics: List[str] = Field(
        description=f"List of topics this feedback belongs to. Must be from: {TOPIC_KEYS + [OTHER_LABEL]}"
    )


def classify_with_gemini(feedback: str, attempt: int = 1) -> List[str]:
    """Classify a single feedback comment to one or more topics with retry logic."""
    
    topics_str = "\n".join(f"- {t}: {TOPIC_DEFS[t]}" for t in TOPIC_KEYS)
    
    prompt = f"""You are classifying a course evaluation comment into one or more topics.

ALLOWED TOPICS:
{topics_str}
- {OTHER_LABEL}: Generic praise or comments that don't fit the above categories.

RULES:
1. Assign to ALL applicable topics (can be 1 or multiple).
2. If a comment is generic praise with no substance (e.g., "Best professor ever", "Eric Wu is my goat"), assign ONLY to "{OTHER_LABEL}". Do NOT assign to any other topic if "{OTHER_LABEL}" is selected.
3. Topic names must be exact matches from the list above.

Return valid JSON only.

FEEDBACK:
"{feedback}"
""".strip()

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": CommentClassification.model_json_schema(),
            },
        )

        result = CommentClassification.model_validate_json(response.text)
        
        # Validate topics exist
        valid_topics = [t for t in result.topics if t in TOPICS]
        if not valid_topics:
            valid_topics = [OTHER_LABEL]
        
        return valid_topics
    
    except Exception as e:
        if attempt < MAX_RETRIES:
            backoff = INITIAL_BACKOFF ** attempt
            print(f"  Attempt {attempt} failed: {type(e).__name__}. Retrying in {backoff}s...")
            time.sleep(backoff)
            return classify_with_gemini(feedback, attempt + 1)
        else:
            print(f"  Failed after {MAX_RETRIES} attempts. Assigning to Other.")
            return [OTHER_LABEL]


def main() -> None:
    # Prepare accumulator
    bucket: Dict[str, List[Comment]] = {t: [] for t in TOPICS}

    # Process each feedback
    for idx, feedback in enumerate(FEEDBACK_LIST, 1):
        print(f"Processing {idx}/{len(FEEDBACK_LIST)}: {feedback[:50]}...")
        
        topics = classify_with_gemini(feedback)
        print(f"  → Topics: {topics}")
        
        # Add comment to each assigned topic
        comment = Comment(
            text=feedback,
            sentiment=None
        )
        
        for topic in topics:
            bucket[topic].append(comment)

    # Build final JSON output
    final_output = AllTopics(
        topics=[
            TopicCategorization(topic=t, feedback=bucket[t])
            for t in TOPICS
        ]
    )

    output_path = BASE_DIR / "results" / "Gemini-2.5" / "GEMINI_OUTPUT.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output.model_dump_json(indent=2))

    json_to_dataframe(output_path, output_path.parent / "GEMINI_OUTPUT.csv")

    print(f"\nAnalysis complete. Results saved to {output_path}.")


if __name__ == "__main__":
    main()