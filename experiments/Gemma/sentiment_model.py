from pathlib import Path
from typing import List, Dict
import json
import requests
import time

from data import TOPIC_DEFS, TOPIC_KEYS, SCORING_RUBRIC
from experiments.json_to_sheet import sentiment_json_to_dataframe
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:latest"


def get_rubric_for_topic(topic: str) -> str:
    """Extract and format rubric for a given topic."""
    if topic not in SCORING_RUBRIC:
        return ""
    
    rubric = SCORING_RUBRIC[topic]
    
    # Check if topic has "or" options (like "Resources or Materials")
    if isinstance(rubric, dict):
        # Format as score levels
        formatted = f"RUBRIC FOR '{topic}':\n"
        for score_level, description in sorted(rubric.items()):
            formatted += f"\n  Score {score_level}: {description}\n"
        return formatted
    
    return ""


def classify_sentiment_with_gemma(
    feedback: str, 
    topic: str, 
    attempt: int = 1
) -> Dict:
    """Use Gemma to classify feedback sentiment for a specific topic."""
    
    rubric = get_rubric_for_topic(topic)
    
    prompt = f"""You are evaluating course feedback for a specific topic using a provided rubric.

TOPIC: {topic}
TOPIC DEFINITION: {TOPIC_DEFS.get(topic, "N/A")}

{rubric}

FEEDBACK: "{feedback}"

Task: 
1. Determine if this feedback is POSITIVE, NEGATIVE, or NEUTRAL relative to this topic.
2. Assign a score from 1-5 based on the rubric (1=very negative, 3=neutral, 5=very positive).
3. Explain which rubric level this feedback matches.

Return ONLY valid JSON with these exact keys:
{{"sentiment": "positive|negative|neutral", "score": 3, "reasoning": "brief explanation"}}
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
        print(f"  ERROR on attempt {attempt}: {e}")
        if attempt < 3:
            time.sleep(2 ** attempt)
            return classify_sentiment_with_gemma(feedback, topic, attempt + 1)
        return {
            "sentiment": "neutral",
            "score": 3,
            "reasoning": "Failed to classify"
        }
    
    # Parse JSON
    try:
        json_start = output_text.find("{")
        json_end = output_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            print(f"  Warning: No JSON found in response: {output_text[:100]}")
            raise ValueError("No JSON found")
        json_str = output_text[json_start:json_end]
        parsed = json.loads(json_str)
        
        # Validate required keys
        if "sentiment" not in parsed or "score" not in parsed or "reasoning" not in parsed:
            print(f"  Warning: Missing keys in response: {parsed}")
            raise ValueError("Missing required keys")
        
        # Validate sentiment value
        if parsed["sentiment"].lower() not in ["positive", "negative", "neutral"]:
            parsed["sentiment"] = "neutral"
        
        # Validate score is integer 1-5
        try:
            parsed["score"] = int(parsed["score"])
            if parsed["score"] < 1 or parsed["score"] > 5:
                parsed["score"] = 3
        except (ValueError, TypeError):
            parsed["score"] = 3
        
        return parsed
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return {
            "sentiment": "neutral",
            "score": 3,
            "reasoning": "Failed to parse response"
        }


def main() -> None:
    import time as time_module
    start_time = time_module.time()
    
    # Controlled sentiment benchmark: use the same best topic output for every sentiment model.
    json_path = BASE_DIR / "results" / "Llama3" / "LLAMA_OUTPUT.json"
    
    with open(json_path, "r") as f:
        gemma_output = json.load(f)

    # Output structure
    sentiment_results = {
        "topics": [],
        "metadata": {
            "model": "Gemma",
            "start_time": start_time,
            "end_time": None,
            "total_time": None,
            "num_feedbacks": 0
        }
    }

    total_feedbacks = 0

    # Process each topic
    for topic_item in gemma_output.get("topics", []):
        current_topic = topic_item.get("topic", "N/A")
        feedbacks = topic_item.get("feedback", [])
        total_feedbacks += len(feedbacks)
        
        print(f"\n{'='*60}")
        print(f"TOPIC: {current_topic}")
        print(f"{'='*60}")
        
        topic_sentiments = {
            "topic": current_topic,
            "feedback_with_sentiment": []
        }
        
        for idx, feedback_item in enumerate(feedbacks, 1):
            current_feedback = feedback_item.get("text", "N/A")
            item_start = time_module.time()
            print(f"\n[{idx}/{len(feedbacks)}] {current_feedback[:70]}...")
            
            sentiment = classify_sentiment_with_gemma(current_feedback, current_topic)
            item_time = time_module.time() - item_start
            
            # Safe access with defaults
            sentiment_label = sentiment.get("sentiment", "neutral").upper()
            sentiment_score = sentiment.get("score", 3)
            sentiment_reasoning = sentiment.get("reasoning", "N/A")
            
            print(f"  → Sentiment: {sentiment_label} (Score: {sentiment_score}/5)")
            print(f"  → Time: {item_time:.2f}s")
            print(f"  → Reasoning: {sentiment_reasoning}")
            
            topic_sentiments["feedback_with_sentiment"].append({
                "text": current_feedback,
                "sentiment": sentiment.get("sentiment", "neutral"),
                "score": sentiment.get("score", 3),
                "reasoning": sentiment.get("reasoning", ""),
                "processing_time": item_time
            })
            
            time.sleep(0.5)  # Rate limit
        
        sentiment_results["topics"].append(topic_sentiments)

    end_time = time_module.time()
    total_time = end_time - start_time
    
    sentiment_results["metadata"]["end_time"] = end_time
    sentiment_results["metadata"]["total_time"] = total_time
    sentiment_results["metadata"]["num_feedbacks"] = total_feedbacks
    sentiment_results["metadata"]["avg_time_per_feedback"] = total_time / total_feedbacks if total_feedbacks > 0 else 0

    # Save results
    json_output_path = BASE_DIR / "results" / "Gemma" / "GEMMA_SENTIMENT.json"
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sentiment analysis complete. Saved to {json_output_path}.")
    print(f"Total time: {total_time:.2f}s | Avg per feedback: {sentiment_results['metadata']['avg_time_per_feedback']:.2f}s")

    # Convert to CSV
    csv_output_path = BASE_DIR / "results" / "Gemma" / "GEMMA_SENTIMENT.csv"
    sentiment_json_to_dataframe(json_output_path, csv_output_path)
    print(f"CSV export complete. Saved to {csv_output_path}.")



if __name__ == "__main__":
    main()
