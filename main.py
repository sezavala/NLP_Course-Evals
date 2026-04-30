from pathlib import Path
from data import TOPIC_DEFS, TOPIC_KEYS
import requests
import json
from typing import List, Dict
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parents[0]
OTHER = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER]

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def classify_with_llama(comment: str) -> List[str]:
    """Use llama for topic classification"""
    topics_str = "\n".join(f"- {t}: {TOPIC_DEFS[t]}" for t in TOPIC_KEYS)
    
    prompt = f"""Classify this course feedback into ALL applicable topics if explicitly mentioned.

    TOPICS:
    {topics_str}
    - {OTHER}: Generic praise with no substance.

    RULES:
    1. Assign to ALL applicable topics (can be 1 or more).
    2. Generic praise only (e.g., "Best professor ever", "Eric Wu is my goat") → "{OTHER}" ONLY.
    3. Return ONLY valid JSON, no markdown.

    FEEDBACK: "{comment}"

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
        return [OTHER]
    
    try:
        json_start = output_text.find("{")
        json_end = output_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found")
        json_str = output_text[json_start:json_end]
        parsed = json.loads(json_str)
    except Exception as e:
        print(f"  JSON parse error: {e}")
        return [OTHER]
    
    topics = parsed.get("topics", [OTHER])
    if not isinstance(topics, list):
        topics = [topics]

    if OTHER in topics or not topics:
        return [OTHER]
    
    valid_topics = [t for t in topics if t in TOPICS]
    return valid_topics if valid_topics else [OTHER]


def sentiment_with_llama(comment: str, topic: str) -> Dict:
    """Classify sentiment using llama for a specific topic"""
    topic_def = TOPIC_DEFS.get(topic, topic)
    
    prompt = f"""Analyze the sentiment of this course feedback relative to the topic "{topic}".

    TOPIC: {topic}
    TOPIC DEFINITION: {topic_def}
    
    FEEDBACK: "{comment}"

    Respond with ONLY valid JSON (no markdown):
    {{
        "sentiment": "positive|negative|neutral",
        "score": <integer from 0 to 5>,
        "confidence": <float from 0.0 to 1.0>
    }}

    Where:
    - sentiment: positive (score 4-5), neutral (score 2-3), negative (score 0-1)
    - score: 0-5 rating for this topic (0=very negative, 5=very positive)
    - confidence: 0.0-1.0 confidence in the classification
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
        print(f"  SENTIMENT ERROR: {e}")
        return {"sentiment": "neutral", "score": 3, "confidence": 0.0}
    
    try:
        json_start = output_text.find("{")
        json_end = output_text.rfind("}") + 1
        if json_start == -1 or json_end == 0:
            raise ValueError("No JSON found")
        json_str = output_text[json_start:json_end]
        parsed = json.loads(json_str)
        
        score = max(0, min(5, int(parsed.get("score", 3))))
        sentiment = parsed.get("sentiment", "neutral").lower()
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        
        return {
            "sentiment": sentiment,
            "score": score,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"  Sentiment parse error: {e}")
        return {"sentiment": "neutral", "score": 3, "confidence": 0.0}


def extract_strengths_weaknesses(results: List[Dict], top_n: int = 3) -> tuple:
    """Extract key strengths and weaknesses from analyzed comments"""
    strengths = defaultdict(int)
    weaknesses = defaultdict(int)
    
    for result in results:
        topics = result["topics"]
        if OTHER in topics:
            continue
            
        for topic, sentiments in result["topic_sentiments"].items():
            for sent in sentiments:
                if sent["sentiment"] == "positive" and sent["score"] >= 4:
                    strengths[topic] += 1
                elif sent["sentiment"] == "negative" and sent["score"] <= 1:
                    weaknesses[topic] += 1
    
    # Get top strengths and weaknesses
    top_strengths = sorted(strengths.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_weaknesses = sorted(weaknesses.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    return [t[0] for t in top_strengths], [t[0] for t in top_weaknesses]


def calculate_all_topic_scores(results: List[Dict]) -> Dict[str, float]:
    """Calculate average scores for ALL individual topics from data.py"""
    topic_scores = defaultdict(list)
    
    for result in results:
        for topic, sentiments in result["topic_sentiments"].items():
            for sent in sentiments:
                topic_scores[topic].append(sent["score"])
    
    # Include all topics from TOPIC_KEYS, even if not mentioned (score = 3.0)
    all_scores = {}
    for topic in TOPIC_KEYS:
        if topic in topic_scores and topic_scores[topic]:
            all_scores[topic] = round(sum(topic_scores[topic]) / len(topic_scores[topic]), 1)
        else:
            all_scores[topic] = 3.0  # Default neutral score
    
    return all_scores


def analysis_pipeline(course_id: str, raw_comments: List[str]) -> Dict:
    """Process feedback through classification and sentiment analysis"""
    
    results = []
    all_scores = []

    for idx, feedback in enumerate(raw_comments, 1):
        print(f"\n[{idx}/{len(raw_comments)}] Processing feedback...")
        
        # Classify topics
        topics = classify_with_llama(feedback)
        print(f"  Topics: {topics}")
        
        # Get sentiment for each topic
        topic_sentiments = {}
        for topic in topics:
            if topic != OTHER:
                sentiment = sentiment_with_llama(feedback, topic)
                if topic not in topic_sentiments:
                    topic_sentiments[topic] = []
                topic_sentiments[topic].append(sentiment)
                all_scores.append(sentiment["score"])
                print(f"    {topic}: {sentiment['sentiment']} ({sentiment['score']}/5)")
        
        results.append({
            "feedback": feedback,
            "topics": topics,
            "topic_sentiments": topic_sentiments
        })
    
    # Calculate scores
    all_topic_scores = calculate_all_topic_scores(results)
    overall_score = round(sum(all_scores) / len(all_scores), 1) if all_scores else 3.0
    strengths, weaknesses = extract_strengths_weaknesses(results)
    
    # Build output
    output = {
        "course_id": course_id,
        "overall_score": overall_score,
        "category_scores": [
            {"category": topic, "score": score}
            for topic, score in all_topic_scores.items()
        ],
        "key_strengths": strengths,
        "key_weaknesses": weaknesses,
        "analyzed_comments": [
            {
                "text": r["feedback"],
                "topics": r["topics"],
                "rating": round(sum(s["score"] for topic_sents in r["topic_sentiments"].values() for s in topic_sents) / max(1, sum(len(ts) for ts in r["topic_sentiments"].values())), 1)
            }
            for r in results
        ]
    }
    
    return output


if __name__ == "__main__":
    example = {
        "course_id": "CHEM_14A",
        "raw_comments": [
            "Professor Wu is a fantastic educator who clearly cares about student success. Lectures are well-organized and engaging.",
            "The pace was too fast and the workload was overwhelming. Exams were extremely difficult.",
            "Great course organization and clear explanations. Could use better learning materials."
        ]
    }

    output = analysis_pipeline(example["course_id"], example["raw_comments"])
    print("\n" + "="*80)
    print(json.dumps(output, indent=2))