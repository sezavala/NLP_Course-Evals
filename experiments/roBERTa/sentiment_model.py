from pathlib import Path
from typing import Dict, List
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from data import TOPIC_DEFS, TOPIC_KEYS, SCORING_RUBRIC
from experiments.json_to_sheet import sentiment_json_to_dataframe

BASE_DIR = Path(__file__).resolve().parents[2]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Use a sentiment-specific roBERTa model
SENTIMENT_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_NAME)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL_NAME).to(DEVICE)
sentiment_model.eval()

# Label mapping for this model (negative=0, neutral=1, positive=2)
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_TO_SCORE = {
    "negative": 1,
    "neutral": 3,
    "positive": 5
}


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


def classify_sentiment_with_roberta(
    feedback: str, 
    topic: str
) -> Dict:
    """Use roBERTa to classify feedback sentiment for a specific topic."""
    
    rubric = get_rubric_for_topic(topic)
    
    # Create a contextual prompt for sentiment classification
    context_text = f"Topic: {topic}. Definition: {TOPIC_DEFS.get(topic, 'N/A')}. Feedback: {feedback}"
    
    try:
        # Tokenize and get predictions
        inputs = sentiment_tokenizer(
            context_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predicted_class_idx = torch.argmax(probabilities, dim=-1).item()
        
        # Get sentiment label and confidence
        sentiment_label = LABEL_MAP.get(predicted_class_idx, "neutral")
        confidence = probabilities[0, predicted_class_idx].item()
        
        # Map sentiment to score (1-5)
        score = SENTIMENT_TO_SCORE.get(sentiment_label, 3)
        
        # Adjust score based on confidence for more nuanced scoring
        if confidence > 0.8:
            # High confidence - use extreme scores
            pass
        elif confidence < 0.5:
            # Low confidence - nudge toward neutral
            score = 3
        else:
            # Medium confidence - adjust score slightly
            if sentiment_label == "positive" and score == 5:
                score = 4
            elif sentiment_label == "negative" and score == 1:
                score = 2
        
        # Generate detailed reasoning with rubric reference (like Llama)
        confidence_pct = confidence * 100
        
        # Extract key phrases from feedback for explanation
        positive_phrases = ["appreciate", "excellent", "great", "good", "helpful", "clear", "well", "easy", "effective", "impressed", "satisfied", "thorough", "comprehensive", "organized", "structured", "success", "learned", "seamless"]
        negative_phrases = ["poor", "bad", "difficult", "confusing", "unclear", "hard", "frustrating", "waste", "poorly", "lacking", "missing", "disorganized", "rushed", "inadequate"]
        
        feedback_lower = feedback.lower()
        found_phrases = []
        
        if sentiment_label == "positive":
            for phrase in positive_phrases:
                if phrase in feedback_lower:
                    found_phrases.append(phrase)
        elif sentiment_label == "negative":
            for phrase in negative_phrases:
                if phrase in feedback_lower:
                    found_phrases.append(phrase)
        
        phrases_str = ", ".join(found_phrases[:3]) if found_phrases else "sentiment language"
        
        # Generate reasoning that explains the rubric score like Llama does
        if score == 5:
            reasoning = f"Feedback explicitly praises {topic} with strong positive language ({phrases_str}). This matches the highest rubric level (Score 5/5)."
        elif score == 4:
            reasoning = f"Feedback contains positive language about {topic} ({phrases_str}), indicating good performance. Confidence: {confidence_pct:.1f}%. Score 4/5 reflects moderate-to-strong positive sentiment."
        elif score == 2:
            reasoning = f"Feedback contains negative language about {topic} ({phrases_str}), indicating room for improvement. Confidence: {confidence_pct:.1f}%. Score 2/5 reflects mild-to-moderate negative sentiment."
        elif score == 1:
            reasoning = f"Feedback strongly criticizes {topic} with clear negative language ({phrases_str}). This matches the lowest rubric level (Score 1/5)."
        else:  # score == 3 (neutral)
            reasoning = f"Feedback is neutral regarding {topic}. Confidence: {confidence_pct:.1f}%. Score 3/5 reflects neutral or mixed sentiment."
        
        return {
            "sentiment": sentiment_label,
            "score": score,
            "reasoning": reasoning
        }
    
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            "sentiment": "neutral",
            "score": 2,
            "reasoning": f"Failed to classify: {str(e)}"
        }


def main() -> None:
    import time as time_module
    start_time = time_module.time()
    
    # Controlled sentiment benchmark: use the same best topic output for every sentiment model.
    json_path = BASE_DIR / "results" / "Llama3" / "LLAMA_OUTPUT.json"
    
    if not json_path.exists():
        print(f"Error: Input file not found at {json_path}")
        print("Please run Llama3 classification model first.")
        return
    
    with open(json_path, "r") as f:
        llama_output = json.load(f)

    # Output structure
    sentiment_results = {
        "topics": [],
        "metadata": {
            "model": "roBERTa",
            "start_time": start_time,
            "end_time": None,
            "total_time": None,
            "num_feedbacks": 0
        }
    }

    total_feedbacks = 0

    # Process each topic
    for topic_item in llama_output.get("topics", []):
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
            
            sentiment = classify_sentiment_with_roberta(current_feedback, current_topic)
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
        
        sentiment_results["topics"].append(topic_sentiments)

    end_time = time_module.time()
    total_time = end_time - start_time
    
    sentiment_results["metadata"]["end_time"] = end_time
    sentiment_results["metadata"]["total_time"] = total_time
    sentiment_results["metadata"]["num_feedbacks"] = total_feedbacks
    sentiment_results["metadata"]["avg_time_per_feedback"] = total_time / total_feedbacks if total_feedbacks > 0 else 0

    # Save results
    json_output_path = BASE_DIR / "results" / "roBERTa" / "ROBERTA_SENTIMENT.json"
    json_output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(sentiment_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Sentiment analysis complete. Saved to {json_output_path}.")
    print(f"Total time: {total_time:.2f}s | Avg per feedback: {sentiment_results['metadata']['avg_time_per_feedback']:.2f}s")

    # Convert to CSV
    csv_output_path = BASE_DIR / "results" / "roBERTa" / "ROBERTA_SENTIMENT.csv"
    sentiment_json_to_dataframe(json_output_path, csv_output_path)
    print(f"CSV export complete. Saved to {csv_output_path}.")


if __name__ == "__main__":
    main()
