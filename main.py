from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import requests

from data import FEEDBACK_LIST, SCORING_RUBRIC, TOPIC_DEFS, TOPIC_KEYS

BASE_DIR = Path(__file__).resolve().parents[0]
OTHER = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER]

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("No JSON object found")
    return json.loads(text[json_start:json_end])


def call_ollama(prompt: str, temperature: float = 0.1, timeout: int = 90) -> str:
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def format_topics() -> str:
    return "\n".join(f"- {topic}: {TOPIC_DEFS[topic]}" for topic in TOPIC_KEYS)


def format_rubric(topic: str) -> str:
    rubric = SCORING_RUBRIC.get(topic, {})
    if not isinstance(rubric, dict):
        return ""
    return "\n".join(f"{score}: {description}" for score, description in sorted(rubric.items()))


def classify_with_llama(comment: str) -> list[str]:
    """Classify a course-evaluation comment into explicit instructional topics."""
    prompt = f"""You are a careful course-evaluation coder.

Assign ONLY the topics that are explicitly supported by the feedback text.
Do not infer a topic from general praise. Prefer fewer labels when evidence is weak.

ALLOWED TOPICS:
{format_topics()}
- {OTHER}: Generic praise, broad approval, or comments with no specific instructional detail.

BOUNDARY RULES:
- Use "Course organization and structure" only for organization, structure, navigation, sequencing, or course design.
- Use "Pace" only when speed, rushing, slowing down, keeping up, or pacing is explicitly mentioned.
- Use "Workload" only when workload, time burden, difficulty load, or amount of work is explicitly mentioned.
- Use "Student engagement and participation" only for participation, engagement activities, entertainment, discussion, questions, or interactive opportunities.
- Use "Clarity of explanations" only for explaining, lecturing clearly, making concepts understandable, or examples that clarify content.
- Use "Effectiveness of assignments" only for homework, problem sets, assignments, practice tasks, or their learning value.
- Use "Classroom atmosphere" only for the emotional class environment, welcoming climate, motivation, or supportiveness.
- Use "Instructor's communication and availability" only for responsiveness, office hours, availability, accommodations, announcements, or communication.
- Use "Inclusivity and sense of belonging" only for inclusion, belonging, accessibility, different learning styles, feeling welcome, or respect.
- Use "Assessment" only for exams, tests, quizzes, assessment fairness, assessment difficulty, or alignment with material.
- Use "Grading and feedback" only for grading, partial credit, grade policy, or feedback on performance.
- Use "Learning resources and materials" only for notes, slides, recordings, review sessions, study resources, or posted materials.
- If the comment is only generic praise, choose only "{OTHER}".
- If "{OTHER}" is selected, it must be the only topic.

Return ONLY valid JSON with exact topic names:
{{"topics": ["Topic name"]}}

FEEDBACK:
\"\"\"{comment}\"\"\"
"""

    try:
        parsed = extract_json_object(call_ollama(prompt))
    except Exception as exc:
        print(f"  Classification error: {exc}")
        return [OTHER]

    topics = parsed.get("topics", [OTHER])
    if not isinstance(topics, list):
        topics = [topics]

    valid_topics = []
    for topic in topics:
        if topic in TOPICS and topic not in valid_topics:
            valid_topics.append(topic)

    if not valid_topics:
        return [OTHER]
    if OTHER in valid_topics:
        return [OTHER]
    return valid_topics


def sentiment_with_llama(comment: str, topic: str) -> dict[str, Any]:
    """Score a comment for one topic using the topic-specific rubric."""
    prompt = f"""You are scoring one course-evaluation comment for one topic.

Use the rubric exactly. The numeric score is rubric-specific, not generic sentiment.
For Pace and Workload, score 5 means the condition supports learning well; score 1 means the condition makes learning difficult.

TOPIC: {topic}
TOPIC DEFINITION: {TOPIC_DEFS.get(topic, topic)}

RUBRIC:
{format_rubric(topic)}

FEEDBACK:
\"\"\"{comment}\"\"\"

TASK:
1. Decide whether the feedback is positive, neutral, or negative relative to this topic.
2. Assign the best matching integer rubric score from 1 to 5.
3. Give one brief reason grounded in the comment text.
4. Provide confidence from 0.0 to 1.0.

Return ONLY valid JSON:
{{
  "sentiment": "positive|negative|neutral",
  "score": 1,
  "confidence": 0.0,
  "reasoning": "brief explanation"
}}
"""

    try:
        parsed = extract_json_object(call_ollama(prompt))
        score = max(1, min(5, int(parsed.get("score", 3))))
        sentiment = str(parsed.get("sentiment", "neutral")).strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = "neutral"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        reasoning = str(parsed.get("reasoning", "")).strip()
    except Exception as exc:
        print(f"  Sentiment error for {topic}: {exc}")
        return {
            "sentiment": "neutral",
            "score": 3,
            "confidence": 0.0,
            "reasoning": "Failed to score with model.",
        }

    return {
        "sentiment": sentiment,
        "score": score,
        "confidence": confidence,
        "reasoning": reasoning,
    }


def summarize_topic_with_llama(topic: str, comments: list[dict[str, Any]], average_score: float | None) -> str:
    if not comments:
        return f"Summary of {topic}: No comments were assigned to this topic."

    scored_comments = [
        {
            "score": item.get("score"),
            "sentiment": item.get("sentiment"),
            "text": item.get("feedback", ""),
        }
        for item in comments
    ]
    prompt = f"""Summarize the course evaluation evidence for one topic.

TOPIC: {topic}
AVERAGE SCORE: {average_score if average_score is not None else "N/A"}
RUBRIC:
{format_rubric(topic) if topic != OTHER else "No rubric score for generic comments."}

COMMENTS WITH SCORES:
{json.dumps(scored_comments, indent=2)}

Write 1-2 concise sentences beginning exactly with:
Summary of {topic}:

Mention the main pattern and any recurring concern. Do not invent evidence.
"""

    try:
        summary = call_ollama(prompt, temperature=0.2, timeout=120).strip()
    except Exception as exc:
        print(f"  Summary error for {topic}: {exc}")
        return f"Summary of {topic}: Summary unavailable due to model error."

    if not summary.startswith(f"Summary of {topic}:"):
        summary = f"Summary of {topic}: {summary}"
    return summary


def write_combined_csv(output: dict[str, Any], csv_path: Path) -> None:
    rows = []
    for topic_item in output["categories"]:
        topic = topic_item["topic"]
        for comment in topic_item["comments"]:
            rows.append(
                {
                    "Course ID": output["course_id"],
                    "Overall Score": output["overall_score"],
                    "Topic": topic,
                    "Topic Average Score": topic_item["average_score"],
                    "Feedback": comment["feedback"],
                    "Sentiment": comment.get("sentiment", ""),
                    "Score": comment.get("score", ""),
                    "Confidence": comment.get("confidence", ""),
                    "Reasoning": comment.get("reasoning", ""),
                    "Topic Summary": topic_item["summary"],
                }
            )

    fieldnames = [
        "Course ID",
        "Overall Score",
        "Topic",
        "Topic Average Score",
        "Feedback",
        "Sentiment",
        "Score",
        "Confidence",
        "Reasoning",
        "Topic Summary",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analysis_pipeline(
    course_id: str,
    raw_comments: list[str],
    output_dir: Path | None = None,
    write_files: bool = True,
) -> dict[str, Any]:
    """Produce a combined classification, sentiment, score, and summary report."""
    output_dir = output_dir or BASE_DIR / "results" / "combined"
    start_time = time.time()
    topic_comments: dict[str, list[dict[str, Any]]] = {topic: [] for topic in TOPICS}
    all_scores = []

    for idx, feedback in enumerate(raw_comments, 1):
        print(f"\n[{idx}/{len(raw_comments)}] Processing feedback...")
        topics = classify_with_llama(feedback)
        print(f"  Topics: {topics}")

        for topic in topics:
            if topic == OTHER:
                topic_comments[OTHER].append(
                    {
                        "feedback": feedback,
                        "sentiment": "",
                        "score": "",
                        "confidence": "",
                        "reasoning": "Generic or non-actionable feedback; no rubric score assigned.",
                    }
                )
                continue

            scored = sentiment_with_llama(feedback, topic)
            all_scores.append(scored["score"])
            topic_comments[topic].append({"feedback": feedback, **scored})
            print(f"    {topic}: {scored['sentiment']} ({scored['score']}/5)")

    categories = []
    category_scores = []
    topic_summaries = []
    for topic in TOPIC_KEYS:
        comments = topic_comments[topic]
        scores = [item["score"] for item in comments if isinstance(item.get("score"), int)]
        average_score = round(sum(scores) / len(scores), 2) if scores else None
        summary = summarize_topic_with_llama(topic, comments, average_score)
        categories.append(
            {
                "topic": topic,
                "average_score": average_score,
                "comment_count": len(comments),
                "summary": summary,
                "comments": comments,
            }
        )
        category_scores.append(
            {
                "category": topic,
                "average_score": average_score,
                "comment_count": len(comments),
            }
        )
        topic_summaries.append({"topic": topic, "summary": summary})

    other_summary = summarize_topic_with_llama(OTHER, topic_comments[OTHER], None)
    categories.append(
        {
            "topic": OTHER,
            "average_score": None,
            "comment_count": len(topic_comments[OTHER]),
            "summary": other_summary,
            "comments": topic_comments[OTHER],
        }
    )
    topic_summaries.append({"topic": OTHER, "summary": other_summary})

    overall_score = round(sum(all_scores) / len(all_scores), 2) if all_scores else None
    output = {
        "course_id": course_id,
        "model": MODEL,
        "overall_score": overall_score,
        "category_scores": category_scores,
        "topic_summaries": topic_summaries,
        "categories": categories,
        "metadata": {
            "num_comments": len(raw_comments),
            "num_scored_topic_comments": len(all_scores),
            "total_time_seconds": round(time.time() - start_time, 2),
        },
    }

    if write_files:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{course_id}_COMBINED_REPORT.json"
        csv_path = output_dir / f"{course_id}_COMBINED_REPORT.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        write_combined_csv(output, csv_path)
        print(f"\nSaved combined JSON to {json_path}")
        print(f"Saved combined CSV to {csv_path}")

    return output


if __name__ == "__main__":
    output = analysis_pipeline("CHEM_14A", FEEDBACK_LIST)
    print("\n" + "=" * 80)
    print(json.dumps(output, indent=2))
