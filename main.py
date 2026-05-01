from __future__ import annotations

import csv
import json
import re
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
    return clean_topic_summary(topic, summary)


def clean_topic_summary(topic: str, summary: str) -> str:
    """Keep exactly one topic-summary heading and remove model preambles."""
    marker = f"Summary of {topic}:"
    text = summary.replace("\r\n", "\n").replace("\r", "\n").strip()
    marker_pattern = re.compile(rf"Summary\s+of\s+{re.escape(topic)}\s*:", re.IGNORECASE)
    marker_matches = list(marker_pattern.finditer(text))
    if marker_matches:
        text = text[marker_matches[-1].end():].strip()

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    return f"{marker} {text}" if text else marker


def write_combined_csv(output: dict[str, Any], csv_path: Path) -> None:
    rows = []
    summary_by_topic = {
        item["topic"]: item["summary"]
        for item in output.get("topic_summaries", [])
        if "topic" in item and "summary" in item
    }
    for topic_item in output["categories"]:
        topic = topic_item["topic"]
        topic_summary = summary_by_topic.get(topic, topic_item.get("summary", ""))
        comments = topic_item["comments"]
        if not comments:
            rows.append(
                {
                    "Course ID": output["course_id"],
                    "Overall Score": output["overall_score"],
                    "Topic": topic,
                    "Topic Average Score": topic_item["average_score"],
                    "Feedback": "",
                    "Sentiment": "",
                    "Score": "",
                    "Confidence": "",
                    "Reasoning": "",
                    "Topic Summary": topic_summary,
                }
            )
            continue

        for comment_idx, comment in enumerate(comments):
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
                    "Topic Summary": topic_summary if comment_idx == 0 else "",
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


def load_feedback_from_json(json_data: dict[str, Any]) -> tuple[str, list[str]]:
    """Load course_id and feedback from JSON input."""
    course_id = json_data.get("course_id", "UNKNOWN")
    raw_comments = json_data.get("raw_comments", [])
    
    if not isinstance(raw_comments, list):
        raise ValueError("raw_comments must be a list")
    
    print(f"Loaded course_id: {course_id}")
    print(f"Loaded {len(raw_comments)} feedback items")
    
    return course_id, raw_comments


if __name__ == "__main__":
    # Exact JSON input
    json_input = {
        "course_id": "CHEM_14A_Fall2025",
        "raw_comments": [
            "Presentations and lectures were very clear.",
            "I felt like Ramachandran really cared about the students doing well and provided all of us ample opportunity to supplement when we were struggling.",
            "Overall the lectures for this class were really good I understood most of the lectures and was able to follow along easily.",
            "Professor Ramachandran is an engaging lecturer and very obviously cares about students understanding the material.",
            "I know that many of us struggle with such dense course but I believe that Professor Ramachandran has made is extremely doable and under- standable for everyone.",
            "The strengths of this course is that we got through all the material so it was excellent on time management.",
            "I think it was confusing at times when assignments and quizzes would be thrown at us at times different than originally specified at the beginning of the quarter.",
            "I believe Dr.Ramachandran has a very organized course and truly cares about the success of her students.",
            "However, I believe that the course could be improved in terms of the course resources that are posted online on CCLE.",
            "Overall, I love this professor, and I can’t wait to have her again in the Fall for 14C!",
            "Ramachandran really cares about us students, and makes lots of efforts to let us learn real knowledge.",
            "Ramachandran genuinely cares for the well being of her students and is constantly trying to improve her teaching methods.",
            "The instructor really cares about her students and what she teaches, which is apparent through her lectures and the resources she provides for her students.",
            "She made the hardest concepts seem very manageable and did a very good job at organizing her course.",
            "The strengths that this professor has performed was teaching skills, knowledge of the material, communication, and concern of the students.",
            "The instructor takes the time togo over any question that is asked during lecture, does her best to make sure that students are understanding the material in case there is any confusion.",
            "Professor’s teaching is always clear.",
            "I love her lectures, and I can always understand the concepts after her explanation.",
            "This was the first time I felt welcomed and interested in such material.",
            "I really believe that a lot of that has to do with Professor Rama chandra n’s teaching style and I appreciate that very much.",
            "Professor Ramachandran is an engaging lecturer and very obviously cares about students understanding the material.",
            "I think her exams and grading system are fair but it would be very helpful if there could be a timer on the screen during exams instead of just a minute warning.",
            "I got a B+ in 14A which is supposed to be the ’weeder’ class, which means I passed my midterms and my final.",
            "I also do not think partial credit was fair.",
            "The professor very clearly represents a concern for student learning and Is always very welcoming of students to attend office hours or set time aside to meet with her as she did, which was a huge time commitment on her part.",
            "The material on the exams was always fair, my only problem was the time constraint as it resulted in a very stressful environment that made it very likely for students to blank out during the exam, more time should be allotted for exams.",
            "I think weekly graded homeworks should be included for a small portion of the grade.",
            "The strengths of the professor are in her way of teaching.",
            "The only weakness of her class in general is that once one grade is low, it is incredibly difficult to raise your grade.",
            "Rama chandra n made it clear from day one that she is very concerned with helping students succeed as much as possible.",
            "I learned a lot and worked very hard but my grade doesn’t reflect it.",
            "Her grading scale is a bit harsh and she does not give too much partial credit.",
            "Just please— consider this.",
            "The strengths of this course is that we got through all the material so it was excellent on time management.",
            "I think weekly graded homeworks should be included for a small portion of the grade.",
            "I feel like organic chemistry is just being thrown into the class material and we do not have much time to go over it before the final.",
            "I thought she taught really well.",
            "I think the implementation of clicker questions really gave me an idea of what exam questions would be like, so clickers were very helpful and they should be used more often!",
            "I found the formatting of the power points to be inconsistent and lacking in explanation which made it difficult to review them before quizzes and tests without rewatching the entire bruin cast.",
            "Overall, I really enjoyed Ramachandran as a professor and felt confident in a subject I did not think I would be.",
            "Ramachandran actually cares that we do well and that we understand the course inside and out.",
            "I have no complaints, I am extremely happy with this course and the professor.",
            "I think in general Dr.",
            "Ramachandran is a very intelligent woman and is always prepared to teach and help those who do not understand any concepts.",
            "Tends to get ex- tremely anxious during exams (weakness).",
            "Professor Rama chandra nisa professor who deserves her high Bruin Walk ratings, as she has shown more concern about her students than any professor I’ve had at UCLA.",
            "She is very good at communication and makes chemistry very easy to understand.",
            "Strength is the amount of care Dr.",
            "Ramachandran puts into her course and making sure students feel free to ask questions and ask for explana- tion when they don’t understand.",
            "Ramachandran really cares about student learning and clearly makes an effort to provide resources and support to students.",
            "She is very understanding and very approachable.",
            "You can tell that she cares about her students and tries her best for us to be successful.",
            "The professor is great at explaining conceptual topics which is relevant to many students as the MCAT composes mainly of conceptual questions.",
            "I would request is that she give reminders for graded clicker question in an email before lecture like she did the first time.",
            "I thinkthe formatting of the power points to be inconsistent andlacking in explanation which made it difficult to review them before quizzes and tests without rewatching the entire bruin cast.",
            "It would be nice if she could adopt a policy where students had higher chances for redemption.",
            "Overall, I had a fantastic time in class.",
            "She encourages discussion, ask- ing questions and provide incentives such as worksheets for people to go to her office hours.",
            "Professor Ramachandran genuinely care about student learning and im- provement, and she made chemistry more bearable.",
            "She is open and welcoming, and always responds super quickly to discussion posts and emails.",
            "She really was invested in student learning which I appreciated a lot as my last chemistry class was not as student-focused and I felt behind all the time.",
            "I think Professor Rama chandra n really cares about her students learning and it shows.",
            "She doesn’t just go through the lecture and hope you understand.",
            "Ramachandran really cares about student learning and clearly makes an effort to provide resources and support to students.",
            "The professor really cares about her students and wants them to succeed.",
            "She is very kind and understanding and does what she can to ensure her students are engaging in the class and doing well.",
            "I appreciate how hard she works and how much she cares.",
            "I wanted to go to her office hours but I was always intimidated by the number of people there, but I will definitely start trying when I take her class in the fall.",
            "I think everything about Dr.",
            "she makes sure we are always up to date with our grades and what we are learning in class by posting everything on CCLE.",
            "Overall, she provides a lot of practice and opportunities for students to learn and grow.",
            "I think in general Dr.",
            "Ramachandran is a very intelligent woman and is always prepared to teach and help those who do not understand any concepts.",
            "Professor Ramachandran completely changed Chemistry for me and I am so thankful I came across this class.",
            "Professor Rama chandra n expressed genuine efforts in catering the course to her students needs, she set aside time to meet with us and often asked for our feedback throughout the quarter.",
            "Ramachandran really cares about us students, and makes lots of efforts to let us learn real knowledge.",
            "Professor Ramachandran completely changed Chemistry for me and I am so thankful I came across this class.",
            "Ramachandran really cares about student learning and clearly makes an effort to provide resources and support to students.",
            "Strengths of this professor include her real concern for how the class was doing and providing resources to students who needed the help.",
            "I think more example problems and clicker questions would help.",
            "I would say however that the level of questions on the tests seem to exceed what is taught in class and at times I felt unprepared for the midterms because lack of adequate resources and uncertainty about what I would actually be expected to do on an exam.",
            "However, I found the formatting of the power points to be inconsistent and lacking in explanation which made it difficult to review them before quizzes and tests without rewatching the entire bruin cast."
        ]
    }
    
    course_id, raw_comments = load_feedback_from_json(json_input)
    output = analysis_pipeline(course_id, raw_comments)
    print("\n" + "=" * 80)
    output_path = BASE_DIR / "results" / "ML_OUTPUT.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Combined analysis complete. Saved to {output_path}.")
