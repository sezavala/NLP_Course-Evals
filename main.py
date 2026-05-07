from __future__ import annotations

import csv
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

from data import SCORING_RUBRIC, TOPIC_DEFS, TOPIC_KEYS

BASE_DIR = Path(__file__).resolve().parents[0]
OTHER = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER]

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

CLASSIFICATION_BASELINE_PATH = BASE_DIR / "HUMAN_CATEGORIZED_OUTPUT.csv"
SENTIMENT_BASELINE_PATH = BASE_DIR / "HUMAN_SENTIMENT_BASELINE.csv"
RAG_CLASSIFICATION_EXAMPLE_COUNT = 4
RAG_SENTIMENT_EXAMPLE_COUNT = 3
RAG_MIN_SIMILARITY = 0.06
OLLAMA_MAX_RETRIES = 2
MIN_RELIABLE_CATEGORY_COMMENTS = 5
LOW_CONFIDENCE_THRESHOLD = 0.55

# Topic-specific confidence thresholds for mismatch filtering
CONFIDENCE_THRESHOLDS = {
    "Assessment": 0.6,
    "Workload": 0.65,
    "Pace": 0.55,
    "Clarity of explanations": 0.5,
    "Classroom atmosphere": 0.5,
    "Course organization and structure": 0.55,
    "default": 0.5,
}

RETRIEVAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "class",
    "course",
    "did",
    "do",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "instructor",
    "is",
    "it",
    "me",
    "my",
    "of",
    "on",
    "or",
    "professor",
    "she",
    "students",
    "that",
    "the",
    "this",
    "to",
    "very",
    "was",
    "were",
    "with",
}

TOPIC_EVIDENCE_PATTERNS = {
    "Course organization and structure": [
        r"\borganiz(?:e|ed|ation|ing)\b",
        r"\bstructure(?:d)?\b",
        r"\bschedul(?:e|ed|ing)\b",
        r"\bsequenc(?:e|ed|ing)\b",
        r"\bnavigation\b",
        r"\boriginally specified\b",
    ],
    "Pace": [
        r"\bpace(?:d)?\b",
        r"\bfast\b",
        r"\bslow\b",
        r"\brushed?\b",
        r"\btoo quickly\b",
        r"\bkeep up\b",
        r"\btime constraint\b",
        r"\bnot enough time\b",
        r"\bmore time\b",
        r"\bdo not have much time\b",
        r"\bbefore the final\b",
    ],
    "Workload": [
        r"\bworkload\b",
        r"\bamount of work\b",
        r"\btoo much work\b",
        r"\bmanageable workload\b",
        r"\btime burden\b",
        r"\bconsume(?:d)? too much\b",
        r"\boverwhel(?:m|med|ming)\b",
        r"\bpacked.*full\b",
        r"\btoo many.*assignments\b",
        r"\bcan't keep up\b",
        r"\bburden\b",
        r"\btoo much to handle\b",
    ],
    "Student engagement and participation": [
        r"\bengag(?:e|ed|ing|ement)\b",
        r"\bengaging lecturer\b",
        r"\bengaging lectures?\b",
        r"\bparticipat(?:e|ed|ion)\b",
        r"\bdiscussion\b",
        r"\bencourag(?:e|ed|es|ing) discussion\b",
        r"\bask(?:ing)? questions\b",
        r"\bgo over any question\b",
        r"\bquestions? .* lecture\b",
        r"\bfeel free to ask\b",
        r"\binteractive\b",
        r"\bclicker questions?\b",
        r"\bclickers?\b",
        r"\bworksheets?\b",
        r"\boffice hours\b",
    ],
    "Clarity of explanations": [
        r"\bclear(?:ly)?\b",
        r"\bexplain(?:s|ed|ing|ation|ations)?\b",
        r"\bunderstand(?:able|ing)?\b",
        r"\bunderstood\b",
        r"\bfollow along\b",
        r"\bmanageable\b",
        r"\bdigestible\b",
        r"\bstraightforward\b",
        r"\bbreak(?:ing)? down\b",
        r"\beasy to understand\b",
        r"\bmade .* understandable\b",
        r"\bmade .* doable\b",
        r"\btaught really well\b",
    ],
    "Effectiveness of assignments": [
        r"\bassignments?\b",
        r"\bhomeworks?\b",
        r"\bproblem sets?\b",
        r"\bpractice problems?\b",
        r"\bexample problems?\b",
        r"\bworksheets?\b",
        r"\bclicker questions?\b",
        r"\bclickers? were helpful\b",
        r"\bgave me an idea of what exam questions\b",
    ],
    "Classroom atmosphere": [
        r"\batmosphere\b",
        r"\benvironment\b",
        r"\bwelcom(?:e|ing)\b",
        r"\bsupportive\b",
        r"\bcomfortable\b",
        r"\bdemotivating\b",
        r"\bstressful environment\b",
        r"\benergy\b",
        r"\bvibe\b",
        r"\bintimidating\b",
        r"\brelaxed\b",
        r"\btone of the class\b",
    ],
    "Instructor's communication and availability": [
        r"\bcommunicat(?:e|ed|ion|ive)\b",
        r"\brespond(?:s|ed|ing)?\b",
        r"\bemails?\b",
        r"\bdiscussion posts?\b",
        r"\boffice hours\b",
        r"\bavailable\b",
        r"\bapproachable\b",
        r"\bset aside time\b",
        r"\bmeet with\b",
        r"\btakes? the time\b",
        r"\bgo over any question\b",
        r"\bup to date\b",
        r"\breminders?\b",
        r"\baccommodations?\b",
    ],
    "Inclusivity and sense of belonging": [
        r"\binclus(?:ive|ion|ivity)\b",
        r"\bbelonging\b",
        r"\bwelcom(?:e|ed|ing)\b",
        r"\baccessible\b",
        r"\blearning styles?\b",
        r"\brespect(?:ful|ed)?\b",
        r"\bcatering\b",
    ],
    "Assessment": [
        r"\bassessments?\b",
        r"\bexams?\b",
        r"\btests?\b",
        r"\bquizzes?\b",
        r"\bmidterms?\b",
        r"\bfinal\b",
        r"\bexam questions?\b",
    ],
    "Grading and feedback": [
        r"\bgrad(?:e|ed|es|ing)\b",
        r"\bgrading system\b",
        r"\bpartial credit\b",
        r"\bfeedback\b",
        r"\bredemption\b",
        r"\bgrade policy\b",
    ],
    "Learning resources and materials": [
        r"\bresources?\b",
        r"\bmaterials?\b",
        r"\bnotes?\b",
        r"\bslides?\b",
        r"\bpower\s*points?\b",
        r"\bbruin\s*cast\b",
        r"\brecordings?\b",
        r"\breview sessions?\b",
        r"\bposted online\b",
        r"\bccle\b",
        r"\blecture notes?\b",
        r"\bstudy materials?\b",
        r"\bflashcards?\b",
        r"\bpractice exams?\b",
    ],
}


def normalize_comment(comment: str) -> str:
    return re.sub(r"\s+", " ", comment).strip()


def canonical_comment_key(comment: str) -> str:
    text = unicodedata.normalize("NFKD", comment).encode("ascii", "ignore").decode("ascii")
    text = text.casefold()
    text = re.sub(r"\bpower\s+points?\b", "powerpoint", text)
    text = re.sub(r"\bbruin\s+cast\b", "bruincast", text)
    return " ".join(re.findall(r"[a-z0-9]+", text))


def dedupe_comments(raw_comments: list[str]) -> tuple[list[str], int]:
    """Remove only exact duplicate comments after whitespace normalization."""
    seen_keys = set()
    unique_comments = []
    duplicate_count = 0
    for comment in raw_comments:
        normalized = normalize_comment(comment)
        if not normalized:
            continue
        if normalized in seen_keys:
            print(normalized)
            duplicate_count += 1
            continue
        seen_keys.add(normalized)
        unique_comments.append(normalized)
    return unique_comments, duplicate_count


def retrieval_tokens(text: str) -> set[str]:
    """Tokenize comments for lightweight example retrieval."""
    return {
        token
        for token in canonical_comment_key(text).split()
        if len(token) > 2 and token not in RETRIEVAL_STOPWORDS
    }


def retrieval_similarity(left: str, right: str) -> float:
    """Score comment similarity using token overlap plus fuzzy full-text matching."""
    left_key = canonical_comment_key(left)
    right_key = canonical_comment_key(right)
    if not left_key or not right_key:
        return 0.0
    if left_key == right_key:
        return 1.0

    left_tokens = retrieval_tokens(left)
    right_tokens = retrieval_tokens(right)
    if left_tokens and right_tokens:
        overlap = left_tokens & right_tokens
        containment = len(overlap) / min(len(left_tokens), len(right_tokens))
        jaccard = len(overlap) / len(left_tokens | right_tokens)
    else:
        containment = 0.0
        jaccard = 0.0

    fuzzy_ratio = SequenceMatcher(None, left_key, right_key).ratio()
    return (0.55 * containment) + (0.25 * jaccard) + (0.20 * fuzzy_ratio)


def is_truthy_label(value: Any) -> bool:
    text = str(value or "").strip().casefold()
    return bool(text) and text not in {"0", "false", "n", "nan", "no", "none"}


def truncate_example_text(text: str, max_chars: int = 420) -> str:
    text = normalize_comment(text)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def load_classification_examples(
    path: Path = CLASSIFICATION_BASELINE_PATH,
) -> list[dict[str, Any]]:
    """Load human topic labels used as retrieval examples for classification."""
    if not path.exists():
        print(f"RAG classification baseline not found: {path}")
        return []

    examples = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feedback = normalize_comment(row.get("Feedback", ""))
            if not feedback:
                continue

            topics = [topic for topic in TOPICS if is_truthy_label(row.get(topic))]
            if not topics:
                topics = [OTHER]
            if OTHER in topics and len(topics) > 1:
                topics = [topic for topic in topics if topic != OTHER] or [OTHER]

            examples.append({"feedback": feedback, "topics": topics})

    return examples


def load_sentiment_examples(
    path: Path = SENTIMENT_BASELINE_PATH,
) -> list[dict[str, Any]]:
    """Load human topic-specific sentiment scores used as retrieval examples."""
    if not path.exists():
        print(f"RAG sentiment baseline not found: {path}")
        return []

    examples = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feedback = normalize_comment(row.get("Feedback", ""))
            topic = str(row.get("Topic", "")).strip()
            sentiment = str(row.get("Sentiment", "neutral")).strip().lower()
            reasoning = normalize_comment(row.get("Reasoning", ""))
            if not feedback or topic not in TOPIC_KEYS:
                continue
            if sentiment not in {"positive", "negative", "neutral"}:
                sentiment = "neutral"
            try:
                score = max(1, min(5, int(row.get("Score", 3))))
            except (TypeError, ValueError):
                score = 3

            examples.append(
                {
                    "feedback": feedback,
                    "topic": topic,
                    "sentiment": sentiment,
                    "score": score,
                    "reasoning": reasoning,
                }
            )

    return examples


def retrieve_similar_examples(
    comment: str,
    examples: list[dict[str, Any]],
    limit: int,
    topic: str | None = None,
    exclude_exact_match: bool = True,
) -> list[dict[str, Any]]:
    """Return the most similar human-labeled examples for a target comment."""
    if not examples or limit <= 0:
        return []

    comment_key = canonical_comment_key(comment)
    scored_examples = []
    for example in examples:
        if topic is not None and example.get("topic") != topic:
            continue

        example_feedback = str(example.get("feedback", ""))
        example_key = canonical_comment_key(example_feedback)
        if exclude_exact_match and comment_key == example_key:
            continue

        similarity = retrieval_similarity(comment, example_feedback)
        if similarity < RAG_MIN_SIMILARITY:
            continue

        scored_examples.append((similarity, example))

    scored_examples.sort(key=lambda item: item[0], reverse=True)
    retrieved = []
    for similarity, example in scored_examples[:limit]:
        retrieved_example = dict(example)
        retrieved_example["similarity"] = round(similarity, 3)
        retrieved.append(retrieved_example)
    return retrieved


def format_classification_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "[]"

    compact_examples = [
        {
            "similarity": example.get("similarity", 0.0),
            "feedback": truncate_example_text(str(example.get("feedback", ""))),
            "human_topics": example.get("topics", []),
        }
        for example in examples
    ]
    return json.dumps(compact_examples, indent=2)


def format_sentiment_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "[]"

    compact_examples = [
        {
            "similarity": example.get("similarity", 0.0),
            "feedback": truncate_example_text(str(example.get("feedback", ""))),
            "human_sentiment": example.get("sentiment", "neutral"),
            "human_score": example.get("score", 3),
            "human_reasoning": truncate_example_text(str(example.get("reasoning", "")), 180),
        }
        for example in examples
    ]
    return json.dumps(compact_examples, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("No JSON object found")
    return json.loads(text[json_start:json_end])


def call_ollama(
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 90,
    max_retries: int = OLLAMA_MAX_RETRIES,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
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
        except requests.RequestException as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Ollama call failed: {last_error}")


def format_topics() -> str:
    return "\n".join(f"- {topic}: {TOPIC_DEFS[topic]}" for topic in TOPIC_KEYS)


def format_rubric(topic: str) -> str:
    rubric = SCORING_RUBRIC.get(topic, {})
    if not isinstance(rubric, dict):
        return ""
    return "\n".join(f"{score}: {description}" for score, description in sorted(rubric.items()))


def has_topic_evidence(comment: str, topic: str) -> bool:
    if topic == OTHER:
        return True
    patterns = TOPIC_EVIDENCE_PATTERNS.get(topic, [])
    text = comment.casefold()
    return any(re.search(pattern, text) for pattern in patterns)


def looks_like_generic_only_comment(comment: str) -> bool:
    """Catch very short generic praise before it gets forced into a real topic."""
    if any(has_topic_evidence(comment, topic) for topic in TOPIC_KEYS):
        return False

    tokens = retrieval_tokens(comment)
    if len(tokens) > 15:
        return False

    text = comment.casefold()
    concrete_keywords = [
        "assign",
        "exam",
        "lecture",
        "class",
        "material",
        "discussion",
        "grade",
        "feedback",
        "office",
        "resource",
        "classroom",
        "pace",
        "workload",
        "assessment",
        "quiz",
        "test",
    ]
    has_concrete = any(kw in text for kw in concrete_keywords)
    if not has_concrete and len(tokens) < 8:
        return True

    generic_patterns = [
        r"\b(best|great|excellent|amazing|incredible|fantastic|good|wonderful|outstanding|awesome)\b.*\b(professor|instructor|teacher|lecturer)\b",
        r"\b(professor|instructor|teacher|lecturer)\b.*\b(best|great|excellent|amazing|incredible|fantastic|good|wonderful|outstanding|awesome)\b",
        r"\b(no complaints|love this class|goat)\b",
    ]
    return any(re.search(pattern, text) for pattern in generic_patterns)


def filter_topics_by_evidence(
    comment: str,
    topics: list[str],
    mode: str = "soft",
) -> list[str]:
    valid_topics = []
    for topic in topics:
        if topic in TOPICS and topic not in valid_topics:
            valid_topics.append(topic)

    if not valid_topics:
        return [OTHER]
    if valid_topics == [OTHER]:
        return [OTHER]

    non_other_topics = [topic for topic in valid_topics if topic != OTHER]
    if not non_other_topics:
        return [OTHER]

    if mode == "strict":
        filtered_topics = [
            topic for topic in non_other_topics if has_topic_evidence(comment, topic)
        ]
        return filtered_topics or [OTHER]

    if looks_like_generic_only_comment(comment):
        return [OTHER]

    return non_other_topics


def sentiment_from_score(score: int) -> str:
    if score <= 2:
        return "negative"
    if score >= 4:
        return "positive"
    return "neutral"


def classify_with_llama(
    comment: str,
    classification_examples: list[dict[str, Any]] | None = None,
    evidence_filter_mode: str = "soft",
) -> dict[str, Any]:
    """Classify a course-evaluation comment into plausible instructional topics."""
    retrieved_examples = retrieve_similar_examples(
        comment,
        classification_examples or [],
        limit=RAG_CLASSIFICATION_EXAMPLE_COUNT,
    )
    prompt = f"""You are a high-recall course-evaluation topic coder.

Assign ALL topics that are plausibly supported by words, close paraphrases, or concrete teaching/course details in the feedback text.
A comment may discuss multiple topics—include each topic with plausible evidence, even if the evidence is brief.
Prefer recall over precision at this stage. A later validation step will remove weak or incorrect topic assignments.
Do not infer a topic from general praise, student success, caring, or broad support alone.
If broad praise also contains concrete topic clues, assign those topics.
Do not overuse "{OTHER}". Use it only when the feedback is generic praise or has no instructional detail.

    ALLOWED TOPICS:
    {format_topics()}
    - {OTHER}: Generic praise, broad approval, or comments with no specific instructional detail.

    RETRIEVED HUMAN-CODED REFERENCE EXAMPLES:
    {format_classification_examples(retrieved_examples)}

    HOW TO USE THE REFERENCE EXAMPLES:
    - Use them as calibration for boundary cases and similar wording.
    - Do not copy labels unless the target feedback contains similar evidence.
    - Similar examples can help you recognize concrete topic clues even when wording differs.

    BOUNDARY RULES:
    - Use "Course organization and structure" for organization, structure, navigation, sequencing, course design, scheduling, or unclear logistics.
    - Use "Pace" for speed, rushing, slowing down, keeping up, pacing, or insufficient time for material/exams.
    - Use "Workload" for workload, time burden, difficulty load, amount of work, or feeling overwhelmed by assignments/quizzes.
    - Use "Student engagement and participation" only for participation, engagement activities, entertainment, discussion, questions, or interactive opportunities.
    - Use "Clarity of explanations" only for explaining, lecturing clearly, making concepts understandable, or examples that clarify content.
    - Use "Effectiveness of assignments" only for homework, problem sets, assignments, practice tasks, or their learning value.
    - Use "Classroom atmosphere" only for the emotional class environment, welcoming climate, motivation, or supportiveness.
    - Use "Instructor's communication and availability" only for responsiveness, office hours, availability, accommodations, announcements, or communication.
    - Use "Inclusivity and sense of belonging" only for inclusion, belonging, accessibility, different learning styles, feeling welcome, or respect.
    - Use "Assessment" only for exams, tests, quizzes, assessment fairness, assessment difficulty, or alignment with material.
    - Use "Grading and feedback" only for grading, partial credit, grade policy, or feedback on performance.
    - Use "Learning resources and materials" only for notes, slides, recordings, review sessions, study resources, or posted materials.
    - "Engaging lecturer" is Student engagement and participation.
    - "Understood", "easy to understand", or "follow along" is Clarity of explanations.
    - Helpful clicker questions can be both Student engagement and participation AND Effectiveness of assignments.
    - Going over questions in lecture can be both Student engagement and participation AND Clarity of explanations.
    - Brief but concrete wording like "clear teaching", "fair exams", "organized", "helpful resources", or "fast lectures" is enough.
    - If the comment mentions multiple distinct instructional topics, assign all of them.
    - If the comment is only generic praise, choose only "{OTHER}".
    - If "{OTHER}" is selected, it must be the only topic.

    Return ONLY valid JSON with exact topic names:
    {{"topics": ["Topic 1", "Topic 2"], "evidence": {{"Topic 1": "phrase from feedback", "Topic 2": "phrase from feedback"}}}}

    FEEDBACK:
    \"\"\"{comment}\"\"\"
    """

    try:
        parsed = extract_json_object(call_ollama(prompt))
    except Exception as exc:
        print(f"  Classification error: {exc}")
        return {
            "topics": [OTHER],
            "classification_status": "model_error",
            "classification_reasoning": "Failed to classify with model; excluded from rubric scoring.",
        }

    topics = parsed.get("topics", [OTHER])
    if not isinstance(topics, list):
        topics = [topics]

    valid_topics = []
    for topic in topics:
        if isinstance(topic, dict):
            topic = topic.get("topic") or topic.get("name")
        topic = str(topic).strip()
        if topic in TOPICS and topic not in valid_topics:
            valid_topics.append(topic)

    if not valid_topics:
        return {"topics": [OTHER], "classification_status": "classified"}
    if valid_topics == [OTHER]:
        return {"topics": [OTHER], "classification_status": "classified"}

    filtered = filter_topics_by_evidence(comment, valid_topics, mode=evidence_filter_mode)
    if valid_topics != filtered:
        print(f"    LLM assigned {valid_topics}; evidence filter kept {filtered}.")

    return {"topics": filtered, "classification_status": "classified"}


def sentiment_with_llama(
    comment: str,
    topic: str,
    sentiment_examples: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score a comment for one topic using the topic-specific rubric."""
    retrieved_examples = retrieve_similar_examples(
        comment,
        sentiment_examples or [],
        limit=RAG_SENTIMENT_EXAMPLE_COUNT,
        topic=topic,
    )
    prompt = f"""You are scoring one course-evaluation comment for one topic.

Use the rubric exactly. The numeric score is rubric-specific, not generic sentiment.
Score only the evidence that is relevant to the given TOPIC. Ignore praise or criticism about other topics.
For Pace and Workload, score 5 means the condition supports learning well; score 1 means the condition makes learning difficult.

    TOPIC: {topic}
    TOPIC DEFINITION: {TOPIC_DEFS.get(topic, topic)}

    RUBRIC:
    {format_rubric(topic)}

    RETRIEVED HUMAN-SCORED EXAMPLES FOR THIS SAME TOPIC:
    {format_sentiment_examples(retrieved_examples)}

    HOW TO USE THE REFERENCE EXAMPLES:
    - Use them as calibration for the rubric scale.
    - Do not copy a score unless the target feedback gives similar topic-specific evidence.
    - If the target feedback is brief or indirect, avoid extreme scores unless the wording is clearly extreme.

    FEEDBACK:
    \"\"\"{comment}\"\"\"

    TASK:
    1. Decide whether the feedback contains concrete evidence for this exact topic.
    2. If the topic is not supported, set topic_supported to false, score to null, sentiment to null, and explain briefly.
    3. If the topic is supported, decide whether the feedback is positive, neutral, or negative relative to this topic.
    4. Assign the best matching integer rubric score from 1 to 5.
    5. Give one brief reason grounded in exact text from the comment.
    6. Provide confidence from 0.0 to 1.0.

    Return ONLY valid JSON:
    {{
    "topic_supported": true,
    "sentiment": "positive|negative|neutral",
    "score": 1,
    "confidence": 0.0,
    "reasoning": "brief explanation"
    }}
    If topic_supported is false, use JSON null for sentiment and score.
    """

    try:
        parsed = extract_json_object(call_ollama(prompt))
        raw_supported = parsed.get("topic_supported", True)
        if isinstance(raw_supported, bool):
            topic_supported = raw_supported
        else:
            topic_supported = str(raw_supported).strip().lower() not in {"false", "0", "no"}
        raw_score = parsed.get("score")
        score = None if raw_score is None else max(1, min(5, int(raw_score)))
        sentiment = str(parsed.get("sentiment", "")).strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = None
        if topic_supported and isinstance(score, int):
            sentiment = sentiment_from_score(score)
        else:
            topic_supported = False
            score = None
            sentiment = None
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
        reasoning = str(parsed.get("reasoning", "")).strip()
        is_mismatched = not topic_supported or check_topic_mismatch(reasoning, topic)
        
    except Exception as exc:
        print(f"  Sentiment error for {topic}: {exc}")
        return {
            "topic_supported": None,
            "sentiment": None,
            "score": None,
            "confidence": 0.0,
            "reasoning": "Failed to score with model; excluded from averages.",
            "scoring_status": "model_error",
            "is_mismatched": False,
        }

    result = {
        "topic_supported": topic_supported,
        "sentiment": sentiment,
        "score": score,
        "confidence": confidence,
        "reasoning": reasoning,
        "scoring_status": "scored",
        "is_mismatched": is_mismatched,
    }
    
    # Debug: flag potential mismatches
    if is_mismatched and confidence > 0.5:
        print(f"    [MISMATCH] Topic {topic} confidence {confidence}: {reasoning[:60]}...")
    
    return result


def check_topic_mismatch(reasoning: str, topic: str) -> bool:
    """Check if sentiment reasoning indicates the comment doesn't actually relate to the assigned topic."""
    text = reasoning.lower()
    
    # Critical patterns: "but no explicit" or "implies but doesn't"
    critical_patterns = [
        r"but\s+there\s+is\s+no\s+(?:explicit\s+)?",
        r"but\s+no\s+(?:explicit\s+)?",
        r"(?:only\s+)?mentions?\s+.*(?:not|but\s+not)\s+",
        r"(?:doesn't|does\s+not)\s+(?:explicitly\s+)?mention",
        r"(?:doesn't|does\s+not)\s+(?:explicitly\s+)?discuss",
        r"(?:doesn't|does\s+not)\s+(?:explicitly\s+)?address",
        r"implies\s+.*but\s+(?:doesn't|does\s+not)",
        r"mentions\s+.*but\s+(?:doesn't|does\s+not|isn't)",
    ]
    
    # Standard patterns: Handle "does not" form
    standard_patterns = [
        r"does not\s+(?:explicitly\s+)?praise",
        r"does not\s+(?:explicitly\s+)?criticize",
        r"does not\s+(?:explicitly\s+)?relate",
        r"no\s+(?:explicit\s+)?evidence",
        r"not\s+(?:explicitly\s+)?specific",
        r"unrelated",
        r"cannot determine",
        r"not\s+(?:directly\s+)?relevant",
        r"tangential\s+to",
        r"no\s+evidence\s+(?:about|of)",
        r"comment\s+(?:doesn't|does not|couldn't|could not)\s+address",
    ]
    
    all_patterns = critical_patterns + standard_patterns
    
    for pattern in all_patterns:
        if re.search(pattern, text):
            return True
    
    return False


def summarize_topic_with_llama(
    topic: str,
    comments: list[dict[str, Any]],
    average_score: float | None,
) -> str:
    if not comments:
        return f"Summary of {topic}: No comments were assigned to this topic."
    if len(comments) == 1:
        return summarize_single_comment(topic, comments[0])

    scored_count = sum(1 for item in comments if isinstance(item.get("score"), int))
    model_error_count = sum(1 for item in comments if item.get("scoring_status") == "model_error")
    sentiment_counts = {
        sentiment: sum(1 for item in comments if item.get("sentiment") == sentiment)
        for sentiment in ("positive", "neutral", "negative")
    }
    scored_comments = [
        {
            "score": item.get("score"),
            "sentiment": item.get("sentiment"),
            "topic_supported": item.get("topic_supported"),
            "scoring_status": item.get("scoring_status", "unscored"),
            "text": item.get("feedback", ""),
        }
        for item in comments
    ]
    exact_prefix = build_topic_summary_prefix(
        topic,
        len(comments),
        scored_count,
        average_score,
        sentiment_counts,
        model_error_count,
    )
    prompt = f"""Summarize the course evaluation evidence for one topic.

    TOPIC: {topic}
    COMMENT COUNT: {len(comments)}
    SCORED COMMENT COUNT: {scored_count}
    AVERAGE SCORE: {average_score if average_score is not None else "N/A"}
    MODEL SCORING ERRORS: {model_error_count}
    SENTIMENT COUNTS:
    {json.dumps(sentiment_counts, indent=2)}
    RUBRIC:
    {format_rubric(topic) if topic != OTHER else "No rubric score for generic comments."}

    COMMENTS WITH SCORES:
    {json.dumps(scored_comments, indent=2)}

    Write 1 concise sentence of qualitative themes only.

    Rules:
    - Do not restate comment counts, sentiment counts, scores, averages, or percentages.
    - Refer to assigned comments, not students/respondents.
    - Do not mention a concern unless at least one listed comment states it.
    - Do not say "majority" unless the sentiment counts support it.
    - Do not repeat rubric dimensions unless the comments explicitly mention them.
    - Do not include meta-notes about following instructions.
    - Do not mention the absence of concerns as a concern.
    """

    try:
        summary = call_ollama(
            prompt,
            temperature=0.2,
            timeout=120,
        ).strip()
    except Exception as exc:
        print(f"  Summary error for {topic}: {exc}")
        return f"{exact_prefix} Themes unavailable due to model error."

    if not summary.startswith(f"Summary of {topic}:"):
        summary = f"Summary of {topic}: {summary}"
    cleaned = clean_topic_summary(topic, summary)
    body = cleaned.removeprefix(f"Summary of {topic}:").strip()
    return f"{exact_prefix} {body}" if body else exact_prefix


def summarize_single_comment(topic: str, comment: dict[str, Any]) -> str:
    feedback = normalize_comment(str(comment.get("feedback", "")))
    if len(feedback) > 180:
        feedback = feedback[:177].rstrip() + "..."
    score = comment.get("score")
    sentiment = comment.get("sentiment")
    if sentiment is None:
        sentiment = "unscored"
    score_text = f" with a score of {score}/5" if isinstance(score, int) else ""
    return f'Summary of {topic}: One assigned comment was {sentiment}{score_text}, citing: "{feedback}"'


def build_topic_summary_prefix(
    topic: str,
    comment_count: int,
    scored_count: int,
    average_score: float | None,
    sentiment_counts: dict[str, int],
    model_error_count: int = 0,
) -> str:
    if topic == OTHER:
        return (
            f"Summary of {topic}: {comment_count} generic or uncategorized comments; "
            "excluded from rubric averages."
        )

    score_text = f"average {average_score}/5" if average_score is not None else "no rubric average"
    prefix = (
        f"Summary of {topic}: {comment_count} assigned comments; "
        f"{scored_count} scored; {score_text}; "
        f"{sentiment_counts.get('positive', 0)} positive, "
        f"{sentiment_counts.get('neutral', 0)} neutral, "
        f"{sentiment_counts.get('negative', 0)} negative."
    )
    if model_error_count:
        prefix += f" {model_error_count} model scoring errors were excluded from averages."
    return prefix


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
    text = re.sub(r"(?i)\bNote:\s*.*$", "", text).strip()
    return f"{marker} {text}" if text else marker


def mean_score(scores: list[int | float]) -> float | None:
    return round(sum(scores) / len(scores), 2) if scores else None


def reliability_for_topic(comments: list[dict[str, Any]]) -> tuple[str, list[str]]:
    scored_count = sum(1 for item in comments if isinstance(item.get("score"), int))
    model_error_count = sum(1 for item in comments if item.get("scoring_status") == "model_error")
    low_confidence_count = sum(
        1
        for item in comments
        if isinstance(item.get("score"), int)
        and isinstance(item.get("confidence"), (int, float))
        and item.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
    )

    notes = []
    if scored_count == 0:
        notes.append("No scored rubric comments.")
    elif scored_count < MIN_RELIABLE_CATEGORY_COMMENTS:
        notes.append(
            f"Low sample: {scored_count} scored comments; use as directional evidence only."
        )
    if model_error_count:
        notes.append(f"{model_error_count} model scoring errors excluded from averages.")
    if low_confidence_count:
        notes.append(f"{low_confidence_count} low-confidence scored comments.")

    if scored_count == 0:
        return "unscored", notes
    if model_error_count:
        return "needs_review", notes
    if scored_count < MIN_RELIABLE_CATEGORY_COMMENTS:
        return "low_sample", notes
    if low_confidence_count:
        return "mixed_confidence", notes
    return "reliable", notes


def pluralize(count: int, singular: str, plural: str | None = None) -> str:
    word = singular if count == 1 else (plural or f"{singular}s")
    return f"{count} {word}"


def build_output_warnings(
    categories: list[dict[str, Any]],
    classification_error_count: int,
    failed_score_count: int,
    other_comment_count: int,
) -> list[str]:
    warnings = []
    low_sample_topics = [
        item["topic"]
        for item in categories
        if item["topic"] != OTHER
        and item.get("comment_count", 0) > 0
        and item.get("reliability") in {"low_sample", "unscored"}
    ]
    if low_sample_topics:
        warnings.append(
            "Low-sample category scores should not be treated as stable professor metrics: "
            + ", ".join(low_sample_topics)
        )
    if classification_error_count:
        verb = "was" if classification_error_count == 1 else "were"
        warnings.append(
            f"{pluralize(classification_error_count, 'feedback item')} failed classification "
            f"and {verb} excluded from rubric scoring."
        )
    if failed_score_count:
        verb = "was" if failed_score_count == 1 else "were"
        warnings.append(
            f"{pluralize(failed_score_count, 'topic assignment')} failed scoring and {verb} excluded."
        )
    if other_comment_count:
        verb = "is" if other_comment_count == 1 else "are"
        warnings.append(
            f"{pluralize(other_comment_count, 'generic or uncategorized comment')} "
            f"{verb} summarized but excluded from rubric averages."
        )
    return warnings


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
                    "Scored Comment Count": topic_item.get("scored_comment_count", 0),
                    "Reliability": topic_item.get("reliability", ""),
                    "Feedback": "",
                    "Classification Status": "",
                    "Topic Supported": "",
                    "Sentiment": "",
                    "Score": "",
                    "Confidence": "",
                    "Scoring Status": "",
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
                    "Scored Comment Count": topic_item.get("scored_comment_count", 0),
                    "Reliability": topic_item.get("reliability", ""),
                    "Feedback": comment["feedback"],
                    "Classification Status": comment.get("classification_status", ""),
                    "Topic Supported": comment.get("topic_supported", ""),
                    "Sentiment": comment.get("sentiment", ""),
                    "Score": comment.get("score", ""),
                    "Confidence": comment.get("confidence", ""),
                    "Scoring Status": comment.get("scoring_status", ""),
                    "Reasoning": comment.get("reasoning", ""),
                    "Topic Summary": topic_summary if comment_idx == 0 else "",
                }
            )

    fieldnames = [
        "Course ID",
        "Overall Score",
        "Topic",
        "Topic Average Score",
        "Scored Comment Count",
        "Reliability",
        "Feedback",
        "Classification Status",
        "Topic Supported",
        "Sentiment",
        "Score",
        "Confidence",
        "Scoring Status",
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
    dedupe_exact_comments: bool = True,
    use_rag: bool = True,
    evidence_filter_mode: str = "soft",
) -> dict[str, Any]:
    """Produce a combined classification, sentiment, score, and summary report."""
    output_dir = output_dir or BASE_DIR / "results" / "combined"
    start_time = time.time()
    input_comment_count = len(raw_comments)
    duplicate_comments_removed = 0
    if dedupe_exact_comments:
        raw_comments, duplicate_comments_removed = dedupe_comments(raw_comments)
        if duplicate_comments_removed:
            print(
                "Removed "
                f"{duplicate_comments_removed} exact duplicate feedback items; "
                f"processing {len(raw_comments)} unique feedback items."
            )

    classification_examples = load_classification_examples() if use_rag else []
    sentiment_examples = load_sentiment_examples() if use_rag else []
    if use_rag:
        print(
            "Loaded RAG examples: "
            f"{len(classification_examples)} classification, "
            f"{len(sentiment_examples)} sentiment."
        )

    topic_comments: dict[str, list[dict[str, Any]]] = {topic: [] for topic in TOPICS}
    assignment_scores: list[int] = []
    per_feedback_scores: dict[str, list[int]] = {}
    classification_error_count = 0
    failed_score_count = 0
    filtered_mismatch_count = 0

    for idx, feedback in enumerate(raw_comments, 1):
        print(f"\n[{idx}/{len(raw_comments)}] Processing feedback...")
        classification = classify_with_llama(
            feedback,
            classification_examples=classification_examples,
            evidence_filter_mode=evidence_filter_mode,
        )
        topics = classification.get("topics", [OTHER])
        classification_status = classification.get("classification_status", "classified")
        classification_reasoning = classification.get("classification_reasoning", "")
        if classification_status == "model_error":
            classification_error_count += 1
        print(f"  Topics: {topics}")

        for topic in topics:
            if topic == OTHER:
                scoring_status = (
                    "classification_error"
                    if classification_status == "model_error"
                    else "not_applicable"
                )
                reasoning = (
                    classification_reasoning
                    if classification_status == "model_error"
                    else "Generic or non-actionable feedback; no rubric score assigned."
                )
                topic_comments[OTHER].append(
                    {
                        "feedback": feedback,
                        "sentiment": None,
                        "score": None,
                        "confidence": None,
                        "classification_status": classification_status,
                        "topic_supported": None,
                        "scoring_status": scoring_status,
                        "reasoning": reasoning,
                    }
                )
                continue

            scored = sentiment_with_llama(
                feedback,
                topic,
                sentiment_examples=sentiment_examples,
            )
            
            # Validation: Skip if sentiment model indicates topic mismatch (using topic-specific threshold)
            is_mismatched = scored.pop("is_mismatched", False)
            threshold = CONFIDENCE_THRESHOLDS.get(topic, CONFIDENCE_THRESHOLDS["default"])
            unsupported_topic = scored.get("topic_supported") is False
            if unsupported_topic or (is_mismatched and scored.get("confidence", 0) > threshold):
                print(
                    f"    {topic}: FILTERED (unsupported topic in validation, "
                    f"conf {scored.get('confidence', 0):.2f})"
                )
                filtered_mismatch_count += 1
                continue
            
            score = scored.get("score")
            if isinstance(score, int):
                assignment_scores.append(score)
                per_feedback_scores.setdefault(feedback, []).append(score)
            elif scored.get("scoring_status") == "model_error":
                failed_score_count += 1
            topic_comments[topic].append(
                {"feedback": feedback, "classification_status": classification_status, **scored}
            )
            if isinstance(score, int):
                print(f"    {topic}: {scored['sentiment']} ({score}/5)")
            else:
                print(f"    {topic}: unscored ({scored.get('scoring_status', 'unknown')})")

    categories = []
    category_scores = []
    topic_summaries = []
    for topic in TOPIC_KEYS:
        comments = topic_comments[topic]
        scores = [item["score"] for item in comments if isinstance(item.get("score"), int)]
        average_score = mean_score(scores)
        reliability, reliability_notes = reliability_for_topic(comments)
        model_error_count = sum(1 for item in comments if item.get("scoring_status") == "model_error")
        low_confidence_count = sum(
            1
            for item in comments
            if isinstance(item.get("score"), int)
            and isinstance(item.get("confidence"), (int, float))
            and item.get("confidence", 0.0) < LOW_CONFIDENCE_THRESHOLD
        )
        summary = summarize_topic_with_llama(
            topic,
            comments,
            average_score,
        )
        categories.append(
            {
                "topic": topic,
                "average_score": average_score,
                "comment_count": len(comments),
                "scored_comment_count": len(scores),
                "model_error_count": model_error_count,
                "low_confidence_count": low_confidence_count,
                "reliability": reliability,
                "reliability_notes": reliability_notes,
                "comments": comments,
            }
        )
        category_scores.append(
            {
                "category": topic,
                "average_score": average_score,
                "comment_count": len(comments),
                "scored_comment_count": len(scores),
                "reliability": reliability,
                "reliability_notes": reliability_notes,
            }
        )
        topic_summaries.append({"topic": topic, "summary": summary})

    other_summary = summarize_topic_with_llama(
        OTHER,
        topic_comments[OTHER],
        None,
    )
    categories.append(
        {
            "topic": OTHER,
            "average_score": None,
            "comment_count": len(topic_comments[OTHER]),
            "scored_comment_count": 0,
            "model_error_count": 0,
            "low_confidence_count": 0,
            "reliability": "not_scored",
            "reliability_notes": [
                "Generic or uncategorized feedback is summarized but excluded from rubric averages."
            ],
            "comments": topic_comments[OTHER],
        }
    )
    topic_summaries.append({"topic": OTHER, "summary": other_summary})

    per_comment_score_means = [
        sum(scores) / len(scores) for scores in per_feedback_scores.values() if scores
    ]
    overall_score = mean_score(per_comment_score_means)
    topic_assignment_overall_score = mean_score(assignment_scores)
    warnings = build_output_warnings(
        categories,
        classification_error_count=classification_error_count,
        failed_score_count=failed_score_count,
        other_comment_count=len(topic_comments[OTHER]),
    )
    output = {
        "course_id": course_id,
        "model": MODEL,
        "overall_score": overall_score,
        "topic_assignment_overall_score": topic_assignment_overall_score,
        "overall_score_method": "mean_of_per_comment_topic_score_means",
        "category_scores": category_scores,
        "topic_summaries": topic_summaries,
        "categories": categories,
        "warnings": warnings,
        "metadata": {
            "num_comments": len(raw_comments),
            "num_input_comments": input_comment_count,
            "num_duplicate_comments_removed": duplicate_comments_removed,
            "dedupe_mode": "exact",
            "num_topic_assignments": sum(len(topic_comments[topic]) for topic in TOPIC_KEYS),
            "num_scored_topic_comments": len(assignment_scores),
            "num_comments_with_scores": len(per_comment_score_means),
            "num_classification_errors": classification_error_count,
            "num_failed_topic_scores": failed_score_count,
            "num_filtered_topic_mismatches": filtered_mismatch_count,
            "num_unscored_other_comments": len(topic_comments[OTHER]),
            "min_reliable_category_comments": MIN_RELIABLE_CATEGORY_COMMENTS,
            "low_confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
            "rag_enabled": use_rag,
            "classification_examples_loaded": len(classification_examples),
            "sentiment_examples_loaded": len(sentiment_examples),
            "classification_examples_per_prompt": RAG_CLASSIFICATION_EXAMPLE_COUNT if use_rag else 0,
            "sentiment_examples_per_prompt": RAG_SENTIMENT_EXAMPLE_COUNT if use_rag else 0,
            "evidence_filter_mode": evidence_filter_mode,
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Combined analysis complete. Saved to {output_path}.")
