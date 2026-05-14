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
MODEL_TASK_MAX_RETRIES = 3

# Topic-specific confidence thresholds for mismatch filtering
CONFIDENCE_THRESHOLDS = {
    "Assessment": 0.6,
    "Workload": 0.65,
    "Pace": 0.55,
    "Clarity of explanations": 0.5,
    "Classroom atmosphere": 0.45,
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
    # Remove accents and any special type of characters (Normalization)
    text = unicodedata.normalize("NFKD", comment).encode("ascii", "ignore").decode("ascii")
    # Convert to lowercase
    text = text.casefold()
    # Extract only alphanumeric tokens and rejoin
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
        # Normalize text and split into individual words
        if len(token) > 2 and token not in RETRIEVAL_STOPWORDS
        # Only keep tokens that are greater than 2 chars and are not stop words
    }


def retrieval_similarity(left: str, right: str) -> float:
    """Score comment similarity using token overlap plus fuzzy full-text matching."""
    # Normalize current evaluation comment
    left_key = canonical_comment_key(left)
    # Normalize RAG comment
    right_key = canonical_comment_key(right)
    # Ensure neither or is empty
    if not left_key or not right_key:
        return 0.0
    # If they match, 100% similarity
    if left_key == right_key:
        return 1.0

    # Tokenize current comment and RAG comment
    left_tokens = retrieval_tokens(left)
    right_tokens = retrieval_tokens(right)
    if left_tokens and right_tokens:
        # Extract tokens that are the same in both tokenizations
        overlap = left_tokens & right_tokens
        # What fraction of the shorter comment appears in the overlap
        containment = len(overlap) / min(len(left_tokens), len(right_tokens))
        # What fraction of all unique tokens are in the overlap
        jaccard = len(overlap) / len(left_tokens | right_tokens)
    else:
        containment = 0.0
        jaccard = 0.0

    # Compare both comments character by character for similarity
    fuzzy_ratio = SequenceMatcher(None, left_key, right_key).ratio()
    # Result is a combined score of containment, jaccard, and fuzzy ratio
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
    # Loop through our HUMAN_CATEGORIZED_OUTPUT.csv file
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract the feedback comment
            feedback = normalize_comment(row.get("Feedback", ""))
            if not feedback:
                continue

            # For every possible topic, check if it has a "✓" or is empty
            topics = [topic for topic in TOPICS if is_truthy_label(row.get(topic))]
            if not topics:
                topics = [OTHER]
            if OTHER in topics and len(topics) > 1:
                topics = [topic for topic in topics if topic != OTHER] or [OTHER]

            # Add feedback comment and its topics to examples list
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
    # Loop through the HUMAN_SENTIMENT_BASELINE.csv file
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract comment, topic to focus on, sentiment, and reasoning for sentiment
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

            # Store extracted information into examples list
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


def example_matches_topic(example: dict[str, Any], topic: str | None) -> bool:
    """Allow topic-filtered retrieval to use multi-label examples."""
    # If no topic filter is provided, allow any example to be considered
    if topic is None:
        return True

    # Classification examples can store multiple human topics in a topics list
    topics = example.get("topics")
    if isinstance(topics, str):
        topics = [item.strip() for item in re.split(r"[;|]", topics) if item.strip()]
    if isinstance(topics, (list, tuple, set)) and topic in topics:
        return True

    # Sentiment examples usually store one topic in a topic field
    example_topic = example.get("topic")
    if isinstance(example_topic, str):
        return example_topic.strip() == topic
    if isinstance(example_topic, (list, tuple, set)):
        return topic in example_topic

    # If neither format contains the topic, skip this example
    return False


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

    # Normalize the current comment
    comment_key = canonical_comment_key(comment)
    scored_examples = []
    # Loop through all RAG examples (baseline)
    for example in examples:
        # Only choose examples that contain our current comments topic or if topic is None
        if not example_matches_topic(example, topic):
            continue

        example_feedback = str(example.get("feedback", ""))
        # Normalize feedback comment from baseline
        example_key = canonical_comment_key(example_feedback)
        # Skip exact matches so the baseline acts as guidance, not an answer key
        if exclude_exact_match and comment_key == example_key:
            continue

        # Calculate a similarity score on RAG comment and current evaluation comment
        similarity = retrieval_similarity(comment, example_feedback)
        # If they aren't similar enough, don't include
        if similarity < RAG_MIN_SIMILARITY:
            continue

        scored_examples.append((similarity, example))

    # Sort scored examples from highest score to lowest
    scored_examples.sort(key=lambda item: item[0], reverse=True)
    retrieved = []
    # Store n amount of RAG examples only (4 in our case)
    for similarity, example in scored_examples[:limit]:
        retrieved_example = dict(example)
        retrieved_example["similarity"] = round(similarity, 3)
        retrieved.append(retrieved_example)
    return retrieved


def format_classification_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "[]"

    # Keep only the fields the classification prompt needs
    compact_examples = [
        {
            "similarity": example.get("similarity", 0.0),
            "feedback": truncate_example_text(str(example.get("feedback", ""))),
            "human_topics": example.get("topics", []),
        }
        for example in examples
    ]
    # Convert examples into formatted JSON so the model sees a clear structure
    return json.dumps(compact_examples, indent=2)


def format_sentiment_examples(examples: list[dict[str, Any]]) -> str:
    if not examples:
        return "[]"

    # Keep only topic-specific sentiment information for the scoring prompt
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
    # Convert examples into formatted JSON so the model can compare rubric scores
    return json.dumps(compact_examples, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first JSON object from an LLM response."""
    # Find the outermost JSON braces in the model response
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    if json_start == -1 or json_end <= json_start:
        raise ValueError("No JSON object found")
    # Parse only the JSON substring, ignoring any extra model text
    return json.loads(text[json_start:json_end])


def call_ollama(
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 90,
    max_retries: int = OLLAMA_MAX_RETRIES,
) -> str:
    last_error: Exception | None = None
    # Retry Ollama requests because local model calls can occasionally fail
    for attempt in range(max_retries + 1):
        try:
            # Send one non-streaming prompt to the local Ollama server
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
            # Wait slightly longer after each failed attempt before retrying
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise

    raise RuntimeError(f"Ollama call failed: {last_error}")


def format_topics() -> str:
    # Format topic definitions as bullet points for the classification prompt
    return "\n".join(f"- {topic}: {TOPIC_DEFS[topic]}" for topic in TOPIC_KEYS)


def format_rubric(topic: str) -> str:
    # Pull the 1-5 scoring rubric for one topic
    rubric = SCORING_RUBRIC.get(topic, {})
    if not isinstance(rubric, dict):
        return ""
    # Format rubric levels in score order for the sentiment prompt
    return "\n".join(f"{score}: {description}" for score, description in sorted(rubric.items()))


def has_topic_evidence(comment: str, topic: str) -> bool:
    # Other does not need concrete topic evidence
    if topic == OTHER:
        return True
    patterns = TOPIC_EVIDENCE_PATTERNS.get(topic, [])
    text = comment.casefold()
    # Check whether any topic-specific evidence pattern appears in the text
    return any(re.search(pattern, text) for pattern in patterns)


def evidence_quote_is_grounded(evidence_quote: str | None, comment: str) -> bool:
    """Check that the model's evidence quote is actually present in the comment."""
    if not evidence_quote:
        return False

    quote_key = canonical_comment_key(evidence_quote)
    comment_key = canonical_comment_key(comment)
    if not quote_key or not comment_key:
        return False
    if quote_key in comment_key:
        return True

    quote_tokens = set(quote_key.split())
    comment_tokens = set(comment_key.split())
    if not quote_tokens:
        return False
    overlap_ratio = len(quote_tokens & comment_tokens) / len(quote_tokens)
    return overlap_ratio >= 0.6


def looks_like_generic_only_comment(comment: str) -> bool:
    """Catch very short generic praise before it gets forced into a real topic."""
    # If any topic evidence exists, do not treat the comment as generic only
    if any(has_topic_evidence(comment, topic) for topic in TOPIC_KEYS):
        return False

    # Longer comments are less likely to be only generic praise
    tokens = retrieval_tokens(comment)
    if len(tokens) > 15:
        return False

    # Look for concrete course words before calling the comment generic
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

    # Catch short praise like "best professor" that has no instructional detail
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
    # Keep only allowed topics and remove duplicates while preserving order
    for topic in topics:
        if topic in TOPICS and topic not in valid_topics:
            valid_topics.append(topic)

    # Fall back to Other when the model gives no valid topics
    if not valid_topics:
        return [OTHER]
    if valid_topics == [OTHER]:
        return [OTHER]

    # Other cannot be mixed with real instructional topics
    non_other_topics = [topic for topic in valid_topics if topic != OTHER]
    if not non_other_topics:
        return [OTHER]

    # Strict mode keeps only topics with regex evidence in the comment
    if mode == "strict":
        filtered_topics = [
            topic for topic in non_other_topics if has_topic_evidence(comment, topic)
        ]
        return filtered_topics or [OTHER]

    # Soft mode only removes topics when the entire comment looks generic
    if looks_like_generic_only_comment(comment):
        return [OTHER]

    return non_other_topics


def add_high_precision_topic_hints(comment: str, topics: list[str]) -> list[str]:
    """Add obvious topics that the model sometimes misses in multi-topic comments."""
    text = comment.casefold()
    hinted_topics = list(topics)
    hint_patterns = {
        "Course organization and structure": [
            r"\borganiz(?:e|ed|ation|ing)\b",
            r"\bstructur(?:e|ed|ing)\b",
        ],
        "Pace": [
            r"\bpace(?:d)?\b",
            r"\brushed?\b",
            r"\btoo (?:fast|slow)\b",
            r"\bnot enough time\b",
        ],
        "Workload": [
            r"\bworkload\b",
            r"\btoo much work\b",
            r"\bmanageable workload\b",
            r"\boverwhelm(?:ed|ing)?\b",
        ],
        "Student engagement and participation": [
            r"\bengag(?:e|ed|ing|ement)\b",
            r"\bparticipat(?:e|ed|ion)\b",
            r"\bdiscussion(?:s)?\b",
            r"\basking questions\b",
            r"\binteractive\b",
            r"\bclicker questions?\b",
        ],
        "Clarity of explanations": [
            r"\bexplain(?:s|ed|ing|ation|ations)?\b",
            r"\bclear(?:ly)?\b",
            r"\beasy to understand\b",
            r"\bfollow along\b",
        ],
        "Effectiveness of assignments": [
            r"\bassignments?\b",
            r"\bhomeworks?\b",
            r"\bproblem sets?\b",
            r"\bpractice (?:problems?|tasks?)\b",
            r"\bworksheets?\b",
            r"\bclicker questions?\b",
        ],
        "Instructor's communication and availability": [
            r"\boffice hours\b",
            r"\bavailable\b",
            r"\bapproachable\b",
            r"\brespond(?:s|ed|ing)?\b",
            r"\bemails?\b",
            r"\bcommunicat(?:e|ed|ion|ive)\b",
        ],
        "Learning resources and materials": [
            r"\bresources?\b",
            r"\bmaterials?\b",
            r"\bnotes?\b",
            r"\bslides?\b",
            r"\brecordings?\b",
            r"\breview sessions?\b",
            r"\bpractice exams?\b",
            r"\bstudy materials?\b",
        ],
    }

    for topic, patterns in hint_patterns.items():
        if topic in hinted_topics:
            continue
        if any(re.search(pattern, text) for pattern in patterns):
            hinted_topics.append(topic)

    return hinted_topics


def sentiment_from_score(score: int) -> str:
    # Map the rubric score back into a sentiment label
    if score <= 2:
        return "negative"
    if score >= 4:
        return "positive"
    return "neutral"


def parse_optional_score(value: Any) -> int | None:
    # Convert model output into an optional integer score from 1 to 5
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "null", "none", "n/a", "na"}:
            return None
        match = re.search(r"\b[1-5]\b", text)
        if not match:
            return None
        value = match.group(0)
    try:
        # Clamp scores into the valid rubric range
        return max(1, min(5, int(value)))
    except (TypeError, ValueError):
        return None


def parse_confidence(value: Any) -> float:
    # Convert model confidence into a float between 0.0 and 1.0
    if value is None:
        return 0.0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "null", "none", "n/a", "na"}:
            return 0.0
        percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
        if percent_match:
            # Convert percent confidence like "85%" into 0.85
            return max(0.0, min(1.0, float(percent_match.group(1)) / 100))
        number_match = re.search(r"\d+(?:\.\d+)?", text)
        if not number_match:
            return 0.0
        value = number_match.group(0)
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0:
        # Convert whole-number confidence like 85 into 0.85
        confidence = confidence / 100
    return max(0.0, min(1.0, confidence))


def classify_with_llama(
    comment: str,
    classification_examples: list[dict[str, Any]] | None = None,
    evidence_filter_mode: str = "soft",
) -> dict[str, Any]:
    """Classify a course-evaluation comment into concrete instructional topics."""
    # Retrieve examples from our baseline that are similar to our current comment
    retrieved_examples = retrieve_similar_examples(
        comment,
        classification_examples or [],
        limit=RAG_CLASSIFICATION_EXAMPLE_COUNT,
    )

    # Use one balanced multi-label prompt instead of one model call per topic
    prompt = f"""You are classifying one course-evaluation comment into instructional topics.

Assign every topic that is directly supported by concrete evidence in the feedback.
Do not require the exact topic words; close paraphrases are okay.
Do not assign a topic from broad praise, broad criticism, or assumptions about student success.
It is okay to assign multiple topics when the comment gives distinct evidence for each one.
Use "{OTHER}" only when the feedback is generic or has no specific instructional detail.
If "{OTHER}" is selected, it must be the only topic.

ALLOWED TOPICS:
{format_topics()}
- {OTHER}: Generic praise, broad approval/disapproval, or no specific instructional detail.

SIMILAR HUMAN-CODED EXAMPLES:
{format_classification_examples(retrieved_examples)}

HOW TO USE THE EXAMPLES:
- Use examples only to calibrate boundaries and multi-topic style.
- Do not copy labels unless this feedback has similar concrete evidence.

BOUNDARY RULES:
- Organization: structure, sequencing, logistics, layout, scheduling, course design.
- Pace: fast/slow movement through material, rushing, keeping up, time pressure.
- Workload: amount of work, burden, difficulty load, too much or manageable work.
- Engagement: participation, discussion, questions, interactive work, activities.
- Clarity: explanations, lectures, examples, understanding concepts.
- Assignments: homework, practice tasks, worksheets, problem sets, usefulness of assigned work.
- Atmosphere: sense of welcoming, belonging, comfort, motivation, stress, support.
- Communication/availability: office hours, responsiveness, announcements, access to instructor.
- Inclusivity/belonging: inclusion, accessibility, respect, feeling welcome across learners.
- Assessment: exams, tests, quizzes, alignment, difficulty, fairness of assessment design.
- Grading/feedback: grades, partial credit, grading policy, feedback on work.
- Resources/materials: notes, slides, recordings, textbooks, review materials, posted resources.

Return ONLY valid JSON in this exact shape:
{{
  "topics": ["Topic 1", "Topic 2"],
  "evidence": {{
    "Topic 1": "short exact phrase from feedback",
    "Topic 2": "short exact phrase from feedback"
  }}
}}

FEEDBACK:
\"\"\"{comment}\"\"\"
"""

    parsed = None
    # Allow for retries since the model can sometimes return incorrect formats.
    for attempt in range(1, MODEL_TASK_MAX_RETRIES + 1):
        try:
            # Call model and ensure result is a JSON object
            parsed = extract_json_object(call_ollama(prompt))
            break
        except Exception as exc:
            print(f"  Classification attempt {attempt}/{MODEL_TASK_MAX_RETRIES} failed: {exc}")

    # If classification fails, keep the comment out of scored topic averages
    if parsed is None:
        return {
            "topics": [OTHER],
            "classification_status": "model_error",
        }

    # Pull topics out of the model response and normalize to a list
    topics = parsed.get("topics", [OTHER])
    if not isinstance(topics, list):
        topics = [topics]

    evidence = parsed.get("evidence", {})
    evidence_by_topic = evidence if isinstance(evidence, dict) else {}
    valid_topics = []
    # Keep only known topic names returned by the model
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

    # If the model gave evidence, keep only topics with evidence grounded in the comment
    if evidence_by_topic:
        grounded_topics = []
        for topic in valid_topics:
            if topic == OTHER:
                continue
            evidence_text = str(evidence_by_topic.get(topic, "")).strip()
            if evidence_text and evidence_quote_is_grounded(evidence_text, comment):
                grounded_topics.append(topic)
            elif not evidence_text and has_topic_evidence(comment, topic):
                grounded_topics.append(topic)
        valid_topics = grounded_topics or [OTHER]

    valid_topics = add_high_precision_topic_hints(comment, valid_topics)

    # Run the lightweight generic-comment filter after the model gives candidate topics
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
    # Retrieve sentiment examples only for the topic currently being scored
    retrieved_examples = retrieve_similar_examples(
        comment,
        sentiment_examples or [],
        limit=RAG_SENTIMENT_EXAMPLE_COUNT,
        topic=topic,
    )
    # Prompt the model to score this one topic instead of the whole comment
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
    5. Provide one short exact evidence quote from the feedback that names or clearly paraphrases this topic.
    6. Give one brief reason grounded in that evidence quote.
    7. Provide confidence from 0.0 to 1.0.

    Return ONLY valid JSON:
    {{
    "topic_supported": true,
    "sentiment": "positive|negative|neutral",
    "score": 1,
    "confidence": 0.0,
    "evidence_quote": "short exact quote from feedback",
    "reasoning": "brief explanation"
    }}
    If topic_supported is false, use JSON null for sentiment, score, and evidence_quote.
    """

    last_error: Exception | None = None
    # Retry because the model may fail to return parseable JSON on the first try
    for attempt in range(1, MODEL_TASK_MAX_RETRIES + 1):
        try:
            # Call model and parse the JSON response for this topic score
            parsed = extract_json_object(call_ollama(prompt))
            break
        except Exception as exc:
            last_error = exc
            print(f"  Sentiment attempt {attempt}/{MODEL_TASK_MAX_RETRIES} for {topic} failed: {exc}")
    else:
        print(f"  Sentiment error for {topic}: {last_error}")
        # Mark failed scoring so it can be excluded from score averages later
        return {
            "topic_supported": None,
            "sentiment": None,
            "score": None,
            "confidence": 0.0,
            "evidence_quote": None,
            "reasoning": "Failed to score with model after retries; excluded from averages.",
            "scoring_status": "model_error",
            "is_mismatched": False,
        }

    try:
        # Normalize whether the model thinks this topic is actually supported
        raw_supported = parsed.get("topic_supported", True)
        if isinstance(raw_supported, bool):
            topic_supported = raw_supported
        else:
            topic_supported = str(raw_supported).strip().lower() not in {"false", "0", "no"}
        # Normalize evidence quote, score, sentiment, confidence, and reasoning
        evidence_quote = parsed.get("evidence_quote")
        evidence_quote = normalize_comment(str(evidence_quote)) if evidence_quote else None
        score = parse_optional_score(parsed.get("score")) if topic_supported else None
        sentiment = str(parsed.get("sentiment", "")).strip().lower()
        if sentiment not in {"positive", "negative", "neutral"}:
            sentiment = None
        confidence = parse_confidence(parsed.get("confidence", 0.0))
        reasoning = str(parsed.get("reasoning", "")).strip()
        quote_is_valid = evidence_quote_is_grounded(evidence_quote, comment)
        # Accept the score only when the model provided grounded evidence
        if topic_supported and isinstance(score, int) and quote_is_valid:
            sentiment = sentiment_from_score(score)
        else:
            if topic_supported and not quote_is_valid:
                reasoning = (
                    "Evidence quote was not grounded in the comment; "
                    f"original model reasoning: {reasoning}"
                )
            topic_supported = False
            score = None
            sentiment = None
        # Track likely topic mismatches so the main pipeline can filter them
        is_mismatched = not topic_supported or check_topic_mismatch(reasoning, topic)
        
    except Exception as exc:
        print(f"  Sentiment parse error for {topic}: {exc}")
        # Parse errors are treated as model errors and excluded from averages
        return {
            "topic_supported": None,
            "sentiment": None,
            "score": None,
            "confidence": 0.0,
            "evidence_quote": None,
            "reasoning": "Failed to score with model; excluded from averages.",
            "scoring_status": "model_error",
            "is_mismatched": False,
        }

    # Return a normalized scoring record for this comment-topic pair
    result = {
        "topic_supported": topic_supported,
        "sentiment": sentiment,
        "score": score,
        "confidence": confidence,
        "evidence_quote": evidence_quote,
        "reasoning": reasoning,
        "scoring_status": "scored",
        "is_mismatched": is_mismatched,
    }
    
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
    # Return a simple summary when no comments landed in this topic
    if not comments:
        return f"Summary of {topic}: No comments were assigned to this topic."
    # Avoid an unnecessary model call for a single comment
    if len(comments) == 1:
        return summarize_single_comment(topic, comments[0])

    # Count scored comments, model errors, and sentiment labels for the summary prompt
    scored_count = sum(1 for item in comments if isinstance(item.get("score"), int))
    model_error_count = sum(1 for item in comments if item.get("scoring_status") == "model_error")
    sentiment_counts = {
        sentiment: sum(1 for item in comments if item.get("sentiment") == sentiment)
        for sentiment in ("positive", "neutral", "negative")
    }
    # Keep the summary prompt focused on evidence and scores only
    scored_comments = [
        {
            "score": item.get("score"),
            "sentiment": item.get("sentiment"),
            "topic_supported": item.get("topic_supported"),
            "evidence_quote": item.get("evidence_quote"),
            "scoring_status": item.get("scoring_status", "unscored"),
            "text": item.get("feedback", ""),
        }
        for item in comments
    ]
    # Build a deterministic prefix so the final summary always includes counts
    exact_prefix = build_topic_summary_prefix(
        topic,
        len(comments),
        scored_count,
        average_score,
        sentiment_counts,
        model_error_count,
    )
    # Ask the model for only the qualitative theme sentence
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
        # Generate the qualitative topic summary
        summary = call_ollama(
            prompt,
            temperature=0.2,
            timeout=120,
        ).strip()
    except Exception as exc:
        print(f"  Summary error for {topic}: {exc}")
        return f"{exact_prefix} Themes unavailable due to model error."

    # Ensure all summaries use the same heading format
    if not summary.startswith(f"Summary of {topic}:"):
        summary = f"Summary of {topic}: {summary}"
    cleaned = clean_topic_summary(topic, summary)
    body = cleaned.removeprefix(f"Summary of {topic}:").strip()
    # Combine deterministic counts with model-written qualitative themes
    return f"{exact_prefix} {body}" if body else exact_prefix


def summarize_single_comment(topic: str, comment: dict[str, Any]) -> str:
    # Build a short summary directly from the one available comment
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
    # Other comments are not scored with the rubric
    if topic == OTHER:
        return (
            f"Summary of {topic}: {comment_count} generic or uncategorized comments; "
            "excluded from rubric averages."
        )

    # Build the numeric part of the summary in code for consistency
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


def public_comment(comment: dict[str, Any]) -> dict[str, Any]:
    # Keep the report output small enough for dashboards and handoff.
    return {
        "feedback": comment.get("feedback") or "",
        "sentiment": comment.get("sentiment") or "",
        "score": comment.get("score") if comment.get("score") is not None else "",
        "confidence": (
            comment.get("confidence") if comment.get("confidence") is not None else ""
        ),
        "reasoning": comment.get("reasoning") or "",
    }


def public_category(category: dict[str, Any]) -> dict[str, Any]:
    # Convert internal category data into the report-friendly category shape
    return {
        "topic": category["topic"],
        "average_score": category["average_score"],
        "comment_count": category["comment_count"],
        "comments": [public_comment(comment) for comment in category["comments"]],
    }


def public_category_score(category: dict[str, Any]) -> dict[str, Any]:
    # Store compact category-level scores for quick summary views
    return {
        "category": category["topic"],
        "average_score": category["average_score"],
        "comment_count": category["comment_count"],
    }


def clean_topic_summary(topic: str, summary: str) -> str:
    """Keep exactly one topic-summary heading and remove model preambles."""
    marker = f"Summary of {topic}:"
    # Normalize whitespace before cleaning the model summary
    text = summary.replace("\r\n", "\n").replace("\r", "\n").strip()
    # If the model repeated the heading, keep only the final body text
    marker_pattern = re.compile(rf"Summary\s+of\s+{re.escape(topic)}\s*:", re.IGNORECASE)
    marker_matches = list(marker_pattern.finditer(text))
    if marker_matches:
        text = text[marker_matches[-1].end():].strip()

    # Remove extra whitespace and trailing model notes
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s*\n\s*", " ", text).strip()
    text = re.sub(r"(?i)\bNote:\s*.*$", "", text).strip()
    return f"{marker} {text}" if text else marker


def mean_score(scores: list[int | float]) -> float | None:
    # Return a rounded mean when scores exist, otherwise leave it blank
    return round(sum(scores) / len(scores), 2) if scores else None


def write_combined_csv(output: dict[str, Any], csv_path: Path) -> None:
    rows = []
    # Look up each topic summary so it can be attached to CSV rows
    summary_by_topic = {
        item["topic"]: item["summary"]
        for item in output.get("topic_summaries", [])
        if "topic" in item and "summary" in item
    }
    # Flatten nested categories and comments into one CSV row per comment
    for topic_item in output["categories"]:
        topic = topic_item["topic"]
        topic_summary = summary_by_topic.get(topic, topic_item.get("summary", ""))
        comments = topic_item["comments"]
        # Keep a topic row even when there are no comments for that topic
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

        # Write every comment assigned to this topic as its own row
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

    # Define column order explicitly so CSV exports stay stable
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
    # Create the output folder and write the CSV file
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

    # Remove comments that are exact duplicates
    if dedupe_exact_comments:
        raw_comments, duplicate_comments_removed = dedupe_comments(raw_comments)

    # Load baseline examples for RAG approach
    classification_examples = load_classification_examples() if use_rag else []
    sentiment_examples = load_sentiment_examples() if use_rag else []

    # ------ Initialize Result Variables ------

    # Dictionary to store comments for each topic
    topic_comments: dict[str, list[dict[str, Any]]] = {topic: [] for topic in TOPICS}
    # Dict of all feedback comments and their scores.
    per_feedback_scores: dict[str, list[int]] = {}
    # Start iterating comments for evaluation
    for feedback in raw_comments:
        # Classify the current feedback into one or more candidate topics
        classification = classify_with_llama(
            feedback,
            classification_examples=classification_examples,
            evidence_filter_mode=evidence_filter_mode,
        )
        # Pull classification result fields with safe defaults
        topics = classification.get("topics", [OTHER])
        classification_status = classification.get("classification_status", "classified")

        # Score each topic assigned to this feedback
        for topic in topics:
            # Other comments are not scored with a rubric
            if topic == OTHER:
                if classification_status == "model_error":
                    continue
                # Store generic feedback separately so it can be summarized but not averaged
                topic_comments[OTHER].append(
                    {
                        "feedback": feedback,
                        "sentiment": None,
                        "score": None,
                        "confidence": None,
                        "classification_status": classification_status,
                        "topic_supported": None,
                        "evidence_quote": None,
                        "scoring_status": "not_applicable",
                        "reasoning": "Generic or non-actionable feedback; no rubric score assigned.",
                    }
                )
                continue

            # Score this feedback for the current topic
            scored = sentiment_with_llama(
                feedback,
                topic,
                sentiment_examples=sentiment_examples,
            )
            
            # Validation: skip if sentiment model indicates topic mismatch.

            # Check if model references a mismatch in reasoning
            is_mismatched = scored.pop("is_mismatched", False)
            # Extract confidence threshold for current topic
            threshold = CONFIDENCE_THRESHOLDS.get(topic, CONFIDENCE_THRESHOLDS["default"])
            # Check if model returned that the comment does not match the topic.
            unsupported_topic = scored.get("topic_supported") is False
            # Use the following factors to determine whether or not to remove topic classification
            if unsupported_topic or (is_mismatched and scored.get("confidence", 0) > threshold):
                continue
            
            # Store valid numeric scores for topic and overall averages
            score = scored.get("score")
            if isinstance(score, int):
                per_feedback_scores.setdefault(feedback, []).append(score)
            # Skip model errors so they do not distort averages
            elif scored.get("scoring_status") == "model_error":
                continue
            else:
                continue
            # Save the scored comment under its topic
            topic_comments[topic].append(
                {"feedback": feedback, "classification_status": classification_status, **scored}
            )

    # Build final topic-level objects after all comments have been processed
    categories = []
    category_scores = []
    topic_summaries = []
    for topic in TOPIC_KEYS:
        # Gather scored records for one topic
        comments = topic_comments[topic]
        scores = [item["score"] for item in comments if isinstance(item.get("score"), int)]
        average_score = mean_score(scores)
        # Generate a topic summary using the scored comments
        summary = summarize_topic_with_llama(
            topic,
            comments,
            average_score,
        )
        # Store full internal category data before converting to public output
        categories.append(
            {
                "topic": topic,
                "average_score": average_score,
                "comment_count": len(comments),
                "comments": comments,
            }
        )
        # Store compact score view and summary view separately for easy access
        category_scores.append(
            public_category_score(categories[-1])
        )
        topic_summaries.append({"topic": topic, "summary": summary})

    # Summarize generic or uncategorized comments outside the scored topic loop
    other_summary = summarize_topic_with_llama(
        OTHER,
        topic_comments[OTHER],
        None,
    )
    # Add Other as a final category, but exclude it from score averages
    categories.append(
        {
            "topic": OTHER,
            "average_score": None,
            "comment_count": len(topic_comments[OTHER]),
            "comments": topic_comments[OTHER],
        }
    )
    topic_summaries.append({"topic": OTHER, "summary": other_summary})

    # Average each feedback's topic scores first, then average across feedback comments
    per_comment_score_means = [
        sum(scores) / len(scores) for scores in per_feedback_scores.values() if scores
    ]
    overall_score = mean_score(per_comment_score_means)
    scored_topic_comment_count = sum(len(topic_comments[topic]) for topic in TOPIC_KEYS)
    # Build the final JSON-compatible output object
    output = {
        "course_id": course_id,
        "model": MODEL,
        "overall_score": overall_score,
        "category_scores": category_scores,
        "topic_summaries": topic_summaries,
        "categories": [public_category(category) for category in categories],
        "metadata": {
            "num_comments": len(raw_comments),
            "num_input_comments": input_comment_count,
            "num_duplicate_comments_removed": duplicate_comments_removed,
            "dedupe_mode": "exact" if dedupe_exact_comments else "none",
            "num_scored_topic_comments": scored_topic_comment_count,
            "total_time_seconds": round(time.time() - start_time, 2),
        },
    }

    # Write JSON and CSV reports when file output is enabled
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
    # Pull the course id, using UNKNOWN if no id is provided
    course_id = json_data.get("course_id", "UNKNOWN")
    # Pull the raw comment list from the input JSON
    raw_comments = json_data.get("raw_comments", [])
    
    # The pipeline expects comments to be provided as a list of strings
    if not isinstance(raw_comments, list):
        raise ValueError("raw_comments must be a list")
    
    # Print a quick confirmation before the expensive model calls begin
    print(f"Loaded course_id: {course_id}")
    print(f"Loaded {len(raw_comments)} feedback items")
    
    return course_id, raw_comments


if __name__ == "__main__":
    # Exact JSON input
    json_input = {
        "course_id": "TEST_THREE",
        "raw_comments": [
            """Professor Wu is one of the best professors I've had since coming to UCLA – I really like his teaching style and how much he cares for his students. I like how the lecture notes are structured, and that he offers both a blank and a filled–in version of the notes so that we can adjust our own learning. His lectures are consistently clear and paced well. I also feel that the course material makes sense in the way that it is taught. During lecture, Professor Wu also answers questions in a very clear way. His exams are also fair – not too challenging, but also not too easy, and also perfectly timed.
I also appreciate how he adjusted the class during the L.A. fires period. It was really nice of him!
Professor Wu is super nice and funny. It is really refreshing to be in his class – I learn a lot without feeling overwhelming pressure. If I could take a class again with him, I 1000% would."""
            ]
    }
    
    # Extract course_id and comments from JSON
    course_id, raw_comments = load_feedback_from_json(json_input)
    # Main pipeline for classification, sentiment, and summary.
    output = analysis_pipeline(course_id, raw_comments)
    print("\n" + "=" * 80)
    # Save a copy of the full output to the project-level results file
    output_path = BASE_DIR / "results" / f"{course_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Combined analysis complete. Saved to {output_path}.")
