from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiments.struct import TopicCategorization, Comment, AllTopics
from experiments.json_to_sheet import json_to_dataframe
from data import TOPIC_DEFS, TOPIC_KEYS, FEEDBACK_LIST

BASE_DIR = Path(__file__).resolve().parents[2]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OTHER_LABEL = "None of the above / Other"
TOPICS = list(TOPIC_KEYS) + [OTHER_LABEL]

# DistilRoBERTa MNLI (faster than full RoBERTa)
TOPIC_MODEL_NAME = "distilroberta-base"
topic_tokenizer = AutoTokenizer.from_pretrained(TOPIC_MODEL_NAME)
topic_model = AutoModelForSequenceClassification.from_pretrained(
    "cross-encoder/nli-distilroberta-base"
).to(DEVICE)
topic_model.eval()

TOPIC_SCORE_THRESHOLD = 0.50
MAX_TOPICS = 5


def _mnli_label_indices(model) -> Tuple[int, int]:
    label2id = {k.lower(): v for k, v in model.config.label2id.items()}
    contradiction_idx = label2id.get("contradiction", 0)
    entailment_idx = label2id.get("entailment", 2)
    return contradiction_idx, entailment_idx


def predict_topics_with_scores(text: str) -> Dict[str, float]:
    """DistilRoBERTa-MNLI zero-shot topic scoring."""
    premises = [text] * len(TOPIC_KEYS)
    hypotheses = [
        f"This feedback is about {TOPIC_DEFS[topic]}."
        for topic in TOPIC_KEYS
    ]

    enc = topic_tokenizer(
        premises,
        hypotheses,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = topic_model(**enc)

    contradiction_idx, entailment_idx = _mnli_label_indices(topic_model)
    pair_logits = outputs.logits[:, [contradiction_idx, entailment_idx]]
    entailment_probs = torch.softmax(pair_logits, dim=1)[:, 1].tolist()

    return {topic: score for topic, score in zip(TOPIC_KEYS, entailment_probs)}


def get_assigned_topics(scores: Dict[str, float]) -> List[str]:
    """Multi-label: assign all topics above threshold."""
    assigned = [
        topic for topic, score in scores.items()
        if score >= TOPIC_SCORE_THRESHOLD
    ]
    return assigned if assigned else [OTHER_LABEL]


def main() -> None:
    bucket: Dict[str, List[Comment]] = {t: [] for t in TOPICS}

    for idx, feedback in enumerate(FEEDBACK_LIST, 1):
        print(f"[{idx}/{len(FEEDBACK_LIST)}] {feedback[:60]}...")
        
        scores = predict_topics_with_scores(feedback)
        assigned_topics = get_assigned_topics(scores)
        
        print(f"  → Topics: {assigned_topics}")
        print(f"  → Scores: {[(t, round(scores[t], 3)) for t in assigned_topics if t != OTHER_LABEL]}")

        comment = Comment(text=feedback, sentiment=None)
        for topic in assigned_topics:
            bucket[topic].append(comment)

    final_output = AllTopics(
        topics=[
            TopicCategorization(topic=t, feedback=bucket[t])
            for t in TOPICS
        ]
    )

    output_path = BASE_DIR / "results" / "DistilroBERTa" / "DISTILROBERTA_OUTPUT.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_output.model_dump_json(indent=2))

    json_to_dataframe(output_path, output_path.parent / "DISTILROBERTA_OUTPUT.csv")

    print(f"\n✓ Classification complete. Saved to {output_path}.")


if __name__ == "__main__":
    main()