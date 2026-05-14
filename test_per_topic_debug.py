#!/usr/bin/env python3
"""Debug per-topic classification."""

from main import classify_topic_confirmed, TOPIC_KEYS

comment = """Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more."""

print("=== PER-TOPIC CONFIRMATION TEST ===\n")

for topic in TOPIC_KEYS:
    result = classify_topic_confirmed(comment, topic)
    status = "✓ YES" if result else "✗ NO"
    print(f"{status:8} {topic}")
