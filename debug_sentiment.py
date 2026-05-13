#!/usr/bin/env python3
"""Debug script to see what sentiment_with_llama finds for each topic."""

import json
from main import sentiment_with_llama, TOPICS

comment = """Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more."""

print("\n=== TESTING SENTIMENT FOR EACH TOPIC ===\n")

# Topics that should be highly supported
print("EXPECTED TOPICS (Should all have topic_supported=true):")
expected = [
    "Course organization and structure",
    "Student engagement and participation",
    "Clarity of explanations",
    "Effectiveness of assignments",
    "Instructor's communication and availability",
    "Learning resources and materials"
]

for topic in expected:
    result = sentiment_with_llama(comment, topic)
    print(f"\n{topic}:")
    print(f"  topic_supported: {result.get('topic_supported')}")
    print(f"  score: {result.get('score')}")
    print(f"  evidence: {result.get('evidence_quote')}")
    print(f"  reasoning: {result.get('reasoning')}")

# Topics that should NOT be supported
print("\n\nSPURIOUS TOPICS (Should all have topic_supported=false):")
spurious = [
    "Pace",
    "Workload",
    "Assessment",
    "Grading and feedback",
    "Inclusivity and sense of belonging",
    "Classroom atmosphere"
]

for topic in spurious:
    result = sentiment_with_llama(comment, topic)
    print(f"\n{topic}:")
    print(f"  topic_supported: {result.get('topic_supported')}")
    print(f"  score: {result.get('score')}")
    print(f"  evidence: {result.get('evidence_quote')}")
    print(f"  reasoning: {result.get('reasoning')}")
