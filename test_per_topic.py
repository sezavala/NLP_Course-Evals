#!/usr/bin/env python3
"""Test the new per-topic classification on a single comment."""

from main import classify_with_llama, TOPIC_KEYS

comment = """Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more."""

print("=== PER-TOPIC CLASSIFICATION TEST ===\n")
print("Comment:", comment[:100] + "...\n")

result = classify_with_llama(comment)
topics = result.get('topics', [])

print(f"Classified topics: {topics}")
print(f"Total: {len(topics)} topics\n")

print("Expected topics:")
expected = [
    "Course organization and structure",
    "Student engagement and participation",
    "Clarity of explanations", 
    "Effectiveness of assignments",
    "Instructor's communication and availability",
    "Learning resources and materials"
]

for exp in expected:
    if exp in topics:
        print(f"  ✓ {exp}")
    else:
        print(f"  ✗ {exp} (MISSING)")

print("\nUnexpected topics (should not be assigned):")
spurious = ["Pace", "Workload", "Assessment", "Grading and feedback", 
            "Inclusivity and sense of belonging", "Classroom atmosphere"]

for spur in spurious:
    if spur in topics:
        print(f"  ✗ {spur} (SHOULD NOT BE HERE)")
    else:
        print(f"  ✓ {spur}")
