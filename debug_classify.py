#!/usr/bin/env python3
"""Debug the raw classification LLM output."""

import json
from main import classify_with_llama, call_ollama, extract_json_object

comment = """Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more."""

print("=== RAW CLASSIFICATION OUTPUT ===\n")

# Build minimal classification prompt to see raw output
from main import format_topics, retrieve_similar_examples, RAG_CLASSIFICATION_EXAMPLE_COUNT, OTHER, format_classification_examples

retrieved_examples = retrieve_similar_examples(comment, [], limit=RAG_CLASSIFICATION_EXAMPLE_COUNT)

prompt = f"""You are a balanced course-evaluation topic coder.

Assign topics that are CLEARLY SUPPORTED by concrete evidence in the feedback.

ALLOWED TOPICS:
{format_topics()}
- {OTHER}: Generic praise, broad approval, or comments with no specific instructional detail.

RETRIEVED HUMAN-CODED REFERENCE EXAMPLES:
{format_classification_examples(retrieved_examples)}

CRITICAL EXAMPLES TO PREVENT HALLUCINATION:
Input: "Explains well, good office hours, organized course."
CORRECT: [Clarity of explanations, Instructor's communication and availability, Course organization and structure]
WRONG: [Assessment, Grading and feedback, Pace, Workload]

Return ONLY valid JSON:
{{"topics": ["Topic 1", "Topic 2"]}}

FEEDBACK:
\"\"\"{comment}\"\"\"
"""

print("Prompt length:", len(prompt))
print("\nCalling model...")

raw_response = call_ollama(prompt)
print("\n=== RAW MODEL RESPONSE ===\n")
print(raw_response)

print("\n=== PARSED JSON ===\n")
parsed = extract_json_object(raw_response)
print(json.dumps(parsed, indent=2))
