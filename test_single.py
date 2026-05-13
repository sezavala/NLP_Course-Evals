import json
import subprocess
import sys

# The comment the user is testing
test_comment = "Eric Wu is a HUGE step up from the last professor who taught chemistry. He is a phenomenal educator who focuses mostly on the logical and understanding parts of chemistry. Taking out most of the memorizing parts, it allows students to fully immerse into thinking how chemistry works. Afterall, it is how the exams, homework, and discussions are structured, solving through deconstruction and reconstruction of the problem. The lectures themselves explains thoroughly through the steps of each step, calculations, or concept. Lectures are beautifully structured with multiple options to engage such as asking questions during the lecture through a QR code, provided downloadable notes, office hours, and plenty more."

# Run main.py with test input
test_data = {
    "course_id": "CHEM_20B_TEST_MULTI",
    "raw_comments": [test_comment]
}

with open('test_input.json', 'w') as f:
    json.dump(test_data, f)

# Execute and see what gets classified
result = subprocess.run(['python3', 'main.py'], 
                       input=json.dumps(test_data),
                       capture_output=True, text=True)

# Parse the output
with open('results/ML_OUTPUT.json') as f:
    data = json.load(f)

print("\n=== CLASSIFICATION RESULT ===")
categories_with_comment = data['categories']
found_topics = [cat['topic'] for cat in categories_with_comment if cat['comment_count'] > 0]

print(f"\nTopics assigned by LLM: {found_topics}")
print(f"\nExpected topics:")
print("  1. Course organization and structure")
print("  2. Student engagement and participation")  
print("  3. Clarity of explanations")
print("  4. Effectiveness of assignments")
print("  5. Instructor's communication and availability")
print("  6. Learning resources and materials")

print(f"\n✓ Found: {len(found_topics)}")
print(f"✗ Missing: {6 - len(found_topics)}")

for topic in found_topics:
    for cat in categories_with_comment:
        if cat['topic'] == topic:
            print(f"\n  {topic}:")
            if cat['comments']:
                print(f"    Score: {cat['comments'][0].get('score')}")
                if 'evidence_quote' in cat['comments'][0]:
                    print(f"    Evidence: {cat['comments'][0].get('evidence_quote')[:60]}...")
