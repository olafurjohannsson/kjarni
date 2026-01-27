from transformers import pipeline

# Initialize the classifier
classifier = pipeline("text-classification", model="/home/olafurj/.cache/kjarni/olafuraron_bert-base-multilingual-uncased-sentiment-safetensors/")

# Test texts
TEST_TEXT_POSITIVE = "I absolutely love this, it's amazing!"
TEST_TEXT_NEGATIVE = "This is terrible, I hate it so much."
TEST_TEXT_NEUTRAL = "The meeting is scheduled for Tuesday."

TEST_TEXT_GERMAN_POSITIVE = "Das ist wunderbar, ich liebe es!"
TEST_TEXT_FRENCH_NEGATIVE = "C'est terrible, je déteste ça."
TEST_TEXT_SPANISH_POSITIVE = "¡Esto es increíble, me encanta!"

TEST_TEXT_TOXIC = "I hate you, you're worthless garbage."
TEST_TEXT_NON_TOXIC = "Thank you for your help today."

# Batch classify (matching your test)
texts = [
    TEST_TEXT_POSITIVE,         # English
    TEST_TEXT_GERMAN_POSITIVE,  # German
    TEST_TEXT_FRENCH_NEGATIVE,  # French
    TEST_TEXT_SPANISH_POSITIVE, # Spanish
]

results = classifier(texts)

# Print results
for i, result in enumerate(results):
    print(f"Batch result: {result['label']} (score: {result['score']})")

print("\n--- Full results ---")
for i, (text, result) in enumerate(zip(texts, results)):
    print(f"{i}: {text[:50]}... -> {result['label']} (score: {result['score']:.4f})")