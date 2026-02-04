# generate_classifier_fixtures.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F

# Model paths based on cache directory
# Prefer olafuraron_ versions where available (safetensors)
models = {
    "distilbert-sentiment": "/home/olafurj/.cache/kjarni/distilbert_distilbert-base-uncased-finetuned-sst-2-english",
    "roberta-sentiment": "/home/olafurj/.cache/kjarni/olafuraron_twitter-roberta-base-sentiment-latest-safetensors",
    "bert-sentiment-multilingual": "/home/olafurj/.cache/kjarni/olafuraron_bert-base-multilingual-uncased-sentiment-safetensors",
    "distilroberta-emotion": "/home/olafurj/.cache/kjarni/olafuraron_emotion-english-distilroberta-base-safetensors",
    "roberta-emotions": "/home/olafurj/.cache/kjarni/SamLowe_roberta-base-go_emotions",
    "toxic-bert": "/home/olafurj/.cache/kjarni/olafuraron_toxic-bert-safetensors",
}

# Test inputs
test_inputs = {
    "positive": "I absolutely love this, it's amazing!",
    "negative": "This is terrible, I hate it so much.",
    "neutral": "The meeting is scheduled for Tuesday.",
    "german_positive": "Das ist wunderbar, ich liebe es!",
    "french_negative": "C'est terrible, je déteste ça.",
    "spanish_positive": "¡Esto es increíble, me encanta!",
    "toxic": "I hate you, you're worthless garbage.",
    "non_toxic": "Thank you for your help today.",
    "happy": "I am so happy and excited for the weekend!",
    "sad": "That was a truly disappointing and sad experience.",
}

results = {}

for model_name, model_path in models.items():
    print(f"\n{'='*60}")
    print(f"=== {model_name} ===")
    print(f"{'='*60}")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        print(f"  Failed to load: {e}")
        continue
    
    # Get labels from model config
    id2label = model.config.id2label
    num_labels = model.config.num_labels
    print(f"  Labels ({num_labels}): {list(id2label.values())}")
    
    # Detect if multi-label (sigmoid) or single-label (softmax)
    # Multi-label models typically have problem_type set or specific architectures
    is_multi_label = getattr(model.config, 'problem_type', None) == 'multi_label_classification'
    # Also check by model name for known multi-label models
    if model_name in ['roberta-emotions', 'toxic-bert']:
        is_multi_label = True
    print(f"  Multi-label: {is_multi_label}")
    
    results[model_name] = {
        "labels": list(id2label.values()),
        "num_labels": num_labels,
        "is_multi_label": is_multi_label,
        "outputs": {}
    }
    
    # Select relevant test inputs for each model
    if model_name == "toxic-bert":
        relevant_inputs = ["toxic", "non_toxic", "positive"]
    elif model_name == "bert-sentiment-multilingual":
        relevant_inputs = ["positive", "negative", "german_positive", "french_negative", "spanish_positive"]
    elif model_name in ["distilroberta-emotion", "roberta-emotions"]:
        relevant_inputs = ["positive", "negative", "happy", "sad", "neutral"]
    else:
        relevant_inputs = ["positive", "negative", "neutral"]
    
    for input_name in relevant_inputs:
        text = test_inputs[input_name]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0]
            
            if is_multi_label:
                # Sigmoid for multi-label
                probs = torch.sigmoid(logits)
            else:
                # Softmax for single-label
                probs = F.softmax(logits, dim=-1)
            
            probs = probs.numpy()
            
            # Get top prediction
            top_idx = probs.argmax()
            top_label = id2label[int(top_idx)]
            top_score = float(probs[top_idx])
            
            # Get all scores
            all_scores = [(id2label[i], float(probs[i])) for i in range(len(probs))]
            all_scores_sorted = sorted(all_scores, key=lambda x: x[1], reverse=True)
        
        results[model_name]["outputs"][input_name] = {
            "text": text,
            "label": top_label,
            "score": top_score,
            "label_index": int(top_idx),
            "all_scores": all_scores,
            "all_scores_sorted": all_scores_sorted[:5],  # Top 5 for display
        }
        
        print(f"\n  {input_name}: \"{text[:50]}...\"" if len(text) > 50 else f"\n  {input_name}: \"{text}\"")
        print(f"    label: {top_label}")
        print(f"    score: {top_score:.8f}")
        print(f"    label_index: {top_idx}")
        if is_multi_label:
            above_threshold = [(l, s) for l, s in all_scores if s > 0.5]
            print(f"    above 0.5: {above_threshold}")

# Output Rust constants
print("\n\n" + "="*80)
print("=== RUST TEST CONSTANTS ===")
print("="*80)

# Test input constants
print("\n// Test inputs")
for name, text in test_inputs.items():
    const_name = f"INPUT_{name.upper()}"
    text_escaped = text.replace('"', '\\"')
    print(f'pub const {const_name}: &str = "{text_escaped}";')

print()

# Model output constants
for model_name, data in results.items():
    model_const = model_name.upper().replace("-", "_")
    print(f"\n// {model_name}")
    print(f"// Labels: {data['labels']}")
    print(f"// Multi-label: {data['is_multi_label']}")
    
    for input_name, output in data["outputs"].items():
        const_prefix = f"{model_const}_{input_name.upper()}"
        label_escaped = output["label"].replace('"', '\\"')
        print(f'pub const {const_prefix}_LABEL: &str = "{label_escaped}";')
        print(f'pub const {const_prefix}_SCORE: f32 = {output["score"]:.8f};')
        print(f'pub const {const_prefix}_LABEL_INDEX: usize = {output["label_index"]};')
    print()

# Also output full score arrays for exact matching
print("\n// Full score arrays for exact matching (single-label models)")
for model_name, data in results.items():
    if data['is_multi_label']:
        continue
    model_const = model_name.upper().replace("-", "_")
    print(f"\n// {model_name} scores")
    for input_name, output in data["outputs"].items():
        const_prefix = f"{model_const}_{input_name.upper()}"
        scores = [f"{s:.8f}" for _, s in output["all_scores"]]
        print(f'pub const {const_prefix}_SCORES: &[f32] = &[{", ".join(scores)}];')