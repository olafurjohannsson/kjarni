"""Reference scores for the classifier and cross-encoder parity tests.

These exercise the classification heads, which the encoder parity reference does
not: it loads AutoModel, which drops the head entirely.

    cd bench && .venv/bin/python head_parity.py
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

torch.set_grad_enabled(False)
cache = os.path.expanduser("~/.cache/kjarni")

TEXTS = [
    "This is absolutely wonderful, thank you.",
    "Worst purchase of my life.",
    "It arrived on time and works as described.",
    "okay I guess, nothing special",
]
QUERY = "when was Iceland settled"
DOCS = [
    "Iceland was settled in the ninth century by Norse seafarers.",
    "Refunds are available within 30 days of purchase.",
    "The settlement of Iceland began around the year 874.",
    "Our support hours are 9am to 5pm on weekdays.",
]

out = {"texts": TEXTS, "query": QUERY, "docs": DOCS, "classifiers": {}, "cross_encoders": {}}

for name in [
    "distilbert_distilbert-base-uncased-finetuned-sst-2-english",
    "SamLowe_roberta-base-go_emotions",
]:
    path = f"{cache}/{name}"
    if not os.path.exists(f"{path}/model.safetensors"):
        print(f"  skipping {name}")
        continue
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
    enc = tok(TEXTS, padding=True, truncation=True, return_tensors="pt")
    logits = m(**enc).logits
    out["classifiers"][name] = {
        "labels": [m.config.id2label[i] for i in range(logits.shape[1])],
        "logits": logits.tolist(),
    }
    print(f"  {name}: {logits.shape[1]} labels")
    del m

name = "cross-encoder_ms-marco-MiniLM-L-6-v2"
path = f"{cache}/{name}"
if os.path.exists(f"{path}/model.safetensors"):
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path, dtype=torch.float32).eval()
    enc = tok([QUERY] * len(DOCS), DOCS, padding=True, truncation=True, return_tensors="pt")
    scores = m(**enc).logits.squeeze(-1)
    out["cross_encoders"][name] = {"scores": scores.tolist()}
    print(f"  {name}: {scores.tolist()}")

with open(os.path.join(os.path.dirname(__file__), "torch_head_parity.json"), "w") as f:
    json.dump(out, f)
print("wrote torch_head_parity.json")
