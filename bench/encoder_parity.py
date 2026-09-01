"""Reference embeddings for the encoder parity test.

Writes bench/torch_encoder_parity.json. Regenerate when the corpus changes:

    cd bench && .venv/bin/python encoder_parity.py
"""
import json, os, torch
from transformers import AutoTokenizer, AutoModel

torch.set_grad_enabled(False)
cache = os.path.expanduser("~/.cache/kjarni")

# Long enough to cross the 1000-token buffered-path threshold at 16 documents.
DOCS = [
    "The history of Iceland begins with settlement in the ninth century. " * 12
    + f"Document {i}."
    for i in range(16)
]
SHORT = "The capital of Iceland is Reykjavik, which sits on the south west coast."

out = {"short": SHORT, "docs": DOCS, "models": {}}
for name in ["sentence-transformers_all-MiniLM-L6-v2", "SamLowe_roberta-base-go_emotions"]:
    path = f"{cache}/{name}"
    if not os.path.exists(f"{path}/model.safetensors"):
        print(f"  skipping {name}: not cached")
        continue
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModel.from_pretrained(path, dtype=torch.float32).eval()

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        h = m(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(v, dim=-1)

    out["models"][name] = {
        "dim": m.config.hidden_size,
        "short": embed([SHORT])[0].tolist(),
        "batch": [row.tolist() for row in embed(DOCS)],
    }
    print(f"  {name}: dim {m.config.hidden_size}")
    del m

with open(os.path.join(os.path.dirname(__file__), "torch_encoder_parity.json"), "w") as f:
    json.dump(out, f)
print("wrote torch_encoder_parity.json")
