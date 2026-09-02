"""PyTorch timings on the same matrix as the Rust encoder benchmark.

Same models, same corpus, same batch sizes and document lengths, so the two
tables line up row for row.

Reports the minimum of N timed iterations after WARMUP untimed ones, matching
crates/kjarni-models/tests/encoder_benchmark.rs. Averaging instead gave a 10%
median and 27% worst-case spread across identical runs, which buried the
differences being measured. Both sides must use the same statistic or the
comparison is biased.
"""
import time, os, torch
from transformers import AutoTokenizer, AutoModel
torch.set_grad_enabled(False)
cache = os.path.expanduser("~/.cache/kjarni")

# The fourth field is the context window. It must match what kjarni feeds the
# model or the two sides are not doing the same work: nomic and bge-m3 take
# 8192 tokens, and truncating either to 512 made torch look flat from 48
# sentences to 100 while kjarni encoded the whole document.
MODELS = [
    ("sentence-transformers_all-MiniLM-L6-v2", "minilm 22M", 64, 512),
    ("distilbert_distilbert-base-uncased-finetuned-sst-2-english", "distilbert 66M", 64, 512),
    ("sentence-transformers_all-mpnet-base-v2", "mpnet 110M", 64, 512),
    ("nomic-ai_nomic-embed-text-v1.5", "nomic 137M", 64, 8192),
    ("BAAI_bge-m3", "bge-m3 567M", 64, 8192),
]

WARMUP = 3
ENCODE_RUNS = 9
BATCH_RUNS = 7

def timed(fn, arg):
    s = time.perf_counter()
    fn(arg)
    return (time.perf_counter() - s) * 1000.0

def docs(n, sentences):
    return ["".join(f"Document number {i} sentence {j} covers refunds, delivery windows and account settings in some detail. "
                    for j in range(sentences)) for i in range(n)]

print(f"\n  torch threads={torch.get_num_threads()}\n")
print(f"  {'model':<16}{'call':>7}{'docs':>6}{'sent':>5}{'ms':>11}")
for path, label, max_docs, ctx in MODELS:
    full = f"{cache}/{path}"
    if not os.path.exists(f"{full}/model.safetensors"):
        print(f"  {label:<16} not cached"); continue
    try:
        tok = AutoTokenizer.from_pretrained(full)
        m = AutoModel.from_pretrained(full, dtype=torch.float32, trust_remote_code=True).eval()
    except Exception as e:
        print(f"  {label:<16} load failed: {type(e).__name__}"); continue

    def embed(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=ctx, return_tensors="pt")
        h = m(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        v = (h * mask).sum(1) / mask.sum(1)
        return torch.nn.functional.normalize(v, dim=-1)

    for sentences in (1, 12, 48, 100):
        t1 = docs(1, sentences)
        for _ in range(WARMUP): embed(t1)
        best = min(timed(embed, t1) for _ in range(ENCODE_RUNS))
        print(f"  {label:<16}{'encode':>7}{1:>6}{sentences:>5}{best:>11.2f}")

    for sentences in (1, 12):
        n = 1
        while n <= max_docs:
            texts = docs(n, sentences)
            for _ in range(WARMUP): embed(texts)
            runs = 4 if n * sentences > 100 else BATCH_RUNS
            best = min(timed(embed, texts) for _ in range(runs))
            print(f"  {label:<16}{'batch':>7}{n:>6}{sentences:>5}{best:>11.2f}")
            n *= 4
    del m
print()
