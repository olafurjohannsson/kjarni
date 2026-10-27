"""Times one engine on one encoder task, in a fresh process, and prints JSON.

    python encoder_worker.py ENGINE TASK

ENGINE: kjarni | torch | onnx
TASK:   embed-minilm | embed-mpnet | rerank | classify

Reports load time, single-input latency, batch throughput and peak RSS. Timings
are the minimum and median of RUNS calls after WARMUP untimed ones; the minimum is
the least noisy estimate of what the engine can do, the median what a caller sees.
"""
import json, os, sys, time, resource
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from corpus import QUERY, documents, reviews

ENGINE, TASK = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None
CACHE = os.path.expanduser("~/.cache/kjarni")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
WARMUP, RUNS_SINGLE, RUNS_BATCH = 5, 50, 9

TASKS = {
    "embed-minilm": dict(kjarni="minilm-l6-v2", hf="sentence-transformers/all-MiniLM-L6-v2",
                         local="sentence-transformers_all-MiniLM-L6-v2", kind="embed"),
    "embed-mpnet": dict(kjarni="mpnet-base-v2", hf="sentence-transformers/all-mpnet-base-v2",
                        local="sentence-transformers_all-mpnet-base-v2", kind="embed"),
    "rerank": dict(kjarni="minilm-l6-v2-cross-encoder", hf="cross-encoder/ms-marco-MiniLM-L6-v2",
                   local="cross-encoder_ms-marco-MiniLM-L-6-v2", kind="rerank"),
    "classify": dict(kjarni="distilbert-sentiment", hf="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
                     local="distilbert_distilbert-base-uncased-finetuned-sst-2-english", kind="classify"),
}
T = TASKS[TASK]
DOCS = documents(64)
TEXTS = reviews(64)


def timed(fn, runs):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(runs):
        s = time.perf_counter(); fn(); ts.append((time.perf_counter() - s) * 1000)
    return dict(min=min(ts), median=float(np.median(ts)))


# ── engines: each returns (single_fn, batch_fn, output_fn, threads) ──

def load_kjarni():
    sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python"))
    import kjarni
    k = T["kind"]
    if k == "embed":
        m = kjarni.Embedder(model=T["kjarni"], quiet=True)
        return (lambda: m.encode(QUERY), lambda: m.encode_batch(DOCS),
                lambda: np.asarray(m.encode_batch(DOCS), dtype=np.float32))
    if k == "rerank":
        m = kjarni.Reranker(model=T["kjarni"], quiet=True)
        def scores():
            r = m.rerank(QUERY, DOCS)
            out = np.zeros(len(DOCS), dtype=np.float32)
            for x in r: out[x.index] = x.score
            return out
        return (lambda: m.score(QUERY, DOCS[0]), lambda: m.rerank(QUERY, DOCS), scores)
    m = kjarni.Classifier(model=T["kjarni"], quiet=True)
    def probs():
        rs = m.classify_batch(TEXTS)
        return np.asarray([[s for _, s in sorted(r.all_scores)] if hasattr(r, "all_scores") else [r.score] for r in rs], dtype=np.float32)
    return (lambda: m.classify(TEXTS[0]), lambda: m.classify_batch(TEXTS), probs)


def load_torch():
    import torch
    torch.set_grad_enabled(False)
    path = os.path.join(CACHE, T["local"])
    k = T["kind"]
    if k == "embed":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(path, device="cpu")
        return (lambda: m.encode(QUERY), lambda: m.encode(DOCS, batch_size=64),
                lambda: m.encode(DOCS, batch_size=64, normalize_embeddings=True).astype(np.float32))
    if k == "rerank":
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(path, device="cpu")
        pairs = [(QUERY, d) for d in DOCS]
        return (lambda: m.predict([(QUERY, DOCS[0])]), lambda: m.predict(pairs, batch_size=64),
                lambda: np.asarray(m.predict(pairs, batch_size=64), dtype=np.float32))
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(path)
    m = AutoModelForSequenceClassification.from_pretrained(path).eval()
    def run(texts):
        enc = tok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return torch.softmax(m(**enc).logits, -1)
    return (lambda: run([TEXTS[0]]), lambda: run(TEXTS), lambda: run(TEXTS).numpy().astype(np.float32))


def load_onnx():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    model = hf_hub_download(T["hf"], "onnx/model.onnx")
    # Some exports ship no tokenizer.json (distilbert-sst2); the model's own,
    # from Kjarni's cache, is the same tokenizer.
    local_tok = os.path.join(CACHE, T["local"], "tokenizer.json")
    tok = Tokenizer.from_file(local_tok if os.path.exists(local_tok) else hf_hub_download(T["hf"], "tokenizer.json"))
    tok.enable_truncation(512)
    tok.enable_padding()
    sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}

    def feed(encs):
        ids = np.array([e.ids for e in encs], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
        f = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            f["token_type_ids"] = np.array([e.type_ids for e in encs], dtype=np.int64)
        return f, mask

    k = T["kind"]
    if k == "embed":
        def emb(texts):
            f, mask = feed(tok.encode_batch(texts))
            h = sess.run(None, f)[0]
            m = mask[..., None].astype(np.float32)
            v = (h * m).sum(1) / m.sum(1)
            return v / np.linalg.norm(v, axis=1, keepdims=True)
        return (lambda: emb([QUERY]), lambda: emb(DOCS), lambda: emb(DOCS).astype(np.float32))
    if k == "rerank":
        def score(docs):
            f, _ = feed(tok.encode_batch([(QUERY, d) for d in docs]))
            return sess.run(None, f)[0][:, 0]
        return (lambda: score(DOCS[:1]), lambda: score(DOCS), lambda: score(DOCS).astype(np.float32))
    def cls(texts):
        f, _ = feed(tok.encode_batch(texts))
        z = sess.run(None, f)[0]
        e = np.exp(z - z.max(1, keepdims=True))
        return e / e.sum(1, keepdims=True)
    return (lambda: cls([TEXTS[0]]), lambda: cls(TEXTS), lambda: cls(TEXTS).astype(np.float32))


def threads():
    if ENGINE == "torch":
        import torch; return torch.get_num_threads()
    if ENGINE == "onnx":
        import psutil; return psutil.cpu_count(logical=False)  # ORT's default intra-op pool
    return int(os.environ.get("RAYON_NUM_THREADS") or os.cpu_count())  # rayon's default


t0 = time.perf_counter()
single, batch, output = {"kjarni": load_kjarni, "torch": load_torch, "onnx": load_onnx}[ENGINE]()
load_ms = (time.perf_counter() - t0) * 1000
first = time.perf_counter(); single(); first_ms = (time.perf_counter() - first) * 1000

res = dict(engine=ENGINE, task=TASK, load_ms=load_ms, first_call_ms=first_ms,
           single_ms=timed(single, RUNS_SINGLE), batch_ms=timed(batch, RUNS_BATCH),
           batch_size=len(DOCS), threads=threads())
res["batch_per_s"] = res["batch_size"] / (res["batch_ms"]["min"] / 1000)
res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
if OUT:
    np.save(OUT, output())
print(json.dumps(res))
