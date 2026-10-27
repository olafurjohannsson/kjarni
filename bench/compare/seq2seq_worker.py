"""Times summarization (distilbart-cnn) or translation (flan-t5-base) on the CPU,
in a fresh process, and records the output so the engines can be checked to have
generated the same text. Prints JSON.

    python seq2seq_worker.py ENGINE TASK      ENGINE: kjarni | torch    TASK: summarize | translate
"""
import json, os, sys, time, resource

ENGINE, TASK = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
CACHE = os.path.expanduser("~/.cache/kjarni")
sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python/tests"))
from test_generation import ARTICLE  # the same article the binding tests use

SENTENCE = "The house is small, but the garden behind it is large and full of old apple trees."
RUNS = 3

t0 = time.perf_counter()
if ENGINE == "kjarni":
    sys.path.insert(0, os.path.join(REPO, "crates/kjarni-ffi/bindings/python"))
    import kjarni
    if TASK == "summarize":
        # The same lengths as PyTorch is given below; beams and the rest come
        # from the checkpoint in both.
        m = kjarni.Summarizer(min_length=56, max_length=142, quiet=True); run = lambda: m.summarize(ARTICLE)
    else:
        m = kjarni.Translator(source="en", target="de", quiet=True); run = lambda: m.translate(SENTENCE)
else:
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    torch.set_grad_enabled(False)
    path = os.path.join(CACHE, "olafuraron_distilbart-cnn-12-6" if TASK == "summarize" else "google_flan-t5-base")
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path).eval()
    text = ARTICLE if TASK == "summarize" else f"translate English to German: {SENTENCE}"
    # The checkpoint's own settings, spelled out. distilbart-cnn keeps them in
    # config.json and leaves generation_config.json null, so a bare generate()
    # falls back to greedy and 20 tokens: a truncated summary, and a timing
    # that looked several times faster than the real thing.
    gen = (dict(num_beams=4, min_length=56, max_length=142, length_penalty=2.0, no_repeat_ngram_size=3,
                early_stopping=True) if TASK == "summarize" else dict(num_beams=1, do_sample=False, max_new_tokens=128))
    def run():
        enc = tok(text, return_tensors="pt", truncation=True)
        out = model.generate(**enc, **gen)
        return tok.decode(out[0], skip_special_tokens=True)
load_ms = (time.perf_counter() - t0) * 1000

output = run()
times = []
for _ in range(RUNS):
    s = time.perf_counter(); run(); times.append(time.perf_counter() - s)
print(json.dumps(dict(engine=ENGINE, task=TASK, load_ms=load_ms, ms=min(times) * 1000, output=output.strip(),
                      peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)))
