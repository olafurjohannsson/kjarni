"""The Python binding. Prints one line of JSON."""
import json, os, statistics, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../crates/kjarni-ffi/bindings/python"))
import kjarni

WARMUP, CALLS = 200, 3000
TEXT = "how do I get a refund for a damaged order?"
embedder = kjarni.Embedder(model="minilm-l6-v2", quiet=True)
for _ in range(WARMUP):
    embedder.encode(TEXT)
times = []
for _ in range(CALLS):
    start = time.perf_counter()
    embedder.encode(TEXT)
    times.append((time.perf_counter() - start) * 1e6)
print(json.dumps(dict(lang="Python", median_us=statistics.median(times), min_us=min(times))))
