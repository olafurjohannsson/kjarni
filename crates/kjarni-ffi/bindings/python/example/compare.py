import time
import numpy as np
from kjarni import Embedder, Classifier, version
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# -------------------------------------------------
# TEST DATA (100+ non-trivial sentences)
# -------------------------------------------------

np.random.seed(42)

topics = [
    "machine learning", "air traffic control", "distributed systems",
    "natural language processing", "computer graphics",
    "financial regulation", "software architecture",
    "real-time systems", "data pipelines", "neural networks"
]

verbs = [
    "optimizes", "evaluates", "analyzes", "processes",
    "transforms", "validates", "predicts", "controls"
]

objects = [
    "large-scale datasets", "complex state transitions",
    "high-dimensional embeddings", "streaming telemetry",
    "regulatory constraints", "fault-tolerant pipelines",
    "user-generated content", "multimodal signals"
]

sentences = [
    f"The {np.random.choice(topics)} system {np.random.choice(verbs)} "
    f"{np.random.choice(objects)} under real-world conditions."
    for _ in range(120)
]

# -------------------------------------------------
# BENCHMARK SETTINGS
# -------------------------------------------------

RUNS = 100
TEXTS = sentences

def benchmark(fn):
    times = []
    for _ in range(RUNS):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    return np.mean(times), np.std(times)

# -------------------------------------------------
# KJARNI (BATCH)
# -------------------------------------------------

print("\n--- Kjarni Embedder (batch) ---")
kjarni = Embedder(quiet=True, model="minilm-l6-v2")

# Warm-up
_ = kjarni.encode_batch(TEXTS)

def kjarni_batch():
    _ = kjarni.encode_batch(TEXTS)

k_mean, k_std = benchmark(kjarni_batch)

k_embedding = np.array(kjarni.encode("Hello, world!"), dtype=np.float32)
print(f"Kjarni dim: {len(k_embedding)}")
print(f"Kjarni first 5: {k_embedding[:5]}")

k_sim = kjarni.similarity("cat", "dog")
print(f"Kjarni similarity(cat, dog): {k_sim:.6f}")

# -------------------------------------------------
# SENTENCE TRANSFORMERS (BATCH)
# -------------------------------------------------

print("\n--- SentenceTransformers (batch) ---")
model_name = "sentence-transformers/all-MiniLM-L6-v2"
st = SentenceTransformer(model_name)

# Warm-up
_ = st.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=False)

def st_batch():
    _ = st.encode(TEXTS, convert_to_numpy=True, normalize_embeddings=False)

st_mean, st_std = benchmark(st_batch)

st_embedding = st.encode("Hello, world!", convert_to_numpy=True)
print(f"ST dim: {len(st_embedding)}")
print(f"ST first 5: {st_embedding[:5]}")

st_cat = st.encode("cat", convert_to_numpy=True)
st_dog = st.encode("dog", convert_to_numpy=True)
st_sim = cos_sim(st_cat, st_dog).item()
print(f"ST similarity(cat, dog): {st_sim:.6f}")


# -------------------------------------------------
# RESULTS
# -------------------------------------------------

print("\n--- Benchmark Results (100 runs, batch size = 120) ---")
print(f"Kjarni batch encode: {k_mean * 1000:.2f} ms ± {k_std * 1000:.2f} ms")
print(f"ST batch encode:     {st_mean * 1000:.2f} ms ± {st_std * 1000:.2f} ms")

speedup = st_mean / k_mean if k_mean > 0 else float("inf")
print(f"\nRelative speed (ST / Kjarni): {speedup:.2f}x")