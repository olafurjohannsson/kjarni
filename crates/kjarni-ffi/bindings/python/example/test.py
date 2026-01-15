from kjarni import Embedder, Classifier, version
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kjarni import Embedder, Classifier, version


def cos_sim(a, b):
    return float(cosine_similarity([a], [b])[0][0])


print(f'Kjarni version: {version()}')

# -------------------------------------------------
# KJARNI
# -------------------------------------------------
print('\n--- Kjarni Embedder ---')
kjarni = Embedder(quiet=True, model='minilm-l6-v2')

text = "Hello, world!"
k_embedding = np.array(kjarni.encode(text), dtype=np.float32)

print(f'Kjarni dim: {len(k_embedding)}')
print(f'Kjarni first 5: {k_embedding[:5]}')

k_sim = kjarni.similarity("cat", "dog")
print(f'Kjarni similarity(cat, dog): {k_sim:.6f}')

# -------------------------------------------------
# SENTENCE TRANSFORMERS
# -------------------------------------------------
print('\n--- SentenceTransformers ---')
model_name = "sentence-transformers/all-MiniLM-L6-v2"

st = SentenceTransformer(model_name)

st_embedding = st.encode(text, convert_to_numpy=True, normalize_embeddings=False)

print(f'ST dim: {len(st_embedding)}')
print(f'ST first 5: {st_embedding[:5]}')

st_cat = st.encode("cat", convert_to_numpy=True)
st_dog = st.encode("dog", convert_to_numpy=True)
st_sim = cos_sim(st_cat, st_dog)

print(f'ST similarity(cat, dog): {st_sim:.6f}')

# -------------------------------------------------
# COMPARISON
# -------------------------------------------------
print('\n--- Comparison ---')

if len(k_embedding) == len(st_embedding):
    diff_l2 = np.linalg.norm(k_embedding - st_embedding)
    diff_cos = cos_sim(k_embedding, st_embedding)

    print(f'L2 distance (Hello, world!): {diff_l2:.6f}')
    print(f'Cosine similarity (Hello, world!): {diff_cos:.6f}')
else:
    print('Embedding dimensions differ – models are not identical')

# -------------------------------------------------
# Classifier sanity check
# -------------------------------------------------
print('\n--- Classifier ---')
classifier = Classifier('distilroberta-emotion', quiet=True)
result = classifier.classify('I love this product!')
print(f'Classification: {result}')

print('\nAll tests completed.')
