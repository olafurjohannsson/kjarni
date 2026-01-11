from sentence_transformers.cross_encoder import CrossEncoder

# Load the model
model = CrossEncoder(
    '/home/olafurj/.cache/kjarni/cross-encoder_ms-marco-MiniLM-L-6-v2/', device='cpu')

# === Example 1: Score a single pair ===
query1 = "How do I train a neural network?"
doc1 = "Neural networks are trained using backpropagation and gradient descent."
score1 = model.predict([(query1, doc1)])
print(f"Relevance score (Python): {score1[0]:.4f}\n")


# === Example 2: Rerank search results ===
query2 = "machine learning algorithms"
documents = [
    "Machine learning algorithms include decision trees, neural networks, and SVMs.",
    "The weather forecast predicts rain tomorrow.",
    "Deep learning is a subset of machine learning using neural networks.",
    "Cooking recipes for Italian pasta dishes.",
]

# Create pairs for batch prediction
pairs = [(query2, doc) for doc in documents]
scores = model.predict(pairs)

# Combine docs and scores and sort
results = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

print("Reranked results (Python):")
for doc, score in results:
    print(f"[Score: {score:.4f}] {doc}")