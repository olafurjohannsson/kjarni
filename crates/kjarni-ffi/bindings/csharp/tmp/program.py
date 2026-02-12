from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
vector = model.encode("Hello world", normalize_embeddings=True)
print(vector[:5])