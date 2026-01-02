import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

# Setup
model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

# Input
sentences = [
    "The cat sits on the mat",
    "A feline rests on a rug",
    "Dogs are playing in the park",
]

# Tokenize
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Inference
with torch.no_grad():
    model_output = model(**encoded_input)

# Hidden States [Batch, Seq, Dim]
token_embeddings = model_output.last_hidden_state
attention_mask = encoded_input['attention_mask']

# --- Pooling Implementations ---

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def max_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # Set padded tokens to large negative value so they aren't picked as max
    token_embeddings[input_mask_expanded == 0] = -1e9
    return torch.max(token_embeddings, 1)[0]

def cls_pooling(token_embeddings):
    return token_embeddings[:, 0]

def last_token_pooling(token_embeddings, attention_mask):
    # Find the index of the last non-padding token
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = token_embeddings.shape[0]
    # Gather the embedding at that index
    return token_embeddings[torch.arange(batch_size), sequence_lengths]

def normalize(embeddings):
    return F.normalize(embeddings, p=2, dim=1)

def fmt_vec(tensor, label):
    # Format the FIRST vector in the batch for Rust copy-paste
    vec = tensor[0].tolist()
    print(f"// {label} (First sentence)")
    print(f"const GOLDEN_{label.upper()}: &[f32] = &[")
    for v in vec:
        print(f"    {v:.8f},")
    print("];\n")

# --- Generate Golden Values ---

# 1. Mean + Normalize (Standard Usage)
mean_emb = mean_pooling(token_embeddings, attention_mask)
mean_norm = normalize(mean_emb)
fmt_vec(mean_norm, "MEAN_NORM")

# 2. Raw Mean (No Normalize)
fmt_vec(mean_emb, "MEAN_RAW")

# 3. CLS + Normalize
cls_emb = cls_pooling(token_embeddings)
cls_norm = normalize(cls_emb)
fmt_vec(cls_norm, "CLS_NORM")

# 4. Max + Normalize
max_emb = max_pooling(token_embeddings, attention_mask)
max_norm = normalize(max_emb)
fmt_vec(max_norm, "MAX_NORM")

# 5. Last Token + Normalize
last_emb = last_token_pooling(token_embeddings, attention_mask)
last_norm = normalize(last_emb)
fmt_vec(last_norm, "LAST_NORM")