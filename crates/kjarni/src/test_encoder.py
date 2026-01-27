import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "olafuraron/emotion-english-distilroberta-base-safetensors"
sad_text = "That was a truly disappointing and sad experience."

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

encoded = tokenizer(sad_text, return_tensors="pt")
print(f"Sad text: {sad_text}")
print(f"Input IDs: {encoded['input_ids'][0].tolist()}")

with torch.no_grad():
    # Step 1: Word embeddings only
    word_emb = model.roberta.embeddings.word_embeddings(encoded['input_ids'])
    
    # Step 2: Add position embeddings
    seq_length = encoded['input_ids'].size(1)
    position_ids = torch.arange(seq_length, dtype=torch.long).unsqueeze(0)
    pos_emb = model.roberta.embeddings.position_embeddings(position_ids)
    embeddings = word_emb + pos_emb
    
    # Step 3: Add token type embeddings
    token_type_ids = torch.zeros_like(encoded['input_ids'])
    token_type_emb = model.roberta.embeddings.token_type_embeddings(token_type_ids)
    embeddings = embeddings + token_type_emb
    
    print(f"\nAfter all embeddings (before LayerNorm):")
    print(f"  First token (first 5): {embeddings[0, 0, :5].tolist()}")
    print(f"  Mean: {embeddings.mean().item():.8f}")
    
    # Step 4: LayerNorm
    embeddings = model.roberta.embeddings.LayerNorm(embeddings)
    print(f"\nAfter LayerNorm:")
    print(f"  First token (first 5): {embeddings[0, 0, :5].tolist()}")
    print(f"  Mean: {embeddings.mean().item():.8f}")
    
    # Step 5: Through encoder
    extended_attention_mask = model.roberta.get_extended_attention_mask(
        encoded['attention_mask'], encoded['input_ids'].shape
    )
    
    hidden_states = embeddings
    for layer in model.roberta.encoder.layer:
        hidden_states = layer(hidden_states, extended_attention_mask)[0]
    
    print(f"\nAfter encoder:")
    print(f"  CLS token (first 5): {hidden_states[0, 0, :5].tolist()}")
    
    # Step 6: Classification head
    logits = model.classifier(hidden_states)
    print(f"\nLogits: {logits[0].tolist()}")