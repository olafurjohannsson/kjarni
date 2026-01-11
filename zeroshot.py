import torch
import math
from transformers import BartForSequenceClassification, BartTokenizer

def main():
    model_name = "facebook/bart-large-mnli"
    print(f"Loading {model_name}...")
    model = BartForSequenceClassification.from_pretrained(model_name)
    model.eval()

    # Constants for BART-Large
    d_model = 1024
    eos_token_id = 2
    
    print("\n--- DEBUGGING DECODER EMBEDDINGS ---")
    
    # 1. Create Input (EOS Token)
    input_ids = torch.tensor([[eos_token_id]], dtype=torch.long)
    print(f"Decoder Input ID: {input_ids.tolist()}")

    with torch.no_grad():
        decoder = model.model.decoder
        
        # 2. Raw Word Embedding
        # Note: HF BartScaledWordEmbedding handles scaling internally if config is set
        # But we want to see the raw weights to compare with Rust
        raw_word_weight = decoder.embed_tokens.weight[eos_token_id]
        print(f"[1] Raw Word Weight [EOS] (first 5): {raw_word_weight[:5].tolist()}")
        
        # 3. Scaled Word Embedding
        # This is what 'embed_tokens(input_ids)' returns
        word_emb = decoder.embed_tokens(input_ids)
        print(f"[2] Scaled Word Embedding (first 5): {word_emb[0, 0, :5].tolist()}")
        
        # Calculate expected scale factor
        expected_scale = math.sqrt(d_model)
        observed_scale = (word_emb[0, 0] / raw_word_weight).mean().item()
        print(f"    -> Observed Scale Factor: {observed_scale:.4f} (Expected: {expected_scale:.4f})")

        # 4. Position Embedding
        # BART uses learned positions. offset=2.
        # Input length 1 -> Position ID 0 -> Lookup Index 2
        pos_emb = decoder.embed_positions(input_ids, past_key_values_length=0)
        print(f"[3] Position Embedding (first 5): {pos_emb[0, 0, :5].tolist()}")
        
        # 5. Sum + LayerNorm (Input to Layer 0)
        hidden = word_emb + pos_emb
        hidden = decoder.layernorm_embedding(hidden)
        
        print(f"[4] Decoder Initial State (Pre-Layer 0) (first 10):")
        print(f"{hidden[0, 0, :10].tolist()}")

if __name__ == "__main__":
    main()