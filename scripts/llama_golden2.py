import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaRMSNorm,
    LlamaAttention,
    LlamaMLP,
    LlamaRotaryEmbedding,
)

def print_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    flat_tensor = tensor.flatten().tolist()
    print(f"Values: vec![")
    for i, val in enumerate(flat_tensor):
        print(f"    {val}f32,", end="")
        if (i + 1) % 4 == 0:
            print("")
    print("]")
    print(f"Mean: {tensor.float().mean().item()}")
    print("-" * (len(name) + 8))
    print("\n")

def generate_llama_layer_golden_values():
    print("=" * 50)
    print(" Generating Golden Values for a Full LlamaDecoderLayer (Manual) ")
    print("=" * 50)

    # --- Config ---
    config = LlamaConfig(
        hidden_size=64, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=128, max_position_embeddings=32,
        rms_norm_eps=1e-5,
    )
    config._attn_implementation = "eager"

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    # --- Components ---
    input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self_attn = LlamaAttention(config, layer_idx=0)
    mlp = LlamaMLP(config)
    post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    rotary_emb = LlamaRotaryEmbedding(config)

    # --- Inputs ---
    seq_len, position_offset = 8, 4
    initial_hidden_states = torch.randn(1, seq_len, config.hidden_size)
    attention_mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float32)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    attention_mask = attention_mask + causal_mask.unsqueeze(0).unsqueeze(0)
    position_ids = torch.arange(position_offset, position_offset + seq_len, dtype=torch.long).unsqueeze(0)
    
    # --- Execute and Capture Intermediate State ---
    
    # -- Attention Block --
    residual = initial_hidden_states
    normalized_states_attn = input_layernorm(initial_hidden_states)
    cos, sin = rotary_emb(normalized_states_attn, position_ids=position_ids)
    position_embeddings = (cos, sin)
    attn_output, _ = self_attn(
        hidden_states=normalized_states_attn,
        attention_mask=attention_mask,
        past_key_values=None,
        position_embeddings=position_embeddings,
    )
    # This is the intermediate value you need
    hidden_states_after_attn = residual + attn_output

    # -- FFN Block --
    residual = hidden_states_after_attn
    normalized_states_ffn = post_attention_layernorm(hidden_states_after_attn)
    ffn_output = mlp(normalized_states_ffn)
    final_output = residual + ffn_output

    # --- Print All Tensors ---
    print("\n\n--- (1) TENSOR FOR 'Llama Layer Input' ---")
    print_tensor(initial_hidden_states, "Llama Layer Input")

    print("\n\n--- (2) TENSOR FOR 'Attention Block Output' / 'FFN Block Input' ---")
    print_tensor(hidden_states_after_attn, "Intermediate State (after Attention)")

    print("\n\n--- (3) TENSOR FOR 'Llama Layer Golden Output' ---")
    print_tensor(final_output, "Llama Layer Golden Output")

    # --- Optional: Print weights if needed ---
    # print("\n\n--- WEIGHT TENSORS ---")
    # ...

if __name__ == "__main__":
    generate_llama_layer_golden_values()