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

    # --- Use a small but realistic config ---
    config = LlamaConfig(
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA
        intermediate_size=128,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
    )
    
    # ✅ THE FINAL FIX: Manually set the attention implementation to prevent the KeyError.
    # This value is normally set when loading a pretrained model.
    config._attn_implementation = "eager"

    torch.manual_seed(42)
    torch.set_default_dtype(torch.float32)

    # --- Manually create the components of a DecoderLayer ---
    input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    self_attn = LlamaAttention(config, layer_idx=0)
    mlp = LlamaMLP(config)
    post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    rotary_emb = LlamaRotaryEmbedding(config)

    # --- Create Inputs ---
    batch_size = 1
    seq_len = 8
    position_offset = 4

    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    attention_mask = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=torch.float32)
    causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    attention_mask = attention_mask + causal_mask.unsqueeze(0).unsqueeze(0)

    position_ids = torch.arange(position_offset, position_offset + seq_len, dtype=torch.long).unsqueeze(0)
    
    # --- Manually Execute the Llama Decoder Layer Logic ---
    residual = hidden_states

    # 1. Pre-Normalization before Attention
    normalized_states = input_layernorm(hidden_states)

    # 2. Get the cos/sin caches from the RoPE module
    cos, sin = rotary_emb(normalized_states, position_ids=position_ids)
    position_embeddings = (cos, sin)

    # 3. Attention block
    # Note: LlamaAttention's forward signature in some versions does not include `use_cache`.
    # It is passed via `kwargs` to the underlying attention interface. We can remove it here for robustness.
    attn_output, _ = self_attn(
        hidden_states=normalized_states,
        attention_mask=attention_mask,
        # position_ids is not an explicit argument in some older versions, it's derived inside.
        # However, it is used by the LlamaDecoderLayer, so we keep it.
        past_key_values=None,
        position_embeddings=position_embeddings,
    )

    # 4. First residual connection
    hidden_states = residual + attn_output

    # 5. Pre-Normalization before FFN
    residual = hidden_states
    normalized_states = post_attention_layernorm(hidden_states)

    # 6. FFN block
    ffn_output = mlp(normalized_states)

    # 7. Second residual connection
    final_output = residual + ffn_output

    # --- Print Values ---
    print_tensor(hidden_states, "Llama Layer Input")
    print("\n\n--- WEIGHT TENSORS ---")
    # Note the .T to match ndarray's [in, out] convention vs PyTorch's [out, in]
    print_tensor(self_attn.q_proj.weight.T, "Q Proj Weight") 
    print_tensor(self_attn.k_proj.weight.T, "K Proj Weight")
    print_tensor(self_attn.v_proj.weight.T, "V Proj Weight")
    print_tensor(self_attn.o_proj.weight.T, "O Proj Weight")
    print_tensor(input_layernorm.weight, "Input LN Weight")
    print_tensor(post_attention_layernorm.weight, "Post Attn LN Weight")
    print_tensor(mlp.gate_proj.weight.T, "Gate Proj Weight")
    print_tensor(mlp.up_proj.weight.T, "Up Proj Weight")
    print_tensor(mlp.down_proj.weight.T, "Down Proj Weight")
    print_tensor(final_output, "Llama Layer Golden Output")

if __name__ == "__main__":
    generate_llama_layer_golden_values()