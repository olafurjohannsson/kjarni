import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention

def print_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    flat_tensor = tensor.flatten().tolist()
    print(f"Values (first 8): {flat_tensor[:8]}")
    print(f"Mean: {tensor.float().mean().item()}")
    print("-" * (len(name) + 8))
    print("\n")

def generate_gqa_golden_values():
    print("=" * 50)
    print(" Generating Golden Values for Attention with GQA ")
    print("=" * 50)

    # --- Mock Config for LlamaAttention ---
    class MockConfig:
        def __init__(self):
            self.hidden_size = 16
            self.num_attention_heads = 4  # Query heads
            self.num_key_value_heads = 2  # GQA: 2 KV heads for 4 Q heads
            self.max_position_embeddings = 32
            self.rope_theta = 10000.0

    torch.manual_seed(42)
    config = MockConfig()
    
    # --- Create Model and Inputs ---
    # use_cache=True is essential as it makes the model return present_key_value
    gqa_layer = LlamaAttention(config, layer_idx=0)
    
    batch_size = 1
    prompt_len = 5
    
    input_tensor = torch.randn(batch_size, prompt_len, config.hidden_size)
    
    # Create an attention mask (1 = attend, 0 = ignore)
    attention_mask = torch.ones(batch_size, 1, prompt_len, prompt_len, dtype=torch.bool)
    # Apply causal masking
    attention_mask = torch.tril(attention_mask)
    
    position_ids = torch.arange(0, prompt_len, dtype=torch.long).unsqueeze(0)

    # --- Execute ---
    # The output tuple contains: (attn_output, attn_weights, present_key_value)
    output_tensor, _, present_key_value = gqa_layer(
        hidden_states=input_tensor,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True
    )
    
    present_key, present_value = present_key_value

    # --- Print Values ---
    print_tensor(input_tensor, "GQA Input")
    print_tensor(gqa_layer.q_proj.weight, "GQA Q Proj Weight")
    print_tensor(gqa_layer.k_proj.weight, "GQA K Proj Weight")
    print_tensor(gqa_layer.v_proj.weight, "GQA V Proj Weight")
    print_tensor(gqa_layer.o_proj.weight, "GQA O Proj Weight")
    print_tensor(output_tensor, "GQA Golden Output (Hidden State)")
    print_tensor(present_key, "GQA Golden Output (New Key Cache)")
    print_tensor(present_value, "GQA Golden Output (New Value Cache)")

if __name__ == "__main__":
    generate_gqa_golden_values()```