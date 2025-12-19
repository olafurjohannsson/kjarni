import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaMLP

def print_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    flat_tensor = tensor.flatten().tolist()
    print(f"Values (first 8): {flat_tensor[:8]}")
    print(f"Mean: {tensor.float().mean().item()}")
    print("-" * (len(name) + 8))
    print("\n")

def generate_swiglu_golden_values():
    print("=" * 50)
    print(" Generating Golden Values for SwiGLU FFN ")
    print("=" * 50)

    # --- Mock Config for LlamaMLP ---
    class MockConfig:
        def __init__(self):
            self.hidden_size = 8
            self.intermediate_size = 16 # Must be multiple of 2
            self.hidden_act = "silu"

    torch.manual_seed(42)
    config = MockConfig()
    
    # --- Create Model and Inputs ---
    swiglu_layer = LlamaMLP(config)
    input_tensor = torch.randn(1, 4, config.hidden_size) # [batch, seq_len, hidden_size]

    # --- Execute ---
    output_tensor, _ = swiglu_layer(input_tensor)

    # --- Print Values ---
    print_tensor(input_tensor, "SwiGLU Input")
    print_tensor(swiglu_layer.gate_proj.weight, "SwiGLU Gate Proj Weight")
    print_tensor(swiglu_layer.up_proj.weight, "SwiGLU Up Proj Weight")
    print_tensor(swiglu_layer.down_proj.weight, "SwiGLU Down Proj Weight")
    print_tensor(output_tensor, "SwiGLU Golden Output")

if __name__ == "__main__":
    generate_swiglu_golden_values()