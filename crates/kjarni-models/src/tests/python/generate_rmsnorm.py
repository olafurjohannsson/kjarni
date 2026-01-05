import torch
import torch.nn as nn

# LLaMA's RMSNorm implementation from Hugging Face for reference
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def print_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    # Flatten and print the first 8 values for easy comparison
    flat_tensor = tensor.flatten().tolist()
    print(f"Values (first 8): {flat_tensor[:8]}")
    print(f"Mean: {tensor.float().mean().item()}")
    print("-" * (len(name) + 8))
    print("\n")


def generate_rmsnorm_golden_values():
    print("=" * 50)
    print(" Generating Golden Values for RMSNorm ")
    print("=" * 50)

    # --- Configuration ---
    hidden_size = 8
    eps = 1e-5
    torch.manual_seed(42) # For reproducibility

    # --- Create Model and Inputs ---
    rmsnorm_layer = LlamaRMSNorm(hidden_size, eps=eps)
    
    # Manually set a predictable weight for the gamma parameter
    gamma_weight = torch.arange(0.5, hidden_size/2 + 0.5, 0.5, dtype=torch.float32)
    rmsnorm_layer.weight.data = gamma_weight
    
    input_tensor = torch.randn(1, 1, hidden_size) # [batch, seq_len, hidden_size]
    
    # --- Execute ---
    output_tensor = rmsnorm_layer(input_tensor)

    # --- Print Values ---
    print_tensor(input_tensor, "RMSNorm Input")
    print_tensor(rmsnorm_layer.weight, "RMSNorm Gamma (Weight)")
    print_tensor(output_tensor, "RMSNorm Golden Output")
    
if __name__ == "__main__":
    generate_rmsnorm_golden_values()