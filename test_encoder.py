import torch
import torch.nn.functional as F
import math

def gelu_new_manual(x):
    """
    Manual implementation of GELU (tanh approximation) for older PyTorch versions.
    Formula: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def generate_golden():
    print("Generating Golden Values for SwiGluFeedForward...")
    
    # 1. Setup Data
    # Batch=1, Seq=2, In=2, Hidden=2, Out=2
    
    # Input: [1, 2, 2]
    x = torch.tensor([[[0.5, -0.5], [0.1, 0.2]]])
    
    # Weights (Out, In)
    w_gate = torch.tensor([
        [0.2, -0.1], 
        [0.3,  0.4]
    ])
    
    w_up = torch.tensor([
        [0.5,  0.1], 
        [-0.2, 0.3]
    ])
    
    w_down = torch.tensor([
        [0.1,  0.2],
        [-0.1, 0.1]
    ])
    
    print("\n=== RUST SETUP DATA ===")
    print(f"// Input [1, 2, 2]")
    print(f"let input_data = vec!{x.flatten().tolist()};")
    print(f"// Gate Weights [2, 2]")
    print(f"let w_gate_data = vec!{w_gate.flatten().tolist()};")
    print(f"// Up Weights [2, 2]")
    print(f"let w_up_data = vec!{w_up.flatten().tolist()};")
    print(f"// Down Weights [2, 2]")
    print(f"let w_down_data = vec!{w_down.flatten().tolist()};")

    # 2. Define Activations
    activations = {
        "Relu": lambda x: F.relu(x),
        "Gelu": lambda x: F.gelu(x), # Standard Exact GELU
        "GeluNew": lambda x: gelu_new_manual(x), # Manual implementation for compatibility
        "SilU": lambda x: F.silu(x),
        "Tanh": lambda x: torch.tanh(x),
    }

    # 3. Compute and Print
    print("\n=== RUST GOLDEN VALUES ===")
    
    for name, act_fn in activations.items():
        # A. Gate Projection
        gate_out = F.linear(x, w_gate)
        
        # B. Up Projection
        up_out = F.linear(x, w_up)
        
        # C. Activation
        activated_gate = act_fn(gate_out)
        
        # D. Element-wise Multiply
        inter = activated_gate * up_out
        
        # E. Down Projection
        out = F.linear(inter, w_down)
        
        flat_out = out.flatten().tolist()
        formatted_out = [float(f"{v:.6f}") for v in flat_out]
        
        print(f"\n// Activation: {name}")
        print(f"// Expected Output")
        print(f"let expected_{name.lower()} = vec!{formatted_out};")

if __name__ == "__main__":
    generate_golden()