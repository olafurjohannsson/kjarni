import torch
import torch.nn as nn
import math

def print_rust(name, tensor):
    data = tensor.detach().numpy().flatten()
    print(f"\n// {name} Shape: {list(tensor.shape)}")
    print(f"let {name}_data = vec![")
    for i in range(0, len(data), 8):
        chunk = data[i:i+8]
        line = ", ".join(f"{x:.6f}" for x in chunk)
        print(f"    {line},")
    print("];")

def make_deterministic(model, seed=42):
    torch.manual_seed(seed)
    with torch.no_grad():
        count = 1
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                # Linear weights
                num = param.numel()
                vals = torch.arange(count, count + num).float() * 0.001
                param.copy_(vals.view(param.shape))
                count += num
            elif 'bias' in name:
                param.fill_(0.01)
            elif 'weight' in name and len(param.shape) == 1:
                # LayerNorm
                param.fill_(1.0)

# ==============================================================================
# MOCK BERT (Simplified for Golden generation)
# ==============================================================================
class MockBert(nn.Module):
    def __init__(self, vocab, hidden, layers):
        super().__init__()
        self.embeddings = nn.Embedding(vocab, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        # Simple Encoder Layer: FC -> Add -> Norm
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden)
            ) for _ in range(layers)
        ])
    
    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.ln1(x)
        for layer in self.layers:
            # Simple residual connection simulation
            # x + Linear(x) -> Norm
            linear, norm = layer[0], layer[1]
            out = linear(x)
            x = norm(x + out)
        return x

def run():
    torch.manual_seed(42)
    VOCAB = 20
    HIDDEN = 4
    LAYERS = 1
    
    model = MockBert(VOCAB, HIDDEN, LAYERS)
    make_deterministic(model)
    model.eval()

    # --- Inputs ---
    # Batch size 2, Max Seq 5
    # Seq 1: [CLS, A, B, C, SEP] -> Length 5
    # Seq 2: [CLS, D, SEP, PAD, PAD] -> Length 3
    input_ids = torch.tensor([
        [1, 5, 6, 7, 2],
        [1, 8, 2, 0, 0]
    ], dtype=torch.long)
    
    # Attention Mask (1=Real, 0=Pad)
    mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0]
    ], dtype=torch.float) # Float for math

    # 1. Hidden States
    with torch.no_grad():
        hidden = model(input_ids)
    
    print("=== ENCODER GOLDEN VALUES ===")
    
    # Dump Weights for Rust init
    print_rust("emb_weight", model.embeddings.weight)
    print_rust("emb_ln_w", model.ln1.weight)
    print_rust("emb_ln_b", model.ln1.bias)
    print_rust("l0_fc_w", model.layers[0][0].weight)
    print_rust("l0_fc_b", model.layers[0][0].bias)
    print_rust("l0_ln_w", model.layers[0][1].weight)
    print_rust("l0_ln_b", model.layers[0][1].bias)

    print_rust("hidden_states", hidden)

    # 2. Pooling Strategies
    
    # A. Mean Pooling (Sum / Count)
    # Mask hidden states first (multiply by 0 for pads)
    mask_expanded = mask.unsqueeze(-1) # [B, S, 1]
    masked_hidden = hidden * mask_expanded
    sum_hidden = torch.sum(masked_hidden, dim=1) # [B, H]
    sum_mask = torch.sum(mask_expanded, dim=1)   # [B, 1]
    pool_mean = sum_hidden / sum_mask
    print_rust("pool_mean", pool_mean)

    # B. CLS Pooling (Index 0)
    pool_cls = hidden[:, 0, :]
    print_rust("pool_cls", pool_cls)

    # C. Max Pooling (Max over seq where mask=1)
    # Set padded values to -inf before max
    neg_inf = torch.zeros_like(hidden) - 1e9
    # If mask is 1, keep hidden. If 0, use neg_inf.
    # Note: mask is [B, S]. hidden is [B, S, H].
    # We need to broadcast mask correctly.
    hidden_for_max = torch.where(mask_expanded.bool(), hidden, neg_inf)
    pool_max, _ = torch.max(hidden_for_max, dim=1)
    print_rust("pool_max", pool_max)

    # D. Last Token Pooling
    # Indices of last real token: [4, 2]
    # We gather from the sequence dim
    last_indices = mask.sum(dim=1).long() - 1 # [4, 2]
    # Gather: [B, H]
    pool_last = torch.stack([hidden[i, idx, :] for i, idx in enumerate(last_indices)])
    print_rust("pool_last", pool_last)

    # 3. Normalization (L2 Norm)
    # We will test normalization on the MEAN pooled output
    norm = torch.norm(pool_mean, p=2, dim=1, keepdim=True)
    normed_mean = pool_mean / norm
    print_rust("normed_mean", normed_mean)

if __name__ == "__main__":
    torch.set_printoptions(precision=6, sci_mode=False)
    run()