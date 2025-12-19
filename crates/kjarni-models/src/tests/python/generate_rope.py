import torch

# RoPE implementation from Hugging Face for reference
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        return self.cos_cached[:seq_len, ...], self.sin_cached[:seq_len, ...]

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def print_tensor(tensor, name):
    print(f"--- {name} ---")
    print(f"Shape: {list(tensor.shape)}")
    flat_tensor = tensor.flatten().tolist()
    print(f"Values (first 8): {flat_tensor[:64]}")
    print(f"Mean: {tensor.float().mean().item()}")
    print("-" * (len(name) + 64))
    print("\n")

def generate_rope_golden_values():
    print("=" * 50)
    print(" Generating Golden Values for RoPE ")
    print("=" * 50)

    # --- Configuration ---
    batch_size = 1
    num_heads = 2
    seq_len = 4
    head_dim = 8
    position_offset = 10
    torch.manual_seed(42)

    # --- Create Model and Inputs ---
    rope_layer = LlamaRotaryEmbedding(dim=head_dim, max_position_embeddings=seq_len + position_offset)
    
    q = torch.randn(batch_size, num_heads, seq_len, head_dim)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # The position IDs are the absolute positions in the sequence
    position_ids = torch.arange(position_offset, position_offset + seq_len, dtype=torch.long).view(1, -1)

    # --- Execute ---
    cos, sin = rope_layer(q, seq_len=position_offset + seq_len)
    q_rotated, k_rotated = apply_rotary_pos_emb(q, k, cos, sin, position_ids)

    # --- Print Values ---
    print_tensor(q, "RoPE Input Q")
    print_tensor(k, "RoPE Input K")
    print(f"Position Offset: {position_offset}\n")
    print_tensor(q_rotated, "RoPE Golden Output Q")
    print_tensor(k_rotated, "RoPE Golden Output K")

if __name__ == "__main__":
    generate_rope_golden_values()