"""Writes a tiny random Qwen 3.5 mixture-of-experts text decoder and what
transformers computes with it, for tests.rs. Shaped as Qwen3.5-35B-A3B is:
every layer routes each token to its top experts and adds a shared expert
behind a sigmoid gate. Layers 0 and 1 store one tensor per expert, as
transformers saves them; layers 2 and 3 store them fused, as Qwen's own
checkpoints do: `experts.gate_up_proj` as [experts, 2 x moe_intermediate,
hidden] and `experts.down_proj` as [experts, hidden, moe_intermediate].

    python make.py   # in this folder, with torch, transformers 5.3 and tokenizers
"""
import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

torch.manual_seed(0)
config = Qwen3_5MoeTextConfig(
    vocab_size=256,
    hidden_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_conv_kernel_dim=4,
    layer_types=["linear_attention", "linear_attention", "full_attention", "linear_attention"],
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=48,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    eos_token_id=1,
)
model = Qwen3_5MoeForCausalLM(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        if name.endswith("A_log"):
            p.uniform_(-1.0, 1.0)
        else:
            p.normal_(0.0, 0.3)
model.save_pretrained(".", safe_serialization=True)

# transformers writes one tensor per expert; Qwen's own checkpoints fuse them.
# Layers 2 and 3 are fused here, so the file holds both forms.
from safetensors.torch import load_file, save_file
state = load_file("model.safetensors")
n, prefix = config.num_experts, "model.language_model.layers"
for layer in (2, 3):
    experts = f"{prefix}.{layer}.mlp.experts"
    take = lambda part: [state.pop(f"{experts}.{e}.{part}.weight") for e in range(n)]
    gate, up, down = take("gate_proj"), take("up_proj"), take("down_proj")
    state[f"{experts}.gate_up_proj"] = torch.stack([torch.cat([g, u]) for g, u in zip(gate, up)])
    state[f"{experts}.down_proj"] = torch.stack(down)
save_file(state, "model.safetensors", metadata={"format": "pt"})

vocab = {f"t{i}": i for i in range(256)}
tok = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
tok.save("tokenizer.json")

def logits(ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0]

def greedy(ids, n):
    with torch.no_grad():
        g = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False, eos_token_id=None)
    return g[0, len(ids):].tolist()

short = [5, 17, 99, 3, 42, 200, 7, 8]
long = [(i * 37 + 11) % 256 for i in range(90)]
expected = {
    "short_ids": short,
    "short_logits": logits(short).tolist(),
    "short_greedy": greedy(short, 12),
    "long_ids": long,
    "long_last_logits": logits(long)[-1].tolist(),
    "long_greedy": greedy(long, 12),
}
with open("expected.json", "w") as f:
    json.dump(expected, f)
