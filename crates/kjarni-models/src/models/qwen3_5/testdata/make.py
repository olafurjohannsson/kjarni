"""Writes a tiny random Qwen 3.5 text decoder and what transformers computes
with it, for tests.rs. Small enough to keep in the repository, and shaped to
reach what Qwen3.5-0.8B does not: value heads sharing key heads, as in the 4B
and larger, an untied LM head, as in the 9B and larger, and a text-only
checkpoint whose tensors sit under `model.`.

    python make.py   # in this folder, with torch, transformers 5.16 and tokenizers
"""
import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

torch.manual_seed(0)
config = Qwen3_5TextConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
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
    max_position_embeddings=512,
    tie_word_embeddings=False,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    eos_token_id=1,
)
model = Qwen3_5ForCausalLM(config).eval()
with torch.no_grad():
    for name, p in model.named_parameters():
        # Random everywhere, norms included: their initial zeros would hide the
        # 1 + weight scaling, and A_log near its init keeps the decay sane.
        if name.endswith("A_log"):
            p.uniform_(-1.0, 1.0)
        else:
            p.normal_(0.0, 0.3)
model.save_pretrained(".", safe_serialization=True)

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
# Longer than one 64-row block of the streaming attention.
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
