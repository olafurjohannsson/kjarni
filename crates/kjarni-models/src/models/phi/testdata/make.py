"""Writes a tiny random Phi-3 decoder and what transformers computes with it,
for tests.rs. Shaped like Phi-3.5-mini where it matters: fused qkv and
gate_up projections, and LongRoPE with separate short and long factor lists.
Its trained context is 64 positions, so a short prompt stays inside it and a
long one reaches past it, as a prompt past 4096 tokens does on Phi-3.5.

    python make.py   # in this folder, with torch, transformers 4.57 and tokenizers
"""
import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import Phi3Config, Phi3ForCausalLM

torch.manual_seed(0)
head_dim = 16
config = Phi3Config(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=4,
    max_position_embeddings=512,
    original_max_position_embeddings=64,
    rope_theta=10000.0,
    rope_scaling={
        "type": "longrope",
        "short_factor": [1.0 + 0.05 * i for i in range(head_dim // 2)],
        "long_factor": [1.5 + 0.5 * i for i in range(head_dim // 2)],
    },
    sliding_window=None,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
    pad_token_id=0,
)
model = Phi3ForCausalLM(config).eval()
with torch.no_grad():
    for p in model.parameters():
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
        g = model.generate(torch.tensor([ids]), max_new_tokens=n, do_sample=False,
                           eos_token_id=None, pad_token_id=0)
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
