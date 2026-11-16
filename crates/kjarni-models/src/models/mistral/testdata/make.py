"""Writes a tiny random Mistral decoder and what transformers computes with it,
for tests.rs. Shaped like Mistral-7B-v0.3: grouped-query attention, an untied
LM head, and no sliding window.

    python make.py   # in this folder, with torch, transformers 4.57 and tokenizers
"""
import json

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import MistralConfig, MistralForCausalLM

torch.manual_seed(0)
config = MistralConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=512,
    rope_theta=1000000.0,
    sliding_window=None,
    tie_word_embeddings=False,
    bos_token_id=1,
    eos_token_id=2,
)
model = MistralForCausalLM(config).eval()
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
    # What `kjarni generate` does with the short prompt as text: it starts with
    # the start-of-text token from the config.
    "bos_short_greedy": greedy([config.bos_token_id] + short, 12),
}
with open("expected.json", "w") as f:
    json.dump(expected, f)
