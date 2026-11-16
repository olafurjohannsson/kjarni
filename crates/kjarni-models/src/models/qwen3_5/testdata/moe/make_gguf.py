"""Writes tiny.gguf: the checkpoint make.py writes, laid out as llama.cpp's
convert_hf_to_gguf.py lays out Qwen 3.5 MoE, for tests.rs. The DeltaNet and
attention tensors change as ../make_gguf.py describes; the experts become
ffn_gate_exps and ffn_up_exps of [experts, moe_intermediate, hidden] and
ffn_down_exps of [experts, hidden, moe_intermediate], the router
ffn_gate_inp, the shared expert ffn_*_shexp, and its gate the vector
ffn_gate_inp_shexp.

The GGUF is written here directly, f32 tensors only, so nothing beyond numpy
is needed.

    python make_gguf.py
"""
import json
import struct

import numpy as np

with open("config.json") as f:
    cfg = json.load(f)


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    out = {}
    for name, info in header.items():
        if name == "__metadata__":
            continue
        assert info["dtype"] == "F32", name
        a, b = info["data_offsets"]
        out[name] = np.frombuffer(data[a:b], dtype=np.float32).reshape(info["shape"]).copy()
    return out


nk, nv = cfg["linear_num_key_heads"], cfg["linear_num_value_heads"]
dk, dv = cfg["linear_key_head_dim"], cfg["linear_value_head_dim"]
per_k = nv // nk
n_experts, size = cfg["num_experts"], cfg["moe_intermediate_size"]


def tiled(x, axis, head_dim):
    """Grouped to tiled value-head order along `axis` (the converter's _reorder_v_heads)."""
    x = np.moveaxis(x, axis, 0)
    shape = x.shape
    x = x.reshape(nk, per_k, head_dim, *shape[1:]).swapaxes(0, 1).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))


names = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight",
    "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight",
    "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.A_log": "ssm_a",
    "linear_attn.dt_bias": "ssm_dt.bias",
    "linear_attn.norm.weight": "ssm_norm.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.shared_expert.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_expert.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_expert.down_proj.weight": "ffn_down_shexp.weight",
    "mlp.shared_expert_gate.weight": "ffn_gate_inp_shexp.weight",
}

hf = read_safetensors("model.safetensors")
prefix = "model.language_model."
tensors = {}
experts = {}  # (layer, part) -> {expert: matrix}
for name, t in hf.items():
    if name == "lm_head.weight":
        tensors["output.weight"] = t
        continue
    name = name[len(prefix):]
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        t = t + 1
    if name == "embed_tokens.weight":
        tensors["token_embd.weight"] = t
        continue
    if name == "norm.weight":
        tensors["output_norm.weight"] = t
        continue
    _, layer, suffix = name.split(".", 2)
    if suffix.startswith("mlp.experts."):
        rest = suffix[len("mlp.experts."):]
        if rest == "gate_up_proj":
            for e in range(n_experts):
                experts.setdefault((layer, "gate"), {})[e] = t[e, :size]
                experts.setdefault((layer, "up"), {})[e] = t[e, size:]
        elif rest == "down_proj":
            for e in range(n_experts):
                experts.setdefault((layer, "down"), {})[e] = t[e]
        else:
            e, part, _ = rest.split(".")
            experts.setdefault((layer, part.removesuffix("_proj")), {})[int(e)] = t
        continue
    if suffix == "mlp.shared_expert_gate.weight":
        t = t[0]
    if suffix == "linear_attn.A_log":
        t = -np.exp(t)
    if suffix == "linear_attn.conv1d.weight":
        t = t.squeeze(1)
    if suffix.startswith("linear_attn.") and nv != nk:
        if suffix == "linear_attn.in_proj_qkv.weight":
            t = np.concatenate([t[: 2 * nk * dk], tiled(t[2 * nk * dk :], 0, dv)])
        elif suffix == "linear_attn.in_proj_z.weight":
            t = tiled(t, 0, dv)
        elif suffix in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
            t = tiled(t, 0, 1)
        elif suffix in ("linear_attn.A_log", "linear_attn.dt_bias"):
            t = tiled(t[:, None], 0, 1)[:, 0]
        elif suffix == "linear_attn.conv1d.weight":
            t = np.concatenate([t[: 2 * nk * dk], tiled(t[2 * nk * dk :], 0, dv)])
        elif suffix == "linear_attn.out_proj.weight":
            t = tiled(t, 1, dv)
    tensors[f"blk.{layer}.{names[suffix]}"] = t
for (layer, part), by_expert in experts.items():
    tensors[f"blk.{layer}.ffn_{part}_exps.weight"] = np.stack([by_expert[e] for e in range(n_experts)])

arch = "qwen35moe"
meta = [
    ("general.architecture", "str", arch),
    (f"{arch}.block_count", "u32", cfg["num_hidden_layers"]),
    (f"{arch}.context_length", "u32", cfg["max_position_embeddings"]),
    (f"{arch}.embedding_length", "u32", cfg["hidden_size"]),
    (f"{arch}.attention.head_count", "u32", cfg["num_attention_heads"]),
    (f"{arch}.attention.head_count_kv", "u32", cfg["num_key_value_heads"]),
    (f"{arch}.attention.key_length", "u32", cfg["head_dim"]),
    (f"{arch}.attention.value_length", "u32", cfg["head_dim"]),
    (f"{arch}.attention.layer_norm_rms_epsilon", "f32", cfg["rms_norm_eps"]),
    (f"{arch}.rope.freq_base", "f32", cfg["rope_parameters"]["rope_theta"]),
    (f"{arch}.rope.dimension_count", "u32", int(cfg["head_dim"] * cfg["partial_rotary_factor"])),
    (f"{arch}.ssm.conv_kernel", "u32", cfg["linear_conv_kernel_dim"]),
    (f"{arch}.ssm.state_size", "u32", dk),
    (f"{arch}.ssm.group_count", "u32", nk),
    (f"{arch}.ssm.time_step_rank", "u32", nv),
    (f"{arch}.ssm.inner_size", "u32", dv * nv),
    (f"{arch}.full_attention_interval", "u32", 4),
    (f"{arch}.attention.recurrent_layers", "bools", [t == "linear_attention" for t in cfg["layer_types"]]),
    (f"{arch}.expert_count", "u32", n_experts),
    (f"{arch}.expert_used_count", "u32", cfg["num_experts_per_tok"]),
    (f"{arch}.expert_feed_forward_length", "u32", size),
    (f"{arch}.expert_shared_feed_forward_length", "u32", cfg["shared_expert_intermediate_size"]),
    ("tokenizer.ggml.eos_token_id", "u32", cfg["eos_token_id"]),
]

ALIGN = 32


def string(s):
    b = s.encode()
    return struct.pack("<Q", len(b)) + b


def value(kind, v):
    if kind == "str":
        return struct.pack("<I", 8) + string(v)
    if kind == "u32":
        return struct.pack("<II", 4, v)
    if kind == "f32":
        return struct.pack("<If", 6, v)
    if kind == "bools":
        return struct.pack("<IIQ", 9, 7, len(v)) + bytes(int(x) for x in v)
    raise ValueError(kind)


order = sorted(tensors)
out = bytearray(b"GGUF" + struct.pack("<IQQ", 3, len(order), len(meta)))
for key, kind, v in meta:
    out += string(key) + value(kind, v)
offset = 0
blobs = []
for name in order:
    t = np.ascontiguousarray(tensors[name], dtype=np.float32)
    # ggml lists dimensions innermost first.
    out += string(name) + struct.pack("<I", t.ndim) + b"".join(struct.pack("<Q", d) for d in t.shape[::-1])
    out += struct.pack("<IQ", 0, offset)
    blob = t.tobytes()
    blob += b"\0" * (-len(blob) % ALIGN)
    blobs.append(blob)
    offset += len(blob)
out += b"\0" * (-len(out) % ALIGN)
with open("tiny.gguf", "wb") as f:
    f.write(out)
    for blob in blobs:
        f.write(blob)
