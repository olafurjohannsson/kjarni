"""Writes tiny.gguf: the checkpoint make.py writes, laid out as llama.cpp's
convert_hf_to_gguf.py lays out Qwen 3.5 (its Qwen3NextModel,
_LinearAttentionVReorderBase and Qwen3_5TextModel classes), for tests.rs.

What the converter does, and this repeats:
- every norm except the DeltaNet's gated norm gets +1;
- A_log becomes -exp(A_log), conv1d loses its unit middle dimension;
- with more value heads than key heads, value heads go from grouped order
  [k0v0 k0v1 k1v0 k1v1] to tiled order [k0v0 k1v0 k0v1 k1v1] in in_proj_qkv's
  v rows, in_proj_z, in_proj_a, in_proj_b, A_log, dt_bias, conv1d's v channels
  and out_proj's columns.

Newer converters also stack attn_qkv over attn_gate and ssm_beta over
ssm_alpha. Layer 0 is written that way and layers 1 and 3 the older way, so
one file covers both.

    pip install gguf   # llama.cpp's gguf-py
    python make_gguf.py
"""
import json
import struct

import numpy as np
import gguf

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


def tiled(x, axis, head_dim):
    """Grouped to tiled value-head order along `axis` (the converter's _reorder_v_heads)."""
    x = np.moveaxis(x, axis, 0)
    shape = x.shape
    x = x.reshape(nk, per_k, head_dim, *shape[1:]).swapaxes(0, 1).reshape(shape)
    return np.ascontiguousarray(np.moveaxis(x, 0, axis))


names = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "post_attention_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
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
}

hf = read_safetensors("model.safetensors")
tensors = {}
for name, t in hf.items():
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        t = t + 1
    if name == "model.embed_tokens.weight":
        tensors["token_embd.weight"] = t
        continue
    if name == "model.norm.weight":
        tensors["output_norm.weight"] = t
        continue
    if name == "lm_head.weight":
        tensors["output.weight"] = t
        continue
    _, _, layer, suffix = name.split(".", 3)
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

# Layer 0 in the merged form newer converters write.
for first, second, merged in [
    ("attn_qkv", "attn_gate", "attn_qkv_gate"),
    ("ssm_beta", "ssm_alpha", "ssm_beta_alpha"),
]:
    a = tensors.pop(f"blk.0.{first}.weight")
    b = tensors.pop(f"blk.0.{second}.weight")
    tensors[f"blk.0.{merged}.weight"] = np.concatenate([a, b])

w = gguf.GGUFWriter("tiny.gguf", "qwen35")
w.add_block_count(cfg["num_hidden_layers"])
w.add_context_length(cfg["max_position_embeddings"])
w.add_embedding_length(cfg["hidden_size"])
w.add_feed_forward_length(cfg["intermediate_size"])
w.add_head_count(cfg["num_attention_heads"])
w.add_head_count_kv(cfg["num_key_value_heads"])
w.add_key_length(cfg["head_dim"])
w.add_value_length(cfg["head_dim"])
w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
w.add_rope_freq_base(cfg["rope_parameters"]["rope_theta"])
w.add_rope_dimension_count(int(cfg["head_dim"] * cfg["partial_rotary_factor"]))
w.add_ssm_conv_kernel(cfg["linear_conv_kernel_dim"])
w.add_ssm_state_size(dk)
w.add_ssm_group_count(nk)
w.add_ssm_time_step_rank(nv)
w.add_ssm_inner_size(dv * nv)
# Deliberately not what layer_types says, so the test shows recurrent_layers wins.
w.add_full_attention_interval(4)
w.add_array("qwen35.attention.recurrent_layers", [t == "linear_attention" for t in cfg["layer_types"]])
w.add_eos_token_id(cfg["eos_token_id"])
for name in sorted(tensors):
    w.add_tensor(name, np.ascontiguousarray(tensors[name], dtype=np.float32))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f"wrote tiny.gguf: {len(tensors)} tensors")
