"""
Quantize a safetensors model to int8 for WASM deployment.

Usage:
    python quantize_model.py --model-dir ./all-MiniLM-L6-v2
    python quantize_model.py --model-dir ./all-MiniLM-L6-v2 --output model_q8.kjq

This reads model.safetensors + config.json from the model directory
and produces a single .kjq file with int8 weights + per-tensor scales.

Format (.kjq):
    Header:
        magic: b"KJQ1" (4 bytes)
        config_json_len: u32 LE
        config_json: [u8; config_json_len]
        tokenizer_json_len: u32 LE
        tokenizer_json: [u8; tokenizer_json_len]
        num_tensors: u32 LE
    Per tensor:
        name_len: u32 LE
        name: [u8; name_len]
        ndim: u32 LE
        shape: [u32 LE; ndim]
        quantized: bool (u8) - 1 if int8+scale, 0 if kept as f32
        If quantized:
            scale: f32 LE
            data: [i8; numel]
        If not quantized:
            data: [f32 LE; numel]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open


# Tensors kept as f32 
# Biases and norms are tiny and sensitive to quantization
SKIP_QUANTIZE_PATTERNS = [
    ".bias",
    "LayerNorm",
    "layer_norm",
    "token_type_embeddings",  # (2, 384)
    "position_ids",           #  just indices
]


def should_quantize(name: str) -> bool:
    """Determine if a tensor should be quantized to int8."""
    for pattern in SKIP_QUANTIZE_PATTERNS:
        if pattern in name:
            return False
    return True


def quantize_tensor(tensor: np.ndarray) -> tuple[np.ndarray, float]:
    """Quantize a float32 tensor to int8 with per-tensor symmetric quantization.
    
    Returns (quantized_int8, scale) where:
        original ≈ quantized_int8.astype(f32) * scale
    """
    abs_max = np.max(np.abs(tensor))
    if abs_max == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    
    scale = abs_max / 127.0
    quantized = np.clip(np.round(tensor / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def _read_bfloat16(safetensors_path, name):
    """Read one bfloat16 tensor and widen it to float32.

    bfloat16 shares float32's exponent and simply truncates the mantissa, so
    placing the 16 bits in the high half of a 32-bit word reconstructs the value
    exactly. No rounding is involved.
    """
    import json as _json
    import struct as _struct

    with open(safetensors_path, "rb") as fh:
        header_len = _struct.unpack("<Q", fh.read(8))[0]
        header = _json.loads(fh.read(header_len))
        meta = header[name]
        start, end = meta["data_offsets"]
        fh.seek(8 + header_len + start)
        raw = fh.read(end - start)

    lo = np.zeros(len(raw) // 2, dtype=np.uint16)
    hi = np.frombuffer(raw, dtype=np.uint16)
    widened = np.stack([lo, hi], axis=1).reshape(-1).view(np.float32)
    return widened.reshape(meta["shape"]).astype(np.float32)


def quantize_model(model_dir: Path, output_path: Path):
    """Read safetensors + config.json, write quantized .kjq file."""
    
    safetensors_path = model_dir / "model.safetensors"
    config_path = model_dir / "config.json"
    tokenizer_path = model_dir / "tokenizer.json"
    
    if not safetensors_path.exists():
        print(f"Error: {safetensors_path} not found")
        sys.exit(1)
    if not config_path.exists():
        print(f"Error: {config_path} not found")
        sys.exit(1)
    if not tokenizer_path.exists():
        print(f"Error: {tokenizer_path} not found")
        sys.exit(1)
    
    # Load config, folding in the sequence length the model was actually tuned for.
    #
    # A .kjq is loaded in the browser, where there is no filesystem, so the packed
    # config is the only channel through which this can travel. Without it the WASM
    # loader falls back to max_position_embeddings and truncates MiniLM at 512 while
    # every native binding truncates at 256, producing different embeddings for the
    # same model and text.
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sbert_path = model_dir / "sentence_bert_config.json"
    max_seq_length = None
    if sbert_path.exists():
        # A present-but-unreadable file is a truncated or redirected download, not a
        # reason to abandon the quantisation. Fall back the way a missing file does.
        try:
            sbert = json.loads(sbert_path.read_text(encoding="utf-8"))
            candidate = sbert.get("max_seq_length")
            if isinstance(candidate, int) and candidate > 0:
                max_seq_length = candidate
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"  Warning: ignoring unreadable {sbert_path.name}: {e}")

    if max_seq_length is not None:
        config["max_seq_length"] = max_seq_length
        print(f"  Sequence length:          {max_seq_length} (from sentence_bert_config.json)")
    else:
        print("  Sequence length:          not specified; loader will use the model config")

    config_json = json.dumps(config)
    config_bytes = config_json.encode("utf-8")
    
    # Load tokenizer
    tokenizer_json = tokenizer_path.read_text(encoding="utf-8")
    tokenizer_bytes = tokenizer_json.encode("utf-8")
    
    # Load tensors
    #
    # numpy has no bfloat16, and safetensors' numpy framework refuses to hand one
    # back, so a bf16 checkpoint fails outright. Most modern chat models ship bf16
    # (Qwen and Llama among them), which made this script unusable for exactly the
    # models people want in a browser. Read those tensors raw and widen them: bf16
    # is the top 16 bits of an f32, so the conversion is a shift, not an
    # approximation.
    tensors = {}
    with safe_open(str(safetensors_path), framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            try:
                tensors[name] = f.get_tensor(name)
            except TypeError as e:
                if "bfloat16" not in str(e):
                    raise
                tensors[name] = _read_bfloat16(safetensors_path, name)

    if not tensors:
        raise SystemExit(f"No tensors could be read from {safetensors_path}")
    
    print(f"Loaded {len(tensors)} tensors from {safetensors_path}")
    
    # Calculate sizes
    original_size = sum(t.nbytes for t in tensors.values())
    quantized_size = 0
    num_quantized = 0
    num_kept = 0
    
    # Write output
    with open(output_path, "wb") as f:
        # Magic
        f.write(b"KJQ1")
        
        # Config
        f.write(struct.pack("<I", len(config_bytes)))
        f.write(config_bytes)
        
        # Tokenizer
        f.write(struct.pack("<I", len(tokenizer_bytes)))
        f.write(tokenizer_bytes)
        
        # Number of tensors
        f.write(struct.pack("<I", len(tensors)))
        
        for name, tensor in sorted(tensors.items()):
            tensor = tensor.astype(np.float32)  # ensure f32
            name_bytes = name.encode("utf-8")
            
            # Name
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)
            
            # Shape
            f.write(struct.pack("<I", len(tensor.shape)))
            for dim in tensor.shape:
                f.write(struct.pack("<I", dim))
            
            if should_quantize(name):
                # Quantized path
                f.write(struct.pack("B", 1))  # quantized = true
                q_data, scale = quantize_tensor(tensor)
                f.write(struct.pack("<f", scale))
                f.write(q_data.tobytes())
                
                quantized_size += 4 + len(q_data.tobytes())  # scale + data
                num_quantized += 1
                
                # Report per-tensor error
                reconstructed = q_data.astype(np.float32) * scale
                max_err = np.max(np.abs(tensor - reconstructed))
                mean_err = np.mean(np.abs(tensor - reconstructed))
                print(f"  Q8  {name:60s} shape={str(tensor.shape):20s} "
                      f"scale={scale:.6f} max_err={max_err:.6f} mean_err={mean_err:.6f}")
            else:
                # Keep as f32
                f.write(struct.pack("B", 0))  # quantized = false
                f.write(tensor.tobytes())
                
                quantized_size += tensor.nbytes
                num_kept += 1
                print(f"  F32 {name:60s} shape={str(tensor.shape):20s} (kept)")
    
    output_size = output_path.stat().st_size
    
    print()
    print(f"Summary:")
    print(f"  Tensors quantized (int8): {num_quantized}")
    print(f"  Tensors kept (f32):       {num_kept}")
    print(f"  Original weights size:    {original_size / 1024 / 1024:.1f} MB")
    print(f"  Tokenizer size:           {len(tokenizer_bytes) / 1024:.0f} KB")
    print(f"  Output file size:         {output_size / 1024 / 1024:.1f} MB")
    print(f"  Compression ratio:        {original_size / output_size:.1f}x")
    print(f"  Saved to:                 {output_path}")


def verify_quantized(model_dir: Path, kjq_path: Path):
    """Optional verification: load both and compare outputs."""
    print(f"\nVerifying quantization accuracy...")
    
    with safe_open(str(model_dir / "model.safetensors"), framework="numpy") as f:
        original = {name: f.get_tensor(name).astype(np.float32) for name in f.keys()}
    
    # Read back the .kjq file
    reconstructed = {}
    with open(kjq_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"KJQ1"
        
        config_len = struct.unpack("<I", f.read(4))[0]
        f.read(config_len)  # skip config
        
        tokenizer_len = struct.unpack("<I", f.read(4))[0]
        f.read(tokenizer_len)  # skip tokenizer
        
        num_tensors = struct.unpack("<I", f.read(4))[0]
        
        for _ in range(num_tensors):
            name_len = struct.unpack("<I", f.read(4))[0]
            name = f.read(name_len).decode("utf-8")
            
            ndim = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<I", f.read(4))[0] for _ in range(ndim))
            numel = 1
            for d in shape:
                numel *= d
            
            is_quantized = struct.unpack("B", f.read(1))[0]
            
            if is_quantized:
                scale = struct.unpack("<f", f.read(4))[0]
                data = np.frombuffer(f.read(numel), dtype=np.int8)
                reconstructed[name] = data.astype(np.float32).reshape(shape) * scale
            else:
                data = np.frombuffer(f.read(numel * 4), dtype=np.float32)
                reconstructed[name] = data.reshape(shape)
    
    # Compare
    max_errors = []
    for name in sorted(original.keys()):
        orig = original[name]
        recon = reconstructed[name]
        max_err = np.max(np.abs(orig - recon))
        max_errors.append((name, max_err))
    
    worst = max(max_errors, key=lambda x: x[1])
    avg = np.mean([e for _, e in max_errors])
    print(f"  Worst tensor error: {worst[0]} = {worst[1]:.6f}")
    print(f"  Average max error:  {avg:.6f}")
    print(f"  Verification passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize safetensors model to int8 for WASM")
    parser.add_argument("--model-dir", type=str, required=True,
                        help="Directory containing model.safetensors and config.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Output .kjq file path (default: model_q8.kjq in model dir)")
    parser.add_argument("--verify", action="store_true",
                        help="Verify quantization accuracy after saving")
    
    args = parser.parse_args()
    model_dir = Path(args.model_dir)
    output_path = Path(args.output) if args.output else model_dir / "model_q8.kjq"
    
    quantize_model(model_dir, output_path)
    
    if args.verify:
        verify_quantized(model_dir, output_path)