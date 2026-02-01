#!/usr/bin/env python3
"""
CPU Baseline Benchmark for Llama 3.2 3B (BF16)

Uses local safetensors model for apples-to-apples comparison with Rust.

Install:
    pip install torch transformers accelerate

Usage:
    python cpu_baseline.py
"""

import time
import sys
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PROMPT = "Describe the theory of relativity in simple terms (max 50 words):\n"
MAX_NEW_TOKENS = 55

# Local safetensors model path
MODEL_PATH = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct/"


def benchmark_transformers():
    """Benchmark using HuggingFace transformers BF16 - same as Rust"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("transformers/torch not installed. Install with:")
        print("  pip install torch transformers accelerate")
        return None

    print("\n" + "=" * 60)
    print("Transformers (BF16) - CPU")
    print("=" * 60)



    print(f"Loading {MODEL_PATH}...")
    load_start = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        local_files_only=True,
    ).eval()

    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f}s")

    # Format prompt
    messages = [{"role": "user", "content": PROMPT}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(formatted, return_tensors="pt")

    # Warmup
    print("Warmup...")
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5, do_sample=False)

    # Benchmark
    print(f"\nGenerating {MAX_NEW_TOKENS} tokens...")
    print("-" * 40)

    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    end = time.perf_counter()

    generated_ids = output_ids[0, input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    tokens_generated = len(generated_ids)

    print(generated_text)
    print("-" * 40)

    elapsed = end - start
    tps = tokens_generated / elapsed

    print(f"\nTokens: {tokens_generated}")
    print(f"Time: {elapsed:.2f}s")
    print(f"Decode TPS: {tps:.1f} t/s")

    return {
        "backend": "transformers BF16",
        "tokens": tokens_generated,
        "time": elapsed,
        "tps": tps,
    }


def print_system_info():
    """Print CPU info for reference"""
    import platform
    import os

    print("=" * 60)
    print("System Info")
    print("=" * 60)
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    
    # Try to get more CPU info on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    print(f"CPU Model: {line.split(':')[1].strip()}")
                    break
    except:
        pass
    
    print(f"Logical CPUs: {os.cpu_count()}")
    
    # Check for torch and print thread info
    try:
        import torch
        print(f"PyTorch threads: {torch.get_num_threads()}")
        print(f"PyTorch inter-op threads: {torch.get_num_interop_threads()}")
    except:
        pass


def main():
    print_system_info()

    result = benchmark_transformers()

    if result:
        print("\n" + "=" * 60)
        print("Comparison")
        print("=" * 60)
        print(f"PyTorch BF16 CPU: {result['tps']:.1f} t/s")
        print(f"Your Rust BF16:   2.5 t/s (from logs)")
        
        ratio = result['tps'] / 2.5
        if ratio > 1:
            print(f"\nPyTorch is {ratio:.1f}x faster")
        else:
            print(f"\nRust is {1/ratio:.1f}x faster")


if __name__ == "__main__":
    main()