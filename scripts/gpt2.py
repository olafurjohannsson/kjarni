#!/usr/bin/env python3
"""
PyTorch GPT-2 benchmark for comparison with kjarni.
Usage: python bench_gpt2.py
"""

import time
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer

def benchmark_gpt2(model_name="distilgpt2", num_tokens=100, device="cpu", dtype=torch.float32):
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Device: {device}, Dtype: {dtype}")
    print(f"Generating {num_tokens} tokens")
    print('='*60)

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    prompt = "The field of Artificial Intelligence has seen a lot of progress"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Prompt tokens: {input_ids.shape[1]}")

    # Warmup
    print("\nWarmup...")
    with torch.no_grad():
        _ = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # Benchmark
    print("Benchmarking...")
    torch.cuda.synchronize() if device == "cuda" else None
    
    start = time.perf_counter()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=False,  # Greedy
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            repetition_penalty=1.1,
        )
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.perf_counter() - start

    # Results
    generated_tokens = output.shape[1] - input_ids.shape[1]
    tokens_per_sec = generated_tokens / elapsed
    ms_per_token = (elapsed / generated_tokens) * 1000

    print(f"\n--- Results ---")
    print(f"Generated {generated_tokens} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"Latency: {ms_per_token:.2f} ms/token")

    # Print generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\n--- Generated Text ---")
    print(generated_text)

    return tokens_per_sec


def benchmark_token_by_token(model_name="distilgpt2", num_tokens=100, device="cpu", dtype=torch.float32):
    """Manual token-by-token generation to match kjarni's loop"""
    print(f"\n{'='*60}")
    print(f"Token-by-token benchmark (matches kjarni)")
    print('='*60)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model = model.to(device)
    model.eval()

    prompt = "The field of Artificial Intelligence has seen a lot of progress"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Warmup
    with torch.no_grad():
        _ = model(input_ids, use_cache=True)

    # Token-by-token generation
    generated_ids = input_ids.clone()
    past_key_values = None
    
    token_times = []
    
    print("\nGenerating...")
    start_total = time.perf_counter()
    
    with torch.no_grad():
        for i in range(num_tokens):
            start_token = time.perf_counter()
            
            if past_key_values is None:
                outputs = model(generated_ids, use_cache=True)
            else:
                outputs = model(generated_ids[:, -1:], past_key_values=past_key_values, use_cache=True)
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # Greedy selection
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            token_time = time.perf_counter() - start_token
            token_times.append(token_time)
            
            if (i + 1) % 20 == 0:
                avg_speed = (i + 1) / (time.perf_counter() - start_total)
                print(f"  Token {i+1}: {token_time*1000:.2f}ms | Avg: {avg_speed:.2f} t/s")

    elapsed = time.perf_counter() - start_total
    tokens_per_sec = num_tokens / elapsed

    print(f"\n--- Results ---")
    print(f"Generated {num_tokens} tokens in {elapsed:.2f}s")
    print(f"Speed: {tokens_per_sec:.2f} tokens/sec")
    print(f"Avg latency: {(elapsed/num_tokens)*1000:.2f} ms/token")
    print(f"First token: {token_times[0]*1000:.2f}ms")
    print(f"Last token: {token_times[-1]*1000:.2f}ms")

    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\n--- Generated Text ---")
    print(generated_text)

    return tokens_per_sec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilgpt2", help="Model name")
    parser.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--bf16", action="store_true", help="Use BF16")
    parser.add_argument("--fp16", action="store_true", help="Use FP16")
    args = parser.parse_args()

    dtype = torch.float32
    if args.bf16:
        dtype = torch.bfloat16
    elif args.fp16:
        dtype = torch.float16

    # Run both benchmarks
    benchmark_gpt2(args.model, args.tokens, args.device, dtype)
    benchmark_token_by_token(args.model, args.tokens, args.device, dtype)
    
    print("\n" + "="*60)
    print("Compare with kjarni:")
    print("  cargo run --release --example gpt2_generation")
    print("="*60)