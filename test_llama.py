#!/usr/bin/env python3
"""
Verify kjarni Llama output against HuggingFace transformers.

Configured to match:
- Model: Llama-3.2-3B-Instruct
- Prompt: "Describe the theory of relativity in simple terms(max 50 words):\n"
- Greedy decoding (for deterministic comparison)
- max_new_tokens: 150

Usage:
    python verify_llama_parity.py
    python verify_llama_parity.py --save-logits
    python verify_llama_parity.py --compare --kjarni-logits kjarni_logits.json
"""

import argparse
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def greedy_decode_with_logits(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 150,
    repetition_penalty: float = 1.0,
    print_tokens: bool = True,
    save_logits: bool = False,
):
    """
    Greedy decoding with detailed logging for verification.
    """
    device = model.device

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask

    print(f"=" * 70)
    print(f"CONFIGURATION")
    print(f"=" * 70)
    print(f"Model: {model.config._name_or_path}")
    print(f"Prompt: {repr(prompt)}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Repetition penalty: {repetition_penalty}")
    print(f"Strategy: Greedy (argmax)")
    print(f"=" * 70)
    print(f"Input token IDs ({input_ids.shape[1]}): {input_ids[0].tolist()}")
    print(f"Decoded back: {repr(tokenizer.decode(input_ids[0]))}")
    print(f"=" * 70)

    generated_ids = []
    all_logits_info = []
    all_generated_ids = input_ids[0].tolist()  # For repetition penalty

    # Use KV cache
    past_key_values = None

    for step in range(max_new_tokens):
        with torch.no_grad():
            if past_key_values is None:
                # Prefill
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Decode
                outputs = model(
                    input_ids=next_token_id.unsqueeze(0).unsqueeze(0),
                    attention_mask=torch.cat(
                        [
                            attention_mask,
                            torch.ones(
                                (1, 1), device=device, dtype=attention_mask.dtype
                            ),
                        ],
                        dim=1,
                    ),
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
                    ],
                    dim=1,
                )

        past_key_values = outputs.past_key_values

        # Get logits for last position
        logits = outputs.logits[0, -1, :].clone()  # [vocab_size]

        # Apply repetition penalty (same as kjarni)
        if repetition_penalty != 1.0:
            for token_id in set(all_generated_ids):
                if logits[token_id] > 0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        # Greedy: select argmax
        next_token_id = logits.argmax(dim=-1)
        next_token = tokenizer.decode([next_token_id.item()])

        generated_ids.append(next_token_id.item())
        all_generated_ids.append(next_token_id.item())

        # Get top-5 for debugging
        top_k = 5
        top_values, top_indices = torch.topk(logits, top_k)
        top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]

        # Softmax for probabilities
        probs = torch.softmax(logits, dim=-1)
        top_probs = probs[top_indices]

        logit_info = {
            "step": step,
            "token_id": next_token_id.item(),
            "token": next_token,
            "logit": logits[next_token_id].item(),
            "prob": probs[next_token_id].item(),
            "top5": [
                {
                    "id": idx.item(),
                    "token": tok,
                    "logit": val.item(),
                    "prob": prob.item(),
                }
                for idx, tok, val, prob in zip(
                    top_indices, top_tokens, top_values, top_probs
                )
            ],
        }
        all_logits_info.append(logit_info)

        if print_tokens:
            print(
                f"Step {step:3d}: id={next_token_id.item():6d}, "
                f"logit={logits[next_token_id].item():8.4f}, "
                f"prob={probs[next_token_id].item():.4f}, "
                f"token={repr(next_token)}"
            )
            if step < 10 or step % 10 == 0:  # Print top-5 for first 10 and every 10th
                for i, t in enumerate(logit_info["top5"]):
                    marker = "→" if i == 0 else " "
                    print(
                        f"    {marker} [{t['id']:6d}] {t['logit']:8.4f} ({t['prob']:.4f}) {repr(t['token'])}"
                    )

        # Check for EOS
        eos_token_ids = tokenizer.eos_token_id
        if isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        elif eos_token_ids is None:
            eos_token_ids = []

        # Llama 3 has multiple EOS tokens
        if hasattr(model.config, "eos_token_id"):
            config_eos = model.config.eos_token_id
            if isinstance(config_eos, list):
                eos_token_ids = config_eos
            elif isinstance(config_eos, int):
                eos_token_ids = [config_eos]

        if next_token_id.item() in eos_token_ids:
            print(f"\n[EOS token {next_token_id.item()} generated at step {step}]")
            break

    # Full generated text
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("=" * 70)
    print(f"Generated tokens ({len(generated_ids)}): {generated_ids[:20]}...")
    print(f"=" * 70)
    print(f"GENERATED TEXT:")
    print(f"=" * 70)
    print(full_text)
    print("=" * 70)

    if save_logits:
        with open("hf_logits.json", "w") as f:
            json.dump(all_logits_info, f, indent=2)
        print(f"\nSaved {len(all_logits_info)} steps to hf_logits.json")

    return generated_ids, all_logits_info


def compare_with_kjarni(
    hf_logits_path: str = "hf_logits.json", kjarni_logits_path: str = "kjarni_logits.json"
):
    """Compare logits from HF and kjarni."""
    with open(hf_logits_path) as f:
        hf_data = json.load(f)

    with open(kjarni_logits_path) as f:
        kjarni_data = json.load(f)

    print("=" * 70)
    print("PARITY CHECK: HuggingFace vs kjarni")
    print("=" * 70)

    min_len = min(len(hf_data), len(kjarni_data))
    mismatches = []
    logit_diffs = []

    for i in range(min_len):
        hf = hf_data[i]
        kj = kjarni_data[i]

        token_match = hf["token_id"] == kj["token_id"]
        logit_diff = abs(hf["logit"] - kj["logit"])
        logit_diffs.append(logit_diff)

        status = "✓" if token_match else "✗"

        print(
            f"Step {i:3d}: {status} "
            f"HF={hf['token_id']:6d} ({repr(hf['token']):15s}) logit={hf['logit']:9.4f} | "
            f"KJ={kj['token_id']:6d} ({repr(kj['token']):15s}) logit={kj['logit']:9.4f} | "
            f"Δ={logit_diff:.6f}"
        )

        if not token_match:
            mismatches.append(i)
            # Show top-5 comparison for mismatches
            print(f"    HF top5: {{[(t['id'], t['logit']:.2f) for t in hf['top5']]}}")
            print(f"    KJ top5: {{[(t['id'], t['logit']:.2f) for t in kj['top5']]}}")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Steps compared: {min_len}")
    print(f"Token matches: {min_len - len(mismatches)}/{min_len}")
    print(f"Avg logit diff: {sum(logit_diffs)/len(logit_diffs):.6f}")
    print(f"Max logit diff: {max(logit_diffs):.6f}")

    if mismatches:
        print(f"\n❌ MISMATCHES at steps: {mismatches}")
    else:
        print(f"\n✅ ALL TOKENS MATCH!")

    if len(hf_data) != len(kjarni_data):
        print(f"\n⚠️  Length mismatch: HF={len(hf_data)}, kjarni={len(kjarni_data)}")


def main():
    parser = argparse.ArgumentParser(description="Verify Llama parity with HuggingFace")
    parser.add_argument(
        "--model",
        type=str,
        default="/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct/",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe the theory of relativity in simple terms(max 50 words):\n",
        help="Prompt (default matches kjarni test)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate (default: 150, matches kjarni)",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,  # Use 1.0 for baseline comparison, then try 1.2
        help="Repetition penalty (default: 1.0 for clean comparison)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--save-logits", action="store_true", help="Save logits to JSON for comparison"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare saved HF and kjarni logits"
    )
    parser.add_argument(
        "--kjarni-logits",
        type=str,
        default="kjarni_logits.json",
        help="Path to kjarni logits JSON",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Only print summary, not each token"
    )

    args = parser.parse_args()

    if args.compare:
        compare_with_kjarni("hf_logits.json", args.kjarni_logits)
        return

    # Load model
    print(f"Loading model: {args.model}")
    print(f"Device: {args.device}, dtype: {args.dtype}")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_map[args.dtype],
        device_map=args.device,
    )
    model.eval()

    print(
        f"Model loaded: {model.config.num_hidden_layers} layers, "
        f"{model.config.hidden_size} hidden, "
        f"{model.config.vocab_size} vocab"
    )

    # Run greedy decoding
    greedy_decode_with_logits(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        print_tokens=not args.quiet,
        save_logits=args.save_logits,
    )


if __name__ == "__main__":
    main()