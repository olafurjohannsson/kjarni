import torch
import time
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION TO MATCH RUST ---
model_path = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct"
prompt = "Describe the theory of relativity in simple terms(max 50 words):\n"
REPETITION_PENALTY = 1.2
MAX_NEW_TOKENS = 150
CORRECT_BOS_TOKEN_ID = 128000

print(f"Loading model from {model_path}...")

# 1. Force Float32 to match Rust CPU backend (ndarray<f32>)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# 2. Tokenization (Manually adding BOS as per your Rust logic)
input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)
input_ids_list.insert(0, CORRECT_BOS_TOKEN_ID)

input_ids = torch.tensor([input_ids_list], dtype=torch.long)
print(f"Input IDs: {input_ids[0].tolist()}")

# Helper to apply repetition penalty (Matches edgetransformers logic)
def apply_repetition_penalty(logits, generated_ids, penalty):
    if penalty == 1.0:
        return logits
    
    # We only penalize tokens that have already been generated
    for token_id in set(generated_ids):
        if logits[0, token_id] < 0:
            logits[0, token_id] *= penalty
        else:
            logits[0, token_id] /= penalty
    return logits

# --- GENERATION ---
print(f"\n--- Generating (Python) ---")
print(prompt, end="", flush=True)

# Keep track of all tokens for repetition penalty
all_generated_ids = input_ids[0].tolist()

with torch.no_grad():
    # 1. Warmup / Prefill
    out = model(input_ids, use_cache=True)
    past_key_values = out.past_key_values
    next_logits = out.logits[:, -1, :]

    # 2. Sample First Token
    next_logits = apply_repetition_penalty(next_logits, all_generated_ids, REPETITION_PENALTY)
    next_token_id = next_logits.argmax(dim=-1) # Shape [1]
    
    # Print First Token
    word = tokenizer.decode(next_token_id, skip_special_tokens=True)
    print(word, end="", flush=True)
    
    all_generated_ids.append(next_token_id.item())
    
    # FIX: Ensure shape is [1, 1]. next_token_id is [1], so we need one unsqueeze.
    next_token_tensor = next_token_id.unsqueeze(0) 

    total_dt = 0

    # 3. Decode Loop
    for i in range(MAX_NEW_TOKENS - 1): # -1 because we already generated one above
        start = time.perf_counter()
        
        # Forward pass for single token
        out = model(
            next_token_tensor, 
            past_key_values=past_key_values, 
            use_cache=True
        )
        
        end = time.perf_counter()
        total_dt += (end - start)

        # Update Cache & Logits
        past_key_values = out.past_key_values
        next_logits = out.logits[:, -1, :]

        # Apply Penalty
        next_logits = apply_repetition_penalty(next_logits, all_generated_ids, REPETITION_PENALTY)

        # Greedy Sample
        next_token_id = next_logits.argmax(dim=-1)
        
        # Decode & Print
        word = tokenizer.decode(next_token_id, skip_special_tokens=True)
        print(word, end="", flush=True)

        # Prepare for next step
        all_generated_ids.append(next_token_id.item())
        next_token_tensor = next_token_id.unsqueeze(0) # [1] -> [1, 1]

print(f"\n\nAverage Speed (Decode Only): {(MAX_NEW_TOKENS-1) / total_dt:.2f} t/s")