import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to your model
model_path = "/home/olafurj/.cache/edgetransformers/meta-llama_Llama-3.2-1B"

print(f"Loading model from {model_path}...")

# Load model in Float32 to match your current Rust CPU implementation
model = AutoModelForCausalLM.from_pretrained(
    model_path,
#    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

# --- YOUR ROBUST TOKENIZATION LOGIC ---
prompt = "The field of Artificial Intelligence has seen a lot of progress"
CORRECT_BOS_TOKEN_ID = 128000

# 1. Encode text only
input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)

# 2. Add BOS
input_ids_list.insert(0, CORRECT_BOS_TOKEN_ID)

# 3. Create Tensor
input_ids = torch.tensor([input_ids_list], dtype=torch.long)
print(f"Input IDs: {input_ids[0].tolist()}")
# --------------------------------------

# Warmup (Prefill)
print("Warming up (Prefill)...")
with torch.no_grad():
    # Run the full prompt
    out = model(input_ids, use_cache=True)
    past_key_values = out.past_key_values
    
    # Greedy pick next token
    next_token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)

print(f"Starting generation loop (10 tokens)...")
total_dt = 0
num_tokens = 10

with torch.no_grad():
    for i in range(num_tokens):
        start = time.perf_counter()
        
        # --- THE HOT PATH ---
        # Run forward pass for exactly 1 token using the KV cache
        out = model(
            next_token, 
            past_key_values=past_key_values, 
            use_cache=True
        )
        # --------------------
        
        # Update cache for next step
        past_key_values = out.past_key_values
        
        # Select next token (Greedy)
        next_token = out.logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
        
        end = time.perf_counter()
        dt = end - start
        total_dt += dt
        
        print(f"Token #{i+1}: {dt*1000:.2f} ms | {1.0/dt:.2f} t/s")

print(f"\nAverage Speed: {num_tokens / total_dt:.2f} t/s")