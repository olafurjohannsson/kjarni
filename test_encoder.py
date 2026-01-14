import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Point to your local cache
model_path = "/home/olafurj/.cache/kjarni/Qwen_Qwen2.5-0.5B-Instruct/"

print(f"Loading model from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.float32, # Matching your likely Rust precision
    device_map="cpu"
)

# ---------------------------------------------------------
# Test 1: Verify the Chat Template formatting
# ---------------------------------------------------------
print("\n=== TEST 1: CHAT TEMPLATE PARITY ===")
messages = [
    {"role": "user", "content": "Hello! Answer with exactly one word: 'Hi'."}
]
# We use the tokenizer to render the string. Compare this EXACTLY to what your Rust code produces.
prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("RAW PROMPT STRING (Copy this to your Rust debug output to compare):")
print(repr(prompt_str))

# ---------------------------------------------------------
# Test 2: Logits Parity (Gate 1)
# ---------------------------------------------------------
print("\n=== TEST 2: LOGITS PARITY ===")
inputs = tokenizer(prompt_str, return_tensors="pt")
with torch.no_grad():
    outputs = model(inputs.input_ids)
    next_token_logits = outputs.logits[0, -1, :]
    
    # Print top 5 tokens and their logits
    probs = torch.softmax(next_token_logits, dim=-1)
    top_k = torch.topk(next_token_logits, 5)
    
    print("Top 5 Predicted Tokens (Rust should match these logits ~1e-4):")
    for i in range(5):
        token_id = top_k.indices[i].item()
        score = top_k.values[i].item()
        token_str = tokenizer.decode([token_id])
        print(f"ID: {token_id:<6} | Logit: {score:.4f} | Token: {repr(token_str)}")

# ---------------------------------------------------------
# Test 3: Conversation Flow (The Failing Test)
# ---------------------------------------------------------
print("\n=== TEST 3: CONVERSATION FLOW ===")

# Turn 1
print("User: Hello! Answer with exactly one word: 'Hi'.")
output_ids = model.generate(
    inputs.input_ids, 
    max_new_tokens=50, 
    do_sample=True, 
    temperature=0.7, # Matching your Rust config
    top_p=0.8,
    top_k=40
)
response_1 = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
print(f"Model: {repr(response_1)}")

# Turn 2 (Manual Append to check context)
messages.append({"role": "assistant", "content": response_1.replace("<|im_end|>", "")})
messages.append({"role": "user", "content": "What word did you just say?"})

prompt_str_2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs_2 = tokenizer(prompt_str_2, return_tensors="pt")
output_ids_2 = model.generate(inputs_2.input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)
response_2 = tokenizer.decode(output_ids_2[0][inputs_2.input_ids.shape[1]:], skip_special_tokens=False)

print("User: What word did you just say?")
print(f"Model: {repr(response_2)}")