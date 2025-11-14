import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Use your existing cached directory
model_path = "/home/olafurj/.cache/edgetransformers/meta-llama_Llama-3.2-1B"

# Set a consistent float precision for direct comparison with Rust's f32
torch.set_default_dtype(torch.float32)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    device_map="cpu",
    local_files_only=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

prompt = "The field of Artificial Intelligence has seen a lot of progress"

# --- DIAGNOSTIC PRINT ---
# This will show us the incorrect value the tokenizer has loaded.
print(f"Tokenizer's configured bos_token_id: {tokenizer.bos_token_id}")
# ------------------------

# --- ROBUST TOKENIZATION LOGIC ---
# The known correct BOS token ID for Llama 3
CORRECT_BOS_TOKEN_ID = 128000

# 1. Tokenize the prompt text ONLY, without adding any special BOS/EOS tokens.
input_ids_list = tokenizer.encode(prompt, add_special_tokens=False)

# 2. Manually prepend the KNOWN CORRECT BOS token ID.
input_ids_list.insert(0, CORRECT_BOS_TOKEN_ID)

# 3. Convert the final, correct list of IDs into the required PyTorch tensor format.
inputs_tensor = torch.tensor([input_ids_list], dtype=torch.long)
# --- END OF LOGIC ---

print("Input IDs being sent to model:", inputs_tensor[0].tolist())

with torch.no_grad():
    outputs = model.generate(
        inputs_tensor,
        max_new_tokens=100,
        repetition_penalty=1.1,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- PYTORCH OUTPUT ---")
print(generated_text)
print("\n--- TOKEN IDs ---")
print(outputs[0].tolist())