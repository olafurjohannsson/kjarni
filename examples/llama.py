# test_llama.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ Use your existing cached directory
model_path = "/home/olafurj/.cache/edgetransformers/meta-llama_Llama-3.2-1B"

model = AutoModelForCausalLM.from_pretrained(
    model_path,  # Local path instead of model ID
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    local_files_only=True,  # Don't try to download
).eval()

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
)

prompt = "The field of Artificial Intelligence has seen a lot of progress"
inputs = tokenizer(prompt, return_tensors="pt")

print("Input IDs:", inputs.input_ids[0].tolist())

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=False,  # Greedy
        pad_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n--- PYTORCH OUTPUT ---")
print(generated_text)
print("\n--- TOKEN IDs ---")
print(outputs[0].tolist())