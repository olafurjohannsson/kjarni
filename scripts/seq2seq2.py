import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# --- CONFIGURATION ---
model_name = "sshleifer/distilbart-cnn-12-6"
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

# --- 1. LOAD MODEL ---
print(f"Loading {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# Print the default generation config to verify parameters
print("\n--- DEFAULT GENERATION CONFIG ---")
print(model.generation_config)

# --- 2. PREPARE INPUTS ---
article = """Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
like C++, Haskell, and Erlang."""

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)

# --- 3. GENERATE ---
print("\n--- GENERATING SUMMARY ---")
with torch.no_grad():
    # We use the generate() method which uses the model's default configuration
    # (Beam Search with num_beams=4, no_repeat_ngram_size=3, etc.)
    summary_ids = model.generate(
        inputs['input_ids'], 
        attention_mask=inputs['attention_mask']
    )

# --- 4. DECODE AND PRINT ---
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("\n--- FINAL GENERATED TEXT ---")
print(summary)

print("\n--- RAW TOKEN IDS ---")
print(summary_ids[0].tolist())