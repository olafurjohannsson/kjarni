import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure we are using float32 for all operations to match Rust's f32
torch.set_default_dtype(torch.float32)

model_name = "sshleifer/distilbart-cnn-12-6"

# --- 1. Load Model and Tokenizer ---
print(f"Loading model and tokenizer for: {model_name}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

# --- 2. Register Hooks ---
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # Handle tuple outputs from layers vs. tensor outputs from norms/projections
        value = output[0] if isinstance(output, tuple) else output
        activation[name] = value.detach()
    return hook

# Input to the first encoder layer
model.model.encoder.layernorm_embedding.register_forward_hook(get_activation('encoder.embedding_norm'))
# Raw output of the first attention block
model.model.encoder.layers[0].self_attn.out_proj.register_forward_hook(get_activation('encoder.layer.0.attn_raw_output'))
# Output after the first attention block's LayerNorm
model.model.encoder.layers[0].self_attn_layer_norm.register_forward_hook(get_activation('encoder.layer.0.attn_norm'))
# Output of the first FFN block (before residual)
model.model.encoder.layers[0].fc2.register_forward_hook(get_activation('encoder.layer.0.ffn_output'))
# Final output of the first encoder layer
model.model.encoder.layers[0].register_forward_hook(get_activation('encoder.layer.0.final_output'))
print("✓ Model and tokenizer loaded.")

# --- 3. Prepare Inputs ---
article = """Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
like C++, Haskell, and Erlang."""

inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
# For BART, the decoder input starts with the EOS token
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]], dtype=torch.long)

# --- 4. Perform a SINGLE Forward Pass ---
print("\nPerforming a single forward pass...")
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)
print("✓ Forward pass complete.")

# --- 5. Print All Captured Values ---
print("\n--- PYTORCH GROUND TRUTH VALUES ---")

# Input to Encoder Layers
print(f"Input to Encoder Layers (After Embed+Norm), Mean: {activation['encoder.embedding_norm'].mean().item():.8f}")

# Layer 0 Intermediate Values
print(f"L0 Raw Self-Attn Output, Mean:                   {activation['encoder.layer.0.attn_raw_output'].mean().item():.8f}")
print(f"L0 After Self-Attn + Norm, Mean:                 {activation['encoder.layer.0.attn_norm'].mean().item():.8f}")
print(f"L0 After FFN (pre-Add/Norm), Mean:               {activation['encoder.layer.0.ffn_output'].mean().item():.8f}")
print(f"L0 Final Output, Mean:                           {activation['encoder.layer.0.final_output'].mean().item():.8f}")

# Final Logits
first_step_logits = outputs.logits[0, 0, :]
print(f"\nFinal Logits Mean:                               {first_step_logits.mean().item():.8f}")
print(f"Logits[1000]:                                    {first_step_logits[1000].item():.8f}")