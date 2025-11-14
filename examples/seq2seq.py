import torch
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Ensure we are using float32 for all operations to match Rust's f32
torch.set_default_dtype(torch.float32)

model_name = "sshleifer/distilbart-cnn-12-6"

# --- 1. Load Model and Tokenizer ---
# We need the full model, not the pipeline, for manual control.
print(f"Loading model and tokenizer for: {model_name}...")
# We explicitly set torch_dtype to float32 to ensure the model weights are loaded in that precision.
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval() # Set the model to evaluation mode

# This dictionary will store the outputs of each layer
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # For encoder/decoder layers, the output is a tuple. We want the hidden state.
        activation[name] = output[0].detach()
    return hook

# Register hooks on the layers you want to inspect
# Let's check the output of the first encoder layer and the first decoder layer
model.model.encoder.layers[0].register_forward_hook(get_activation('encoder.layer.0'))
model.model.decoder.layers[0].register_forward_hook(get_activation('decoder.layer.0'))
model.model.encoder.layers[0].register_forward_hook(get_activation('encoder.layer.0.output'))
model.model.encoder.layers[0].self_attn_layer_norm.register_forward_hook(get_activation('encoder.layer.0.attn_norm'))
model.model.encoder.layers[0].fc2.register_forward_hook(get_activation('encoder.layer.0.ffn_output'))
print("✓ Model and tokenizer loaded.")

# The article to be summarized
article = """Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
like C++, Haskell, and Erlang."""

# --- 2. Manually Encode the Input Article (Encoder Pass) ---
print("\nEncoding the article for the encoder...")
# The tokenizer prepares the input IDs and the attention mask for the model.
# return_tensors='pt' gives us PyTorch tensors.
inputs = tokenizer(article, return_tensors="pt", max_length=1024, truncation=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

print(f"Input Token IDs: {input_ids.tolist()[0]}")
print(f"Encoder Attention Mask Shape: {attention_mask.shape}")

# The encoder processes the input text and creates a rich contextual representation.
# We don't need gradients for inference.
with torch.no_grad():
    encoder_outputs = model.get_encoder()(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
encoder_hidden_states = encoder_outputs.last_hidden_state

print(f"Encoder Output Shape: {encoder_hidden_states.shape}")
print(f"Encoder Output Mean: {encoder_hidden_states.mean().item():.8f}")
print(f"Encoder Output Std Dev: {encoder_hidden_states.std().item():.8f}")


# --- 3. Manually Perform the First Decoding Step (Decoder Pass) ---
print("\nPerforming the first decoding step...")

# The first input to the decoder is always the `decoder_start_token_id`.
# For BART, this is the EOS token ID (value: 2).
decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]], dtype=torch.long)

print(f"Decoder Input Token ID (Step 0): {decoder_input_ids.item()}")

# --- Run the full model pass (encoder + first decoder step) ---
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])
with torch.no_grad():
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids)

# --- Now, print the captured values ---
print("\n--- PYTORCH INTERMEDIATE VALUES ---")
encoder_layer_0_output = activation['encoder.layer.0']
print(f"Encoder Layer 0 Output Mean: {encoder_layer_0_output.mean().item():.8f}")

decoder_layer_0_output = activation['decoder.layer.0']
print(f"Decoder Layer 0 Output Mean: {decoder_layer_0_output.mean().item():.8f}")

print("\n--- PYTORCH INTERMEDIATE VALUES ---")
attn_norm_output = activation['encoder.layer.0.attn_norm']
print(f"Encoder Layer 0, After Self-Attn + Norm, Mean: {attn_norm_output.mean().item():.8f}")

ffn_output = activation['encoder.layer.0.ffn_output']
print(f"Encoder Layer 0, After FFN (pre-Add/Norm), Mean: {ffn_output.mean().item():.8f}")

final_output = activation['encoder.layer.0.output']
print(f"Encoder Layer 0, Final Output, Mean: {final_output.mean().item():.8f}")

# Now, we call the full model. It will internally use the pre-computed encoder_hidden_states
# and pass the decoder_input_ids to the decoder.
with torch.no_grad():
    outputs = model(
        encoder_outputs=encoder_outputs, # Pass the encoder's output
        decoder_input_ids=decoder_input_ids,
        return_dict=True
    )

# The logits tensor has shape [batch_size, sequence_length, vocab_size].
# For our single token input, this is [1, 1, 50264].
# We select the logits for our one and only token.
first_step_logits = outputs.logits[0, 0, :]

# --- 4. Print the Logits for Comparison ---
print("\n--- DEBUG: PYTORCH LOGITS (STEP 0) ---")
print(f"Logits Tensor Shape: {first_step_logits.shape}")
print(f"Logits Mean: {first_step_logits.mean().item():.8f}")
print(f"Logits Std Dev: {first_step_logits.std().item():.8f}")

# Print specific values that you can easily check against your Rust output
print(f"Logits[0]:     {first_step_logits[0].item():.8f}")
print(f"Logits[100]:   {first_step_logits[100].item():.8f}")
print(f"Logits[1000]:  {first_step_logits[1000].item():.8f}")

# To get the full array for a diff tool, you can save it to a file
# np.savetxt("pytorch_logits_step0.txt", first_step_logits.numpy())
# print("\n✓ Logits for step 0 saved to pytorch_logits_step0.txt")

# --- 5. Find the Top Token (to verify our logits make sense) ---
top_token_id = torch.argmax(first_step_logits).item()
top_token_str = tokenizer.decode([top_token_id])
print(f"\nTop token predicted at Step 0: ID={top_token_id}, Token='{top_token_str}'")

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ... (load model, tokenizer, and encode article as before) ...



