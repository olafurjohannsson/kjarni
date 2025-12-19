import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Setup
model_name = "sshleifer/distilbart-cnn-12-6"
print(f"Loading {model_name}...")
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()  # Disable dropout

# Input text (Your exact example)
text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector . Rust was influenced by languages like C++, Haskell, and Erlang ."

# 1. Tokenization
inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"]

print("\n" + "=" * 50)
print("PHASE 1: INPUT TOKENS")
print("=" * 50)
print(f"Input IDs shape: {input_ids.shape}")
print(f"First 10 Input IDs: {input_ids[0, :10].tolist()}")

print("\n" + "=" * 50)
print("PHASE 1.5: EMBEDDINGS")
print("=" * 50)
with torch.no_grad():
    # Handle scaling conditionally
    embed_scale = 1.0
    if hasattr(model.model.encoder, "embed_scale") and model.model.encoder.embed_scale is not None:
        embed_scale = model.model.encoder.embed_scale

    embeds = model.model.shared(input_ids) * embed_scale
    positions = model.model.encoder.embed_positions(input_ids)
    final_embeds = embeds + positions

    print(f"Embeddings Shape: {final_embeds.shape}")
    print(f"Embeddings first 10: {final_embeds[0, 0, :10].tolist()}")

# 2. Encoder Pass
print("\n" + "=" * 50)
print("PHASE 2: ENCODER OUTPUT (Last Hidden State)")
print("=" * 50)
with torch.no_grad():
    encoder = model.get_encoder()
    encoder_outputs = encoder(input_ids)
    last_hidden_state = encoder_outputs.last_hidden_state

    # Comparison Data
    # Compare this against your `encoder_output.last_hidden_state` in Rust
    print(f"Encoder Output Shape: {last_hidden_state.shape}")
    slice_data = last_hidden_state[0, 0, :10].tolist()
    print(f"First token (<s>), first 10 hidden dims:\n{slice_data}")
    print(f"Mean: {last_hidden_state.mean().item():.6f}")
    print(f"Sum: {last_hidden_state.sum().item():.6f}")

# 3. Decoder Step 0 (Logits check)
print("\n" + "=" * 50)
print("PHASE 3: DECODER STEP 0 (Logits)")
print("=" * 50)
# Prepare decoder input: Just the Start Token (2 for BART)
decoder_input_ids = torch.tensor([[model.config.decoder_start_token_id]])

with torch.no_grad():
    # Forward pass through the full model using pre-computed encoder outputs
    outputs = model(
        input_ids=None,  # Not needed when encoder_outputs is provided
        encoder_outputs=encoder_outputs,
        decoder_input_ids=decoder_input_ids,
    )
    logits = outputs.logits  # Shape: [1, 1, vocab_size]

    # Comparison Data
    # Compare this against `logits_2d` in your Rust `beam_step` (Step 0)
    print(f"Logits Shape: {logits.shape}")

    # This checks the final linear projection + bias
    logits_slice = logits[0, 0, :10].tolist()
    print(f"First 10 logits for the first step:\n{logits_slice}")

    # Check for the specific bias vector
    if hasattr(model, "final_logits_bias"):
        print(f"\nModel has final_logits_bias shape: {model.final_logits_bias.shape}")
        print(f"First 10 values of final_logits_bias:\n{model.final_logits_bias[0, :10].tolist()}")
    else:
        print("\nModel does NOT use final_logits_bias parameter")

# 4. Full Generation (Beam Search)
print("\n" + "=" * 50)
print("PHASE 4: FULL BEAM SEARCH GENERATION")
print("=" * 50)

# Exact config from your Rust parameters
gen_kwargs = {
    "num_beams": 4,
    "length_penalty": 2.0,
    "max_length": 142,
    "min_length": 56,
    "no_repeat_ngram_size": 3,
    "early_stopping": True,
}
print(f"Config: {gen_kwargs}")

with torch.no_grad():
    summary_ids = model.generate(input_ids, **gen_kwargs)
    output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print(f"\nPyTorch Output:\n{output_text}")

# 5. Config Check
print("\n" + "=" * 50)
print("CONFIG CHECK")
print("=" * 50)
print(f"scale_embedding: {model.config.scale_embedding}")
