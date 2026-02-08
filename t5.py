"""
Dump per-step logits for greedy BART-large-cnn generation.
Compare with kjarni's decode_step output to find where divergence starts.

Usage:
    python bart_debug_logits.py
"""

import os
import torch
import numpy as np
from transformers import BartForConditionalGeneration, AutoTokenizer

CACHE = os.path.expanduser("~/.cache/kjarni")

model_path = os.path.join(CACHE, "facebook_bart-large-cnn")
print(f"Loading from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BartForConditionalGeneration.from_pretrained(model_path)
model.eval()

input_text = (
    "The Federal Reserve announced today that it would hold interest rates "
    "steady at their current level, citing ongoing concerns about inflation "
    "and economic uncertainty. The decision was widely expected by analysts "
    "and marks the third consecutive meeting where rates have remained unchanged. "
    "Fed Chair Jerome Powell stated that the central bank remains committed to "
    "bringing inflation down to its two percent target but acknowledged that "
    "progress has been slower than anticipated. Markets reacted positively to "
    "the announcement, with the S&P 500 rising half a percent in afternoon trading."
)

# Encode input
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024)
encoder_input_ids = inputs.input_ids
encoder_attention_mask = inputs.attention_mask

print(f"Encoder input: {encoder_input_ids.shape[1]} tokens")

# Run encoder
with torch.no_grad():
    encoder_outputs = model.get_encoder()(
        encoder_input_ids,
        attention_mask=encoder_attention_mask,
    )

print(f"Encoder output shape: {encoder_outputs.last_hidden_state.shape}")
enc_hidden = encoder_outputs.last_hidden_state
print(f"Encoder hidden[0,0,:5]: {enc_hidden[0, 0, :5].tolist()}")
print(f"Encoder hidden[0,-1,:5]: {enc_hidden[0, -1, :5].tolist()}")

# Manual greedy decode step by step
decoder_start_token_id = 2  # BART decoder_start_token_id
forced_bos_token_id = 0     # BART forced_bos_token_id
eos_token_id = 2
max_length = 80
min_length = 20

# Start with decoder_start_token_id
decoder_input_ids = torch.tensor([[decoder_start_token_id]])
past_key_values = None

print(f"\n{'='*80}")
print(f"STEP-BY-STEP GREEDY DECODE (max_length={max_length}, min_length={min_length})")
print(f"{'='*80}\n")

generated_tokens = [decoder_start_token_id]

for step in range(max_length - 1):
    with torch.no_grad():
        outputs = model(
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=True,
        )
    
    logits = outputs.logits[:, -1, :]  # (1, vocab_size)
    past_key_values = outputs.past_key_values
    
    current_len = step + 2  # decoder_start + generated so far
    
    # Apply forced_bos at step 0
    if step == 0:
        next_token = forced_bos_token_id
        logits_display = logits.clone()
    else:
        logits_display = logits.clone()
        
        # Suppress EOS if below min_length
        if current_len < min_length:
            logits[:, eos_token_id] = float('-inf')
        
        next_token = logits.argmax(dim=-1).item()
    
    # Print top 5 logits
    top5_values, top5_indices = logits_display[0].topk(5)
    top5_tokens = [(tokenizer.decode([idx.item()]), idx.item(), val.item()) 
                   for idx, val in zip(top5_indices, top5_values)]
    
    selected_token_str = tokenizer.decode([next_token])
    print(f"Step {step:3d} | len={current_len:3d} | selected: {next_token:5d} '{selected_token_str}' | "
          f"top5: {[(t, id, f'{v:.4f}') for t, id, v in top5_tokens]}")
    
    if next_token == eos_token_id and current_len >= min_length:
        print(f"\n  >>> EOS at step {step}, stopping")
        break
    
    generated_tokens.append(next_token)
    decoder_input_ids = torch.tensor([[next_token]])

# Decode final output
output_tokens = [t for t in generated_tokens if t != decoder_start_token_id]
final_text = tokenizer.decode(output_tokens, skip_special_tokens=True)
print(f"\n{'='*80}")
print(f"Final: {final_text}")
print(f"{'='*80}")