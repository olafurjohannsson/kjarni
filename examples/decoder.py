import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
    """
    This script is a Python equivalent of the Rust text generation logic.
    It manually performs the token-by-token generation loop to make the
    comparison with the Rust code clear.
    """
    # ==================================================================
    # 1. Setup: Load Model and Tokenizer
    # ==================================================================
    print("--- DistilGPT-2 Text Generation Example (Python) ---")
    
    device = "cpu"
    model_name = "distilgpt2"
    
    print(f"Loading {model_name} model on {device}...")
    
    # Equivalent to Tokenizer::from_file(...)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Equivalent to loading weights and building the model from config
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    
    print("✓ Model loaded.")

    # ==================================================================
    # 2. Generation Configuration
    # This mirrors the Rust `GenerationConfig` struct.
    # ==================================================================
    prompt = "The field of Artificial Intelligence has seen a lot of progress"
    max_new_tokens = 100
    temperature = 0.7
    top_k = 50
    top_p = 0.9
    repetition_penalty = 1.1
    
    # Get special token IDs from the tokenizer
    eos_token_id = tokenizer.eos_token_id
    
    # ==================================================================
    # 3. The Generation Loop
    # This section is a direct translation of `generate_text_streaming`
    # or your new `TextGenerator::generate` function.
    # ==================================================================
    
    print("\n--- PROMPT ---")
    print(prompt)
    print("\nGenerating text...")

    # --- Tokenize the prompt ---
    # Equivalent to `tokenizer.encode(...)`
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    
    # This will hold all generated tokens, starting with the prompt
    generated_tokens = input_ids.clone()
    
    # --- Initialize the Key/Value Cache ---
    # `past_key_values` is the Python equivalent of your `past` or `cache` variable.
    past_key_values = None
    
    # We use `torch.no_grad()` to disable gradient calculations for efficiency.
    with torch.no_grad():
        # --- Prompt Processing Pass (Optional but efficient) ---
        # Process the whole prompt at once to fill the cache.
        # This is what your `if !tokens.is_empty()` block does.
        outputs = model(input_ids=input_ids)
        past_key_values = outputs.past_key_values
        
        # The input for the main loop is now just the very last token of the prompt
        current_token = input_ids[:, -1:]

        # --- Token-by-token Generation Loop ---
        for _ in range(max_new_tokens):
            
            # --- Forward Pass ---
            # Equivalent to `model.forward(&input_array, past)`
            # We only pass the *newest* token, but the model uses the `past_key_values`
            # to "remember" the entire sequence.
            outputs = model(input_ids=current_token, past_key_values=past_key_values)
            
            logits = outputs.logits
            past_key_values = outputs.past_key_values # Update the cache

            # --- Get Logits for the Next Token ---
            # Equivalent to `logits.slice(s![0, -1, ..])` or `s![0, 0, ..]`
            # We get the logits for the last token in the sequence (which is our only token).
            next_token_logits = logits[:, -1, :]
            
            # --- Apply Repetition Penalty ---
            # Equivalent to `apply_repetition_penalty(...)`
            # This logic is identical to your Rust implementation.
            if repetition_penalty != 1.0:
                for token_id in generated_tokens[0]:
                    logit = next_token_logits[0, token_id]
                    if logit < 0:
                        next_token_logits[0, token_id] = logit * repetition_penalty
                    else:
                        next_token_logits[0, token_id] = logit / repetition_penalty

            # --- Apply Temperature, Top-K, Top-P Sampling ---
            # This whole block is equivalent to `sample_token(...)`
            
            # 1. Temperature
            # if temperature != 1.0:
            #     next_token_logits = next_token_logits / temperature

            # # 2. Top-K and Top-P
            # # We filter the logits, setting low-probability ones to -inf
            # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            # 3. Sample
            # Convert filtered logits to probabilities and sample one token.
            # Equivalent to `softmax_1d` and `sample_from_probs`
            # probs = F.softmax(filtered_logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            # --- Update for Next Iteration ---
            generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
            current_token = next_token # The next input is the token we just generated

            # --- Check for End of Sequence ---
            if next_token.item() == eos_token_id:
                break
    
    # ==================================================================
    # 4. Decode and Print
    # Equivalent to `tokenizer.decode(...)`
    # ==================================================================
    generated_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    print("\n--- GENERATED TEXT ---")
    print(generated_text)


# This helper function is a direct port of the Top-K/Top-P logic.
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    - Top-k: keeps only top k tokens with highest probability (sets others to -inf).
    - Top-p: keeps the smallest set of tokens with cumulative probability >= top_p.
    """
    top_k = min(top_k, logits.size(-1))  # Safety check

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

if __name__ == "__main__":
    main()
