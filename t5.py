from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/home/olafurj/.cache/kjarni/google_flan-t5-base")
print(f"EOS token ID: {tokenizer.eos_token_id}")
print(f"EOS token: {tokenizer.eos_token}")

# Encode "Eiffel Tower" to see tokens
tokens = tokenizer.encode("Eiffel Tower", add_special_tokens=False)
print(f"'Eiffel Tower' tokens: {tokens}")

# Check what comes after in a generation
text = "Eiffel Tower</s>"  # with EOS
tokens = tokenizer.encode(text, add_special_tokens=False)
print(f"'Eiffel Tower</s>' tokens: {tokens}")