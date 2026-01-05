from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained("sshleifer/distilbart-cnn-12-6")
print(f"decoder_start_token_id: {model.config.decoder_start_token_id}")
print(f"forced_bos_token_id: {model.config.forced_bos_token_id}")
print(f"bos_token_id: {model.config.bos_token_id}")
print(f"force_bos_token_to_be_generated: {model.config.force_bos_token_to_be_generated}")
print("forced_eos_token_id:", model.config.forced_eos_token_id)
print("normalize_embedding: ", model.config.normalize_embedding)