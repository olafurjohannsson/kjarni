import torch
import numpy as np
from transformers import WhisperModel, WhisperProcessor, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer
import soundfile as sf

# Load audio
audio, sr = sf.read("crates/kjarni-models/examples/hideyowife.wav")
print(f"Audio: {len(audio)} samples, sr={sr}")
if sr != 16000:
    import resampy
    audio = resampy.resample(audio, sr, 16000)
    sr = 16000

# Get mel spectrogram from HF
feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-base")
mel_input = feature_extractor(audio, sampling_rate=sr, return_tensors="pt").input_features
print(f"=== Mel Spectrogram ===")
print(f"Shape: {mel_input.shape}")
print(f"Min/Max: {mel_input.min().item():.6f} / {mel_input.max().item():.6f}")
print(f"mel[0, 0, :10]: {mel_input[0, 0, :10].tolist()}")
print(f"mel[0, 0, 100:110]: {mel_input[0, 0, 100:110].tolist()}")
print()

# Load HF model (whisper-base)
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.eval()

# Add this to the Python script
with torch.no_grad():
    decoder = model.model.decoder
    
    decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]])
    
    # 1. Token embeddings
    token_embeds = decoder.embed_tokens(decoder_input_ids)
    print(f"=== Decoder token embeddings ===")
    print(f"Shape: {token_embeds.shape}")
    print(f"[0, 0, :10]: {token_embeds[0, 0, :10].tolist()}")
    print(f"[0, -1, :10]: {token_embeds[0, -1, :10].tolist()}")
    
    # 2. Position embeddings  
    # Whisper's embed_positions takes input_ids directly (uses their length)
    # It has a built-in offset. Let's get raw weights instead:
    pos_weights = decoder.embed_positions.weight
    print(f"\n=== Decoder position embedding weights ===")
    print(f"Shape: {pos_weights.shape}")
    print(f"[0, :10]: {pos_weights[0, :10].tolist()}")
    print(f"[1, :10]: {pos_weights[1, :10].tolist()}")
    print(f"[2, :10]: {pos_weights[2, :10].tolist()}")
    print(f"[3, :10]: {pos_weights[3, :10].tolist()}")

    # What embed_positions actually returns:
    # Whisper has past_key_values_length offset
    positions = torch.arange(4).unsqueeze(0)
    pos_embeds = decoder.embed_positions(positions)
    print(f"\n=== Decoder embed_positions output ===")
    print(f"Shape: {pos_embeds.shape}")
    print(f"[0, :10]: {pos_embeds[0, :10].tolist()}")
    print(f"[1, :10]: {pos_embeds[1, :10].tolist()}")

    # Combined
    combined = decoder.embed_tokens(decoder_input_ids) + pos_embeds
    print(f"\n=== Combined (token + pos) ===")
    print(f"[0, 0, :10]: {combined[0, 0, :10].tolist()}")
    print(f"[0, -1, :10]: {combined[0, -1, :10].tolist()}")
# with torch.no_grad():
#     # === ENCODER ===
#     encoder = model.model.encoder
    
#     # 1. Conv frontend
#     inputs_embeds = torch.nn.functional.gelu(encoder.conv1(mel_input))
#     print(f"=== After conv1 + GELU ===")
#     print(f"Shape: {inputs_embeds.shape}")
#     print(f"Min/Max: {inputs_embeds.min().item():.6f} / {inputs_embeds.max().item():.6f}")
#     print(f"[0, :5, 0]: {inputs_embeds[0, :5, 0].tolist()}")
#     print()
    
#     inputs_embeds = torch.nn.functional.gelu(encoder.conv2(inputs_embeds))
#     print(f"=== After conv2 + GELU ===")
#     print(f"Shape: {inputs_embeds.shape}")
#     print(f"Min/Max: {inputs_embeds.min().item():.6f} / {inputs_embeds.max().item():.6f}")
#     print(f"[0, :5, 0]: {inputs_embeds[0, :5, 0].tolist()}")
#     print()
    
#     inputs_embeds = inputs_embeds.permute(0, 2, 1)
#     print(f"=== After permute (before pos embed) ===")
#     print(f"Shape: {inputs_embeds.shape}")
#     print(f"[0, 0, :10]: {inputs_embeds[0, 0, :10].tolist()}")
#     print()
    
#     # 2. Position embeddings
#     embed_pos = encoder.embed_positions.weight[:1500]
#     print(f"=== Position embeddings ===")
#     print(f"Shape: {embed_pos.shape}")
#     print(f"pos[0, :10]: {embed_pos[0, :10].tolist()}")
#     print(f"pos[1, :10]: {embed_pos[1, :10].tolist()}")
#     print()
    
#     inputs_embeds = inputs_embeds + embed_pos
#     print(f"=== After adding positions ===")
#     print(f"Shape: {inputs_embeds.shape}")
#     print(f"[0, 0, :10]: {inputs_embeds[0, 0, :10].tolist()}")
#     print(f"Min/Max: {inputs_embeds.min().item():.6f} / {inputs_embeds.max().item():.6f}")
#     print()
    
#     # 3. Full encoder output
#     encoder_output = model.model.encoder(mel_input)
#     enc_hidden = encoder_output.last_hidden_state
#     print(f"=== Encoder output ===")
#     print(f"Shape: {enc_hidden.shape}")
#     print(f"Min/Max: {enc_hidden.min().item():.6f} / {enc_hidden.max().item():.6f}")
#     print(f"Mean: {enc_hidden.mean().item():.6f}")
#     print(f"[0, 0, :10]: {enc_hidden[0, 0, :10].tolist()}")
#     print(f"[0, 0, -10:]: {enc_hidden[0, 0, -10:].tolist()}")
#     print()
    
#     # === DECODER ===
#     decoder_input_ids = torch.tensor([[50258, 50259, 50359, 50363]])
    
#     decoder_output = model.model.decoder(
#         input_ids=decoder_input_ids,
#         encoder_hidden_states=enc_hidden,
#     )
#     dec_hidden = decoder_output.last_hidden_state
#     print(f"=== Decoder output (after prompt) ===")
#     print(f"Shape: {dec_hidden.shape}")
#     print(f"[0, -1, :10]: {dec_hidden[0, -1, :10].tolist()}")
#     print()
    
#     # Logits
#     logits = model.proj_out(dec_hidden)
#     last_logits = logits[0, -1]
#     print(f"=== Logits (last position) ===")
#     top_values, top_indices = torch.topk(last_logits, 10)
#     for val, idx in zip(top_values.tolist(), top_indices.tolist()):
#         print(f"  Token {idx}: {val:.4f}")
#     print()
    
#     # Generate 20 tokens
#     next_token = last_logits.argmax().item()
#     generated = [next_token]
    
#     for step in range(20):
#         all_ids = torch.tensor([[50258, 50259, 50359, 50363] + generated])
#         out = model(
#             encoder_outputs=(enc_hidden,),
#             decoder_input_ids=all_ids,
#         )
#         next_logits = out.logits[0, -1]
#         next_token = next_logits.argmax().item()
#         generated.append(next_token)
#         if next_token == 50257:
#             break
    
#     tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-base")
#     print(f"Generated tokens: {generated}")
#     print(f"Decoded: {tokenizer.decode(generated, skip_special_tokens=True)}")