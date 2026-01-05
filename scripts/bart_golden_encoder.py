import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# Setup
model_name = "sshleifer/distilbart-cnn-12-6"
print(f"Loading {model_name}...")
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)
model.eval()

# Input
text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency . It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector . Rust was influenced by languages like C++, Haskell, and Erlang ."
input_ids = tokenizer(text, return_tensors="pt")["input_ids"]

print("\n" + "="*50)
print("GENERATING GOLDEN VALUES FOR RUST TEST")
print("="*50)
# At the top, after tokenizing
input_ids = input_ids[:, :10]  # Use only first 10 tokens
print(f"input_ids: {input_ids[0].tolist()}")

with torch.no_grad():
    # Get embeddings
    embeds = model.model.shared(input_ids)
    embed_scale = model.model.encoder.embed_scale if hasattr(model.model.encoder, "embed_scale") else 1.0
    embeds = embeds * embed_scale
    positions = model.model.encoder.embed_positions(input_ids)
    hidden = embeds + positions
    hidden = model.model.encoder.layernorm_embedding(hidden)
    
    print(f"\n// Layer 0 Input")
    print(f"let layer0_input = vec!{hidden[0, 0, :10].tolist()};")
    
    layer0 = model.model.encoder.layers[0]
    
    # Self-attention output (before residual)
    attn_out, _ = layer0.self_attn(hidden_states=hidden, attention_mask=None, output_attentions=False)
    print(f"\n// After self_attn (before residual)")
    print(f"let attn_out = vec!{attn_out[0, 0, :10].tolist()};")
    
    # After residual + layernorm
    hidden_post_attn = layer0.self_attn_layer_norm(hidden + attn_out)
    print(f"\n// After attn residual + layernorm")
    print(f"let post_attn_ln = vec!{hidden_post_attn[0, 0, :10].tolist()};")

    
    # After getting post_attn_ln...
    fc1_out = layer0.fc1(hidden_post_attn)
    print(f"\n// After FC1 (before activation)")
    print(f"let fc1_out = vec!{fc1_out[0, 0, :10].tolist()};")

    fc1_gelu = layer0.activation_fn(fc1_out)
    print(f"\n// After FC1 + GELU (before FC2)")  
    print(f"let fc1_gelu = vec!{fc1_gelu[0, 0, :10].tolist()};")

    
    # FFN
    ffn_out = layer0.fc2(layer0.activation_fn(layer0.fc1(hidden_post_attn)))
    print(f"\n// After FFN (before residual)")
    print(f"let ffn_out = vec!{ffn_out[0, 0, :10].tolist()};")
    
    # Final
    layer0_out = layer0.final_layer_norm(hidden_post_attn + ffn_out)
    print(f"\n// Layer 0 Final Output")
    print(f"let layer0_output = vec!{layer0_out[0, 0, :10].tolist()};")
    print(f"\n// FC1 weight shape: {layer0.fc1.weight.shape}")  # Should be [4096, 1024]
    print(f"// FC1 bias shape: {layer0.fc1.bias.shape}")        # Should be [4096]
    print(f"// FC1 weight[0, 0:5]: {layer0.fc1.weight[0, 0:5].tolist()}")
    print(f"// FC1 bias[0:5]: {layer0.fc1.bias[0:5].tolist()}")
    # After getting hidden (layer0_input)
    with torch.no_grad():
        layer0 = model.model.encoder.layers[0]
        
        # Get Q, K, V
        q = layer0.self_attn.q_proj(hidden)
        k = layer0.self_attn.k_proj(hidden)
        v = layer0.self_attn.v_proj(hidden)
        
        # Reshape to heads
        batch, seq_len, _ = hidden.shape
        num_heads = layer0.self_attn.num_heads
        head_dim = layer0.self_attn.head_dim
        
        q_heads = q.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
        k_heads = k.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
        v_heads = v.view(batch, seq_len, num_heads, head_dim).transpose(1, 2)  # [B, H, S, D]
        
        print(f"\n// q_heads shape: {q_heads.shape}")
        print(f"// k_heads shape: {k_heads.shape}")
        
        # Scores = Q @ K^T
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1))
        print(f"// scores shape: {scores.shape}")
        print(f"// scores[0,0,0,0:5]: {scores[0, 0, 0, :5].tolist()}")
        
        # Scaled
        scale = layer0.self_attn.scaling
        print(f"// scale_factor: {scale}")
        scaled = scores * scale
        print(f"// scaled_scores[0,0,0,0:5]: {scaled[0, 0, 0, :5].tolist()}")
        # Softmax
        attn_weights = torch.nn.functional.softmax(scaled, dim=-1)
        print(f"// softmax[0,0,0,0:5]: {attn_weights[0, 0, 0, :5].tolist()}")
        
        # Context
        context = torch.matmul(attn_weights, v_heads)
        print(f"// context[0,0,0,0:5]: {context[0, 0, 0, :5].tolist()}")
        
        # Reshape back
        context_reshaped = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        
        # Output projection
        attn_output = layer0.self_attn.out_proj(context_reshaped)
        print(f"// attn_output[0,0,0:10]: {attn_output[0, 0, :10].tolist()}")