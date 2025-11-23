use edgetransformers::traits::LanguageModelConfig;

#[derive(Debug)]
pub struct VramEstimate {
    pub model_weights_mb: f64,
    pub kv_cache_mb: f64,
    pub activations_mb: f64,
    pub working_memory_mb: f64,
    pub total_mb: f64,
}

impl VramEstimate {
    pub fn print(&self) {
        println!("\n=== VRAM Usage Estimate ===");
        println!("Model Weights:    {:.2} MB ({:.2} GB)", self.model_weights_mb, self.model_weights_mb / 1024.0);
        println!("KV Cache:         {:.2} MB ({:.2} GB)", self.kv_cache_mb, self.kv_cache_mb / 1024.0);
        println!("Activations:      {:.2} MB ({:.2} GB)", self.activations_mb, self.activations_mb / 1024.0);
        println!("Working Memory:   {:.2} MB ({:.2} GB)", self.working_memory_mb, self.working_memory_mb / 1024.0);
        println!("──────────────────────────────");
        println!("Total Estimate:   {:.2} MB ({:.2} GB)", self.total_mb, self.total_mb / 1024.0);
        println!("===========================\n");
    }

    pub fn total_gb(&self) -> f64 {
        self.total_mb / 1024.0
    }
}

pub fn estimate_llama_vram(
    config: &dyn LanguageModelConfig,
    max_seq_len: usize,
    batch_size: usize,
) -> VramEstimate {
    let hidden_size = config.hidden_size();
    let num_layers = config.num_hidden_layers();
    let vocab_size = config.vocab_size();
    let num_kv_heads = config.num_key_value_heads();
    let head_dim = hidden_size / config.num_attention_heads();
    
    // 1. Model Weights (fp16 = 2 bytes per parameter)
    let embedding_params = vocab_size * hidden_size;
    let attention_params_per_layer = 4 * hidden_size * hidden_size; // Q, K, V, O projections
    let ffn_params_per_layer = 8 * hidden_size * hidden_size; // Typical FFN is 4x hidden, with gate
    let layer_params = (attention_params_per_layer + ffn_params_per_layer) * num_layers;
    let total_params = embedding_params + layer_params;
    let model_weights_bytes = total_params * 2; // fp16
    let model_weights_mb = model_weights_bytes as f64 / 1_048_576.0;

    // 2. KV Cache (fp32 = 4 bytes)
    // Shape: [num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]
    // We store both K and V, so multiply by 2
    let kv_cache_bytes = 2 * num_layers * batch_size * num_kv_heads * max_seq_len * head_dim * 4;
    let kv_cache_mb = kv_cache_bytes as f64 / 1_048_576.0;

    // 3. Activations (rough estimate for single forward pass)
    // Main contributors: attention scores, intermediate FFN states
    let attention_scores_per_layer = batch_size * config.num_attention_heads() * max_seq_len * max_seq_len * 4;
    let ffn_intermediate = batch_size * max_seq_len * (hidden_size * 4) * 4; // FFN intermediate is ~4x hidden
    let activations_per_layer = attention_scores_per_layer + ffn_intermediate;
    let activations_bytes = activations_per_layer * num_layers;
    let activations_mb = activations_bytes as f64 / 1_048_576.0;

    // 4. Working memory (for temporary buffers, copies, etc.)
    // Conservative estimate: 20% of model weights
    let working_memory_mb = model_weights_mb * 0.2;

    let total_mb = model_weights_mb + kv_cache_mb + activations_mb + working_memory_mb;

    VramEstimate {
        model_weights_mb,
        kv_cache_mb,
        activations_mb,
        working_memory_mb,
        total_mb,
    }
}

/// Estimates VRAM for any decoder model
pub fn estimate_decoder_vram(
    hidden_size: usize,
    num_layers: usize,
    vocab_size: usize,
    num_kv_heads: usize,
    num_attention_heads: usize,
    max_seq_len: usize,
    batch_size: usize,
) -> VramEstimate {
    let head_dim = hidden_size / num_attention_heads;
    
    // Model weights (fp16)
    let embedding_params = vocab_size * hidden_size;
    let attention_params_per_layer = 4 * hidden_size * hidden_size;
    let ffn_params_per_layer = 8 * hidden_size * hidden_size;
    let layer_params = (attention_params_per_layer + ffn_params_per_layer) * num_layers;
    let total_params = embedding_params + layer_params;
    let model_weights_mb = (total_params * 2) as f64 / 1_048_576.0;

    // KV Cache (fp32)
    let kv_cache_mb = (2 * num_layers * batch_size * num_kv_heads * max_seq_len * head_dim * 4) as f64 / 1_048_576.0;

    // Activations (rough estimate)
    let attention_scores_per_layer = batch_size * num_attention_heads * max_seq_len * max_seq_len * 4;
    let ffn_intermediate = batch_size * max_seq_len * (hidden_size * 4) * 4;
    let activations_mb = ((attention_scores_per_layer + ffn_intermediate) * num_layers) as f64 / 1_048_576.0;

    // Working memory
    let working_memory_mb = model_weights_mb * 0.2;

    let total_mb = model_weights_mb + kv_cache_mb + activations_mb + working_memory_mb;

    VramEstimate {
        model_weights_mb,
        kv_cache_mb,
        activations_mb,
        working_memory_mb,
        total_mb,
    }
}

/// Check if there's enough VRAM before loading
pub fn check_vram_available(
    estimate: &VramEstimate,
    available_vram_bytes: Option<u64>,
) -> Result<(), String> {
    if let Some(available) = available_vram_bytes {
        let available_gb = available as f64 / 1_073_741_824.0;
        let required_gb = estimate.total_gb();

        if required_gb > available_gb {
            return Err(format!(
                "Insufficient VRAM! Required: {:.2} GB, Available: {:.2} GB (shortfall: {:.2} GB)",
                required_gb,
                available_gb,
                required_gb - available_gb
            ));
        }

        // Warn if using >90% of available VRAM
        let usage_percent = (required_gb / available_gb) * 100.0;
        if usage_percent > 90.0 {
            println!("⚠️  WARNING: Will use {:.1}% of available VRAM - may be unstable!", usage_percent);
        } else {
            println!("✓ VRAM check passed: {:.2} GB required, {:.2} GB available ({:.1}% usage)",
                     required_gb, available_gb, usage_percent);
        }
    } else {
        println!("⚠️  Could not determine available VRAM - proceeding without check");
    }

    Ok(())
}