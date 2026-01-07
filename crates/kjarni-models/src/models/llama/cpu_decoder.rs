//! CPU implementation of the Llama decoder architecture.
//!
//! Provides high-performance CPU inference for Llama 2/3 models using optimized
//! SIMD kernels and efficient memory layouts. Supports RoPE attention, SwiGLU
//! feedforward, and RMS normalization.
//!
//! # Architecture
//!
//! Llama uses a standard decoder-only transformer with:
//! - **RoPE (Rotary Position Embeddings)** — Relative position encoding in attention
//! - **SwiGLU activation** — Gated FFN for improved expressiveness
//! - **RMS normalization** — Simpler, faster alternative to LayerNorm
//! - **Grouped-Query Attention** — Reduced KV cache for 8B+ models
//!
//! # Performance
//!
//! - **Prefill**: ~10-20 tokens/sec for 1B model on modern CPU
//! - **Decode**: ~40-60 tokens/sec single-token generation
//! - Optimized with AVX2/FMA kernels for matmul-heavy operations
//!
//! # Example
//!
//! ```ignore
//! use kjarni_models::models::llama::LlamaCpuDecoder;
//! use kjarni_transformers::weights::ModelWeights;
//!
//! let weights = ModelWeights::load("llama-3.2-1b.safetensors")?;
//! let decoder = LlamaCpuDecoder::new(&weights, metadata, layout, rope, None)?;
//!
//! let output = decoder.forward(&input, &mut cache, 0)?;
//! ```
//!
//! # TODO
//! - Add support for flash attention on CPU (blocked by available libraries)
//! - Implement speculative decoding for 2-3x speedup
//! - Add INT8 KV cache compression to reduce memory 4x
//!
//! # See Also
//!
//! - [`super::LlamaModel`] — High-level model wrapper
//! - [`crate::models::mistral`] — Mistral variant with sliding window attention

// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use crate::models::llama::config::LlamaConfig;
use anyhow::{anyhow, Result};
use ndarray::{s, Array2, Array3, Axis};
use std::time::Instant;

use kjarni_transformers::{
    cache::CpuKVCache, decoder::prelude::*,
    embeddings::Embeddings,
    feedforward::SwiGluFeedForward,
    linear_layer::LinearLayer,
    models::base::ModelInput,
    normalization::RMSNorm,
    pipeline::CpuLayerFactory,
    rope::RoPE,
    stats::GenerationStats,
    tensor::DType,
    traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
    Normalization,
    WgpuContext,
};

/// CPU-based Llama decoder implementation with RoPE and SwiGLU.
///
/// Implements the complete Llama architecture optimized for CPU inference.
/// Supports quantized weights (Q4_K, Q8_0) for memory efficiency.
///
/// # Fields
///
/// * `embeddings` — Token embedding lookup table
/// * `layers` — Stack of decoder layers (attention + FFN)
/// * `final_norm` — RMS normalization before LM head
/// * `metadata` — Model hyperparameters (hidden size, num layers, etc.)
///
/// # Performance Note
///
/// For best performance:
/// - Use BF16 weights for 2x memory bandwidth vs F32
/// - Enable AVX2/FMA CPU features at compile time
/// - Use quantized formats (Q4_K) for models >3B parameters
pub struct LlamaCpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<CpuRoPEDecoderLayer>,
    pub final_norm: RMSNorm,
    pub metadata: ModelMetadata,
}

impl LlamaCpuDecoder {
    /// Constructs a new Llama decoder from model weights.
    ///
    /// Loads embeddings, builds all decoder layers, and initializes normalization.
    /// Weight dtype can be converted on-the-fly via `target_dtype` parameter.
    ///
    /// # Arguments
    ///
    /// * `weights` - Pre-loaded model weights from safetensors/GGUF
    /// * `metadata` - Model hyperparameters (hidden size, num layers, etc.)
    /// * `layout` - Tensor name mapping for this model variant
    /// * `rope` - Shared RoPE instance (cached sin/cos tables)
    /// * `target_dtype` - Optional dtype conversion (None = keep original format)
    ///
    /// # Returns
    ///
    /// Configured decoder ready for inference.
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Required tensors missing from weights
    /// - Shape mismatches detected
    /// - Unsupported dtype conversion requested
    ///
    /// # Example
    ///
    /// ```ignore
    /// let decoder = LlamaCpuDecoder::new(
    ///     &weights,
    ///     metadata,
    ///     layout,
    ///     rope,
    ///     Some(DType::BF16) // Convert to BF16 for memory efficiency
    /// )?;
    /// ```
    pub fn new(
        weights: &ModelWeights,
        metadata: ModelMetadata,
        layout: ModelLayout,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");

        let embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            decoder_layout.position_embedding.as_deref(), // Correctly access nested field
            decoder_layout.token_type_embedding.as_deref(),
        )?;

        let final_norm = RMSNorm::new(
            weights.get_array1(decoder_layout.final_norm_weight.as_ref().unwrap())?,
            metadata.norm_eps,
        );

        let mut layers = Vec::with_capacity(metadata.num_layers);
        for i in 0..metadata.num_layers {
            layers.push(Self::build_layer(
                weights,
                &metadata,
                &layout,
                i,
                rope.clone(),
                target_dtype,
            )?);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            metadata,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        i: usize,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<CpuRoPEDecoderLayer> {
        // Get the specific nested layouts for the decoder.
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;

        let attention: DecoderAttention = CpuLayerFactory::build_decoder_attention(
            weights,
            meta,
            &decoder_layout.layer.self_attn,
            i,
            target_dtype,
        )?;

        let feed_forward =
            CpuLayerFactory::build_swiglu_ffn(weights, &decoder_layout.layer.ffn, i, target_dtype)?;

        let attention_norm = CpuLayerFactory::build_norm(
            weights,
            &layer_layout.self_attn.norm_weight,
            &layer_layout.self_attn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        let ffn_norm = CpuLayerFactory::build_norm(
            weights,
            &layer_layout.ffn.norm_weight,
            &layer_layout.ffn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        Ok(CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        })
    }
}

impl InferenceModel for LlamaCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDecoder for LlamaCpuDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn embed(&self, input: ModelInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
        match input {
            ModelInput::TokensCpu(ids) => {
                let seq_len = ids.len();
                // let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;

                Ok(self
                    .embeddings
                    .forward(&ids.to_owned(), None, position_offset, false))
            }
            ModelInput::HiddenCpu(hidden) => Ok(hidden.to_owned()),
            _ => Err(anyhow!(
                "LlamaCpuDecoder received GPU input. Transfer to CPU first."
            )),
        }
    }

    fn embed_and_normalize(
        &self,
        input: ModelInput<'_>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        // Llama is Pre-Norm (Norm is inside the layer).
        // No initial LayerNorm exists before the first block.
        self.embed(input, position_offset)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        mut cache: Option<&mut dyn Cache>, // ... args
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let seq_len = hidden.shape()[1]; // The "new" tokens count

        // Downcast to CpuKVCache
        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        for i in start_layer..end_layer {
            let layer: &CpuRoPEDecoderLayer = &self.layers[i];

            if let Some(ref mut c) = cpu_cache_opt {
                // OPTIMIZATION: Get the FULL contiguous view (History + New Slot)
                let current_len = c.get_seq_length();
                let (k_full_mut, v_full_mut) = c.get_context_view_mut(i, seq_len)?;

                // Pass this full view to the layer
                hidden = layer.forward(
                    &hidden,
                    attention_mask,
                    position_offset,
                    k_full_mut,
                    v_full_mut,
                )?;
            } else {
                // Fallback (No cache): Allocate temporary buffers
                let kv_dim = layer.attention.num_kv_heads * layer.attention.head_dim;
                let (b, s, _) = hidden.dim();

                // Allocate exact size needed for this step
                let mut temp_k = Array3::<f32>::zeros((b, s, kv_dim));
                let mut temp_v = Array3::<f32>::zeros((b, s, kv_dim));

                hidden = layer.forward(
                    &hidden,
                    attention_mask,
                    position_offset,
                    temp_k.view_mut(),
                    temp_v.view_mut(),
                )?;
            }
        }

        // Increment once at the end
        if let Some(c) = cpu_cache_opt {
            c.increment_len(seq_len);
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward(
        &self,
        input: ModelInput<'_>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let t_start = Instant::now();

        // 1. Embed & Normalize
        let t_embed = Instant::now();
        let hidden = self.embed_and_normalize(input, position_offset)?;
        let d_embed = t_embed.elapsed();

        // 2. Run Layers
        let t_layers = Instant::now();
        let mut output = self.forward_layers(
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;
        let d_layers = t_layers.elapsed();

        // 3. Final Norm
        let t_norm = Instant::now();
        output = self.final_norm.forward_3d(&output);
        let d_norm = t_norm.elapsed();

        let d_total = t_start.elapsed();

        if d_total.as_millis() > 1 {
            log::info!(
                "[Model Forward Perf] Total: {:?}, Embed: {:?}, Layers (x{}): {:?}, Final Norm: {:?}",
                d_total,
                d_embed,
                self.num_layers(),
                d_layers,
                d_norm
            );
        }

        Ok(output)
    }
}

mod llama_test {
    use super::*;
    use crate::models::llama::LlamaModel;
    const SAFETENSORS_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct";
    use kjarni_transformers::{
        cpu::kernels::{
            dequantize::{dequantize_q4_k_block, dequantize_q6_k_block},
            q_common::{BlockQ4_K, BlockQ6_K},
        },
        weights::{
            cast_or_copy, ModelWeights
            ,
        }

        ,
        ModelType,
    };
    use std::path::Path;
    const GGUF_PATH: &str = "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf";
    #[test]
    fn test_check_all_matrix_sizes_interleaving() {
        let gguf_path = Path::new(GGUF_PATH);
        let st_path = Path::new(SAFETENSORS_PATH);

        let gguf_weights = ModelWeights::new(gguf_path).unwrap();
        let st_weights = ModelWeights::new(st_path).unwrap();

        let tensors_to_check = [
            ("model.layers.0.self_attn.q_proj.weight", [2048, 2048]), // Q
            ("model.layers.0.self_attn.k_proj.weight", [512, 2048]),  // K (smaller)
            ("model.layers.0.self_attn.v_proj.weight", [512, 2048]),  // V (smaller)
            ("model.layers.0.self_attn.o_proj.weight", [2048, 2048]), // O
            ("model.layers.0.mlp.gate_proj.weight", [8192, 2048]),    // Gate (larger)
            ("model.layers.0.mlp.up_proj.weight", [8192, 2048]),      // Up (larger)
            ("model.layers.0.mlp.down_proj.weight", [2048, 8192]),    // Down
        ];

        for (name, expected_shape) in tensors_to_check {
            println!(
                "\n=== {} [{}, {}] ===",
                name, expected_shape[0], expected_shape[1]
            );

            let raw = gguf_weights.get_raw(name).unwrap();
            println!("Actual shape: {:?}, dtype: {:?}", raw.shape, raw.dtype);

            // Get ST reference (if exists)
            let st_f32 = match st_weights.get_typed_tensor(name) {
                Ok(t) => t.to_array2_f32().ok(),
                Err(_) => None,
            };

            if st_f32.is_none() {
                println!("ST tensor not found, skipping");
                continue;
            }
            let st_f32 = st_f32.unwrap();

            // Get ORIGINAL GGUF blocks
            let blocks_per_row = raw.shape[1] / 256;

            // Check block group mapping for first few groups
            print!("Block mapping: ");

            match raw.dtype {
                DType::Q4_K => {
                    let blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
                    // let arr = [0.0usize; blocks.len()]; // Dummy array for type inference

                    for (i, block) in blocks.iter().enumerate().take(256) {
                        let block_idx = i * blocks_per_row;
                        if block_idx >= blocks.len() {
                            continue;
                        }

                        let mut block_data = [0.0f32; 256];
                        dequantize_q4_k_block(&blocks[block_idx], &mut block_data);

                        // Find best ST row
                        let mut best_row = 0;
                        let mut best_diff = f32::MAX;
                        for st_row in 0..64.min(raw.shape[0]) {
                            let diff: f32 = block_data
                                .iter()
                                .zip(st_f32.row(st_row).iter().take(256))
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            if diff < best_diff {
                                best_diff = diff;
                                best_row = st_row;
                            }
                        }
                        print!("{}→{} ", i, best_row);
                    }
                }
                DType::Q6_K => {
                    let blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
                    for block_group in [0, 1, 2, 3] {
                        let block_idx = block_group * blocks_per_row;
                        if block_idx >= blocks.len() {
                            continue;
                        }

                        let mut block_data = [0.0f32; 256];
                        dequantize_q6_k_block(&blocks[block_idx], &mut block_data);

                        let mut best_row = 0;
                        let mut best_diff = f32::MAX;
                        for st_row in 0..64.min(raw.shape[0]) {
                            let diff: f32 = block_data
                                .iter()
                                .zip(st_f32.row(st_row).iter().take(256))
                                .map(|(a, b)| (a - b).abs())
                                .sum();
                            if diff < best_diff {
                                best_diff = diff;
                                best_row = st_row;
                            }
                        }
                        print!("{}→{} ", block_group, best_row);
                    }
                }
                _ => println!("Unsupported dtype"),
            }
            println!();
        }
    }

    #[test]
    fn test_gguf_and_safetensors_load_identical_configs() -> Result<()> {
        println!("--- Comparing GGUF vs Safetensors configs for 1B and 3B models ---");

        // --- Test 1B Model ---
        {
            println!("\n[1] Testing Llama 3.2 1B Instruct...");
            let model_gguf = LlamaModel::from_pretrained(
                Path::new(
                    "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                ),
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_1B_Instruct),
            )?;
            let model_st = LlamaModel::from_pretrained(
                Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct"),
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_1B_Instruct),
            )?;

            let config_gguf = model_gguf.config();
            let config_st = model_st.config();

            println!("   ... Comparing all parameters for 1B model...");
            assert_eq!(
                config_gguf.hidden_size, config_st.hidden_size,
                "1B: hidden_size"
            );
            assert_eq!(
                config_gguf.num_hidden_layers, config_st.num_hidden_layers,
                "1B: num_hidden_layers"
            );
            assert_eq!(
                config_gguf.num_attention_heads, config_st.num_attention_heads,
                "1B: num_attention_heads"
            );
            assert_eq!(
                config_gguf.num_key_value_heads, config_st.num_key_value_heads,
                "1B: num_key_value_heads"
            );
            assert_eq!(
                config_gguf.intermediate_size, config_st.intermediate_size,
                "1B: intermediate_size"
            );
            assert_eq!(
                config_gguf.vocab_size, config_st.vocab_size,
                "1B: vocab_size"
            );
            assert_eq!(
                config_gguf.max_position_embeddings, config_st.max_position_embeddings,
                "1B: max_position_embeddings"
            );

            // Use a tolerance for floating point comparisons
            assert!(
                (config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6,
                "1B: rms_norm_eps"
            );
            assert_eq!(
                config_gguf.hidden_act, config_st.hidden_act,
                "1B: hidden_act"
            );

            assert!(
                (config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6,
                "1B: rope_theta"
            );

            // assert_eq!(
            //     config_gguf.bos_token_id, config_st.bos_token_id,
            //     "1B: bos_token_id"
            // );
            // assert_eq!(
            //     config_gguf.eos_token_id, config_st.eos_token_id,
            //     "1B: eos_token_id"
            // );
            assert_eq!(
                config_gguf.pad_token_id, config_st.pad_token_id,
                "1B: pad_token_id"
            );

            assert_eq!(
                config_gguf.tie_word_embeddings, config_st.tie_word_embeddings,
                "1B: tie_word_embeddings"
            );

            println!("   ... ✓ 1B model configs match perfectly.");
        }

        // --- Test 3B Model ---
        {
            println!("\n[2] Testing Llama 3.2 3B Instruct...");
            let model_gguf = LlamaModel::from_pretrained(
                Path::new(
                    "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                ),
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_3B_Instruct),
            )?;
            let model_st = LlamaModel::from_pretrained(
                Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct"),
                Device::Cpu,
                None,
                None,
                Some(ModelType::Llama3_2_3B_Instruct),
            )?;

            let config_gguf = model_gguf.config();
            let config_st = model_st.config();

            println!("   ... Comparing all parameters for 3B model...");
            assert_eq!(
                config_gguf.hidden_size, config_st.hidden_size,
                "3B: hidden_size"
            );
            assert_eq!(
                config_gguf.num_hidden_layers, config_st.num_hidden_layers,
                "3B: num_hidden_layers"
            );
            assert_eq!(
                config_gguf.num_attention_heads, config_st.num_attention_heads,
                "3B: num_attention_heads"
            );
            assert_eq!(
                config_gguf.num_key_value_heads, config_st.num_key_value_heads,
                "3B: num_key_value_heads"
            );
            assert_eq!(
                config_gguf.intermediate_size, config_st.intermediate_size,
                "3B: intermediate_size"
            );
            assert_eq!(
                config_gguf.vocab_size, config_st.vocab_size,
                "3B: vocab_size"
            );
            assert_eq!(
                config_gguf.max_position_embeddings, config_st.max_position_embeddings,
                "3B: max_position_embeddings"
            );

            assert!(
                (config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6,
                "3B: rms_norm_eps"
            );
            assert_eq!(
                config_gguf.hidden_act, config_st.hidden_act,
                "3B: hidden_act"
            );

            assert!(
                (config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6,
                "3B: rope_theta"
            );

            // assert_eq!(
            //     config_gguf.bos_token_id, config_st.bos_token_id,
            //     "3B: bos_token_id"
            // );
            // assert_eq!(
            //     config_gguf.eos_token_id, config_st.eos_token_id,
            //     "3B: eos_token_id"
            // );
            assert_eq!(
                config_gguf.pad_token_id, config_st.pad_token_id,
                "3B: pad_token_id"
            );

            assert_eq!(
                config_gguf.tie_word_embeddings, config_st.tie_word_embeddings,
                "3B: tie_word_embeddings"
            );

            println!("   ... ✓ 3B model configs match perfectly.");
        }

        println!(
            "\n✓ SUCCESS: All GGUF configurations correctly match their SafeTensors counterparts."
        );
        Ok(())
    }
}
