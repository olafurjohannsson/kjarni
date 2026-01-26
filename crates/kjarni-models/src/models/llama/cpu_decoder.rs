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

use std::sync::Arc;
use anyhow::{Result};
use ndarray::{Array2, Array3};

use kjarni_transformers::{
    WgpuContext, activations::Activation, cache::CpuKVCache, decoder::prelude::*, normalization::RMSNorm, pipeline::CpuLayerFactory, rope::RoPE, tensor::DType, traits::{Cache, Device, InferenceModel, ModelLayout, ModelMetadata}, weights::ModelWeights
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
    // pub embeddings: Embeddings,
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
        if target_dtype.is_some() {
            println!("Creating Llama CPU Decoder with dtype {:?}", target_dtype);
        } else {
            println!("No dtype set on Llama CPU Decoder!");
        }

        let start_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();
        println!("  [LlamaCpu] Pre-Layers RAM: {:.2} MB", start_ram);

        // let embeddings = Embeddings::from_weights(
        //     weights,
        //     &layout.token_embedding,
        //     decoder_layout.position_embedding.as_deref(), // Correctly access nested field
        //     decoder_layout.token_type_embedding.as_deref(),
        //     target_dtype,
        // )?;

        let mid_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();
        println!(
            "  [LlamaCpu] Post-Embeddings RAM: {:.2} MB (Delta: {:.2} MB)",
            mid_ram,
            mid_ram - start_ram
        );

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
        let end_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();
        println!(
            "  [LlamaCpu] Post-Layers RAM: {:.2} MB (Delta: {:.2} MB)",
            end_ram,
            end_ram - mid_ram
        );
        Ok(Self {
            // embeddings,
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
        let factory = CpuLayerFactory::new(weights).with_target_dtype(target_dtype);
        let attention: DecoderAttention = factory.build_decoder_attention(
            meta,
            &decoder_layout.layer.self_attn,
            i,
        )?;

        let feed_forward =
            factory.build_swiglu_ffn(&decoder_layout.layer.ffn, Activation::SilU, i)?;

        let attention_norm = factory.build_norm(
            &layer_layout.self_attn.norm_weight,
            &layer_layout.self_attn.norm_bias,
            meta.norm_eps,
            i,
        )?;

        let ffn_norm = factory.build_norm(
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
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn head_dim(&self) -> usize {
        0
    }
    fn hidden_size(&self) -> usize {
        0
    }
    fn num_attention_heads(&self) -> usize {
        0
    }

    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        Ok(self.final_norm.forward_3d(hidden_states))
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
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let mut output = self.forward_layers(
            &hidden_states,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;
        output = self.final_norm.forward_3d(&output);
        Ok(output)
    }
}

mod llama_test {
    use super::*;
    use crate::models::llama::LlamaModel;
    const SAFETENSORS_PATH: &str = "/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct";
    use kjarni_transformers::{
        ModelType,
        cpu::kernels::{
            dequantize::{dequantize_q4_k_block, dequantize_q6_k_block},
            q_common::{BlockQ4_K, BlockQ6_K},
        },
        weights::{ModelWeights, cast_or_copy},
    };
    use ndarray::ArrayView1;
    use std::path::Path;
    const GGUF_PATH: &str = "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf";

    // Helper to calculate cosine similarity between two 1D slices
    fn cosine_similarity(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }
        dot / (norm_a * norm_b)
    }

    #[test]
    fn test_gguf_safetensors_alignment_parity() -> Result<()> {
        // Define paths (use environment variables or hardcoded local paths)
        let gguf_path =
            Path::new("/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6/model-q4_k.gguf");
        let st_path = Path::new("/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6");

        if !gguf_path.exists() || !st_path.exists() {
            println!("Skipping test: Model files not found for comparison.");
            return Ok(());
        }

        let gguf_weights = ModelWeights::new(gguf_path)?;
        let st_weights = ModelWeights::new(st_path)?;

        let tensors_to_check = [
            // {Name}, {Is Interleaved in GGUF?}
            ("model.layers.0.self_attn.q_proj.weight", true), // Q (Needs Reordering)
            ("model.layers.0.self_attn.k_proj.weight", true), // K (Needs Reordering)
            ("model.layers.0.self_attn.v_proj.weight", false), // V (Standard)
            ("model.layers.0.self_attn.o_proj.weight", false), // O (Standard)
            ("model.layers.0.mlp.gate_proj.weight", false),   // Gate
        ];

        for (name, is_interleaved) in tensors_to_check {
            println!(
                "\n=== Checking Tensor: {} (Interleaved: {}) ===",
                name, is_interleaved
            );

            // 1. Load Ground Truth (SafeTensors)
            if !st_weights.contains(name) {
                println!("Skipping {}: Not found in SafeTensors", name);
                continue;
            }
            let st_arr = st_weights.get_array2(name)?;

            // 2. Load GGUF (This triggers the reordering logic inside get_array2)
            if !gguf_weights.contains(name) {
                println!("Skipping {}: Not found in GGUF", name);
                continue;
            }
            let gguf_arr = gguf_weights.get_array2(name)?;

            // 3. Check Shapes
            assert_eq!(
                gguf_arr.shape(),
                st_arr.shape(),
                "Shape mismatch for {}",
                name
            );

            // 4. Verify Content (Row by Row)
            // We check the first 64 rows (covering at least one super-block)
            let rows_to_check = 64.min(gguf_arr.nrows());
            let mut total_sim = 0.0;

            for i in 0..rows_to_check {
                let sim = cosine_similarity(gguf_arr.row(i), st_arr.row(i));
                total_sim += sim;

                // Thresholds:
                // Q4_K vs F32 should have high similarity (>0.90 usually).
                // If reordering is wrong, similarity will be near 0 (random vectors).
                if sim < 0.80 {
                    println!(
                        "!! FAILURE at Row {}: Similarity = {:.4} (Expected > 0.8)",
                        i, sim
                    );

                    // Debug print first few values
                    println!(
                        "  GGUF: {:?}",
                        gguf_arr.row(i).iter().take(5).collect::<Vec<_>>()
                    );
                    println!(
                        "  ST  : {:?}",
                        st_arr.row(i).iter().take(5).collect::<Vec<_>>()
                    );

                    if is_interleaved {
                        panic!(
                            "Interleaved layer '{}' failed parity check at row {}. \
                            This indicates `gguf_conversion.rs` reordering logic is incorrect.",
                            name, i
                        );
                    } else {
                        panic!(
                            "Standard layer '{}' failed parity check. Basic loading is broken.",
                            name
                        );
                    }
                }
            }

            let avg_sim = total_sim / rows_to_check as f32;
            println!("  ✅ Passed. Avg Cosine Similarity: {:.4}", avg_sim);

            // Stricter assertion on average to ensure overall quality
            assert!(avg_sim > 0.90, "Average similarity too low for {}", name);
        }

        Ok(())
    }

    
    // #[test]
    // fn test_gguf_and_safetensors_load_identical_configs() -> Result<()> {
    //     use kjarni_transformers::weights::clear_mmap_cache;
    //     use std::path::Path;

    //     println!("--- Comparing GGUF vs Safetensors configs for 1B and 3B models ---");

    //     // Define paths (Hardcoded based on your environment)
    //     let gguf_1b_path = Path::new(
    //         "/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
    //     );
    //     let st_1b_path = Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct");

    //     let gguf_3b_path = Path::new(
    //         "/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    //     );
    //     let st_3b_path = Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-3B-Instruct");

    //     // --- Test 1B Model ---
    //     {
    //         println!("\n[1] Testing Llama 3.2 1B Instruct...");

    //         if !gguf_1b_path.exists() || !st_1b_path.exists() {
    //             println!("Skipping 1B test: Files not found.");
    //         } else {
    //             let start_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();

    //             let model_gguf = LlamaModel::from_pretrained(
    //                 gguf_1b_path,
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_1B_Instruct),
    //             )?;
    //             let end_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();

    //             println!("Model 1 Loaded. Delta RAM: {:.2} MB", end_ram - start_ram);

    //             let model_st = LlamaModel::from_pretrained(
    //                 st_1b_path,
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_1B_Instruct),
    //             )?;

    //             let end_ram2 = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();
    //             println!("Model 2 Loaded. Delta RAM: {:.2} MB", end_ram2 - end_ram);

    //             let config_gguf = model_gguf.config();
    //             let config_st = model_st.config();

    //             println!("   ... Comparing all parameters for 1B model...");
    //             assert_eq!(
    //                 config_gguf.hidden_size, config_st.hidden_size,
    //                 "1B: hidden_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_hidden_layers, config_st.num_hidden_layers,
    //                 "1B: num_hidden_layers"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_attention_heads, config_st.num_attention_heads,
    //                 "1B: num_attention_heads"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_key_value_heads, config_st.num_key_value_heads,
    //                 "1B: num_key_value_heads"
    //             );
    //             assert_eq!(
    //                 config_gguf.intermediate_size, config_st.intermediate_size,
    //                 "1B: intermediate_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.vocab_size, config_st.vocab_size,
    //                 "1B: vocab_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.max_position_embeddings, config_st.max_position_embeddings,
    //                 "1B: max_position_embeddings"
    //             );

    //             assert!(
    //                 (config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6,
    //                 "1B: rms_norm_eps"
    //             );
    //             assert_eq!(
    //                 config_gguf.hidden_act, config_st.hidden_act,
    //                 "1B: hidden_act"
    //             );
    //             assert!(
    //                 (config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6,
    //                 "1B: rope_theta"
    //             );

    //             assert_eq!(
    //                 config_gguf.pad_token_id, config_st.pad_token_id,
    //                 "1B: pad_token_id"
    //             );
    //             assert_eq!(
    //                 config_gguf.tie_word_embeddings, config_st.tie_word_embeddings,
    //                 "1B: tie_word_embeddings"
    //             );

    //             println!("   ... ✓ 1B model configs match perfectly.");
    //         }
    //         // `model_gguf` and `model_st` DROP here.
    //         // Arc counts to mmap decrease, but cache still holds them.
    //     }

    //     // CRITICAL: Clear cache now that the structs are dropped.
    //     // This frees the 1B model RAM before loading the 3B model.
    //     println!("   ... Clearing mmap cache to free 1B model memory ...");
    //     clear_mmap_cache();

    //     // --- Test 3B Model ---
    //     {
    //         println!("\n[2] Testing Llama 3.2 3B Instruct...");
    //         if !gguf_3b_path.exists() || !st_3b_path.exists() {
    //             println!("Skipping 3B test: Files not found.");
    //         } else {
    //             let start_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();

    //             let model_gguf = LlamaModel::from_pretrained(
    //                 gguf_3b_path,
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_3B_Instruct),
    //             )?;
    //             let end_ram = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();

    //             println!("Model 3 Loaded. Delta RAM: {:.2} MB", end_ram - start_ram);

    //             let model_st = LlamaModel::from_pretrained(
    //                 st_3b_path,
    //                 Device::Cpu,
    //                 None,
    //                 None,
    //                 Some(ModelType::Llama3_2_3B_Instruct),
    //             )?;
    //             let end_ram2 = kjarni_transformers::utils::alloc_stats::get_current_ram_usage_mb();
    //             println!("Model 4 Loaded. Delta RAM: {:.2} MB", end_ram2 - end_ram);

    //             let config_gguf = model_gguf.config();
    //             let config_st = model_st.config();

    //             println!("   ... Comparing all parameters for 3B model...");
    //             assert_eq!(
    //                 config_gguf.hidden_size, config_st.hidden_size,
    //                 "3B: hidden_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_hidden_layers, config_st.num_hidden_layers,
    //                 "3B: num_hidden_layers"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_attention_heads, config_st.num_attention_heads,
    //                 "3B: num_attention_heads"
    //             );
    //             assert_eq!(
    //                 config_gguf.num_key_value_heads, config_st.num_key_value_heads,
    //                 "3B: num_key_value_heads"
    //             );
    //             assert_eq!(
    //                 config_gguf.intermediate_size, config_st.intermediate_size,
    //                 "3B: intermediate_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.vocab_size, config_st.vocab_size,
    //                 "3B: vocab_size"
    //             );
    //             assert_eq!(
    //                 config_gguf.max_position_embeddings, config_st.max_position_embeddings,
    //                 "3B: max_position_embeddings"
    //             );

    //             assert!(
    //                 (config_gguf.rms_norm_eps - config_st.rms_norm_eps).abs() < 1e-6,
    //                 "3B: rms_norm_eps"
    //             );
    //             assert_eq!(
    //                 config_gguf.hidden_act, config_st.hidden_act,
    //                 "3B: hidden_act"
    //             );
    //             assert!(
    //                 (config_gguf.rope_theta - config_st.rope_theta).abs() < 1e-6,
    //                 "3B: rope_theta"
    //             );

    //             assert_eq!(
    //                 config_gguf.pad_token_id, config_st.pad_token_id,
    //                 "3B: pad_token_id"
    //             );
    //             assert_eq!(
    //                 config_gguf.tie_word_embeddings, config_st.tie_word_embeddings,
    //                 "3B: tie_word_embeddings"
    //             );

    //             println!("   ... ✓ 3B model configs match perfectly.");
    //         }
    //     }

    //     // Final cleanup (good practice)
    //     println!("   ... Clearing mmap cache ...");
    //     clear_mmap_cache();

    //     println!(
    //         "\n✓ SUCCESS: All GGUF configurations correctly match their SafeTensors counterparts."
    //     );
    //     Ok(())
    // }
}
