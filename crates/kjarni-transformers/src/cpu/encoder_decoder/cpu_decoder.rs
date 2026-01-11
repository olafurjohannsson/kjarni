//! Unified Seq2Seq decoder supporting BART, T5, Whisper, and similar architectures.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4, s};
use std::sync::Arc;

pub use crate::encoder_decoder::config::{PositionEncodingType, Seq2SeqDecoderConfig};

use crate::cpu::encoder_decoder::{
    decoder_cross_attn_layer::CrossDecoderLayer, relative_position_bias::T5RelativePositionBias,
};

use crate::{
    Normalization, WgpuContext,
    cache::{Cache, CpuBeamKVCache},
    embeddings::Embeddings,
    encoder_decoder::traits::{CpuCrossAttentionKVCache, CpuCrossDecoder, CpuCrossDecoderOutput},
    models::base::ModelLoadConfig,
    pipeline::Seq2SeqFactory,
    traits::{
        Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    },
    weights::ModelWeights,
};

// =============================================================================
// Decoder Output
// =============================================================================

/// Output from the Seq2Seq decoder (for internal use, separate from trait output).
#[derive(Debug)]
pub struct DecoderOutput {
    /// Hidden states from the final layer. Shape: [batch, seq, hidden]
    pub last_hidden_state: Array3<f32>,
    /// New self-attention K/V pairs for each layer.
    pub new_self_attn_kv: Vec<(Array3<f32>, Array3<f32>)>,
}

// =============================================================================
// Seq2Seq Decoder (CPU)
// =============================================================================

/// Unified transformer decoder for seq2seq models.
///
/// Supports BART, T5, Whisper, mBART, and similar architectures.
/// Configuration is driven by `ModelConfig` and `Seq2SeqDecoderConfig`.
///
/// # Example
///
/// ```ignore
/// // From a BartConfig
/// let decoder = Seq2SeqCPUDecoder::new(
///     &weights,
///     &bart_config,
///     Seq2SeqDecoderConfig::bart(),
///     load_config,
/// )?;
///
/// // Forward with pre-computed cross-attention cache
/// let cross_kv = decoder.precompute_cross_attention_kv(&encoder_output)?;
/// let output = decoder.forward(
///     input_ids,
///     &encoder_hidden_states,
///     Some(&cross_kv),
///     cache,
/// )?;
/// ```
pub struct Seq2SeqCPUDecoder {
    /// Token embeddings
    embeddings: Embeddings,

    /// Embedding layer normalization (BART, Whisper)
    embed_norm: Option<Normalization>,

    /// Transformer layers
    layers: Vec<CrossDecoderLayer>,

    /// Final layer normalization (T5, Whisper)
    final_norm: Option<Normalization>,

    /// Position encoding implementation
    position_encoding: PositionEncoding,

    /// Use pre-norm (T5, Whisper) vs post-norm (BART)
    pre_norm: bool,

    /// Model metadata
    pub meta: ModelMetadata,

    /// Model layout (for reference)
    pub layout: ModelLayout,
}

impl Seq2SeqCPUDecoder {
    /// Create decoder from ModelConfig.
    ///
    /// # Arguments
    /// * `weights` - Model weights
    /// * `config` - Model configuration implementing `ModelConfig`
    /// * `decoder_config` - Seq2Seq-specific decoder configuration
    /// * `load_config` - Weight loading options
    pub fn new<C: ModelConfig>(
        weights: &ModelWeights,
        config: &C,
        decoder_config: Seq2SeqDecoderConfig,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();

        let decoder_layout = layout
            .decoder
            .as_ref()
            .ok_or_else(|| anyhow!("Model layout has no decoder component"))?;

        let factory = Seq2SeqFactory::new(weights).with_load_config(&load_config);

        // 1. Build embeddings
        let embeddings: Embeddings = factory.build_embeddings(
            &layout.token_embedding,
            decoder_layout.position_embedding.as_deref(),
        )?;

        // 2. Build embedding normalization
        let embed_norm = if decoder_config.normalize_embeddings {
            Self::build_embed_norm(&factory, decoder_layout, &meta)?
        } else {
            None
        };

        // 3. Build transformer layers
        let layers: Vec<CrossDecoderLayer> = (0..meta
            .decoder_layers
            .expect("Invalid cross decoder layers"))
            .map(|i| factory.build_decoder_layer(decoder_layout, &meta, i, decoder_config.pre_norm))
            .collect::<Result<Vec<_>>>()?;

        // 4. Build final layer normalization
        let final_norm = if decoder_config.final_layer_norm {
            Self::build_final_norm(&factory, decoder_layout, &meta)?
        } else {
            None
        };

        // 5. Build position encoding
        let position_encoding = Self::build_position_encoding(
            weights,
            &decoder_config.position_encoding,
            decoder_layout.position_embedding.as_deref(),
            &meta,
        )?;

        Ok(Self {
            embeddings,
            embed_norm,
            layers,
            final_norm,
            position_encoding,
            pre_norm: decoder_config.pre_norm,
            meta,
            layout,
        })
    }

    /// Build embedding normalization from layout.
    fn build_embed_norm(
        factory: &Seq2SeqFactory,
        decoder_layout: &crate::traits::DecoderLayout,
        meta: &ModelMetadata,
    ) -> Result<Option<Normalization>> {
        match (
            &decoder_layout.embedding_norm_weight,
            &decoder_layout.embedding_norm_bias,
        ) {
            (Some(w), Some(b)) => Ok(Some(factory.build_norm(
                w,
                Some(b.as_str()),
                meta.normalization_strategy.clone(),
                meta.norm_eps,
                0,
            )?)),
            (Some(w), None) => {
                // RMSNorm (T5)
                Ok(Some(factory.build_norm(
                    w,
                    None,
                    NormalizationStrategy::RMSNorm,
                    meta.norm_eps,
                    0,
                )?))
            }
            _ => Ok(None),
        }
    }

    /// Build final layer normalization from layout.
    fn build_final_norm(
        factory: &Seq2SeqFactory,
        decoder_layout: &crate::traits::DecoderLayout,
        meta: &ModelMetadata,
    ) -> Result<Option<Normalization>> {
        match &decoder_layout.final_norm_weight {
            Some(w) => Ok(Some(factory.build_norm(
                w,
                decoder_layout.final_norm_bias.as_deref(),
                meta.normalization_strategy.clone(),
                meta.norm_eps,
                0,
            )?)),
            None => Ok(None),
        }
    }

    /// Build position encoding based on config.
    fn build_position_encoding(
        weights: &ModelWeights,
        encoding_type: &PositionEncodingType,
        pos_embedding_name: Option<&str>,
        meta: &ModelMetadata,
    ) -> Result<PositionEncoding> {
        match encoding_type {
            PositionEncodingType::Learned { offset } => {
                let pos_name = pos_embedding_name.ok_or_else(|| {
                    anyhow!("Learned positions require position_embedding in layout")
                })?;
                let embeddings = weights.get_array2(pos_name)?;
                Ok(PositionEncoding::Learned {
                    embeddings,
                    offset: *offset,
                })
            }
            PositionEncodingType::RelativeBias {
                num_buckets,
                max_distance,
            } => {
                let bias = T5RelativePositionBias::new(
                    weights,
                    "decoder",
                    false, // is_bidirectional = false
                    *num_buckets,
                    *max_distance,
                )?;
                Ok(PositionEncoding::RelativeBias { bias })
            }
            PositionEncodingType::Sinusoidal => {
                let cache = create_sinusoidal_embeddings(meta.max_seq_len, meta.hidden_size);
                Ok(PositionEncoding::Sinusoidal { cache })
            }
            PositionEncodingType::None => Ok(PositionEncoding::None),
        }
    }

    // =========================================================================
    // Forward Pass
    // =========================================================================

    /// Full forward pass for decoder.
    ///
    /// # Arguments
    /// * `input_ids` - Decoder input token IDs [batch, seq]
    /// * `encoder_hidden_states` - Encoder output [batch, enc_seq, hidden]
    /// * `attention_mask` - Decoder self-attention mask
    /// * `cross_attention_mask` - Encoder-decoder attention mask
    /// * `cache` - Self-attention KV cache (updated in-place)
    /// * `cross_kv_cache` - Pre-computed cross-attention KV (optional)
    /// * `position_offset` - Position offset for autoregressive decoding
    pub fn forward(
        &self,
        input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        cross_attention_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        position_offset: usize,
    ) -> Result<DecoderOutput> {
        println!("\n[DECODER DEBUG] Number of layers: {}", self.layers.len());
        println!("[DECODER DEBUG] Input shape: {:?}", input_ids.dim());
        println!(
            "[DECODER DEBUG] Encoder hidden shape: {:?}",
            encoder_hidden_states.dim()
        );
        
        // 1. Embeddings
        let mut hidden = self.embed(input_ids, position_offset);
        if position_offset == 0 {
            println!("\n--- RUST DECODER DEBUG ---");
            // Slice the first 5 values of the embedding (before norm/pos)
            println!(
                "[2] Scaled Word Embedding (first 5): {:?}",
                hidden.slice(s![0, 0, ..5])
            );
        }
        // 2. Apply position encoding (for sinusoidal - learned handled in embed)
        hidden = self.apply_position_encoding(hidden, position_offset)?;

        // 3. Embedding normalization
        if let Some(norm) = &self.embed_norm {
            hidden = norm.forward(&hidden);
        }

        // 4. Compute position bias (T5)
        let position_bias = self.compute_position_bias(&hidden, position_offset)?;

        // 5. Get CPU cache if available
        let cpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<CpuBeamKVCache>());

        // 6. Transformer layers
        let mut new_self_attn_kvs = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            // Get past K/V for this layer
            let past_kv = cpu_cache.and_then(|c| c.get(i));
            let past_kv_views = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));

            // Get cross-attention K/V for this layer
            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));
            let pre_mean = hidden.mean().unwrap_or(0.0);
            let (new_hidden, new_kv) = layer.forward(
                &hidden,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
                past_kv_views,
                cross_kv_for_layer,
                position_bias.as_ref(),
            )?;
            let post_mean = hidden.mean().unwrap_or(0.0);
            hidden = new_hidden;
            println!(
                "[DECODER DEBUG] Layer {} | Pre-mean: {:.6} -> Post-mean: {:.6}",
                i, pre_mean, post_mean
            );
            new_self_attn_kvs.push(new_kv);
        }

        // 7. Final layer normalization
        if let Some(norm) = &self.final_norm {
            hidden = norm.forward(&hidden);
        }
        if position_offset == 0 {
            println!(
                "[4] Decoder Initial State (Pre-Layer 0) (first 10): {:?}",
                hidden.slice(s![0, 0, ..10])
            );
        }
        Ok(DecoderOutput {
            last_hidden_state: hidden,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }

    // =========================================================================
    // Helper Methods
    // =========================================================================

    /// Embed input tokens.
    fn embed(&self, input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32> {
        self.embeddings.forward(
            input_ids,
            None, // No token type IDs
            position_offset + self.position_offset(),
            self.meta.scale_embeddings,
        )
    }

    /// Get position offset for learned embeddings.
    fn position_offset(&self) -> usize {
        match &self.position_encoding {
            PositionEncoding::Learned { offset, .. } => *offset,
            _ => 0,
        }
    }

    /// Apply position encoding to hidden states (for sinusoidal).
    fn apply_position_encoding(
        &self,
        hidden: Array3<f32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        match &self.position_encoding {
            PositionEncoding::Sinusoidal { cache } => {
                let (batch, seq_len, hidden_size) = hidden.dim();
                let mut result = hidden;

                for b in 0..batch {
                    for s in 0..seq_len {
                        let pos = s + position_offset;
                        if pos < cache.nrows() {
                            for h in 0..hidden_size {
                                result[[b, s, h]] += cache[[pos, h]];
                            }
                        }
                    }
                }
                Ok(result)
            }
            // Learned positions handled in embeddings, RelativeBias in attention
            _ => Ok(hidden),
        }
    }

    /// Compute position bias for attention (T5).
    fn compute_position_bias(
        &self,
        hidden: &Array3<f32>,
        position_offset: usize, // Add this parameter
    ) -> Result<Option<Array4<f32>>> {
        match &self.position_encoding {
            PositionEncoding::RelativeBias { bias } => {
                let query_len = hidden.dim().1;
                // Key length is the total history plus the current tokens
                let key_len = position_offset + query_len;

                // We need a version of compute that understands the query starts at position_offset
                Ok(Some(bias.compute_with_offset(
                    query_len,
                    key_len,
                    position_offset,
                )?))
            }
            _ => Ok(None),
        }
    }

    /// Number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    /// Access to layers (for pre-computing cross-attention).
    pub fn layers(&self) -> &[CrossDecoderLayer] {
        &self.layers
    }
}

// =============================================================================
// Position Encoding Implementations (same as encoder, could be shared)
// =============================================================================

/// Runtime position encoding implementation.
pub enum PositionEncoding {
    /// Learned absolute positions (BART, Whisper)
    Learned {
        embeddings: Array2<f32>,
        offset: usize,
    },
    /// Relative position bias (T5)
    RelativeBias { bias: T5RelativePositionBias },
    /// Sinusoidal positions
    Sinusoidal { cache: Array2<f32> },
    /// No position encoding
    None,
}

/// Create sinusoidal position embeddings.
fn create_sinusoidal_embeddings(max_len: usize, dim: usize) -> Array2<f32> {
    let mut embeddings = Array2::<f32>::zeros((max_len, dim));

    for pos in 0..max_len {
        for i in 0..dim / 2 {
            let angle = pos as f32 / 10000_f32.powf(2.0 * i as f32 / dim as f32);
            embeddings[[pos, 2 * i]] = angle.sin();
            embeddings[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    embeddings
}

impl Seq2SeqCPUDecoder {}

// =============================================================================
// Trait Implementations
// =============================================================================

impl InferenceModel for Seq2SeqCPUDecoder {
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

#[async_trait]
impl CpuCrossDecoder for Seq2SeqCPUDecoder {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn layers(&self) -> &Vec<CrossDecoderLayer> {
        &self.layers
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    fn forward2(
        &self,
        decoder_input_ids: &Array2<u32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>, // Padding in the decoder
        encoder_padding_mask: Option<&Array2<f32>>, // Padding in the encoder (NEW)
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
    ) -> Result<CpuCrossDecoderOutput> {
        // 1. Calculate offset
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());

        // 2. Embed
        let hidden = self.embed_and_normalize(decoder_input_ids, position_offset)?;

        // 3. Run all layers
        self.forward_layers2(
            &hidden,
            encoder_hidden_states,
            decoder_padding_mask,
            encoder_padding_mask,
            cache,
            cross_kv_cache,
            0,
            self.num_layers(),
        )
    }

    fn precompute_cross_attention_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<CpuCrossAttentionKVCache> {
        let mut cache_vec = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (k, v) = layer.precompute_cross_kv(encoder_hidden_states)?;
            cache_vec.push((k, v));
        }
        Ok(CpuCrossAttentionKVCache(cache_vec))
    }

    fn embed(&self, decoder_input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32> {
        self.embeddings.forward(
            decoder_input_ids,
            None,
            position_offset + self.position_offset(),
            self.meta.scale_embeddings,
        )
    }

    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = self.embed(input_ids, position_offset);

        // Apply sinusoidal positions if needed
        hidden = self.apply_position_encoding(hidden, position_offset)?;

        if let Some(norm) = &self.embed_norm {
            hidden = norm.forward(&hidden);
        }
        Ok(hidden)
    }

    fn forward_layers2(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_padding_mask: Option<&Array2<f32>>,
        encoder_padding_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput> {
        let cpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<CpuBeamKVCache>());
        let position_offset = cpu_cache.map_or(0, |c| c.get_seq_length());

        let seq_len = hidden_states.dim().1;
        let total_len = position_offset + seq_len;

        // 1. Create the Causal Mask [seq_len, total_len]
        let causal_mask = crate::utils::masks::create_causal_mask(seq_len, total_len);

        // 2. Combine with Decoder Padding Mask (Self-Attention Mask)
        let self_attn_mask = match decoder_padding_mask {
            Some(pad) => {
                // If pad is [batch, total_len], we broadcast and multiply
                // This ensures we are causal AND ignoring padding
                Some(combine_masks(&causal_mask, pad))
            }
            None => Some(causal_mask),
        };

        // 3. Encoder Padding Mask is used for Cross-Attention
        let cross_attn_mask = encoder_padding_mask;

        // 4. Compute T5 Position Bias (Using query_offset fix)
        let position_bias = self.compute_position_bias(hidden_states, position_offset)?;

        let mut hidden = hidden_states.clone();
        let mut new_self_attn_kvs = Vec::with_capacity(end_layer - start_layer);

        for i in start_layer..end_layer {
            let layer = &self.layers[i];
            let past_kv = cpu_cache.and_then(|c| c.get(i));
            let past_kv_views = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));
            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));

            // 5. Pass DIFFERENT masks to the layer
            let (new_hidden, new_kv) = layer.forward(
                &hidden,
                encoder_hidden_states,
                self_attn_mask.as_ref(), // causal + pad
                cross_attn_mask,         // encoder pad only
                past_kv_views,
                cross_kv_for_layer,
                position_bias.as_ref(),
            )?;

            hidden = new_hidden;
            new_self_attn_kvs.push(new_kv);
        }

        if end_layer == self.layers.len() {
            if let Some(norm) = &self.final_norm {
                hidden = norm.forward(&hidden);
            }
        }

        Ok(CpuCrossDecoderOutput {
            last_hidden_state: hidden,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: Option<&Array2<f32>>, // padding mask
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput> {
        let cpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<CpuBeamKVCache>());
        // let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let position_offset = cpu_cache.map_or(0, |c| c.get_seq_length());
        let position_bias = self.compute_position_bias(hidden_states, position_offset)?;

        let mut hidden = hidden_states.clone();
        // log::error!("embed sum: {:?}", hidden.sum());
        let mut new_self_attn_kvs = Vec::with_capacity(end_layer - start_layer);

        if start_layer >= self.layers.len() || end_layer > self.layers.len() {
            return Err(anyhow!("Layer indices out of bounds"));
        }

        // let seq_len = hidden_states.dim().1;

        // let model_mask = self.get_decoder_mask(seq_len, position_offset);
        // let effective_mask = match (model_mask, decoder_attention_mask) {
        //     (Some(m), Some(p)) => Some(combine_masks(&m, p)), // Helper to merge them
        //     (Some(m), None) => Some(m),
        //     (None, Some(p)) => Some(p.clone()),
        //     (None, None) => None,
        // };
        for i in start_layer..end_layer {
            let layer = &self.layers[i];

            let past_kv = cpu_cache.and_then(|c| c.get(i));
            let past_kv_views = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));

            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));

            let (new_hidden, new_kv) = layer.forward(
                &hidden,
                encoder_hidden_states,
                //effective_mask,
                decoder_attention_mask,
                None, // cross_mask
                past_kv_views,
                cross_kv_for_layer,
                position_bias.as_ref(),
            )?;

            hidden = new_hidden;
            new_self_attn_kvs.push(new_kv);

            // log::debug!("layer {} output sum: {:?}", i, hidden.sum());
        }

        // Apply final norm if this includes the last layer
        if end_layer == self.layers.len() {
            if let Some(norm) = &self.final_norm {
                hidden = norm.forward(&hidden);
            }
        }

        Ok(CpuCrossDecoderOutput {
            last_hidden_state: hidden,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }
}
fn combine_masks(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    // Standard element-wise multiplication works for 1.0/0.0 masks
    a * b
}

#[cfg(test)]
mod seq2seq_decoder_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::cache::CpuBeamKVCache;
    use crate::encoder_decoder::traits::CpuCrossDecoder; // Import trait for forward2
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{
        AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata, NormalizationStrategy,
    };
    use crate::weights::ModelWeights;
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // =========================================================================
    //  1. Mock Configuration & Layout
    // =========================================================================

    #[derive(Debug, Clone)]
    struct MockConfig {
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        is_prenorm: bool,
        no_pos_emb_in_layout: bool,
    }

    impl ModelConfig for MockConfig {
        fn model_type(&self) -> &str {
            "mock"
        }
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }

        fn metadata(&self) -> ModelMetadata {
            ModelMetadata {
                decoder_layers: Some(self.num_layers),
                hidden_size: self.hidden_size,
                num_layers: 0, // Encoder layers (unused here)
                num_attention_heads: self.num_heads,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                vocab_size: self.vocab_size,
                max_seq_len: 1024,
                norm_eps: 1e-5,
                activation: Activation::Gelu,
                rope_theta: None,
                rope_scaling: None,
                scale_embeddings: false,
                normalize_embedding: false,
                extra_pos_embeddings: 0,
                is_prenorm: self.is_prenorm,
                transpose_ffn_weights: false,
                transpose_attention_weights: false,
                normalization_strategy: NormalizationStrategy::LayerNorm,
                no_scale_qk: false,
            }
        }

        fn layout(&self) -> ModelLayout {
            ModelLayout {
                token_embedding: "token_emb".to_string(),
                lm_head: "lm_head".to_string(),
                encoder: None,
                decoder: Some(DecoderLayout {
                    position_embedding: if self.no_pos_emb_in_layout {
                        None
                    } else {
                        Some("pos_emb".to_string())
                    },
                    token_type_embedding: None,
                    embedding_norm_weight: Some("embed_ln.weight".to_string()),
                    embedding_norm_bias: Some("embed_ln.bias".to_string()),
                    final_norm_weight: Some("final_ln.weight".to_string()),
                    final_norm_bias: Some("final_ln.bias".to_string()),
                    layer: DecoderLayerLayout {
                        self_attn: AttentionLayout {
                            q_weight: "l0.sa_q.weight".to_string(),
                            q_bias: None,
                            k_weight: "l0.sa_k.weight".to_string(),
                            k_bias: None,
                            v_weight: "l0.sa_v.weight".to_string(),
                            v_bias: None,
                            o_weight: "l0.sa_o.weight".to_string(),
                            o_bias: None,
                            norm_weight: "l0.sa_ln.weight".to_string(),
                            norm_bias: Some("l0.sa_ln.bias".to_string()),
                        },
                        cross_attn: Some(AttentionLayout {
                            q_weight: "l0.ca_q.weight".to_string(),
                            q_bias: None,
                            k_weight: "l0.ca_k.weight".to_string(),
                            k_bias: None,
                            v_weight: "l0.ca_v.weight".to_string(),
                            v_bias: None,
                            o_weight: "l0.ca_o.weight".to_string(),
                            o_bias: None,
                            norm_weight: "l0.ca_ln.weight".to_string(),
                            norm_bias: Some("l0.ca_ln.bias".to_string()),
                        }),
                        ffn: FeedForwardLayout {
                            up_weight: "l0.fc1.weight".to_string(),
                            up_bias: None,
                            down_weight: "l0.fc2.weight".to_string(),
                            down_bias: None,
                            gate_weight: None,
                            gate_bias: None,
                            norm_weight: "l0.ffn_ln.weight".to_string(),
                            norm_bias: Some("l0.ffn_ln.bias".to_string()),
                        },
                    },
                }),
            }
        }
    }

    // =========================================================================
    //  2. Helpers
    // =========================================================================

    fn create_model_weights(
        weights_map: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<usize>>,
    ) -> Result<(ModelWeights, TempDir)> {
        let dir = tempfile::tempdir()?;
        let stored_data: Vec<(String, Vec<usize>, Vec<u8>)> = weights_map
            .into_iter()
            .map(|(k, v)| {
                let shape = shapes.get(&k).unwrap().clone();
                let bytes: Vec<u8> = v.iter().flat_map(|f| f.to_le_bytes()).collect();
                (k, shape, bytes)
            })
            .collect();

        let mut tensors = HashMap::new();
        for (k, shape, bytes) in &stored_data {
            tensors.insert(
                k.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensors, &None, &file_path)?;
        let weights = ModelWeights::new(dir.path())?;

        Ok((weights, dir))
    }

    // BART Golden Data Provider
    fn get_bart_decoder_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // 1. Embeddings
        w.insert(
            "token_emb".into(),
            vec![
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.010, 0.011, 0.012,
                0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020, 0.021, 0.022, 0.023, 0.024,
                0.025, 0.026, 0.027, 0.028, 0.029, 0.030, 0.031, 0.032, 0.033, 0.034, 0.035, 0.036,
                0.037, 0.038, 0.039, 0.040,
            ],
        );
        s.insert("token_emb".into(), vec![10, 4]);

        let pos_data: Vec<f32> = (0..1024 * 4).map(|i| 0.041 + (i as f32 * 0.001)).collect();
        w.insert("pos_emb".into(), pos_data);
        s.insert("pos_emb".into(), vec![1024, 4]);

        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        // 2. Layer 0 - Self Attention
        w.insert(
            "l0.sa_q.weight".into(),
            vec![
                4.137, 4.138, 4.139, 4.140, 4.141, 4.142, 4.143, 4.144, 4.145, 4.146, 4.147, 4.148,
                4.149, 4.150, 4.151, 4.152,
            ],
        );
        s.insert("l0.sa_q.weight".into(), vec![4, 4]);

        w.insert(
            "l0.sa_k.weight".into(),
            vec![
                4.153, 4.154, 4.155, 4.156, 4.157, 4.158, 4.159, 4.160, 4.161, 4.162, 4.163, 4.164,
                4.165, 4.166, 4.167, 4.168,
            ],
        );
        s.insert("l0.sa_k.weight".into(), vec![4, 4]);

        w.insert(
            "l0.sa_v.weight".into(),
            vec![
                4.169, 4.170, 4.171, 4.172, 4.173, 4.174, 4.175, 4.176, 4.177, 4.178, 4.179, 4.180,
                4.181, 4.182, 4.183, 4.184,
            ],
        );
        s.insert("l0.sa_v.weight".into(), vec![4, 4]);

        w.insert(
            "l0.sa_o.weight".into(),
            vec![
                4.185, 4.186, 4.187, 4.188, 4.189, 4.190, 4.191, 4.192, 4.193, 4.194, 4.195, 4.196,
                4.197, 4.198, 4.199, 4.200,
            ],
        );
        s.insert("l0.sa_o.weight".into(), vec![4, 4]);

        w.insert("l0.sa_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.sa_ln.weight".into(), vec![4]);
        w.insert("l0.sa_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.sa_ln.bias".into(), vec![4]);

        // 3. Layer 0 - Cross Attention
        w.insert(
            "l0.ca_q.weight".into(),
            vec![
                4.201, 4.202, 4.203, 4.204, 4.205, 4.206, 4.207, 4.208, 4.209, 4.210, 4.211, 4.212,
                4.213, 4.214, 4.215, 4.216,
            ],
        );
        s.insert("l0.ca_q.weight".into(), vec![4, 4]);

        w.insert(
            "l0.ca_k.weight".into(),
            vec![
                4.217, 4.218, 4.219, 4.220, 4.221, 4.222, 4.223, 4.224, 4.225, 4.226, 4.227, 4.228,
                4.229, 4.230, 4.231, 4.232,
            ],
        );
        s.insert("l0.ca_k.weight".into(), vec![4, 4]);

        w.insert(
            "l0.ca_v.weight".into(),
            vec![
                4.233, 4.234, 4.235, 4.236, 4.237, 4.238, 4.239, 4.240, 4.241, 4.242, 4.243, 4.244,
                4.245, 4.246, 4.247, 4.248,
            ],
        );
        s.insert("l0.ca_v.weight".into(), vec![4, 4]);

        w.insert(
            "l0.ca_o.weight".into(),
            vec![
                4.249, 4.250, 4.251, 4.252, 4.253, 4.254, 4.255, 4.256, 4.257, 4.258, 4.259, 4.260,
                4.261, 4.262, 4.263, 4.264,
            ],
        );
        s.insert("l0.ca_o.weight".into(), vec![4, 4]);

        w.insert("l0.ca_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ca_ln.weight".into(), vec![4]);
        w.insert("l0.ca_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ca_ln.bias".into(), vec![4]);

        // 4. Layer 0 - FFN
        let fc1_data: Vec<f32> = (0..32).map(|i| 4.265 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);

        let fc2_data: Vec<f32> = (0..32).map(|i| 4.297 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);

        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        (w, s)
    }

    // =========================================================================
    //  3. Test
    // =========================================================================

    #[test]
    fn test_decoder_bart_golden() -> Result<()> {
        let (weights_map, shapes) = get_bart_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false, // BART has learned positions
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: false, // Match config
        };

        let decoder = Seq2SeqCPUDecoder::new(
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Inputs
        // Decoder Input: [1, 2] (Batch=1, Seq=2)
        let dec_ids = Array2::from_shape_vec((1, 2), vec![1u32, 2]).unwrap();

        // Encoder Output: [1, 3, 4] (Batch, EncSeq, Hidden)
        let enc_out_data = vec![
            0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
            0.461657, 0.267351, 0.534905, 0.809357,
        ];
        let enc_hidden = Array3::from_shape_vec((1, 3, 4), enc_out_data).unwrap();

        // Masks (Optionally passed to forward2 via `decoder_padding_mask` / `encoder_padding_mask`)
        // The trait method `forward2` generates the causal mask internally.
        // We just need to pass None for padding if we assume full sequences.
        // Or if we want to match the Python script exactly, Python used explicit masks.
        // forward2 will generate causal mask [2, 2] which matches Python's torch.tril(ones).

        // Cache Setup
        let mut cache = CpuBeamKVCache::new(1, 1, 10, 4);
        // We need to initialize the Cross Attention cache since `forward2` expects it if optimization is used
        // But for a single pass test without previous state, we can compute it on the fly or precompute.
        // Let's precompute it properly using the decoder method.
        let cross_cache = decoder.precompute_cross_attention_kv(&enc_hidden)?;

        // Run Forward
        let output = decoder.forward2(
            &dec_ids,
            &enc_hidden,
            None, // decoder_padding_mask (None = all ones)
            None, // encoder_padding_mask (None = all ones)
            Some(&mut cache),
            Some(&cross_cache),
        )?;

        // Check Output
        let golden_data = vec![
            -1.331633, -0.437210, 0.457207, 1.351637, -1.331632, -0.437217, 0.457217, 1.351632,
        ];
        let golden = Array3::from_shape_vec((1, 2, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("BART Decoder Max Diff: {}", max_diff);
        assert!(max_diff < 1e-4, "BART decoder mismatch");

        Ok(())
    }

    #[test]
    fn test_decoder_whisper_golden() -> Result<()> {
        let (weights_map, shapes) = get_whisper_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true,
            no_pos_emb_in_layout: true, // Whisper uses Sinusoidal, no learned pos weights
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::Sinusoidal,
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: true,
        };

        let decoder = Seq2SeqCPUDecoder::new(
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        let dec_ids = Array2::from_shape_vec((1, 2), vec![1u32, 2]).unwrap();

        let enc_out_data = vec![
            0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
            0.461657, 0.267351, 0.534905, 0.809357,
        ];
        let enc_hidden = Array3::from_shape_vec((1, 3, 4), enc_out_data).unwrap();

        let mut cache = CpuBeamKVCache::new(1, 1, 10, 4);
        let cross_cache = decoder.precompute_cross_attention_kv(&enc_hidden)?;

        let output = decoder.forward2(
            &dec_ids,
            &enc_hidden,
            None,
            None,
            Some(&mut cache),
            Some(&cross_cache),
        )?;

        // Golden Data (Generated by Python script for Whisper scenario)
        let golden_data = vec![
            -0.995535, 1.004425, -0.984425, 1.015535, 0.646058, -0.145752, -1.544646, 1.084340,
        ];
        let golden = Array3::from_shape_vec((1, 2, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Whisper Decoder Max Diff: {}", max_diff);
        // Relaxed tolerance for Whisper due to GELU/Sinusoidal approximation differences
        assert!(max_diff < 1e-3, "Whisper decoder mismatch");

        Ok(())
    }

    #[test]
    fn test_decoder_t5_golden() -> Result<()> {
        let (weights_map, shapes) = get_t5_decoder_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true,
            no_pos_emb_in_layout: true, // T5 uses Relative Bias, no learned pos weights
        };

        let dec_config = Seq2SeqDecoderConfig {
            position_encoding: PositionEncodingType::RelativeBias {
                num_buckets: 32,
                max_distance: 128,
            },
            normalize_embeddings: true,
            final_layer_norm: true,
            pre_norm: true, // Match config
        };

        let decoder = Seq2SeqCPUDecoder::new(
            &model_weights,
            &config,
            dec_config,
            ModelLoadConfig::default(),
        )?;

        // Inputs
        let dec_ids = Array2::from_shape_vec((1, 2), vec![1u32, 2]).unwrap();
        // Encoder Output: [1, 3, 4]
        let enc_out_data = vec![
            0.336690, 0.128809, 0.234462, 0.230333, -1.122856, -0.186328, 2.208201, -0.637997,
            0.461657, 0.267351, 0.534905, 0.809357,
        ];
        let enc_hidden = Array3::from_shape_vec((1, 3, 4), enc_out_data).unwrap();

        let mut cache = CpuBeamKVCache::new(1, 1, 10, 4);
        let cross_cache = decoder.precompute_cross_attention_kv(&enc_hidden)?;

        // Run Forward
        let output = decoder.forward2(
            &dec_ids,
            &enc_hidden,
            None,
            None,
            Some(&mut cache),
            Some(&cross_cache),
        )?;

        // Golden Data (Generated by Python script for T5 scenario)
        let golden_data = vec![
            -1.331581, -0.437194, 0.457194, 1.351581, -1.331581, -0.437193, 0.457194, 1.351580,
        ];
        let golden = Array3::from_shape_vec((1, 2, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("T5 Decoder Max Diff: {}", max_diff);
        assert!(max_diff < 1e-4, "T5 decoder mismatch");

        Ok(())
    }
    fn gen_weights_helper(
        w: &mut HashMap<String, Vec<f32>>,
        s: &mut HashMap<String, Vec<usize>>,
        name: &str,
        shape: Vec<usize>,
        start_count: &mut usize,
    ) {
        let size: usize = shape.iter().product();
        let data: Vec<f32> = (*start_count..*start_count + size)
            .map(|i| i as f32 * 0.001)
            .collect();
        *start_count += size;
        w.insert(name.to_string(), data);
        s.insert(name.to_string(), shape);
    }

    fn get_t5_decoder_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();
        let mut count = 1;

        // 1. Embeddings
        gen_weights_helper(&mut w, &mut s, "token_emb", vec![10, 4], &mut count);

        // 2. Relative Position Bias
        gen_weights_helper(
            &mut w,
            &mut s,
            "decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight",
            vec![32, 2],
            &mut count,
        );

        // 3. Layer 0
        // Self Attention
        gen_weights_helper(&mut w, &mut s, "l0.sa_q.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_k.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_v.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_o.weight", vec![4, 4], &mut count);

        let ln_w = vec![1.0; 4];
        let ln_b = vec![0.01; 4];

        w.insert("l0.sa_ln.weight".into(), ln_w.clone());
        s.insert("l0.sa_ln.weight".into(), vec![4]);
        w.insert("l0.sa_ln.bias".into(), ln_b.clone());
        s.insert("l0.sa_ln.bias".into(), vec![4]);

        // Cross Attention
        gen_weights_helper(&mut w, &mut s, "l0.ca_q.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_k.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_v.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_o.weight", vec![4, 4], &mut count);

        w.insert("l0.ca_ln.weight".into(), ln_w.clone());
        s.insert("l0.ca_ln.weight".into(), vec![4]);
        w.insert("l0.ca_ln.bias".into(), ln_b.clone());
        s.insert("l0.ca_ln.bias".into(), vec![4]);

        // FFN
        gen_weights_helper(&mut w, &mut s, "l0.fc1.weight", vec![8, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.fc2.weight", vec![4, 8], &mut count);

        w.insert("l0.ffn_ln.weight".into(), ln_w.clone());
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), ln_b.clone());
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        // 4. Other Norms
        w.insert("embed_ln.weight".into(), ln_w.clone());
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), ln_b.clone());
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), ln_w.clone());
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), ln_b.clone());
        s.insert("final_ln.bias".into(), vec![4]);

        (w, s)
    }

    fn get_whisper_decoder_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>)
    {
        let mut w = HashMap::new();
        let mut s = HashMap::new();
        let mut count = 1;

        // 1. Embeddings
        gen_weights_helper(&mut w, &mut s, "token_emb", vec![10, 4], &mut count);

        // 2. Layer 0
        // Self Attention
        gen_weights_helper(&mut w, &mut s, "l0.sa_q.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_k.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_v.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.sa_o.weight", vec![4, 4], &mut count);

        let ln_w = vec![1.0; 4];
        let ln_b = vec![0.01; 4];

        w.insert("l0.sa_ln.weight".into(), ln_w.clone());
        s.insert("l0.sa_ln.weight".into(), vec![4]);
        w.insert("l0.sa_ln.bias".into(), ln_b.clone());
        s.insert("l0.sa_ln.bias".into(), vec![4]);

        // Cross Attention
        gen_weights_helper(&mut w, &mut s, "l0.ca_q.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_k.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_v.weight", vec![4, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.ca_o.weight", vec![4, 4], &mut count);

        w.insert("l0.ca_ln.weight".into(), ln_w.clone());
        s.insert("l0.ca_ln.weight".into(), vec![4]);
        w.insert("l0.ca_ln.bias".into(), ln_b.clone());
        s.insert("l0.ca_ln.bias".into(), vec![4]);

        // FFN
        gen_weights_helper(&mut w, &mut s, "l0.fc1.weight", vec![8, 4], &mut count);
        gen_weights_helper(&mut w, &mut s, "l0.fc2.weight", vec![4, 8], &mut count);

        w.insert("l0.ffn_ln.weight".into(), ln_w.clone());
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), ln_b.clone());
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        // 3. Global Norms
        w.insert("embed_ln.weight".into(), ln_w.clone());
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), ln_b.clone());
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), ln_w.clone());
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), ln_b.clone());
        s.insert("final_ln.bias".into(), vec![4]);

        (w, s)
    }
}
