//! Unified Seq2Seq decoder supporting BART, T5, Whisper, and similar architectures.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
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
        let embeddings = factory.build_embeddings(
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
        let layers = (0..meta.decoder_layers.expect("Invalid deocder layers"))
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
        // 1. Embeddings
        let mut hidden = self.embed(input_ids, position_offset);

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

            let (new_hidden, new_kv) = layer.forward(
                &hidden,
                encoder_hidden_states,
                attention_mask,
                cross_attention_mask,
                past_kv_views,
                cross_kv_for_layer,
                position_bias.as_ref(),
            )?;

            hidden = new_hidden;
            new_self_attn_kvs.push(new_kv);
        }

        // 7. Final layer normalization
        if let Some(norm) = &self.final_norm {
            hidden = norm.forward(&hidden);
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
