//! Unified Seq2Seq encoder supporting BART, T5, Whisper, and similar architectures.
//!
//! This encoder uses `ModelConfig` and `ModelLayout` to configure itself,
//! and accepts `ModelInput` for flexible input handling.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use std::sync::Arc;

use crate::{
    Normalization, WgpuContext, embeddings::Embeddings, encoder::{encoder_layer::EncoderLayer, prelude::*}, gpu_ops::{GpuFrameContext, GpuTensor}, models::base::{ModelInput, ModelLoadConfig}, pipeline::Seq2SeqFactory, traits::{Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy}, weights::ModelWeights
};

pub use super::config::{Seq2SeqEncoderConfig, PositionEncodingType};

// =============================================================================
// Encoder Output
// =============================================================================

/// Output from the Seq2Seq encoder.
#[derive(Debug)]
pub struct EncoderOutput {
    /// Hidden states from the final layer. Shape: [batch, seq, hidden]
    pub last_hidden_state: Array3<f32>,
    /// Hidden states from all layers (if requested).
    pub all_hidden_states: Option<Vec<Array3<f32>>>,
    /// Attention weights from all layers (if requested).
    pub attentions: Option<Vec<Array4<f32>>>,
}

// =============================================================================
// Seq2Seq Encoder (CPU)
// =============================================================================

/// Unified transformer encoder for seq2seq models.
///
/// Supports BART, T5, Whisper, mBART, and similar architectures.
/// Configuration is driven by `ModelConfig` and `Seq2SeqEncoderConfig`.
///
/// # Example
///
/// ```ignore
/// // From a BartConfig
/// let encoder = Seq2SeqEncoder::new(
///     &weights,
///     &bart_config,
///     Seq2SeqEncoderConfig::bart(),
///     load_config,
/// )?;
///
/// // Forward with tokens
/// let input = ModelInput::from_tokens(&token_ids);
/// let output = encoder.forward(input, &attention_mask)?;
///
/// // Forward with pre-computed hidden states (e.g., from audio frontend)
/// let input = ModelInput::from_hidden(audio_hidden_states.view());
/// let output = encoder.forward(input, &attention_mask)?;
/// ```
pub struct Seq2SeqCPUEncoder {
    /// Token embeddings (None for audio-only models like Whisper encoder)
    embeddings: Option<Embeddings>,

    /// Embedding layer normalization
    embed_norm: Option<Normalization>,

    /// Transformer layers
    layers: Vec<EncoderLayer>,

    /// Final layer normalization
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

impl Seq2SeqCPUEncoder {
    /// Create encoder from ModelConfig.
    ///
    /// # Arguments
    /// * `weights` - Model weights
    /// * `config` - Model configuration implementing `ModelConfig`
    /// * `encoder_config` - Seq2Seq-specific configuration
    /// * `load_config` - Weight loading options
    pub fn new<C: ModelConfig>(
        weights: &ModelWeights,
        config: &C,
        encoder_config: Seq2SeqEncoderConfig,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();

        let encoder_layout = layout.encoder.as_ref()
            .ok_or_else(|| anyhow!("Model layout has no encoder component"))?;

        let factory = Seq2SeqFactory::new(weights)
            .with_load_config(&load_config);

        // 1. Build embeddings (optional - Whisper doesn't need token embeddings)
        let embeddings = if let Some(pos_name) = &encoder_layout.position_embedding {
            Some(factory.build_embeddings(&layout.token_embedding, Some(pos_name))?)
        } else {
            // Check if we should still build token embeddings without positions
            // (for models that handle positions differently)
            match encoder_config.position_encoding {
                PositionEncodingType::RelativeBias { .. } |
                PositionEncodingType::Sinusoidal |
                PositionEncodingType::None => {
                    // These don't use learned position embeddings in the Embeddings struct
                    Some(factory.build_embeddings(&layout.token_embedding, None)?)
                }
                PositionEncodingType::Learned { .. } => {
                    // Learned positions should have been in the layout
                    return Err(anyhow!("Learned position encoding requires position_embedding in layout"));
                }
            }
        };

        // 2. Build embedding normalization
        let embed_norm = if encoder_config.normalize_embeddings {
            if let (Some(w), Some(b)) = (
                &encoder_layout.embedding_norm_weight,
                &encoder_layout.embedding_norm_bias,
            ) {
                Some(factory.build_norm(
                    w,
                    Some(b.as_str()),
                    meta.normalization_strategy.clone(),
                    meta.norm_eps,
                    0, // Not layer-indexed
                )?)
            } else if let Some(w) = &encoder_layout.embedding_norm_weight {
                Some(factory.build_norm(
                    w,
                    None,
                    NormalizationStrategy::RMSNorm,
                    meta.norm_eps,
                    0,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // 3. Build transformer layers
        let layers = (0..meta.num_layers)
            .map(|i| factory.build_encoder_layer(encoder_layout, &meta, i))
            .collect::<Result<Vec<_>>>()?;

        // 4. Build final layer normalization
        let final_norm = if encoder_config.final_layer_norm {
            if let Some(w) = &encoder_layout.final_norm_weight {
                Some(factory.build_norm(
                    w,
                    encoder_layout.final_norm_bias.as_deref(),
                    meta.normalization_strategy.clone(),
                    meta.norm_eps,
                    0,
                )?)
            } else {
                None
            }
        } else {
            None
        };

        // 5. Build position encoding
        let position_encoding = match encoder_config.position_encoding {
            PositionEncodingType::Learned { offset } => {
                let pos_name = encoder_layout.position_embedding.as_ref()
                    .ok_or_else(|| anyhow!("Learned positions require position_embedding in layout"))?;
                let embeddings = weights.get_array2(pos_name)?;
                PositionEncoding::Learned { embeddings, offset }
            }
            PositionEncodingType::RelativeBias { num_buckets, max_distance } => {
                let bias = T5RelativePositionBias::new(
                    weights,
                    "encoder", // T5 prefix
                    meta.num_attention_heads,
                    num_buckets,
                    max_distance,
                )?;
                PositionEncoding::RelativeBias { bias }
            }
            PositionEncodingType::Sinusoidal => {
                let cache = create_sinusoidal_embeddings(meta.max_seq_len, meta.hidden_size);
                PositionEncoding::Sinusoidal { cache }
            }
            PositionEncodingType::None => PositionEncoding::None,
        };

        Ok(Self {
            embeddings,
            embed_norm,
            layers,
            final_norm,
            position_encoding,
            pre_norm: meta.is_prenorm,
            meta,
            layout,
        })
    }

    /// Forward pass accepting ModelInput.
    ///
    /// # Arguments
    /// * `input` - Either token IDs or pre-computed hidden states
    /// * `attention_mask` - Attention mask [batch, seq], 1.0 for real tokens
    ///
    /// # Returns
    /// Encoder output containing final hidden states
    pub fn forward(
        &self,
        input: ModelInput,
        attention_mask: &Array2<f32>,
    ) -> Result<EncoderOutput> {
        // 1. Get initial hidden states based on input type
        let (mut hidden, is_raw_hidden) = match input {
            ModelInput::TokensCpu(tokens) => {
                let embeddings = self.embeddings.as_ref()
                    .ok_or_else(|| anyhow!("No embeddings available for token input"))?;
                
                // Convert view to owned for embedding lookup
                let tokens_owned = tokens.to_owned();
                let e = embeddings.forward(&tokens_owned, None, self.position_offset(), false);
                (e, false)
            }
            ModelInput::HiddenCpu(hidden_states) => {
                (hidden_states.to_owned(), true)
            }
            ModelInput::TokensGpu(_) | ModelInput::HiddenGpu(_) => {
                return Err(anyhow!("GPU input not supported by CPU encoder. Use Seq2SeqGpuEncoder."));
            }
        };

        if !is_raw_hidden {
            // 2. Add position encoding (for non-learned positions)
            hidden = self.apply_position_encoding(hidden)?;
        }

        // 3. Embedding layer normalization
        if let Some(norm) = &self.embed_norm {
            hidden = norm.forward(&hidden);
        }

        // 4. Compute relative position bias (T5)
        let position_bias = self.compute_position_bias(&hidden)?;

        // 5. Transformer layers
        for layer in &self.layers {
            hidden = layer.forward(
                hidden,
                attention_mask,
                position_bias.as_ref(),
                self.pre_norm,
                None, // output_attentions
            )?;
        }

        // 6. Final layer normalization
        if let Some(norm) = &self.final_norm {
            hidden = norm.forward(&hidden);
        }

        Ok(EncoderOutput {
            last_hidden_state: hidden,
            all_hidden_states: None,
            attentions: None,
        })
    }

    /// Forward pass for token input (convenience method).
    pub fn forward_tokens(
        &self,
        input_ids: &Array2<u32>,
        attention_mask: &Array2<f32>,
    ) -> Result<EncoderOutput> {
        self.forward(ModelInput::TokensCpu(input_ids.view()), attention_mask)
    }

    /// Forward pass for hidden state input (convenience method).
    ///
    /// Use this for audio models where hidden states come from an AudioPipeline.
    pub fn forward_hidden_states(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
    ) -> Result<EncoderOutput> {
        self.forward(ModelInput::HiddenCpu(hidden_states.view()), attention_mask)
    }

    /// Get position offset for learned embeddings.
    fn position_offset(&self) -> usize {
        match &self.position_encoding {
            PositionEncoding::Learned { offset, .. } => *offset,
            _ => 0,
        }
    }

    /// Apply position encoding to hidden states.
    fn apply_position_encoding(&self, hidden: Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq_len, hidden_size) = hidden.dim();

        match &self.position_encoding {
            PositionEncoding::Learned { embeddings, offset } => {
                // Learned positions are typically added during embedding lookup
                // If they weren't, add them here
                let mut result = hidden;
                for b in 0..batch {
                    for s in 0..seq_len {
                        let pos = s + offset;
                        if pos < embeddings.nrows() {
                            for h in 0..hidden_size {
                                result[[b, s, h]] += embeddings[[pos, h]];
                            }
                        }
                    }
                }
                Ok(result)
            }
            PositionEncoding::Sinusoidal { cache } => {
                let mut result = hidden;
                for b in 0..batch {
                    for s in 0..seq_len.min(cache.nrows()) {
                        for h in 0..hidden_size {
                            result[[b, s, h]] += cache[[s, h]];
                        }
                    }
                }
                Ok(result)
            }
            PositionEncoding::RelativeBias { .. } | PositionEncoding::None => {
                // RelativeBias is applied in attention, not here
                Ok(hidden)
            }
        }
    }

    /// Compute position bias for attention (T5).
    fn compute_position_bias(&self, hidden: &Array3<f32>) -> Result<Option<Array4<f32>>> {
        match &self.position_encoding {
            PositionEncoding::RelativeBias { bias } => {
                let seq_len = hidden.dim().1;
                Ok(Some(bias.compute(seq_len, seq_len)?))
            }
            _ => Ok(None),
        }
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
}

// =============================================================================
// Position Encoding Implementations
// =============================================================================

/// Runtime position encoding implementation.
pub enum PositionEncoding {
    /// Learned absolute positions (BART)
    Learned {
        embeddings: Array2<f32>,
        offset: usize,
    },
    /// Relative position bias (T5)
    RelativeBias {
        bias: T5RelativePositionBias,
    },
    /// Sinusoidal positions (Whisper)
    Sinusoidal {
        cache: Array2<f32>,
    },
    /// No position encoding
    None,
}

/// T5-style relative position bias.
pub struct T5RelativePositionBias {
    embeddings: Array2<f32>,
    num_buckets: usize,
    max_distance: usize,
}

impl T5RelativePositionBias {
    pub fn new(
        weights: &ModelWeights,
        prefix: &str,
        num_heads: usize,
        num_buckets: usize,
        max_distance: usize,
    ) -> Result<Self> {
        // Try different naming conventions
        let embeddings = weights
            .get_array2(&format!("{}.block.0.layer.0.SelfAttention.relative_attention_bias.weight", prefix))
            .or_else(|_| weights.get_array2(&format!("{}.layer.0.SelfAttention.relative_attention_bias.weight", prefix)))?;

        Ok(Self {
            embeddings,
            num_buckets,
            max_distance,
        })
    }

    pub fn compute(&self, query_len: usize, key_len: usize) -> Result<Array4<f32>> {
        let num_heads = self.embeddings.nrows();
        let mut bias = Array4::<f32>::zeros((1, num_heads, query_len, key_len));

        for q in 0..query_len {
            for k in 0..key_len {
                let relative_pos = k as i32 - q as i32;
                let bucket = self.relative_position_bucket(relative_pos);

                for h in 0..num_heads {
                    bias[[0, h, q, k]] = self.embeddings[[h, bucket]];
                }
            }
        }

        Ok(bias)
    }

    fn relative_position_bucket(&self, relative_position: i32) -> usize {
        let n = (-relative_position).max(0);
        let max_exact = self.num_buckets / 2;
        let is_small = n < max_exact as i32;

        let bucket = if is_small {
            n as usize
        } else {
            let val = ((n as f32 / max_exact as f32).ln()
                / (self.max_distance as f32 / max_exact as f32).ln()
                * (self.num_buckets - max_exact) as f32) as usize;
            (max_exact + val).min(self.num_buckets - 1)
        };

        let final_bucket = if relative_position > 0 {
            bucket + self.num_buckets / 2
        } else {
            bucket
        };

        final_bucket.min(self.num_buckets - 1)
    }
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

// =============================================================================
// Trait Implementations
// =============================================================================

impl InferenceModel for Seq2SeqCPUEncoder {
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
impl CpuEncoder for Seq2SeqCPUEncoder {
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings
            .as_ref()
            .map(|e| e.forward(input_ids, token_type_ids, self.position_offset(), self.meta.scale_embeddings))
            .unwrap_or_else(|| {
                Array3::zeros((input_ids.nrows(), input_ids.ncols(), self.meta.hidden_size))
            })
    }

    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        let hidden = self.embed(input_ids, token_type_ids);
        if let Some(norm) = &self.embed_norm {
            norm.forward(&hidden)
        } else {
            hidden
        }
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let position_bias = self.compute_position_bias(&hidden)?;

        for layer in self.layers.iter().take(end_layer).skip(start_layer) {
            hidden = layer.forward(
                hidden,
                attention_mask,
                position_bias.as_ref(),
                self.pre_norm,
                None,
            )?;
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
}