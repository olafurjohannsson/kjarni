//! Unified Seq2Seq encoder supporting BART, T5, Whisper, and similar architectures.
//!
//! This encoder uses `ModelConfig` and `ModelLayout` to configure itself,
//! and accepts `ModelInput` for flexible input handling.

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Array4};
use std::{any::Any, sync::Arc};

pub use crate::encoder_decoder::config::{PositionEncodingType, Seq2SeqEncoderConfig};
use crate::{
    Embeddings, Normalization, WgpuContext,
    cpu::encoder::{CpuEncoderOps, encoder_layer::EncoderLayer, prelude::*},
    models::base::{ModelInput, ModelLoadConfig},
    pipeline::Seq2SeqFactory,
    traits::{
        Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    },
    weights::ModelWeights,
};
use crate::{
    cpu::{
        encoder::buffers::EncoderBuffers,
        encoder_decoder::relative_position_bias::T5RelativePositionBias,
    },
    traits::CpuTransformerCore,
};


// Encoder Output


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


// Seq2Seq Encoder (CPU)


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
    pub embeddings: Option<Embeddings>,

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

        let encoder_layout = layout
            .encoder
            .as_ref()
            .ok_or_else(|| anyhow!("Model layout has no encoder component"))?;

        let factory = Seq2SeqFactory::new(weights).with_load_config(&load_config);

        // 1. Build embeddings (optional - Whisper doesn't need token embeddings)
        let embeddings = if let Some(pos_name) = &encoder_layout.position_embedding {
            Some(factory.build_embeddings(&layout.token_embedding, Some(pos_name))?)
        } else {
            // Check if we should still build token embeddings without positions
            // (for models that handle positions differently)
            match encoder_config.position_encoding {
                PositionEncodingType::RelativeBias { .. }
                | PositionEncodingType::Sinusoidal
                | PositionEncodingType::None => {
                    // These don't use learned position embeddings in the Embeddings struct
                    Some(factory.build_embeddings(&layout.token_embedding, None)?)
                }
                PositionEncodingType::Learned { .. } => {
                    // Learned positions should have been in the layout
                    return Err(anyhow!(
                        "Learned position encoding requires position_embedding in layout"
                    ));
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
                let pos_name = encoder_layout.position_embedding.as_ref().ok_or_else(|| {
                    anyhow!("Learned positions require position_embedding in layout")
                })?;
                let embeddings = weights.get_array2(pos_name)?;
                PositionEncoding::Learned { embeddings, offset }
            }
            PositionEncodingType::RelativeBias {
                num_buckets,
                max_distance,
            } => {
                let bias = T5RelativePositionBias::new(
                    weights,
                    "encoder",
                    true, // Encoders ARE bidirectional
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
    pub fn get_layers(&self) -> &Vec<EncoderLayer> {
        &self.layers
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
        let mut hidden = match input {
            ModelInput::TokensCpu(tokens) => {
                let embeddings = self
                    .embeddings
                    .as_ref()
                    .ok_or_else(|| anyhow!("No embeddings available for token input"))?;
                let tokens_owned = tokens.to_owned();

                // Embed tokens (includes positions if learned embeddings)
                let mut h = embeddings.forward(
                    &tokens_owned,
                    None,
                    self.position_offset(),
                    self.meta.scale_embeddings,
                );

                // Apply sinusoidal position encoding (for T5-style, not BART)
                h = self.apply_position_encoding(h)?;

                // Embedding layer normalization
                if let Some(norm) = &self.embed_norm {
                    h = norm.forward(&h);
                }

                h
            }
            ModelInput::HiddenCpu(hidden_states) => hidden_states.to_owned(),
            ModelInput::TokensGpu(_) | ModelInput::HiddenGpu(_) => {
                return Err(anyhow!(
                    "GPU input not supported by CPU encoder. Use Seq2SeqGpuEncoder."
                ));
            }
        };

        // if !is_raw_hidden {
        //     // 2. Add position encoding (for non-learned positions)
        // }
        // hidden = self.apply_position_encoding(hidden)?;

        // // 3. Embedding layer normalization
        // if let Some(norm) = &self.embed_norm {
        //     hidden = norm.forward(&hidden);
        // }

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
    pub fn position_offset(&self) -> usize {
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


// Position Encoding Implementations


/// Runtime position encoding implementation.
pub enum PositionEncoding {
    /// Learned absolute positions (BART)
    Learned {
        embeddings: Array2<f32>,
        offset: usize,
    },
    /// Relative position bias (T5)
    RelativeBias { bias: T5RelativePositionBias },
    /// Sinusoidal positions (Whisper)
    Sinusoidal { cache: Array2<f32> },
    /// No position encoding
    None,
}

/// Create sinusoidal position embeddings.
fn create_sinusoidal_embeddings(max_len: usize, dim: usize) -> Array2<f32> {
    let mut embeddings = Array2::<f32>::zeros((max_len, dim));

    for pos in 0..max_len {
        for i in 0..dim / 2 {
            // Python: denom = 10000.0 ** (2 * i / dim)
            // Rust: 10000_f32.powf(...)
            // CRITICAL: Ensure integer division (2*i)/dim is floating point division in calculation
            let exponent = (2 * i) as f32 / dim as f32;
            let denom = 10000_f32.powf(exponent);

            let angle = pos as f32 / denom;

            embeddings[[pos, 2 * i]] = angle.sin();
            embeddings[[pos, 2 * i + 1]] = angle.cos();
        }
    }

    embeddings
}


// Trait Implementations


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
impl CpuTransformerCore for Seq2SeqCPUEncoder {
    fn final_norm(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        if let Some(norm) = &self.final_norm {
            Ok(norm.forward(hidden_states))
        } else {
            Ok(hidden_states.clone())
        }
    }
    fn embed_norm(&self, hidden: &Array3<f32>) -> Result<Array3<f32>> {
        // DON'T apply position encoding here - already done in embed_tokens!
        if let Some(norm) = &self.embed_norm {
            Ok(norm.forward(hidden))
        } else {
            Ok(hidden.clone())
        }
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn num_attention_heads(&self) -> usize {
        self.meta.num_attention_heads
    }
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
}

#[async_trait]
impl CpuEncoder for Seq2SeqCPUEncoder {
    fn create_buffers(&self, max_batch: usize, max_seq: usize) -> EncoderBuffers {
        EncoderBuffers::new_auto(
            max_batch,
            max_seq,
            self.meta.hidden_size,
            self.meta.num_attention_heads,
            self.meta.intermediate_size,
        )
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
    fn forward(&self, hidden_states: &Array3<f32>, mask: &Array2<f32>) -> Result<CpuEncoderOutput> {
        // DON'T do embedding here - it's already done!
        let output = self.forward_layers(hidden_states, mask, 0, self.num_layers())?;
        let normed = self.final_norm(&output)?;
        Ok(CpuEncoderOutput {
            last_hidden_state: normed,
        })
    }
}

#[cfg(test)]
mod seq2seq_encoder_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{
        AttentionLayout, EncoderLayerLayout, EncoderLayout, FeedForwardLayout, ModelConfig,
        ModelLayout, ModelMetadata, NormalizationStrategy,
    };
    use crate::weights::ModelWeights;
    use anyhow::Result;
    use ndarray::{Array2, Array3};
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ... [MockConfig struct remains the same] ...
    #[derive(Debug, Clone)]
    struct MockConfig {
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        is_prenorm: bool,
        // Allow overriding layout for T5/Whisper specific needs
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
                decoder_layers: None,
                hidden_size: self.hidden_size,
                num_layers: self.num_layers,
                num_attention_heads: self.num_heads,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                vocab_size: self.vocab_size,
                intermediate_size: 0,
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
                encoder: Some(EncoderLayout {
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
                    layer: EncoderLayerLayout {
                        self_attn: AttentionLayout {
                            q_weight: "l0.q.weight".to_string(),
                            q_bias: None,
                            k_weight: "l0.k.weight".to_string(),
                            k_bias: None,
                            v_weight: "l0.v.weight".to_string(),
                            v_bias: None,
                            o_weight: "l0.o.weight".to_string(),
                            o_bias: None,
                            norm_weight: "l0.attn_ln.weight".to_string(),
                            norm_bias: Some("l0.attn_ln.bias".to_string()),
                        },
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
                decoder: None,
            }
        }
    }

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

    fn get_bart_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

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

        w.insert(
            "l0.q.weight".into(),
            vec![
                4.137, 4.138, 4.139, 4.140, 4.141, 4.142, 4.143, 4.144, 4.145, 4.146, 4.147, 4.148,
                4.149, 4.150, 4.151, 4.152,
            ],
        );
        s.insert("l0.q.weight".into(), vec![4, 4]);

        w.insert(
            "l0.k.weight".into(),
            vec![
                4.153, 4.154, 4.155, 4.156, 4.157, 4.158, 4.159, 4.160, 4.161, 4.162, 4.163, 4.164,
                4.165, 4.166, 4.167, 4.168,
            ],
        );
        s.insert("l0.k.weight".into(), vec![4, 4]);

        w.insert(
            "l0.v.weight".into(),
            vec![
                4.169, 4.170, 4.171, 4.172, 4.173, 4.174, 4.175, 4.176, 4.177, 4.178, 4.179, 4.180,
                4.181, 4.182, 4.183, 4.184,
            ],
        );
        s.insert("l0.v.weight".into(), vec![4, 4]);

        w.insert(
            "l0.o.weight".into(),
            vec![
                4.185, 4.186, 4.187, 4.188, 4.189, 4.190, 4.191, 4.192, 4.193, 4.194, 4.195, 4.196,
                4.197, 4.198, 4.199, 4.200,
            ],
        );
        s.insert("l0.o.weight".into(), vec![4, 4]);

        w.insert("l0.attn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.attn_ln.weight".into(), vec![4]);
        w.insert("l0.attn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.attn_ln.bias".into(), vec![4]);
        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        let start = 4.201;
        let fc1_data: Vec<f32> = (0..32).map(|i| start + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);

        let start_fc2 = start + 0.032;
        let fc2_data: Vec<f32> = (0..32).map(|i| start_fc2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);

        (w, s)
    }

    fn get_whisper_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        w.insert("token_emb".into(), vec![0.0; 40]);
        s.insert("token_emb".into(), vec![10, 4]);

        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        w.insert(
            "l0.q.weight".into(),
            vec![
                0.041, 0.042, 0.043, 0.044, 0.045, 0.046, 0.047, 0.048, 0.049, 0.050, 0.051, 0.052,
                0.053, 0.054, 0.055, 0.056,
            ],
        );
        s.insert("l0.q.weight".into(), vec![4, 4]);

        w.insert(
            "l0.k.weight".into(),
            vec![
                0.057, 0.058, 0.059, 0.060, 0.061, 0.062, 0.063, 0.064, 0.065, 0.066, 0.067, 0.068,
                0.069, 0.070, 0.071, 0.072,
            ],
        );
        s.insert("l0.k.weight".into(), vec![4, 4]);

        w.insert(
            "l0.v.weight".into(),
            vec![
                0.073, 0.074, 0.075, 0.076, 0.077, 0.078, 0.079, 0.080, 0.081, 0.082, 0.083, 0.084,
                0.085, 0.086, 0.087, 0.088,
            ],
        );
        s.insert("l0.v.weight".into(), vec![4, 4]);

        w.insert(
            "l0.o.weight".into(),
            vec![
                0.089, 0.090, 0.091, 0.092, 0.093, 0.094, 0.095, 0.096, 0.097, 0.098, 0.099, 0.100,
                0.101, 0.102, 0.103, 0.104,
            ],
        );
        s.insert("l0.o.weight".into(), vec![4, 4]);

        w.insert("l0.attn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.attn_ln.weight".into(), vec![4]);
        w.insert("l0.attn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.attn_ln.bias".into(), vec![4]);
        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        let start = 0.105;
        let fc1_data: Vec<f32> = (0..32).map(|i| start + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);
        let start_fc2 = start + 0.032;
        let fc2_data: Vec<f32> = (0..32).map(|i| start_fc2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);

        (w, s)
    }

    fn get_t5_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

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

        let rel_data: Vec<f32> = (0..32 * 2).map(|i| 0.041 + (i as f32 * 0.001)).collect();
        w.insert(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            rel_data,
        );
        s.insert(
            "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            vec![32, 2],
        );

        w.insert(
            "l0.q.weight".into(),
            vec![
                0.105, 0.106, 0.107, 0.108, 0.109, 0.110, 0.111, 0.112, 0.113, 0.114, 0.115, 0.116,
                0.117, 0.118, 0.119, 0.120,
            ],
        );
        s.insert("l0.q.weight".into(), vec![4, 4]);

        w.insert(
            "l0.k.weight".into(),
            vec![
                0.121, 0.122, 0.123, 0.124, 0.125, 0.126, 0.127, 0.128, 0.129, 0.130, 0.131, 0.132,
                0.133, 0.134, 0.135, 0.136,
            ],
        );
        s.insert("l0.k.weight".into(), vec![4, 4]);

        w.insert(
            "l0.v.weight".into(),
            vec![
                0.137, 0.138, 0.139, 0.140, 0.141, 0.142, 0.143, 0.144, 0.145, 0.146, 0.147, 0.148,
                0.149, 0.150, 0.151, 0.152,
            ],
        );
        s.insert("l0.v.weight".into(), vec![4, 4]);

        w.insert(
            "l0.o.weight".into(),
            vec![
                0.153, 0.154, 0.155, 0.156, 0.157, 0.158, 0.159, 0.160, 0.161, 0.162, 0.163, 0.164,
                0.165, 0.166, 0.167, 0.168,
            ],
        );
        s.insert("l0.o.weight".into(), vec![4, 4]);

        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        w.insert("l0.attn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.attn_ln.weight".into(), vec![4]);
        w.insert("l0.attn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.attn_ln.bias".into(), vec![4]);
        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        let start = 0.169;
        let fc1_data: Vec<f32> = (0..32).map(|i| start + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);
        let start_fc2 = start + 0.032;
        let fc2_data: Vec<f32> = (0..32).map(|i| start_fc2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);

        (w, s)
    }

    #[test]
    fn test_scenario_a_bart_postnorm_learned() -> Result<()> {
        let (weights_map, shapes) = get_bart_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: false,
            no_pos_emb_in_layout: false,
        };

        let enc_config = Seq2SeqEncoderConfig {
            position_encoding: PositionEncodingType::Learned { offset: 0 },
            normalize_embeddings: true,
            final_layer_norm: true,
        };

        let encoder = Seq2SeqCPUEncoder::new(
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        let input_ids = Array2::from_shape_vec((1, 3), vec![1u32, 5, 9]).unwrap();
        let mask = Array2::from_elem((1, 3), 1.0);

        let output = encoder.forward_tokens(&input_ids, &mask)?;

        let golden_data = vec![
            -1.331634, -0.437211, 0.457210, 1.351635, -1.331635, -0.437211, 0.457214, 1.351632,
            -1.331633, -0.437213, 0.457211, 1.351635,
        ];
        let golden = Array3::from_shape_vec((1, 3, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("BART Max Diff: {}", max_diff);
        assert!(max_diff < 1e-4);
        Ok(())
    }

    #[test]
    fn test_scenario_b_whisper_prenorm_sinusoidal() -> Result<()> {
        let (weights_map, shapes) = get_whisper_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true,
            no_pos_emb_in_layout: true,
        };

        let enc_config = Seq2SeqEncoderConfig {
            position_encoding: PositionEncodingType::Sinusoidal,
            normalize_embeddings: true,
            final_layer_norm: true,
        };

        let encoder = Seq2SeqCPUEncoder::new(
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        let input_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000,
                0.900000, 1.000000, 1.100000, 1.200000,
            ],
        )
        .unwrap();
        let mask = Array2::from_elem((1, 3), 1.0);

        let output = encoder.forward_hidden_states(&input_hidden, &mask)?;

        let golden_data = vec![
            -1.153330, 0.814141, -0.794141, 1.173330, 0.247473, -0.264928, -1.361826, 1.419281,
            0.621133, -1.347122, -0.484890, 1.250878,
        ];
        let golden = Array3::from_shape_vec((1, 3, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("Whisper Max Diff: {}", max_diff);
        assert!(max_diff < 1e-3);
        Ok(())
    }

    #[test]
    fn test_scenario_c_t5_prenorm_rel_bias() -> Result<()> {
        let (weights_map, shapes) = get_t5_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true,
            no_pos_emb_in_layout: true,
        };

        let enc_config = Seq2SeqEncoderConfig {
            position_encoding: PositionEncodingType::RelativeBias {
                num_buckets: 32,
                max_distance: 128,
            },
            normalize_embeddings: true,
            final_layer_norm: true,
        };

        let encoder = Seq2SeqCPUEncoder::new(
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        let input_ids = Array2::from_shape_vec((1, 3), vec![1u32, 5, 9]).unwrap();
        let mask = Array2::from_elem((1, 3), 1.0);

        let output = encoder.forward_tokens(&input_ids, &mask)?;

        let golden_data = vec![
            -1.331581, -0.437194, 0.457194, 1.351581, -1.331581, -0.437193, 0.457194, 1.351580,
            -1.331581, -0.437193, 0.457193, 1.351581,
        ];
        let golden = Array3::from_shape_vec((1, 3, 4), golden_data).unwrap();

        let diff = (&output.last_hidden_state - &golden).mapv(|x| x.abs());
        let max_diff = diff.fold(0.0f32, |a, &b| a.max(b));

        println!("T5 Max Diff: {}", max_diff);
        assert!(max_diff < 1e-4);
        Ok(())
    }
}
