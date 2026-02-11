//! Unified Seq2Seq GPU encoder

use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3};
use std::sync::Arc;
use wgpu::CommandEncoder;

use crate::{
    WgpuContext,
    cpu::encoder::GpuEncoder,
    encoder_decoder::config::{PositionEncodingType, Seq2SeqEncoderConfig},
    gpu::{
        GpuTensor, GpuTensorPool,
        normalization::{
            GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
        },
    },
    gpu_ops::blocks::{
        GpuFeedForwardWeights, GpuFeedForwardWeightsStd, attention::GpuAttentionWeights,
        encoder::GpuEncoderLayer,
    },
    models::base::ModelLoadConfig,
    tensor::DType,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

pub struct Seq2SeqGPUEncoder {
    context: Arc<WgpuContext>,

    embed_layer_norm: Option<GpuNormalization>,
    embed_ln_weights: Option<GpuNormalizationWeights>,

    layers: Vec<GpuEncoderLayer>,

    final_layer_norm: Option<GpuNormalization>,
    final_ln_weights: Option<GpuNormalizationWeights>,

    
    pre_norm: bool,

    sinusoidal_cache: Option<Array2<f32>>, // cpu cache

    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl Seq2SeqGPUEncoder {
    /// Create encoder
    pub fn new<C: ModelConfig>(
        context: &Arc<WgpuContext>,
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
        let target_dt = load_config.target_dtype;

        log::info!(
            "Building Seq2SeqGpuEncoder: {} layers, hidden_size={}, pre_norm={}, embed_norm={}, final_norm={}",
            meta.num_layers,
            meta.hidden_size,
            meta.is_prenorm,
            encoder_config.normalize_embeddings,
            encoder_config.final_layer_norm
        );

        let (embed_layer_norm, embed_ln_weights) = if encoder_config.normalize_embeddings {
            if let (Some(w_key), Some(b_key)) = (
                &encoder_layout.embedding_norm_weight,
                &encoder_layout.embedding_norm_bias,
            ) {
                log::debug!("Loading embedding LayerNorm: {} / {}", w_key, b_key);
                let weights_gpu = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        w_key,
                        target_dt,
                        "embed_ln_w",
                    )?,
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        b_key,
                        target_dt,
                        "embed_ln_b",
                    )?,
                )?);
                let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
                (Some(norm), Some(weights_gpu))
            } else if let Some(w_key) = &encoder_layout.embedding_norm_weight {
                log::debug!("Loading embedding RMSNorm: {}", w_key);
                let weights_gpu = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        w_key,
                        target_dt,
                        "embed_ln_w",
                    )?,
                    GpuTensor::zeros(
                        context,
                        vec![meta.hidden_size],
                        target_dt.unwrap_or(DType::F32),
                        "embed_ln_b_zero",
                    )?,
                )?);
                let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
                (Some(norm), Some(weights_gpu))
            } else {
                log::warn!("normalize_embeddings=true but no embedding norm weights in layout");
                (None, None)
            }
        } else {
            (None, None)
        };

        // transformer layers
        let layers = Self::build_layers(context, weights, &meta, &layout, &load_config)?;

        // optional final norm
        let (final_layer_norm, final_ln_weights) = if encoder_config.final_layer_norm {
            if let Some(w_key) = &encoder_layout.final_norm_weight {
                let b_key = encoder_layout.final_norm_bias.as_deref();

                log::debug!("Loading final LayerNorm: {} / {:?}", w_key, b_key);

                let weights_gpu = if let Some(b) = b_key {
                    GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                        GpuTensor::from_model_weights(
                            context,
                            weights,
                            w_key,
                            target_dt,
                            "final_ln_w",
                        )?,
                        GpuTensor::from_model_weights(
                            context,
                            weights,
                            b,
                            target_dt,
                            "final_ln_b",
                        )?,
                    )?)
                } else {
                    // RMSNorm (T5 style) - no bias
                    GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                        GpuTensor::from_model_weights(
                            context,
                            weights,
                            w_key,
                            target_dt,
                            "final_ln_w",
                        )?,
                        GpuTensor::zeros(
                            context,
                            vec![meta.hidden_size],
                            target_dt.unwrap_or(DType::F32),
                            "final_ln_b_zero",
                        )?,
                    )?)
                };

                let norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));
                (Some(norm), Some(weights_gpu))
            } else {
                log::warn!("final_layer_norm=true but no final norm weights in layout");
                (None, None)
            }
        } else {
            (None, None)
        };
        let sinusoidal_cache = match &encoder_config.position_encoding {
            PositionEncodingType::Sinusoidal => Some(create_sinusoidal_embeddings(
                meta.max_seq_len,
                meta.hidden_size,
            )),
            _ => None,
        };

        log::debug!(
            "Seq2SeqGpuEncoder built: {} layers, embed_norm={}, final_norm={}",
            layers.len(),
            embed_layer_norm.is_some(),
            final_layer_norm.is_some()
        );

        Ok(Self {
            context: context.clone(),
            embed_layer_norm,
            embed_ln_weights,
            sinusoidal_cache,
            layers,
            final_layer_norm,
            final_ln_weights,
            pre_norm: meta.is_prenorm,
            meta,
            layout,
        })
    }

    fn build_layers(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        load_config: &ModelLoadConfig,
    ) -> Result<Vec<GpuEncoderLayer>> {
        let encoder_layout = layout
            .encoder
            .as_ref()
            .expect("Model layout must have an encoder section");
        let target_dt = load_config.target_dtype;

        let mut layers = Vec::with_capacity(meta.num_layers);

        for i in 0..meta.num_layers {
            log::debug!("Building encoder layer {}", i);

            let self_attn_weights = GpuAttentionWeights::from_encoder_self_attn_layout(
                context,
                weights,
                layout,
                i,
                target_dt,
                meta.hidden_size,
            )?;
            let sa_norm_w = encoder_layout
                .layer
                .self_attn
                .norm_weight
                .replace("{}", &i.to_string());
            let sa_norm_b = encoder_layout
                .layer
                .self_attn
                .norm_bias
                .as_ref()
                .map(|b| b.replace("{}", &i.to_string()));

            let self_attn_ln_weights = if let Some(b_key) = sa_norm_b {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context, weights, &sa_norm_w, target_dt, "sa_ln_w",
                    )?,
                    GpuTensor::from_model_weights(context, weights, &b_key, target_dt, "sa_ln_b")?,
                )?)
            } else {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context, weights, &sa_norm_w, target_dt, "sa_ln_w",
                    )?,
                    GpuTensor::zeros(
                        context,
                        vec![meta.hidden_size],
                        target_dt.unwrap_or(DType::F32),
                        "sa_ln_b_zero",
                    )?,
                )?)
            };

            let fc1_w_key = encoder_layout
                .layer
                .ffn
                .up_weight
                .replace("{}", &i.to_string());
            let fc2_w_key = encoder_layout
                .layer
                .ffn
                .down_weight
                .replace("{}", &i.to_string());

            let raw_fc1_w = weights.get_array2(&fc1_w_key)?;
            let raw_fc2_w = weights.get_array2(&fc2_w_key)?;

            let fc1_w = if meta.transpose_ffn_weights {
                raw_fc1_w.t().as_standard_layout().to_owned()
            } else {
                raw_fc1_w
            };
            let fc2_w = if meta.transpose_ffn_weights {
                raw_fc2_w.t().as_standard_layout().to_owned()
            } else {
                raw_fc2_w
            };

            // Handle optional FFN biases
            let fc1_b = if let Some(b_template) = &encoder_layout.layer.ffn.up_bias {
                weights.get_array1(&b_template.replace("{}", &i.to_string()))?
            } else {
                ndarray::Array1::zeros(fc1_w.nrows())
            };

            let fc2_b = if let Some(b_template) = &encoder_layout.layer.ffn.down_bias {
                weights.get_array1(&b_template.replace("{}", &i.to_string()))?
            } else {
                ndarray::Array1::zeros(fc2_w.nrows())
            };

            let ff_weights = GpuFeedForwardWeights::Standard(GpuFeedForwardWeightsStd::new(
                GpuTensor::from_ndarray(context, &fc1_w)?,
                GpuTensor::from_ndarray(context, &fc1_b)?,
                GpuTensor::from_ndarray(context, &fc2_w)?,
                GpuTensor::from_ndarray(context, &fc2_b)?,
            )?);

            // ================================================================
            // FFN LAYER NORM
            // ================================================================
            let ffn_norm_w = encoder_layout
                .layer
                .ffn
                .norm_weight
                .replace("{}", &i.to_string());
            let ffn_norm_b = encoder_layout
                .layer
                .ffn
                .norm_bias
                .as_ref()
                .map(|b| b.replace("{}", &i.to_string()));

            let ffn_ln_weights = if let Some(b_key) = ffn_norm_b {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &ffn_norm_w,
                        target_dt,
                        "ffn_ln_w",
                    )?,
                    GpuTensor::from_model_weights(context, weights, &b_key, target_dt, "ffn_ln_b")?,
                )?)
            } else {
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_model_weights(
                        context,
                        weights,
                        &ffn_norm_w,
                        target_dt,
                        "ffn_ln_w",
                    )?,
                    GpuTensor::zeros(
                        context,
                        vec![meta.hidden_size],
                        target_dt.unwrap_or(DType::F32),
                        "ffn_ln_b_zero",
                    )?,
                )?)
            };

            // ================================================================
            // BUILD LAYER
            // ================================================================
            layers.push(GpuEncoderLayer::new(
                context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                meta.activation,
                meta,
            )?);
        }

        Ok(layers)
    }

    // ========================================================================
    // ACCESSORS
    // ========================================================================

    /// Get the WGPU context.
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden size.
    pub fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    /// Whether this encoder uses pre-norm architecture.
    pub fn is_prenorm(&self) -> bool {
        self.pre_norm
    }

    /// Whether this encoder has embedding layer norm.
    pub fn has_embed_norm(&self) -> bool {
        self.embed_layer_norm.is_some()
    }

    /// Whether this encoder has final layer norm.
    pub fn has_final_norm(&self) -> bool {
        self.final_layer_norm.is_some()
    }
    /// Uses CPU fallback since this is only for testing/rare use cases.
    pub fn apply_sinusoidal_positions(
        &self,
        hidden_states: &Array3<f32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        match &self.sinusoidal_cache {
            Some(cache) => {
                let mut result = hidden_states.clone();
                let (batch, seq_len, hidden_size) = result.dim();

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
            None => Ok(hidden_states.clone()),
        }
    }

    pub fn has_sinusoidal(&self) -> bool {
        self.sinusoidal_cache.is_some()
    }
}

// ============================================================================
// GpuEncoder TRAIT IMPLEMENTATION
// ============================================================================

impl GpuEncoder for Seq2SeqGPUEncoder {
    fn embed(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: crate::models::base::ModelInput<'_>,
        token_type_ids: Option<crate::models::base::ModelInput<'_>>,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }
    fn embed_and_normalize(
        &self,
        cmd_encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: crate::models::base::ModelInput<'_>,
        token_type_ids: Option<crate::models::base::ModelInput<'_>>,
    ) -> Result<GpuTensor> {
        unimplemented!()
    }
    fn embed_norm(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        match (&self.embed_layer_norm, &self.embed_ln_weights) {
            (Some(norm), Some(weights)) => {
                let output = pool.get(hidden_states.shape().to_vec());
                norm.encode(cmd_encoder, weights, hidden_states, &output);
                Ok(output)
            }
            _ => {
                // No embed norm (e.g., T5) - return input unchanged
                Ok(hidden_states.clone())
            }
        }
    }

    fn forward_layers(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        // Validate layer range
        if start_layer > end_layer {
            return Err(anyhow!(
                "Invalid layer range: start ({}) > end ({})",
                start_layer,
                end_layer
            ));
        }
        if end_layer > self.layers.len() {
            return Err(anyhow!(
                "Layer range end ({}) exceeds number of layers ({})",
                end_layer,
                self.layers.len()
            ));
        }

        // Handle empty range
        if start_layer == end_layer {
            return Ok(hidden_states.clone());
        }

        // Process layers
        let mut hidden = hidden_states.clone();
        for (i, layer) in self.layers[start_layer..end_layer].iter().enumerate() {
            let layer_idx = start_layer + i;
            log::trace!("Processing encoder layer {}", layer_idx);

            // Use pre-norm or post-norm based on architecture
            hidden = if self.pre_norm {
                layer.forward_prenorm(cmd_encoder, &hidden, attention_mask, pool)?
            } else {
                layer.forward_postnorm(cmd_encoder, &hidden, attention_mask, pool)?
            };
        }

        Ok(hidden)
    }

    fn final_norm(
        &self,
        cmd_encoder: &mut CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
    ) -> Result<GpuTensor> {
        match (&self.final_layer_norm, &self.final_ln_weights) {
            (Some(norm), Some(weights)) => {
                let output = pool.get(hidden_states.shape().to_vec());
                norm.encode(cmd_encoder, weights, hidden_states, &output);
                Ok(output)
            }
            _ => {
                // No final norm (e.g., BART) - return input unchanged
                Ok(hidden_states.clone())
            }
        }
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }
}

// Add this helper function (same as CPU decoder has):
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

#[cfg(test)]
mod seq2seq_gpu_encoder_tests {
    use super::*;
    use crate::activations::Activation;
    use crate::cpu::encoder::CpuEncoder;
    use crate::gpu::GpuFrameContext;
    use crate::models::base::ModelLoadConfig;
    use crate::traits::{
        AttentionLayout, CpuTransformerCore, EncoderLayerLayout, EncoderLayout, FeedForwardLayout,
        ModelConfig, ModelLayout, ModelMetadata, NormalizationStrategy,
    };
    use crate::weights::ModelWeights;
    use ndarray::{Array2, Array3};
    use safetensors::tensor::{Dtype, TensorView};
    use std::collections::HashMap;
    use tempfile::TempDir;

    // ========================================================================
    // Test Helpers
    // ========================================================================

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new()
            .await
            .expect("Failed to create WGPU context")
    }

    fn assert_gpu_cpu_close(
        gpu_result: &Array3<f32>,
        cpu_result: &Array3<f32>,
        atol: f32,
        context: &str,
    ) {
        assert_eq!(
            gpu_result.shape(),
            cpu_result.shape(),
            "{}: Shape mismatch - GPU: {:?}, CPU: {:?}",
            context,
            gpu_result.shape(),
            cpu_result.shape()
        );

        let max_diff = gpu_result
            .iter()
            .zip(cpu_result.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < atol,
            "{}: Max diff {} exceeds tolerance {}.\nGPU first 5: {:?}\nCPU first 5: {:?}",
            context,
            max_diff,
            atol,
            gpu_result.iter().take(5).collect::<Vec<_>>(),
            cpu_result.iter().take(5).collect::<Vec<_>>()
        );

        println!("{}: PASSED (max_diff={})", context, max_diff);
    }

    // ========================================================================
    // Mock Config (same as CPU tests)
    // ========================================================================

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
                decoder_layers: None,
                hidden_size: self.hidden_size,
                num_layers: self.num_layers,
                num_attention_heads: self.num_heads,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                vocab_size: self.vocab_size,
                intermediate_size: self.hidden_size * 2, // 4 * 2 = 8 for our test
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
                problem_type: None,
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
                            q_weight: "l{}.q.weight".to_string(),
                            q_bias: None,
                            k_weight: "l{}.k.weight".to_string(),
                            k_bias: None,
                            v_weight: "l{}.v.weight".to_string(),
                            v_bias: None,
                            o_weight: "l{}.o.weight".to_string(),
                            o_bias: None,
                            norm_weight: "l{}.attn_ln.weight".to_string(),
                            norm_bias: Some("l{}.attn_ln.bias".to_string()),
                        },
                        ffn: FeedForwardLayout {
                            up_weight: "l{}.fc1.weight".to_string(),
                            up_bias: Some("l{}.fc1.bias".to_string()),
                            down_weight: "l{}.fc2.weight".to_string(),
                            down_bias: Some("l{}.fc2.bias".to_string()),
                            gate_weight: None,
                            gate_bias: None,
                            norm_weight: "l{}.ffn_ln.weight".to_string(),
                            norm_bias: Some("l{}.ffn_ln.bias".to_string()),
                        },
                    },
                }),
                decoder: None,
            }
        }
    }

    // ========================================================================
    // Weight Creation Helper
    // ========================================================================

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

    // ========================================================================
    // Golden Data Generators (same as CPU tests)
    // ========================================================================

    fn get_bart_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // Token embeddings
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

        // Position embeddings
        let pos_data: Vec<f32> = (0..1024 * 4).map(|i| 0.041 + (i as f32 * 0.001)).collect();
        w.insert("pos_emb".into(), pos_data);
        s.insert("pos_emb".into(), vec![1024, 4]);

        // Embed LayerNorm
        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);

        // Final LayerNorm
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        // Layer 0 - Attention weights
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

        // Layer 0 - Attention LayerNorm
        w.insert("l0.attn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.attn_ln.weight".into(), vec![4]);
        w.insert("l0.attn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.attn_ln.bias".into(), vec![4]);

        // Layer 0 - FFN LayerNorm
        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        // Layer 0 - FFN weights
        let start = 4.201;
        let fc1_data: Vec<f32> = (0..32).map(|i| start + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);
        w.insert("l0.fc1.bias".into(), vec![0.01; 8]);
        s.insert("l0.fc1.bias".into(), vec![8]);

        let start_fc2 = start + 0.032;
        let fc2_data: Vec<f32> = (0..32).map(|i| start_fc2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);
        w.insert("l0.fc2.bias".into(), vec![0.01; 4]);
        s.insert("l0.fc2.bias".into(), vec![4]);

        (w, s)
    }

    fn get_whisper_golden_data() -> (HashMap<String, Vec<f32>>, HashMap<String, Vec<usize>>) {
        let mut w = HashMap::new();
        let mut s = HashMap::new();

        // Dummy token_emb (Whisper encoder doesn't use it for hidden input)
        w.insert("token_emb".into(), vec![0.0; 40]);
        s.insert("token_emb".into(), vec![10, 4]);

        // Embed LayerNorm
        w.insert("embed_ln.weight".into(), vec![1.0; 4]);
        s.insert("embed_ln.weight".into(), vec![4]);
        w.insert("embed_ln.bias".into(), vec![0.01; 4]);
        s.insert("embed_ln.bias".into(), vec![4]);

        // Final LayerNorm
        w.insert("final_ln.weight".into(), vec![1.0; 4]);
        s.insert("final_ln.weight".into(), vec![4]);
        w.insert("final_ln.bias".into(), vec![0.01; 4]);
        s.insert("final_ln.bias".into(), vec![4]);

        // Layer 0 - Attention weights
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

        // Layer 0 - Attention LayerNorm
        w.insert("l0.attn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.attn_ln.weight".into(), vec![4]);
        w.insert("l0.attn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.attn_ln.bias".into(), vec![4]);

        // Layer 0 - FFN LayerNorm
        w.insert("l0.ffn_ln.weight".into(), vec![1.0; 4]);
        s.insert("l0.ffn_ln.weight".into(), vec![4]);
        w.insert("l0.ffn_ln.bias".into(), vec![0.01; 4]);
        s.insert("l0.ffn_ln.bias".into(), vec![4]);

        // Layer 0 - FFN weights
        let start = 0.105;
        let fc1_data: Vec<f32> = (0..32).map(|i| start + (i as f32 * 0.001)).collect();
        w.insert("l0.fc1.weight".into(), fc1_data);
        s.insert("l0.fc1.weight".into(), vec![8, 4]);
        w.insert("l0.fc1.bias".into(), vec![0.01; 8]);
        s.insert("l0.fc1.bias".into(), vec![8]);

        let start_fc2 = start + 0.032;
        let fc2_data: Vec<f32> = (0..32).map(|i| start_fc2 + (i as f32 * 0.001)).collect();
        w.insert("l0.fc2.weight".into(), fc2_data);
        s.insert("l0.fc2.weight".into(), vec![4, 8]);
        w.insert("l0.fc2.bias".into(), vec![0.01; 4]);
        s.insert("l0.fc2.bias".into(), vec![4]);

        (w, s)
    }

    #[tokio::test]
    async fn test_gpu_scenario_a_bart_postnorm() -> Result<()> {
        let ctx = get_test_context().await;
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

        let encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        // Verify encoder configuration
        assert!(!encoder.is_prenorm(), "BART should be post-norm");
        assert!(encoder.has_embed_norm(), "BART should have embed norm");
        assert!(
            encoder.has_final_norm(),
            "Test config has final norm enabled"
        );
        assert_eq!(encoder.num_layers(), 1);
        assert_eq!(encoder.hidden_size(), 4);
        let input_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.046, 0.052, 0.058, 0.064, // token 1 + pos 0
                0.062, 0.076, 0.090, 0.104, // token 5 + pos 1
                0.078, 0.100, 0.122, 0.144, // token 9 + pos 2
            ],
        )?;
        let mask = Array2::from_elem((1, 3), 1.0);

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_hidden)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        let gpu_output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();
            let normed = encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            let output = encoder.forward2(cmd_enc, pool_ref, &normed, &mask_gpu)?;
            frame.finish();
            output.last_hidden_state.to_ndarray_3d::<f32>().await?
        };
        let golden_data = vec![
            -1.331634, -0.437211, 0.457210, 1.351635, -1.331635, -0.437211, 0.457214, 1.351632,
            -1.331633, -0.437213, 0.457211, 1.351635,
        ];
        let golden = Array3::from_shape_vec((1, 3, 4), golden_data)?;

        assert_gpu_cpu_close(&gpu_output, &golden, 1e-3, "BART Post-Norm GPU");

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_embed_norm_only() -> Result<()> {
        let ctx = get_test_context().await;
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
            final_layer_norm: false,
        };

        let encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        // Input
        let input =
            Array3::from_shape_vec((1, 2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;

        // Run embed_norm
        let gpu_output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            frame.finish();
            normed.to_ndarray_3d::<f32>().await?
        };

        // LayerNorm should normalize each position to mean≈0, var≈1
        // With weight=1, bias=0.01
        let expected = Array3::from_shape_vec(
            (1, 2, 4),
            vec![
                -1.331634 + 0.01,
                -0.437211 + 0.01,
                0.457210 + 0.01,
                1.351635 + 0.01,
                -1.331634 + 0.01,
                -0.437211 + 0.01,
                0.457210 + 0.01,
                1.351635 + 0.01,
            ],
        )?;

        // Check normalization works (rough check)
        let max_diff = gpu_output
            .iter()
            .zip(expected.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 0.1, "embed_norm max diff {} too high", max_diff);
        println!("embed_norm test PASSED (max_diff={})", max_diff);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_forward_layers_range() -> Result<()> {
        let ctx = get_test_context().await;
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
            final_layer_norm: false,
        };

        let encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        let input =
            Array3::from_shape_vec((1, 2, 4), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])?;
        let mask = Array2::from_elem((1, 2), 1.0);

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        // Test forward_layers with range [0, 0] - should return input unchanged
        let output_empty_range = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let out = encoder.forward_layers(cmd_enc, pool_ref, &input_gpu, &mask_gpu, 0, 0)?;
            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        assert_eq!(output_empty_range.shape(), input.shape());
        println!("Empty range test PASSED");

        // Test forward_layers with full range [0, 1]
        let output_full = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let out = encoder.forward_layers(cmd_enc, pool_ref, &input_gpu, &mask_gpu, 0, 1)?;
            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        // Output should be different from input (layer was applied)
        let diff: f32 = output_full
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(diff > 0.1, "Layer should change the output");
        println!("Full range test PASSED (diff from input: {})", diff);

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_no_embed_norm() -> Result<()> {
        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map, shapes)?;

        let config = MockConfig {
            vocab_size: 10,
            hidden_size: 4,
            num_layers: 1,
            num_heads: 2,
            is_prenorm: true, // T5-like
            no_pos_emb_in_layout: true,
        };

        // T5-like: no embed norm
        let enc_config = Seq2SeqEncoderConfig {
            position_encoding: PositionEncodingType::None,
            normalize_embeddings: false,
            final_layer_norm: true,
        };

        let encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        assert!(!encoder.has_embed_norm(), "Should not have embed norm");
        assert!(encoder.has_final_norm(), "Should have final norm");

        let input =
            Array3::from_shape_vec((1, 2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])?;

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;

        // embed_norm should return input unchanged
        let output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let out = encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        // Should be identical to input
        let max_diff = output
            .iter()
            .zip(input.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_diff < 1e-6,
            "embed_norm should be identity when disabled"
        );
        println!("No embed_norm test PASSED");

        Ok(())
    }

    #[tokio::test]
    async fn test_gpu_cpu_parity_bart() -> Result<()> {
        use crate::cpu::encoder_decoder::Seq2SeqCPUEncoder;

        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_bart_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map.clone(), shapes.clone())?;

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

        // Build both CPU and GPU encoders
        let cpu_encoder = Seq2SeqCPUEncoder::new(
            &model_weights,
            &config,
            enc_config.clone(),
            ModelLoadConfig::default(),
        )?;

        let gpu_encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        // Same input for both
        let input_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        )?;
        let mask = Array2::from_elem((1, 3), 1.0);

        // CPU forward
        let cpu_normed = cpu_encoder.embed_norm(&input_hidden)?;
        let cpu_output = cpu_encoder.forward_layers(&cpu_normed, &mask, 0, 1)?;
        let cpu_final = cpu_encoder.final_norm(&cpu_output)?;

        // GPU forward
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_hidden)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        let gpu_final = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = gpu_encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            let output = gpu_encoder.forward_layers(cmd_enc, pool_ref, &normed, &mask_gpu, 0, 1)?;
            let final_out = gpu_encoder.final_norm(cmd_enc, pool_ref, &output)?;
            frame.finish();
            final_out.to_ndarray_3d::<f32>().await?
        };

        assert_gpu_cpu_close(&gpu_final, &cpu_final, 1e-4, "GPU/CPU Parity BART");

        Ok(())
    }
    #[tokio::test]
    async fn test_gpu_scenario_b_whisper_prenorm() -> Result<()> {
        let ctx = get_test_context().await;
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

        let encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        assert!(encoder.is_prenorm(), "Whisper should be pre-norm");
        assert!(encoder.has_embed_norm(), "Whisper should have embed norm");
        assert!(encoder.has_final_norm(), "Whisper should have final norm");
        assert!(
            encoder.has_sinusoidal(),
            "Whisper should have sinusoidal positions"
        );

        // Raw input
        let input_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000,
                0.900000, 1.000000, 1.100000, 1.200000,
            ],
        )?;

        // Apply sinusoidal on CPU before uploading
        let input_with_pos = encoder.apply_sinusoidal_positions(&input_hidden, 0)?;

        let mask = Array2::from_elem((1, 3), 1.0);

        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_with_pos)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        let gpu_output = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            let output = encoder.forward2(cmd_enc, pool_ref, &normed, &mask_gpu)?;
            frame.finish();
            output.last_hidden_state.to_ndarray_3d::<f32>().await?
        };

        let golden_data = vec![
            -1.153330, 0.814141, -0.794141, 1.173330, 0.247473, -0.264928, -1.361826, 1.419281,
            0.621133, -1.347122, -0.484890, 1.250878,
        ];
        let golden = Array3::from_shape_vec((1, 3, 4), golden_data)?;

        assert_gpu_cpu_close(&gpu_output, &golden, 1e-3, "Whisper Pre-Norm GPU");
        Ok(())
    }
    #[tokio::test]
    async fn test_gpu_cpu_parity_whisper() -> Result<()> {
        use crate::cpu::encoder_decoder::Seq2SeqCPUEncoder;

        let ctx = get_test_context().await;
        let (weights_map, shapes) = get_whisper_golden_data();
        let (model_weights, _tmp) = create_model_weights(weights_map.clone(), shapes.clone())?;

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

        // Build both CPU and GPU encoders
        let cpu_encoder = Seq2SeqCPUEncoder::new(
            &model_weights,
            &config,
            enc_config.clone(),
            ModelLoadConfig::default(),
        )?;

        let gpu_encoder = Seq2SeqGPUEncoder::new(
            &ctx,
            &model_weights,
            &config,
            enc_config,
            ModelLoadConfig::default(),
        )?;

        // Same input for both (hidden states, as Whisper would use)
        let input_hidden = Array3::from_shape_vec(
            (1, 3, 4),
            vec![
                0.1, 1.2, 0.3, 1.4, 1.341471, 1.140302, 1.541471, 1.340302, 1.809297, 0.583853,
                2.009297, 0.783853,
            ],
        )?;
        let mask = Array2::from_elem((1, 3), 1.0);

        // CPU forward
        let cpu_normed = cpu_encoder.embed_norm(&input_hidden)?;
        let cpu_output = cpu_encoder.forward_layers(&cpu_normed, &mask, 0, 1)?;
        let cpu_final = cpu_encoder.final_norm(&cpu_output)?;

        // GPU forward
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input_hidden)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        let gpu_final = {
            let pool = ctx.get_inference_pool();
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (cmd_enc, pool_ref) = frame.resources();

            let normed = gpu_encoder.embed_norm(cmd_enc, pool_ref, &input_gpu)?;
            let output = gpu_encoder.forward_layers(cmd_enc, pool_ref, &normed, &mask_gpu, 0, 1)?;
            let final_out = gpu_encoder.final_norm(cmd_enc, pool_ref, &output)?;
            frame.finish();
            final_out.to_ndarray_3d::<f32>().await?
        };

        assert_gpu_cpu_close(&gpu_final, &cpu_final, 1e-4, "GPU/CPU Parity Whisper");

        Ok(())
    }
}
