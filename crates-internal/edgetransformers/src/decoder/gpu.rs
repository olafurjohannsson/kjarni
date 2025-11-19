use crate::Embeddings;
use crate::activations::Activation;
use crate::adaptive_embeddings::EmbeddingSelector;
use crate::cache::GpuKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::decoder::GpuPreNormDecoderLayer;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use crate::gpu_ops::blocks::rms_norm::{GpuRMSNorm, GpuRMSNormWeights};
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
    GpuNormalization, GpuNormalizationWeights, GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
};
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use crate::rope::RoPE;
use crate::traits::{Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerModel};
use crate::weights::ModelWeights;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3, s};
use std::sync::Arc;
use tokio::sync::Mutex;
/// The GPU backend for a generic Transformer Decoder.
pub struct GpuTransformerDecoder {
    // CPU-side embeddings
    word_embeddings: Array2<f32>,
    position_embeddings_cpu: Option<Array2<f32>>,

    embedding_weights: GpuEmbeddingWeights, // Holds GPU tensors
    embeddings: GpuEmbeddings,              // Holds the kernels

    // GPU-side weight buffers
    layers: Vec<GpuPreNormDecoderLayer>,

    // Final LayerNorm components
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,

    config: Arc<dyn DecoderArchitecture + Send + Sync>,

    context: Arc<WgpuContext>,

    cpu_embeddings: Embeddings,

    embedding_selector: EmbeddingSelector,

    gpu_rope: Option<GpuRoPE>,

    pool: Mutex<GpuTensorPool>,
}

impl GpuTransformerDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
        rope: Option<&RoPE>,
    ) -> Result<Self> {
        let gpu_rope = rope
            .map(|r| GpuRoPE::from_cpu_rope(&context, r))
            .transpose()?;

        let (word_w, pos_w, _) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings_cpu = if !pos_w.is_empty() {
            Some(weights.get_array2(pos_w)?)
        } else {
            None
        };
        let cpu_embeddings = Embeddings::new(
            word_embeddings.clone(),
            position_embeddings_cpu.clone(),
            None,
        );
        let vocab_size = word_embeddings.shape()[0];
        let hidden_size = word_embeddings.shape()[1];
        let embedding_selector = EmbeddingSelector::new(&context, vocab_size, hidden_size);
        let embedding_weights = GpuEmbeddingWeights::new(&context, weights, config.as_ref())?;
        let embeddings = GpuEmbeddings::new(&context)?;

        let (norm_w_name, norm_b_name) = config.get_final_layer_norm_names();
        let (final_layer_norm, final_ln_weights) = {
            let gamma = weights.get_array1(norm_w_name)?;
            if norm_b_name.is_empty() {
                (
                    GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps())),
                    GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
                        GpuTensor::from_ndarray(&context, &gamma)?,
                    )?),
                )
            } else {
                let beta = weights.get_array1(norm_b_name)?;
                (
                    GpuNormalization::LayerNorm(GpuLayerNorm::new(
                        &context,
                        config.layer_norm_eps(),
                    )),
                    GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                        GpuTensor::from_ndarray(&context, &gamma)?,
                        GpuTensor::from_ndarray(&context, &beta)?,
                    )?),
                )
            }
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let decoder_layer = Self::build_layer(
                context.clone(),
                weights,
                config.clone(),
                i,
                gpu_rope.as_ref(),
            )?;
            layers.push(decoder_layer);
        }

        Ok(Self {
            word_embeddings,
            position_embeddings_cpu,
            embedding_weights,
            embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            config: config.clone(),
            context: context.clone(),
            cpu_embeddings,
            embedding_selector,
            gpu_rope,
            pool: Mutex::new(GpuTensorPool::new(context)),
        })
    }

    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        i: usize,
        gpu_rope: Option<&GpuRoPE>,
    ) -> Result<GpuPreNormDecoderLayer> {
        let attn_names = config.get_attention_names(i);
        let ffn_names = config.get_feed_forward_names(i);
        let hidden_size = config.hidden_size();
        let kv_dim = config.kv_dim();

        let (q_weight, k_weight, v_weight, o_weight, q_bias, k_bias, v_bias, o_bias) =
            if !attn_names.qkv_weight.is_empty() {
                // GPT-2 style: Combined QKV
                let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
                let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;
                let o_weight = weights.get_array2(&attn_names.output_weight)?;
                let o_bias = weights.get_array1(&attn_names.output_bias)?;

                (
                    qkv_weight.slice(s![.., 0..hidden_size]).to_owned(),
                    qkv_weight
                        .slice(s![.., hidden_size..2 * hidden_size])
                        .to_owned(),
                    qkv_weight
                        .slice(s![.., 2 * hidden_size..3 * hidden_size])
                        .to_owned(),
                    o_weight,
                    qkv_bias.slice(s![0..hidden_size]).to_owned(),
                    qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned(),
                    qkv_bias
                        .slice(s![2 * hidden_size..3 * hidden_size])
                        .to_owned(),
                    o_bias,
                )
            } else {
                // LLaMA style: Separate Q, K, V
                let layer_attn_names = config.get_layer_attention_names(i);
                (
                    weights.get_linear_weight(&layer_attn_names.q_weight)?,
                    weights.get_linear_weight(&layer_attn_names.k_weight)?,
                    weights.get_linear_weight(&layer_attn_names.v_weight)?,
                    weights.get_linear_weight(&layer_attn_names.output_weight)?,
                    // LLaMA has no biases, so we create zero tensors
                    Array1::zeros(hidden_size),
                    Array1::zeros(kv_dim),
                    Array1::zeros(kv_dim),
                    Array1::zeros(hidden_size),
                )
            };

        let self_attn_weights = GpuAttentionWeights::new(
            GpuTensor::from_ndarray(&context, &q_weight)?,
            GpuTensor::from_ndarray(&context, &q_bias)?,
            GpuTensor::from_ndarray(&context, &k_weight)?,
            GpuTensor::from_ndarray(&context, &k_bias)?,
            GpuTensor::from_ndarray(&context, &v_weight)?,
            GpuTensor::from_ndarray(&context, &v_bias)?,
            GpuTensor::from_ndarray(&context, &o_weight)?,
            GpuTensor::from_ndarray(&context, &o_bias)?,
        )?;

        let (self_attn_norm, self_attn_norm_weights) = if !attn_names.qkv_weight.is_empty() {
            // GPT-2 Path: Use the top-level `attn_names`
            let gamma = weights.get_array1(&attn_names.norm_weight)?;
            let beta = weights.get_array1(&attn_names.norm_bias)?;
            (
                GpuNormalization::LayerNorm(GpuLayerNorm::new(&context, config.layer_norm_eps())),
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_ndarray(&context, &gamma)?,
                    GpuTensor::from_ndarray(&context, &beta)?,
                )?),
            )
        } else {
            let layer_attn_names = config.get_layer_attention_names(i);
            let gamma = weights.get_array1(&layer_attn_names.norm_weight)?;
            // LLaMA uses RMSNorm, so there is no bias.
            (
                GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps())),
                GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(GpuTensor::from_ndarray(
                    &context, &gamma,
                )?)?),
            )
        };

        let (feedforward, ff_weights) = if let Some(gate_weight_name) = &ffn_names.gate_weight {
            // SwiGLU (LLaMA)
            let gate_w = weights.get_linear_weight(gate_weight_name)?;
            let up_w = weights.get_linear_weight(&ffn_names.intermediate_weight)?;
            let down_w = weights.get_linear_weight(&ffn_names.output_weight)?;

            let weights_gpu = GpuSwiGLUFFNWeights::new(
                GpuTensor::from_ndarray(&context, &gate_w)?,
                GpuTensor::from_ndarray(&context, &up_w)?,
                GpuTensor::from_ndarray(&context, &down_w)?,
            )?;

            (
                GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(&context)?),
                GpuFeedForwardWeights::SwiGLU(weights_gpu),
            )
        } else {
            // Standard (GPT-2)
            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let fc1_w = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w
            };

            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let fc2_w = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w
            };

            let weights_gpu = GpuFeedForwardWeightsStd::from_ndarrays(
                &context,
                &fc1_w,
                &weights.get_array1(&ffn_names.intermediate_bias)?,
                &fc2_w,
                &weights.get_array1(&ffn_names.output_bias)?,
            )?;
            println!("Activation!: {:?}", config.activation_function());
            (
                GpuFeedForward::Standard(GpuFeedForwardStd::new(
                    &context,
                    Activation::Gelu, // TODO: temp
                )?),
                GpuFeedForwardWeights::Standard(weights_gpu),
            )
        };

        let (ffn_norm, ffn_norm_weights) = {
            let gamma = weights.get_array1(&ffn_names.norm_weight)?;

            if ffn_names.norm_bias.is_empty() {
                // RMSNorm (LLaMA)
                (
                    GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps())),
                    GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
                        GpuTensor::from_ndarray(&context, &gamma)?,
                    )?),
                )
            } else {
                // LayerNorm (GPT-2)
                let beta = weights.get_array1(&ffn_names.norm_bias)?;
                (
                    GpuNormalization::LayerNorm(GpuLayerNorm::new(
                        &context,
                        config.layer_norm_eps(),
                    )),
                    GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                        GpuTensor::from_ndarray(&context, &gamma)?,
                        GpuTensor::from_ndarray(&context, &beta)?,
                    )?),
                )
            }
        };

        // === 5. BUILD LAYER ===
        Ok(GpuPreNormDecoderLayer::new(
            &context,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            config.clone(),
            gpu_rope,
        )?)
    }
}

impl TransformerModel for GpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
#[async_trait(?Send)]
impl Decoder for GpuTransformerDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let pool_guard = self.pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (encoder, pool) = frame.resources();

        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = input.shape()[1];

        // let initial_embeddings_cpu = self.cpu_embeddings.forward(
        //     input,
        //     None,
        //     position_offset,
        //     self.config.scale_embeddings(),
        // );
        // let mut hidden_states = GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?;

        let input_ids_gpu = GpuTensor::from_ndarray(&self.context, input)?;

        let mut hidden_states = self.embeddings.encode(
            encoder,
            &self.embedding_weights,
            &input_ids_gpu,
            None,
            position_offset,
            self.config.as_ref(),
            pool,
        )?;

        let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

        let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuKVCache>());

        for (i, layer) in self.layers.iter().enumerate() {
            if self.gpu_rope.is_none() {
                let (output, _) = layer.forward_gpt2(
                    encoder,
                    &hidden_states,
                    &attention_mask_gpu,
                    i,
                    position_offset,
                    gpu_cache.as_deref_mut(),
                    pool,
                    self.gpu_rope.as_ref(),
                )?;
                hidden_states = output;
            } else {
                let (output, _) = layer
                    .forward_llama(
                        encoder,
                        &hidden_states,
                        &attention_mask_gpu,
                        i,
                        position_offset,
                        gpu_cache.as_deref_mut(),
                        pool,
                        self.gpu_rope.as_ref(),
                    )
                    .await?;
                hidden_states = output;
            }
        }
        let final_ln_output = pool.get(hidden_states.shape().to_vec());
        self.final_layer_norm.encode(
            encoder,
            &self.final_ln_weights,
            &hidden_states,
            &final_ln_output,
        );
        hidden_states = final_ln_output;

        frame.finish();
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        if let Some(cache) = gpu_cache {
            cache.increment_len(seq_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: last_hidden_state_cpu,
            past_key_values: None,
        })
    }

    async fn get_hidden_states(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
    ) -> Result<Array3<f32>> {
        let output = self.forward(input, attention_mask, None).await?;
        Ok(output.last_hidden_state)
    }
}
