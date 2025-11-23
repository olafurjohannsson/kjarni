use crate::models::llama::config::LlamaConfig;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::TransformerConfig;
use edgetransformers::WgpuContext;
use edgetransformers::cache::GpuKVCache;
use edgetransformers::gpu_ops::blocks::decoder::GpuPreNormDecoderLayer;
use edgetransformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use edgetransformers::gpu_ops::blocks::rms_norm::{GpuRMSNorm, GpuRMSNormWeights};
use edgetransformers::gpu_ops::blocks::rope::GpuRoPE;
use edgetransformers::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights, GpuSwiGLUFFN,
    GpuSwiGLUFFNWeights, attention::GpuAttentionWeights,
};
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::traits::DecoderArchitecture;
use edgetransformers::weights::ModelWeights;
use ndarray::Array1;
use ndarray::Array2;
use std::sync::Arc;
use tokio::sync::Mutex;

/// The GPU-native implementation of the Llama decoder architecture.
pub struct LlamaGpuDecoder {
    embedding_weights: GpuEmbeddingWeights,
    embeddings: GpuEmbeddings,
    layers: Vec<GpuPreNormDecoderLayer>,
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,
    gpu_rope: GpuRoPE,
    pool: Mutex<GpuTensorPool>,
    context: Arc<WgpuContext>,
    config: Arc<LlamaConfig>,
}

impl LlamaGpuDecoder {
    /// Creates a new LlamaGpuDecoder directly from Llama-specific components.
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<LlamaConfig>,
        gpu_rope: GpuRoPE,
    ) -> Result<Self> {
        // --- This code is copied and simplified from GpuTransformerDecoder ---

        // 1. Embeddings (Simplified, as Llama has no position embeddings)
        let embedding_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
        let embeddings = GpuEmbeddings::new(context)?;

        // 2. Final Layer Norm (Simplified for Llama's RMSNorm)
        let (norm_w_name, _) = config.get_final_layer_norm_names();
        let final_layer_norm =
            GpuNormalization::RMSNorm(GpuRMSNorm::new(context, config.layer_norm_eps()));
        let final_ln_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_ndarray(context, &weights.get_array1(norm_w_name)?)?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            // Cast the concrete LlamaConfig to the generic trait for the layer builder
            let dyn_config = config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>;
            let decoder_layer = Self::build_layer(
                context.clone(),
                weights,
                dyn_config, // Pass the trait object
                i,
                Some(&gpu_rope),
            )?;
            layers.push(decoder_layer);
        }

        Ok(Self {
            embedding_weights,
            embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            gpu_rope: gpu_rope,
            pool: Mutex::new(GpuTensorPool::new(context.clone())),
            context: context.clone(),
            config: config,
        })
    }

    /// This function is IDENTICAL to your GpuTransformerDecoder::build_layer.
    /// It is generic enough to build Llama layers correctly when given a LlamaConfig.
    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        i: usize,
        gpu_rope: Option<&GpuRoPE>, // Changed to Option<&Arc<...>>
    ) -> Result<GpuPreNormDecoderLayer> {
        let hidden_size = config.hidden_size();
        let kv_dim = config.kv_dim();
        let attn_names = config.get_attention_names(i);
        let ffn_names = config.get_feed_forward_names(i);
        let layer_attn_names = config.get_layer_attention_names(i);

        let q_weight: Array2<f32> = weights.get_linear_weight(&layer_attn_names.q_weight)?;
        let k_weight: Array2<f32> = weights.get_linear_weight(&layer_attn_names.k_weight)?;
        let v_weight: Array2<f32> = weights.get_linear_weight(&layer_attn_names.v_weight)?;
        let o_weight: Array2<f32> = weights.get_linear_weight(&layer_attn_names.output_weight)?;

        // LLaMA has no biases, so we create zero tensors
        let q_bias: Array1<f32> = Array1::zeros(hidden_size);
        let k_bias: Array1<f32> = Array1::zeros(kv_dim);
        let v_bias: Array1<f32> = Array1::zeros(kv_dim);
        let o_bias: Array1<f32> = Array1::zeros(hidden_size);
        
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

        // Llama Path (RMSNorm)
        let self_attn_norm_weights =
            GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(GpuTensor::from_ndarray(
                &context,
                &weights.get_array1(&layer_attn_names.norm_weight)?,
            )?)?);
        let self_attn_norm =
            GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps()));

        // Llama Path (SwiGLU)
        let gate_w = weights.get_linear_weight(ffn_names.gate_weight.as_ref().unwrap())?;
        let up_w = weights.get_linear_weight(&ffn_names.intermediate_weight)?;
        let down_w = weights.get_linear_weight(&ffn_names.output_weight)?;

        let ff_weights = GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(
            GpuTensor::from_ndarray(&context, &gate_w)?,
            GpuTensor::from_ndarray(&context, &up_w)?,
            GpuTensor::from_ndarray(&context, &down_w)?,
        )?);
        let feedforward = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(&context)?);

        // Llama Path (RMSNorm)
        let ffn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.norm_weight)?)?,
        )?);
        let ffn_norm =
            GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps()));

        Ok(GpuPreNormDecoderLayer::new(
            &context,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            config,
            gpu_rope.as_deref(),
        )?)
    }
}

#[async_trait(?Send)]
impl GpuDecoder for LlamaGpuDecoder {
    /// Implements the forward pass for the new generation backend trait.
    /// This is adapted from your old `GpuTransformerDecoder::forward` but now works with GPU-native tensors.
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder, // <-- Pass in the encoder
        pool: &mut edgetransformers::gpu_ops::GpuTensorPool,    // <-- Pass in the pool
        input_ids: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>, // Ignored for Llama
    ) -> Result<GpuTensor> {

        let mut hidden_states = self.embeddings.encode(
            encoder,
            &self.embedding_weights,
            input_ids,
            None, // No token type embeddings for Llama
            position_offset,
            self.config.as_ref(),
            pool,
        )?;

        let mut cache_mut_ref = cache;

        for (i, layer) in self.layers.iter().enumerate() {
            let (output, _) = layer
                .forward_llama(
                    encoder,
                    &hidden_states,
                    attention_mask,
                    i,
                    position_offset,
                    // Pass the mutable reference, which gets re-borrowed each iteration.
                    cache_mut_ref.as_deref_mut(),
                    pool,
                    Some(&self.gpu_rope),
                )
                .await?;
            hidden_states = output;
        }

        let final_ln_output = pool.get(hidden_states.shape().to_vec());
        self.final_layer_norm.encode(
            encoder,
            &self.final_ln_weights,
            &hidden_states,
            &final_ln_output,
        );
        hidden_states = final_ln_output;
        
        Ok(hidden_states)
    }
}
