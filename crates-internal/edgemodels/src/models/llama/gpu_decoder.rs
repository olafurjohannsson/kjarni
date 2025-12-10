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
use edgetransformers::models::base::DecoderInput;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::traits::{DecoderArchitecture, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2};
use std::sync::Arc;

/// The GPU-native implementation of the Llama decoder architecture.
pub struct LlamaGpuDecoder {
    embedding_weights: Option<GpuEmbeddingWeights>,
    embeddings: Option<GpuEmbeddings>,

    layers: Vec<GpuPreNormDecoderLayer>,
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,
    gpu_rope: GpuRoPE,
    context: Arc<WgpuContext>,
    config: Arc<LlamaConfig>,

    cpu_embeddings: Option<edgetransformers::embeddings::Embeddings>,
}

impl LlamaGpuDecoder {
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    /// Helper to load a linear layer lazily.
    ///
    /// Why this exists:
    /// 1. PyTorch weights are [Out, In]. ndarray/MatMul expects [In, Out].
    /// 2. We need to transpose.
    /// 3. To avoid loading 14GB of RAM, we load ONE tensor, convert to F32, transpose, upload, then DROP.
    ///
    /// In the future, we will replace the `to_ndarray_f32` with a GPU Compute Shader transpose
    /// to support full BF16 loading without CPU conversion.
    fn load_linear_lazy(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        name: &str,
    ) -> Result<GpuTensor> {
        let load_f32 = false;
        if load_f32 {
            let raw = weights.get_raw(name)?;
            let arr = raw.to_ndarray_f32()?;
            // Transpose and make contiguous
            let arr_t = arr
                .view()
                .into_dimensionality::<ndarray::Ix2>()?
                // .t()
                .as_standard_layout()
                .to_owned();
            let tensor = GpuTensor::from_ndarray(ctx, &arr_t)?;
            Ok(tensor)
        } else {
            // 1. Get view from disk (0 RAM)
            let raw = weights.get_raw(name)?;
            let tensor = GpuTensor::from_raw(ctx, &raw, name)?;

            Ok(tensor)
        }
    }

    /// Helper for 1D tensors (biases, norms)
    fn load_1d_lazy(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        name: &str,
    ) -> Result<GpuTensor> {
        let raw = weights.get_raw(name)?;
        let arr_dyn = raw.to_ndarray_f32()?;
        let arr_1d = arr_dyn.into_dimensionality::<ndarray::Ix1>()?;
        GpuTensor::from_ndarray(ctx, &arr_1d)
    }

    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<LlamaConfig>,
        gpu_rope: GpuRoPE,
    ) -> Result<Self> {
        let use_cpu_embeddings = true;

        // 1. Embeddings
        let (cpu_embeddings, embedding_weights, embeddings) = if use_cpu_embeddings {
            log::info!("Llama Optimization: Loading Embedding weights to CPU RAM only.");

            let (word_w, _, _) = config.get_embedding_weight_names();
            // We still use the old get_array2 here because CPU embeddings need to persist in RAM
            let word_embeddings = weights.get_array2(word_w)?;

            let cpu_embs =
                edgetransformers::embeddings::Embeddings::new(word_embeddings, None, None);
            (Some(cpu_embs), None, None)
        } else {
            log::info!("Loading Llama Embedding weights to VRAM.");
            let ew = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
            let em = GpuEmbeddings::new(context)?;
            (None, Some(ew), Some(em))
        };

        // 2. Final Layer Norm
        let (norm_w_name, _) = config.get_final_layer_norm_names();
        let final_layer_norm =
            GpuNormalization::RMSNorm(GpuRMSNorm::new(context, config.layer_norm_eps()));

        let final_ln_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            Self::load_1d_lazy(context, weights, norm_w_name)?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let dyn_config = config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>;
            let decoder_layer =
                Self::build_layer(context.clone(), weights, dyn_config, i, Some(&gpu_rope))?;
            layers.push(decoder_layer);
        }

        Ok(Self {
            embedding_weights,
            embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            gpu_rope,
            context: context.clone(),
            config,
            cpu_embeddings,
        })
    }

    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        i: usize,
        gpu_rope: Option<&GpuRoPE>,
    ) -> Result<GpuPreNormDecoderLayer> {
        let hidden_size = config.hidden_size();
        let kv_dim = config.kv_dim();
        let ffn_names = config.get_feed_forward_names(i);
        let layer_attn_names = config.get_layer_attention_names(i);

        // --- LAZY LOADING START ---
        // Instead of reading all 4 massive matrices into RAM, we do them one by one.

        let q_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.q_weight)?;
        let k_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.k_weight)?;
        let v_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.v_weight)?;
        let o_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.output_weight)?;

        // LLaMA has no biases. Use zeros.
        // GpuTensor::zeros is very cheap (allocates VRAM, but no CPU RAM).
        let q_bias = GpuTensor::zeros(
            &context,
            vec![hidden_size],
            edgetransformers::weights::DType::F32,
            "q_bias",
        )?;
        let k_bias = GpuTensor::zeros(
            &context,
            vec![kv_dim],
            edgetransformers::weights::DType::F32,
            "k_bias",
        )?;
        let v_bias = GpuTensor::zeros(
            &context,
            vec![kv_dim],
            edgetransformers::weights::DType::F32,
            "v_bias",
        )?;
        let o_bias = GpuTensor::zeros(
            &context,
            vec![hidden_size],
            edgetransformers::weights::DType::F32,
            "o_bias",
        )?;

        let self_attn_weights = GpuAttentionWeights::new(
            q_tensor, q_bias, k_tensor, k_bias, v_tensor, v_bias, o_tensor, o_bias,
        )?;

        // Llama RMSNorm
        let self_attn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            Self::load_1d_lazy(&context, weights, &layer_attn_names.norm_weight)?,
        )?);
        let self_attn_norm =
            GpuNormalization::RMSNorm(GpuRMSNorm::new(&context, config.layer_norm_eps()));

        // Llama SwiGLU
        let gate_tensor =
            Self::load_linear_lazy(&context, weights, ffn_names.gate_weight.as_ref().unwrap())?;
        let up_tensor = Self::load_linear_lazy(&context, weights, &ffn_names.intermediate_weight)?;
        let down_tensor = Self::load_linear_lazy(&context, weights, &ffn_names.output_weight)?;

        let ff_weights = GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(
            gate_tensor,
            up_tensor,
            down_tensor,
        )?);
        let feedforward = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(&context)?);

        // FFN RMSNorm
        let ffn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
            Self::load_1d_lazy(&context, weights, &ffn_names.norm_weight)?,
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
            gpu_rope,
        )?)
    }
}

#[async_trait(?Send)]
impl GpuDecoder for LlamaGpuDecoder {
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input_ids: DecoderInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // 1. Handle Embeddings (CPU vs GPU path)
        let mut hidden_states = match input_ids {
            DecoderInput::Cpu(ids) => {
                let cpu_embeds = self.cpu_embeddings.as_ref().ok_or_else(|| {
                    anyhow!("CPU input provided but CPU embeddings not initialized")
                })?;

                // Construct ndarray input (Batch=1)
                let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;

                // Perform lookup on CPU.
                let initial_embeddings_cpu = cpu_embeds.forward(
                    &input_array,
                    None,
                    position_offset,
                    self.config.scale_embeddings(),
                );

                // Upload [1, Seq, Hidden] to GPU
                GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?
            }
            DecoderInput::Gpu(ids_tensor) => {
                let gpu_embeds = self.embeddings.as_ref().ok_or_else(|| {
                    anyhow!("GPU input provided but GPU embeddings not initialized")
                })?;
                let gpu_weights = self
                    .embedding_weights
                    .as_ref()
                    .ok_or_else(|| anyhow!("GPU weights not loaded"))?;

                gpu_embeds.encode(
                    encoder,
                    gpu_weights,
                    ids_tensor,
                    None,
                    position_offset,
                    self.config.as_ref(),
                    pool,
                )?
            }
        };

        let mut cache_mut_ref = cache;

        for (i, layer) in self.layers.iter().enumerate() {
            let (output, _) = layer
                .forward_llama(
                    encoder,
                    &hidden_states,
                    attention_mask,
                    i,
                    position_offset,
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
