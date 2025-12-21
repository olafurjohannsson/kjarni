// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::Result;
use async_trait::async_trait;
use ndarray::Array2;

// --- Workspace Crates ---
use kjarni_transformers::{
    TransformerConfig, WgpuContext,
    cache::GpuKVCache,
    decoder::prelude::*,
    embeddings::Embeddings,
    gpu_ops::{
        GpuTensor, GpuTensorPool,
        blocks::{
            GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights,
            GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
            attention::GpuAttentionWeights,
            embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
            rms_norm::{GpuRMSNorm, GpuRMSNormWeights},
            rope::GpuRoPE,
        },
    },
    tensor::DType,
    traits::{DecoderArchitecture, LanguageModelConfig},
    weights::ModelWeights,
};

// --- Crate-Specific ---
use crate::models::llama::config::LlamaConfig;

/// The GPU-native implementation of the Llama decoder architecture.
pub struct LlamaGpuDecoder {
    gpu_embeddings: Option<GpuEmbeddings>,
    gpu_embedding_weights: Option<GpuEmbeddingWeights>,
    cpu_embeddings: Option<Embeddings>,
    layers: Vec<GpuPreNormDecoderLayer>,
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,
    gpu_rope: GpuRoPE,
    context: Arc<WgpuContext>,
    config: Arc<LlamaConfig>,
    load_config: DecoderLoadConfig,
}

impl LlamaGpuDecoder {
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    fn load_linear_lazy(
        ctx: &Arc<WgpuContext>,
        weights: &ModelWeights,
        name: &str,
    ) -> Result<GpuTensor> {
        let raw = weights.get_raw(name)?;
        let tensor = GpuTensor::from_raw(ctx, &raw, name)?;

        Ok(tensor)
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
        load_config: DecoderLoadConfig,
    ) -> Result<Self> {
        let (cpu_embeddings, gpu_embedding_weights, gpu_embeddings) =
            if load_config.offload_embeddings {
                log::info!("Llama Optimization: Loading Embedding weights to CPU RAM only.");

                let (word_w, _, _) = config.get_embedding_weight_names();
                let word_embeddings = weights.get_array2(word_w)?;

                let cpu_embs = Embeddings::new(
                    kjarni_transformers::embeddings::EmbeddingData::F32(word_embeddings),
                    None,
                    None,
                );
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
            gpu_embedding_weights,
            gpu_embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            gpu_rope,
            context: context.clone(),
            config,
            cpu_embeddings,
            load_config,
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

        let q_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.q_weight)?;
        let k_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.k_weight)?;
        let v_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.v_weight)?;
        let o_tensor = Self::load_linear_lazy(&context, weights, &layer_attn_names.output_weight)?;

        let q_bias = GpuTensor::zeros(&context, vec![hidden_size], DType::F32, "q_bias")?;
        let k_bias = GpuTensor::zeros(&context, vec![kv_dim], DType::F32, "k_bias")?;
        let v_bias = GpuTensor::zeros(&context, vec![kv_dim], DType::F32, "v_bias")?;
        let o_bias = GpuTensor::zeros(&context, vec![hidden_size], DType::F32, "o_bias")?;

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
    async fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        match input {
            DecoderInput::TokensCpu(ids) => {
                // This path is for prefill. It should use CPU embeddings if available.
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;
                    let embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        self.config.scale_embeddings(),
                    );
                    GpuTensor::from_ndarray(&self.context, &embeddings_cpu)
                } else {
                    // This case handles prefill when embeddings are on GPU.
                    let gpu_embeds = self.gpu_embeddings.as_ref().unwrap();
                    let gpu_weights = self.gpu_embedding_weights.as_ref().unwrap();
                    let ids_tensor = GpuTensor::from_ndarray(
                        &self.context,
                        &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
                    )?;
                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        &ids_tensor,
                        None,
                        position_offset,
                        self.config.as_ref(),
                        pool,
                    )
                }
            }
            DecoderInput::TokensGpu(ids_tensor) => {
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    let token_id_vec: Vec<u32> =
                        bytemuck::cast_slice(&ids_tensor.read_raw_data().await?).to_vec();
                    let token_id = token_id_vec[0];

                    let input_array = Array2::from_elem((1, 1), token_id);
                    let embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        self.config.scale_embeddings(),
                    );

                    GpuTensor::from_ndarray(&self.context, &embeddings_cpu)
                } else {
                    let gpu_embeds = self.gpu_embeddings.as_ref().unwrap(); // Safe to unwrap here
                    let gpu_weights = self.gpu_embedding_weights.as_ref().unwrap();
                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        ids_tensor,
                        None,
                        position_offset,
                        self.config.as_ref(),
                        pool,
                    )
                }
            }
            DecoderInput::HiddenGpu(t) => Ok(t.clone()),
            DecoderInput::HiddenCpu(t) => GpuTensor::from_ndarray(&self.context, t),
        }
    }

    async fn embed_and_normalize(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        // Llama is Pre-Norm (Norm is inside the layer), so we just embed.
        self.embed(encoder, pool, input, position_offset).await
    }

    fn forward_layers(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        position_offset: usize,
        mut cache: Option<&mut GpuKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<GpuTensor> {
        let mut current_state = hidden_states.clone();

        for i in start_layer..end_layer {
            if i >= self.layers.len() {
                break;
            }
            let layer = &self.layers[i];

            // Re-borrow the cache mutably for each layer
            let layer_cache = cache.as_deref_mut();

            let (output, _) = layer.forward_llama(
                encoder,
                &current_state,
                attention_mask,
                i,
                position_offset,
                layer_cache,
                pool,
                Some(&self.gpu_rope),
            )?;
            current_state = output;
        }

        Ok(current_state)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    // Override default forward because Llama needs Final Layer Norm
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // 1. Embed
        let hidden = self
            .embed_and_normalize(encoder, pool, input, position_offset)
            .await?;

        // 2. Layers
        let mut hidden = self.forward_layers(
            encoder,
            pool,
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;

        // 3. Final Layer Norm (Llama Specific)
        let final_ln_output = pool.get(hidden.shape().to_vec());
        self.final_layer_norm
            .encode(encoder, &self.final_ln_weights, &hidden, &final_ln_output);
        hidden = final_ln_output;

        Ok(hidden)
    }
}
