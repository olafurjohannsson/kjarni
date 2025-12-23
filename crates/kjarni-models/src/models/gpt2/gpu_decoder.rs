// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, s};

// --- Workspace Crates ---
use kjarni_transformers::{
    WgpuContext,
    cache::GpuKVCache,
    decoder::prelude::*,
    embeddings::Embeddings,
    gpu_ops::{
        GpuTensor, GpuTensorPool,
        blocks::{
            GpuFeedForward, GpuFeedForwardStd as GpuStandardFFN, GpuFeedForwardWeights,
            GpuFeedForwardWeightsStd as GpuStandardFFNWeights, GpuNormalization,
            GpuNormalizationWeights,
            attention::GpuAttentionWeights,
            embeddings::{GpuEmbeddingWeights, GpuEmbeddings},
            layer_norm::{GpuLayerNorm, GpuLayerNormWeights},
        },
    },
    models::base::ModelLoadConfig,
    traits::{ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};

// --- Crate-Specific ---
use crate::models::gpt2::config::Gpt2Config;

/// The GPU-native implementation of the GPT-2 decoder architecture.
pub struct Gpt2GpuDecoder {
    // Option: If using CPU embeddings, these are None to save VRAM
    embedding_weights: Option<GpuEmbeddingWeights>,
    embeddings: Option<GpuEmbeddings>,

    layers: Vec<GpuPreNormDecoderLayer>,
    final_layer_norm: GpuNormalization,
    final_ln_weights: GpuNormalizationWeights,

    context: Arc<WgpuContext>,
    config: Arc<Gpt2Config>,

    // Option: If using GPU embeddings, this is None
    cpu_embeddings: Option<Embeddings>,

    load_config: ModelLoadConfig,

    pub meta: ModelMetadata,
    pub layout: ModelLayout,
}

impl Gpt2GpuDecoder {
    pub fn context(&self) -> &Arc<WgpuContext> {
        &self.context
    }

    /// Creates a new Gpt2GpuDecoder directly from GPT-2-specific components.
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<Gpt2Config>,
        load_config: ModelLoadConfig,
    ) -> Result<Self> {
        log::info!("Building GPT-2 GPU decoder...");

        let meta = config.metadata();
        let layout = config.layout();
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("GPT-2 layout must have a decoder section");

        let (cpu_embeddings, embedding_weights, embeddings) = if load_config.offload_embeddings {
            log::info!("Optimization: Loading Embedding weights to CPU RAM only.");

            let word_embeddings = weights.get_array2(&layout.token_embedding)?;
            let position_embeddings_cpu = if let Some(pos_w) = &decoder_layout.position_embedding {
                Some(weights.get_array2(pos_w)?)
            } else {
                None
            };

            let cpu_embs = Embeddings::new(
                kjarni_transformers::embeddings::EmbeddingData::F32(word_embeddings),
                position_embeddings_cpu,
                None,
            );

            (Some(cpu_embs), None, None)
        } else {
            log::info!("Loading Embedding weights to VRAM.");
            let ew = GpuEmbeddingWeights::from_layout(
                context,
                weights,
                &layout.token_embedding,
                decoder_layout.position_embedding.as_deref(),
                None,
                load_config.target_dtype,
            )?;
            let em = GpuEmbeddings::new(context)?;
            (None, Some(ew), Some(em))
        };

        // 2. Final Layer Norm
        let final_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, meta.norm_eps));

        let final_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(
                context,
                &weights.get_array1(decoder_layout.final_norm_weight.as_ref().unwrap())?,
            )?,
            GpuTensor::from_ndarray(
                context,
                &weights.get_array1(decoder_layout.final_norm_bias.as_ref().unwrap())?,
            )?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(meta.num_layers);
        for i in 0..meta.num_layers {
            log::info!("Building GPT-2 layer {}/{}", i + 1, meta.num_layers);
            let decoder_layer =
                Self::build_layer(context.clone(), weights, &meta, &layout, i, load_config)?;
            layers.push(decoder_layer);
        }

        log::info!("✓ GPT-2 GPU decoder built successfully");

        Ok(Self {
            embedding_weights,
            embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            context: context.clone(),
            config,
            cpu_embeddings,
            load_config,
            meta,
            layout,
        })
    }

    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &ModelLayout,
        layer_idx: usize,
        load_config: ModelLoadConfig,
    ) -> Result<GpuPreNormDecoderLayer> {
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("GPT-2 layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let self_attn_layout = &layer_layout.self_attn;
        let ffn_layout = &layer_layout.ffn;

        let idx = layer_idx.to_string();
        let name = |t: &String| t.replace("{}", &idx);
        let opt_name = |t: &Option<String>| t.as_ref().map(|s| s.replace("{}", &idx)).unwrap();

        let hidden_size = meta.hidden_size;

        // Load COMBINED QKV weight and split it (Mapping attn_q to c_attn)
        let qkv_weight = weights.get_array2(&name(&self_attn_layout.q_weight))?;
        let qkv_bias = weights.get_array1(&opt_name(&self_attn_layout.q_bias))?;

        log::info!(
            "Layer {}: QKV weight shape (file): {:?}, expected [In={}, Out={}]",
            layer_idx,
            qkv_weight.shape(),
            hidden_size,
            3 * hidden_size
        );

        // Logic Preserved: Split AND transpose
        let q_weight = qkv_weight
            .slice(s![.., 0..hidden_size])
            .t()
            .as_standard_layout()
            .to_owned();
        let k_weight = qkv_weight
            .slice(s![.., hidden_size..2 * hidden_size])
            .t()
            .as_standard_layout()
            .to_owned();
        let v_weight = qkv_weight
            .slice(s![.., 2 * hidden_size..3 * hidden_size])
            .t()
            .as_standard_layout()
            .to_owned();

        let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
        let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
        let v_bias = qkv_bias
            .slice(s![2 * hidden_size..3 * hidden_size])
            .to_owned();

        // Output projection logic preserved
        let o_weight_raw = weights.get_array2(&name(&self_attn_layout.o_weight))?;
        let o_weight = o_weight_raw.t().as_standard_layout().to_owned();
        let o_bias = weights.get_array1(&opt_name(&self_attn_layout.o_bias))?;

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

        // --- Attention LayerNorm logic preserved ---
        let gamma = weights.get_array1(&name(&self_attn_layout.norm_weight))?;
        let beta = weights.get_array1(&opt_name(&self_attn_layout.norm_bias))?;
        let self_attn_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(&context, meta.norm_eps));
        let self_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &gamma)?,
            GpuTensor::from_ndarray(&context, &beta)?,
        )?);

        // --- FFN weights with transpose logic preserved ---
        let intermediate_w_raw = weights.get_array2(&name(&ffn_layout.up_weight))?;
        let output_w_raw = weights.get_array2(&name(&ffn_layout.down_weight))?;

        let intermediate_w = intermediate_w_raw.t().as_standard_layout().to_owned();
        let output_w = output_w_raw.t().as_standard_layout().to_owned();

        let intermediate_b = weights.get_array1(&opt_name(&ffn_layout.up_bias))?;
        let output_b = weights.get_array1(&opt_name(&ffn_layout.down_bias))?;

        let ff_weights = GpuFeedForwardWeights::Standard(GpuStandardFFNWeights::new(
            GpuTensor::from_ndarray(&context, &intermediate_w)?,
            GpuTensor::from_ndarray(&context, &intermediate_b)?,
            GpuTensor::from_ndarray(&context, &output_w)?,
            GpuTensor::from_ndarray(&context, &output_b)?,
        )?);

        let feedforward = GpuFeedForward::Standard(GpuStandardFFN::new(&context, meta.activation)?);

        // --- FFN LayerNorm logic preserved ---
        let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(
                &context,
                &weights.get_array1(&name(&ffn_layout.norm_weight))?,
            )?,
            GpuTensor::from_ndarray(
                &context,
                &weights.get_array1(&opt_name(&ffn_layout.norm_bias))?,
            )?,
        )?);
        let ffn_norm = GpuNormalization::LayerNorm(GpuLayerNorm::new(&context, meta.norm_eps));

        Ok(GpuPreNormDecoderLayer::new(
            &context,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            meta.hidden_size,
            meta.num_attention_heads,
            meta.num_kv_heads,
        )?)
    }
}

#[async_trait(?Send)]
impl GpuDecoder for Gpt2GpuDecoder {
    async fn embed(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<GpuTensor> {
        match input {
            // Case 1: Tokens on CPU
            DecoderInput::TokensCpu(ids) => {
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    // 1A. CPU Embeddings Loaded: Compute on CPU, Upload Result
                    // (Saves VRAM, slower due to upload of float hidden states)
                    let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;

                    let initial_embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        self.meta.scale_embeddings,
                    );

                    GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)
                } else {
                    // 1B. GPU Embeddings Loaded: Upload Tokens, Compute on GPU
                    // (Fastest standard generation path)
                    let tokens_tensor = GpuTensor::from_ndarray(
                        &self.context,
                        &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
                    )?;

                    let gpu_embeds = self
                        .embeddings
                        .as_ref()
                        .ok_or_else(|| anyhow!("Embeddings not loaded on CPU or GPU"))?;
                    let gpu_weights = self.embedding_weights.as_ref().unwrap();
                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        &tokens_tensor,
                        None,
                        position_offset,
                        self.meta.hidden_size,
                        self.meta.extra_pos_embeddings,
                        self.meta.scale_embeddings,
                        pool,
                    )
                }
            }
            // Case 2: Tokens already on GPU (Optimized Beam Search)
            DecoderInput::TokensGpu(ids_tensor) => {
                let gpu_embeds = self
                    .embeddings
                    .as_ref()
                    .ok_or_else(|| anyhow!("GPU input provided but embeddings are on CPU"))?;
                let gpu_weights = self.embedding_weights.as_ref().unwrap();

                gpu_embeds.encode(
                    encoder,
                    gpu_weights,
                    &ids_tensor,
                    None,
                    position_offset,
                    self.meta.hidden_size,
                    self.meta.extra_pos_embeddings,
                    self.meta.scale_embeddings,
                    pool,
                )
            }
            // Case 3: Pre-computed Hidden States (Multimodal / Prefix Tuning)
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
        // GPT-2 standard architecture does not have a LayerNorm *before* the first block.
        // It only has norms inside blocks (Pre-Norm) and a final norm.
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

            // Re-borrow cache mutably
            let layer_cache = cache.as_deref_mut();

            // GPT-2 specific layer call
            let (output, _) = layer.forward_gpt2(
                encoder,
                &current_state,
                attention_mask,
                i,
                position_offset,
                layer_cache,
                pool,
                None, // GPT-2 does not use RoPE
            )?;
            current_state = output;
        }

        Ok(current_state)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn hidden_size(&self) -> usize {
        self.meta.hidden_size
    }

    // Override default forward to include Final Layer Norm
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

        // 3. Final Layer Norm
        let final_ln_output = pool.get(hidden.shape().to_vec());
        self.final_layer_norm
            .encode(encoder, &self.final_ln_weights, &hidden, &final_ln_output);

        Ok(final_ln_output)
    }
}
