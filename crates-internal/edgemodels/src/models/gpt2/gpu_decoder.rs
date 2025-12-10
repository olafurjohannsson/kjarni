use crate::models::gpt2::config::Gpt2Config;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::TransformerConfig;
use edgetransformers::WgpuContext;
use edgetransformers::cache::GpuKVCache;
use edgetransformers::gpu_ops::blocks::decoder::GpuPreNormDecoderLayer;
use edgetransformers::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use edgetransformers::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use edgetransformers::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd as GpuStandardFFN, GpuFeedForwardWeights,
    GpuFeedForwardWeightsStd as GpuStandardFFNWeights, GpuNormalization, GpuNormalizationWeights,
    attention::GpuAttentionWeights,
};
use edgetransformers::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool};
use edgetransformers::models::base::DecoderInput;
use edgetransformers::models::base::DecoderLoadConfig;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::traits::{DecoderArchitecture, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2, s};
use std::sync::Arc;
use tokio::sync::Mutex;

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
    cpu_embeddings: Option<edgetransformers::embeddings::Embeddings>,
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
        load_config: DecoderLoadConfig,
    ) -> Result<Self> {
        log::info!("Building GPT-2 GPU decoder...");

        let (cpu_embeddings, embedding_weights, embeddings) = if load_config.offload_embeddings {
            log::info!("Optimization: Loading Embedding weights to CPU RAM only.");

            let (word_w, pos_w, _) = config.get_embedding_weight_names();
            let word_embeddings = weights.get_array2(word_w)?;
            let position_embeddings_cpu = if !pos_w.is_empty() {
                Some(weights.get_array2(pos_w)?)
            } else {
                None
            };

            let cpu_embs = edgetransformers::embeddings::Embeddings::new(
                word_embeddings,
                position_embeddings_cpu,
                None,
            );

            (Some(cpu_embs), None, None)
        } else {
            log::info!("Loading Embedding weights to VRAM.");
            let ew = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
            let em = GpuEmbeddings::new(context)?;
            (None, Some(ew), Some(em))
        };

        // 2. Final Layer Norm (GPT-2 uses standard LayerNorm)
        let (norm_w_name, norm_b_name) = config.get_final_layer_norm_names();
        let final_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
        let final_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(context, &weights.get_array1(norm_w_name)?)?,
            GpuTensor::from_ndarray(context, &weights.get_array1(norm_b_name)?)?,
        )?);

        // 3. Decoder Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            log::info!(
                "Building GPT-2 layer {}/{}",
                i + 1,
                config.num_hidden_layers()
            );
            let dyn_config = config.clone() as Arc<dyn DecoderArchitecture + Send + Sync>;
            let decoder_layer = Self::build_layer(context.clone(), weights, dyn_config, i)?;
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
        })
    }

    /// Build a single GPT-2 decoder layer
    fn build_layer(
        context: Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        layer_idx: usize,
    ) -> Result<GpuPreNormDecoderLayer> {
        let hidden_size = config.hidden_size();
        let intermediate_size = config.intermediate_size();

        let attn_names = config.get_attention_names(layer_idx);

        // Load COMBINED QKV weight and split it
        let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
        let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;

        log::info!(
            "Layer {}: QKV weight shape (file): {:?}, expected [In={}, Out={}]",
            layer_idx,
            qkv_weight.shape(),
            hidden_size,
            3 * hidden_size
        );

        // Split AND transpose: file is [In, Out], GPU needs [Out, In]
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

        // Verify [Out, In] layout for GPU
        assert_eq!(
            q_weight.shape(),
            &[hidden_size, hidden_size],
            "Q weight must be [Out={}, In={}] for GPU, got {:?}",
            hidden_size,
            hidden_size,
            q_weight.shape()
        );
        assert_eq!(
            k_weight.shape(),
            &[hidden_size, hidden_size],
            "K weight must be [Out={}, In={}] for GPU, got {:?}",
            hidden_size,
            hidden_size,
            k_weight.shape()
        );
        assert_eq!(
            v_weight.shape(),
            &[hidden_size, hidden_size],
            "V weight must be [Out={}, In={}] for GPU, got {:?}",
            hidden_size,
            hidden_size,
            v_weight.shape()
        );

        log::info!(
            "Layer {}: Q/K/V weights transposed to [Out, In]: {:?}",
            layer_idx,
            q_weight.shape()
        );

        // Split biases (no transpose needed for 1D)
        let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
        let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
        let v_bias = qkv_bias
            .slice(s![2 * hidden_size..3 * hidden_size])
            .to_owned();

        // Output projection - transpose from [In, Out] to [Out, In]
        let o_weight_raw = weights.get_array2(&attn_names.output_weight)?;
        log::info!(
            "Layer {}: O weight shape (file): {:?}",
            layer_idx,
            o_weight_raw.shape()
        );

        let o_weight = o_weight_raw.t().as_standard_layout().to_owned();
        let o_bias = weights.get_array1(&attn_names.output_bias)?;

        assert_eq!(
            o_weight.shape(),
            &[hidden_size, hidden_size],
            "Output weight must be [Out={}, In={}] for GPU, got {:?}",
            hidden_size,
            hidden_size,
            o_weight.shape()
        );

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

        // --- Attention LayerNorm (unchanged) ---
        let (self_attn_norm, self_attn_norm_weights) = {
            let gamma = weights.get_array1(&attn_names.norm_weight)?;
            let beta = weights.get_array1(&attn_names.norm_bias)?;
            (
                GpuNormalization::LayerNorm(GpuLayerNorm::new(&context, config.layer_norm_eps())),
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_ndarray(&context, &gamma)?,
                    GpuTensor::from_ndarray(&context, &beta)?,
                )?),
            )
        };

        // --- FFN weights with transpose ---
        let ffn_names = config.get_feed_forward_names(layer_idx);

        let intermediate_w_raw = weights.get_array2(&ffn_names.intermediate_weight)?;
        let output_w_raw = weights.get_array2(&ffn_names.output_weight)?;

        log::info!(
            "Layer {}: FFN intermediate weight (file): {:?}, output weight (file): {:?}",
            layer_idx,
            intermediate_w_raw.shape(),
            output_w_raw.shape()
        );

        // Transpose: [In, Out] -> [Out, In]
        let intermediate_w = intermediate_w_raw.t().as_standard_layout().to_owned();
        let output_w = output_w_raw.t().as_standard_layout().to_owned();

        assert_eq!(
            intermediate_w.shape(),
            &[intermediate_size, hidden_size],
            "FFN intermediate must be [Out={}, In={}] for GPU, got {:?}",
            intermediate_size,
            hidden_size,
            intermediate_w.shape()
        );
        assert_eq!(
            output_w.shape(),
            &[hidden_size, intermediate_size],
            "FFN output must be [Out={}, In={}] for GPU, got {:?}",
            hidden_size,
            intermediate_size,
            output_w.shape()
        );

        let intermediate_b = weights.get_array1(&ffn_names.intermediate_bias)?;
        let output_b = weights.get_array1(&ffn_names.output_bias)?;

        let ff_weights = GpuFeedForwardWeights::Standard(GpuStandardFFNWeights::new(
            GpuTensor::from_ndarray(&context, &intermediate_w)?,
            GpuTensor::from_ndarray(&context, &intermediate_b)?,
            GpuTensor::from_ndarray(&context, &output_w)?,
            GpuTensor::from_ndarray(&context, &output_b)?,
        )?);

        let feedforward =
            GpuFeedForward::Standard(GpuStandardFFN::new(&context, config.activation_function())?);

        // --- FFN LayerNorm (unchanged) ---
        let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.norm_weight)?)?,
            GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.norm_bias)?)?,
        )?);
        let ffn_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(&context, config.layer_norm_eps()));

        log::info!(
            "✓ Layer {} built: attn [{}, {}], ffn [{}, {}] → [{}, {}]",
            layer_idx,
            hidden_size,
            hidden_size,
            intermediate_size,
            hidden_size,
            hidden_size,
            intermediate_size
        );

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
            None, // GPT-2 doesn't use RoPE
        )?)
    }
}

#[async_trait(?Send)]
impl GpuDecoder for Gpt2GpuDecoder {
    async fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pool: &mut edgetransformers::gpu_ops::GpuTensorPool,
        input_ids: DecoderInput<'_>, // The flexible input
        attention_mask: &GpuTensor,
        position_offset: usize,
        cache: Option<&mut GpuKVCache>,
        _encoder_hidden_states: Option<&GpuTensor>,
    ) -> Result<GpuTensor> {
        // 1. Handle Embeddings (CPU vs GPU path)
        let mut hidden_states = match input_ids {
            // CASE 1: CPU Input IDs
            DecoderInput::Cpu(ids) => {
                if let Some(cpu_embeds) = &self.cpu_embeddings {
                    // Sub-case 1A: CPU Weights (Hybrid Mode) -> Lookup on CPU
                    let input_array = Array2::from_shape_vec((1, ids.len()), ids.to_vec())?;
                    let initial_embeddings_cpu = cpu_embeds.forward(
                        &input_array,
                        None,
                        position_offset,
                        self.config.scale_embeddings(),
                    );
                    GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?
                } else {
                    // Sub-case 1B: GPU Weights (Pure GPU Mode) -> Upload IDs -> GPU Kernel
                    let input_tensor = GpuTensor::from_ndarray(
                        &self.context,
                        &Array2::from_shape_vec((1, ids.len()), ids.to_vec())?,
                    )?;

                    let gpu_embeds = self
                        .embeddings
                        .as_ref()
                        .ok_or_else(|| anyhow!("Fatal: Embeddings not loaded on CPU or GPU"))?;
                    let gpu_weights = self.embedding_weights.as_ref().unwrap();

                    gpu_embeds.encode(
                        encoder,
                        gpu_weights,
                        &input_tensor,
                        None,
                        position_offset,
                        self.config.as_ref(),
                        pool,
                    )?
                }
            }

            // CASE 2: GPU Input IDs
            DecoderInput::Gpu(ids_tensor) => {
                // Must have GPU weights
                let gpu_embeds = self
                    .embeddings
                    .as_ref()
                    .ok_or_else(|| anyhow!("GPU input provided but embeddings are on CPU"))?;
                let gpu_weights = self.embedding_weights.as_ref().unwrap();

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

        // 2. Pass through each transformer layer
        for (i, layer) in self.layers.iter().enumerate() {
            let (output, _) = layer.forward_gpt2(
                encoder,
                &hidden_states,
                attention_mask,
                i,
                position_offset,
                cache_mut_ref.as_deref_mut(),
                pool,
                None,
            )?;
            hidden_states = output;
        }

        // 3. Final layer norm
        let final_ln_output = pool.get(hidden_states.shape().to_vec());
        self.final_layer_norm.encode(
            encoder,
            &self.final_ln_weights,
            &hidden_states,
            &final_ln_output,
        );

        // Returns GPU tensor. Data stays on VRAM.
        Ok(final_ln_output)
    }
}
