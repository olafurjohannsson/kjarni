use crate::cache::GpuKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::attention::TempStorage;
use crate::gpu_ops::blocks::decoder::GpuPreNormDecoderLayer;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::ffn::GpuFeedForwardWeights;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::traits::{Cache, Decoder, DecoderArchitecture, DecoderOutput, Device, TransformerModel};
use crate::weights::ModelWeights;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3, s};
use std::sync::Arc;

/// The GPU backend for a generic Transformer Decoder.
pub struct GpuTransformerDecoder {
    // CPU-side embeddings
    word_embeddings: Array2<f32>,
    position_embeddings: Array2<f32>,

    embedding_weights: GpuEmbeddingWeights, // Holds GPU tensors
    embeddings: GpuEmbeddings,              // Holds the kernels

    // GPU-side weight buffers
    layers: Vec<GpuPreNormDecoderLayer>,

    // Final LayerNorm components
    final_layer_norm: GpuLayerNorm,
    final_ln_weights: GpuLayerNormWeights,

    slicer: GpuSlice,

    config: Arc<dyn DecoderArchitecture + Send + Sync>,

    context: Arc<WgpuContext>,

    cpu_embeddings: crate::Embeddings,
}

impl GpuTransformerDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<dyn DecoderArchitecture + Send + Sync>,
        context: Arc<WgpuContext>,
    ) -> Result<Self> {
        let slicer = GpuSlice::new(&context);

        // Load CPU-side embeddings
        let (word_w, pos_w, _) = config.get_embedding_weight_names();
        let word_embeddings = weights.get_array2(word_w)?;
        let position_embeddings = weights.get_array2(pos_w)?;

        let cpu_embeddings =
            crate::Embeddings::new(word_embeddings.clone(), position_embeddings.clone(), None);

        let embedding_weights: GpuEmbeddingWeights =
            GpuEmbeddingWeights::new(&context, weights, config.as_ref())?;
        let embeddings: GpuEmbeddings = GpuEmbeddings::new(&context)?;

        let (norm_w_name, norm_b_name) = config.get_final_layer_norm_names();
        let final_ln_gamma_cpu = weights.get_array1(norm_w_name)?;
        let final_ln_beta_cpu = weights.get_array1(norm_b_name)?;

        let final_ln_weights = GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(&context, &final_ln_gamma_cpu)?,
            GpuTensor::from_ndarray(&context, &final_ln_beta_cpu)?,
        )?;
        let final_layer_norm = GpuLayerNorm::new(&context, config.layer_norm_eps());

        let mut layers = Vec::with_capacity(config.num_hidden_layers());
        for i in 0..config.num_hidden_layers() {
            let attn_names = config.get_attention_names(i);
            let ffn_names = config.get_feed_forward_names(i);
            let hidden_size = config.hidden_size();

            // 1. Load Attention weights from CPU tensors
            let qkv_weight = weights.get_array2(&attn_names.qkv_weight)?;
            let qkv_bias = weights.get_array1(&attn_names.qkv_bias)?;
            let q_weight = qkv_weight.slice(s![.., 0..hidden_size]).to_owned();
            let k_weight = qkv_weight
                .slice(s![.., hidden_size..2 * hidden_size])
                .to_owned();
            let v_weight = qkv_weight
                .slice(s![.., 2 * hidden_size..3 * hidden_size])
                .to_owned();
            let q_bias = qkv_bias.slice(s![0..hidden_size]).to_owned();
            let k_bias = qkv_bias.slice(s![hidden_size..2 * hidden_size]).to_owned();
            let v_bias = qkv_bias
                .slice(s![2 * hidden_size..3 * hidden_size])
                .to_owned();
            let attn_output_w = weights.get_array2(&attn_names.output_weight)?;
            let attn_output_b = weights.get_array1(&attn_names.output_bias)?;

            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray(&context, &q_weight)?,
                GpuTensor::from_ndarray(&context, &q_bias)?,
                GpuTensor::from_ndarray(&context, &k_weight)?,
                GpuTensor::from_ndarray(&context, &k_bias)?,
                GpuTensor::from_ndarray(&context, &v_weight)?,
                GpuTensor::from_ndarray(&context, &v_bias)?,
                GpuTensor::from_ndarray(&context, &attn_output_w)?,
                GpuTensor::from_ndarray(&context, &attn_output_b)?,
            )?;

            // 2. Load Attention LayerNorm weights
            let attn_ln_gamma = weights.get_array1(&attn_names.norm_weight)?;
            let attn_ln_beta = weights.get_array1(&attn_names.norm_bias)?;
            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &attn_ln_gamma)?,
                GpuTensor::from_ndarray(&context, &attn_ln_beta)?,
            )?;

            // 3. Load Feed-Forward weights, applying transposition logic first
            let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
            let fc1_w_cpu = if config.transpose_ffn_weights() {
                intermediate_w.t().as_standard_layout().to_owned()
            } else {
                intermediate_w
            };
            let output_w = weights.get_array2(&ffn_names.output_weight)?;
            let fc2_w_cpu = if config.transpose_ffn_weights() {
                output_w.t().as_standard_layout().to_owned()
            } else {
                output_w
            };

            let ff_weights = GpuFeedForwardWeights::new(
                GpuTensor::from_ndarray(&context, &fc1_w_cpu)?,
                GpuTensor::from_ndarray(
                    &context,
                    &weights.get_array1(&ffn_names.intermediate_bias)?,
                )?,
                GpuTensor::from_ndarray(&context, &fc2_w_cpu)?,
                GpuTensor::from_ndarray(&context, &weights.get_array1(&ffn_names.output_bias)?)?,
            )?;

            // 4. Load FFN LayerNorm weights
            let ffn_ln_gamma = weights.get_array1(&ffn_names.norm_weight)?;
            let ffn_ln_beta = weights.get_array1(&ffn_names.norm_bias)?;
            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&context, &ffn_ln_gamma)?,
                GpuTensor::from_ndarray(&context, &ffn_ln_beta)?,
            )?;

            // 5. Create the specialized GPU decoder layer
            layers.push(GpuPreNormDecoderLayer::new(
                &context,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                config.clone(),
            )?);
        }

        Ok(Self {
            word_embeddings,
            position_embeddings,
            embedding_weights,
            embeddings,
            layers,
            final_layer_norm,
            final_ln_weights,
            slicer,
            config: config.clone(),
            context,
            cpu_embeddings,
        })
    }
}

impl TransformerModel for GpuTransformerDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
}

#[async_trait]
impl Decoder for GpuTransformerDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Decoder Forward"),
                });
        let mut temp = TempStorage::new(self.context.clone());
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = input.shape()[1];

        let initial_embeddings_cpu = self.cpu_embeddings.forward(
            input,
            None,
            position_offset,
            self.config.scale_embeddings(),
        );

        let mut hidden_states = GpuTensor::from_ndarray(&self.context, &initial_embeddings_cpu)?;
        // let input_ids_gpu = GpuTensor::from_ndarray(&self.context, input)?;

        // let mut hidden_states = self.embeddings.encode(
        //     &mut encoder,
        //     &self.embedding_weights,
        //     &input_ids_gpu,
        //     None, // <-- Pass None for token_type_ids
        //     self.config.as_ref(), // Your config should ensure scale_embeddings() is false
        //     &mut temp,
        // )?;

        let attention_mask_gpu = GpuTensor::from_ndarray(&self.context, attention_mask)?;

        let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuKVCache>());

        for (i, layer) in self.layers.iter().enumerate() {
            let (output, _) = layer.forward(
                &mut encoder,
                &hidden_states,
                &attention_mask_gpu,
                i,
                position_offset,
                gpu_cache.as_deref_mut(),
                &mut temp,
            )?;

            hidden_states = output;
        }
        let final_ln_output = temp.get(hidden_states.shape().to_vec());
        self.final_layer_norm.encode(
            &mut encoder,
            &self.final_ln_weights,
            &hidden_states,
            &final_ln_output,
        );
        hidden_states = final_ln_output;

        temp.reclaim();
        self.context.queue.submit(Some(encoder.finish()));
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
