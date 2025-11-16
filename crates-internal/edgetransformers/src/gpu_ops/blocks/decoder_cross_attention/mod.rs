use crate::Cache;
use crate::GpuKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights, TempStorage};
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
    GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuTensor, Kernel};
use crate::traits::{
    CrossAttentionDecoder as CrossAttentionDecoderTrait, DecoderOutput, Device,
    EncoderDecoderArchitecture, LanguageModelConfig, TransformerConfig, TransformerModel,
};
use crate::weights::ModelWeights;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{Array2, Array3};
use std::sync::Arc;

// This is just a data container, like the CPU version.
pub struct GpuCrossAttentionDecoderLayer {
    pub self_attn: GpuAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub cross_attn: GpuAttention,
    pub cross_attn_weights: GpuAttentionWeights,
    pub cross_attn_norm: GpuNormalization,
    pub cross_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
}

// This is the main orchestrator struct.
pub struct GpuCrossAttentionDecoder {
    pub layers: Vec<GpuCrossAttentionDecoderLayer>,
    pub embeddings: GpuEmbeddings,
    pub embedding_weights: GpuEmbeddingWeights,
    pub embed_layer_norm: GpuNormalization,
    pub embed_ln_weights: GpuNormalizationWeights,
    pub context: Arc<WgpuContext>,
    pub config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
}

impl GpuCrossAttentionDecoder {
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn EncoderDecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;

        // --- 1. Load Decoder Embeddings and Initial LayerNorm ---
        let embedding_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
        let embeddings = GpuEmbeddings::new(context)?;

        let (embed_norm_w, embed_norm_b) = config.get_decoder_embedding_ln_names();
        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_w)?)?,
            GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_b)?)?,
        )?);

        // --- 2. Loop and Build Decoder Layers ---
        let mut layers = Vec::with_capacity(config.num_decoder_layers());
        for i in 0..config.num_decoder_layers() {
            // --- Load Self-Attention components for layer `i` ---
            let self_attn_names = config.get_decoder_self_attention_names(i);
            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&self_attn_names.q_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&self_attn_names.q_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&self_attn_names.k_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&self_attn_names.k_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&self_attn_names.v_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&self_attn_names.v_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&self_attn_names.output_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&self_attn_names.output_bias)?,
                )?,
            )?;
            let self_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let self_attn_norm_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&self_attn_names.norm_weight)?,
                    )?,
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&self_attn_names.norm_bias)?,
                    )?,
                )?);
            let self_attn = GpuAttention::new(context, hidden_size, num_heads, num_heads);

            // --- Load Cross-Attention components for layer `i` ---
            let cross_attn_names = config.get_decoder_cross_attention_names(i);
            let cross_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&cross_attn_names.q_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&cross_attn_names.q_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&cross_attn_names.k_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&cross_attn_names.k_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&cross_attn_names.v_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&cross_attn_names.v_bias)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_linear_weight(&cross_attn_names.output_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&cross_attn_names.output_bias)?,
                )?,
            )?;
            let cross_attn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let cross_attn_norm_weights =
                GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&cross_attn_names.norm_weight)?,
                    )?,
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&cross_attn_names.norm_bias)?,
                    )?,
                )?);
            let cross_attn = GpuAttention::new(context, hidden_size, num_heads, num_heads);

            // --- Load FFN components for layer `i` ---
            let ffn_names = config.get_decoder_feed_forward_names(i);
            let (feedforward, ff_weights) = {
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

                let weights_gpu = GpuFeedForwardWeightsStd::new(
                    GpuTensor::from_ndarray::<f32, _>(context, &fc1_w)?,
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&ffn_names.intermediate_bias)?,
                    )?,
                    GpuTensor::from_ndarray::<f32, _>(context, &fc2_w)?,
                    GpuTensor::from_ndarray::<f32, _>(
                        context,
                        &weights.get_array1(&ffn_names.output_bias)?,
                    )?,
                )?;
                (
                    GpuFeedForward::Standard(GpuFeedForwardStd::new(
                        context,
                        crate::activations::Activation::Gelu,
                    )?),
                    GpuFeedForwardWeights::Standard(weights_gpu),
                )
            };
            let ffn_norm =
                GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
            let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&ffn_names.norm_weight)?,
                )?,
                GpuTensor::from_ndarray::<f32, _>(
                    context,
                    &weights.get_array1(&ffn_names.norm_bias)?,
                )?,
            )?);

            // --- Push the fully constructed layer struct ---
            layers.push(GpuCrossAttentionDecoderLayer {
                self_attn,
                self_attn_weights,
                self_attn_norm,
                self_attn_norm_weights,
                cross_attn,
                cross_attn_weights,
                cross_attn_norm,
                cross_attn_norm_weights,
                feedforward,
                ff_weights,
                ffn_norm,
                ffn_norm_weights,
                add: GpuAdd::new(context),
            });
        }

        // --- 3. Construct and return the final Self ---
        Ok(Self {
            layers,
            embeddings,
            embedding_weights,
            embed_layer_norm,
            embed_ln_weights,
            context: context.clone(),
            config,
        })
    }
}

impl GpuCrossAttentionDecoderLayer {
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoder_hidden_states: &GpuTensor,
        encoder_hidden_states: &GpuTensor,
        decoder_attn_mask: &GpuTensor,
        encoder_attn_mask: Option<&GpuTensor>, // GPU version of the mask
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        temp: &mut TempStorage,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
        // Returns (final_output, new_k_for_cache, new_v_for_cache)

        // --- 1. Self-Attention Block (Post-Norm) ---
        let residual = decoder_hidden_states;
        let (self_attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
            encoder,
            residual, // Input is the raw hidden state
            None,     // K/V source is the same as Q for self-attention
            &self.self_attn_weights,
            Some(decoder_attn_mask),
            true, // Self-attention in a decoder is always causal
            cached_kv,
            None, // No RoPE in models like BART/T5
            temp,
        );
        let hidden_states_after_add1 = temp.get(residual.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &self_attn_output],
            &hidden_states_after_add1,
        );

        let hidden_states_after_norm1 = temp.get(hidden_states_after_add1.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            &hidden_states_after_add1,
            &hidden_states_after_norm1,
        );

        // --- 2. Cross-Attention Block (Post-Norm) ---
        let residual = &hidden_states_after_norm1;
        let cross_attn_output = self.cross_attn.forward(
            encoder,
            residual,                    // Query (Q) is from the decoder's current state
            Some(encoder_hidden_states), // Key (K) and Value (V) are from the encoder's output
            &self.cross_attn_weights,
            encoder_attn_mask,
            false, // Cross-attention is NOT causal
            None,  // No RoPE
            temp,
        );
        let hidden_states_after_add2 = temp.get(residual.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &cross_attn_output],
            &hidden_states_after_add2,
        );

        let hidden_states_after_norm2 = temp.get(hidden_states_after_add2.shape().to_vec());
        self.cross_attn_norm.encode(
            encoder,
            &self.cross_attn_norm_weights,
            &hidden_states_after_add2,
            &hidden_states_after_norm2,
        );

        // --- 3. Feed-Forward Block (Post-Norm) ---
        let residual = &hidden_states_after_norm2;
        // Note: Your standard FFN encode takes 3D input, which is correct here.
        let ffn_output = temp.get(residual.shape().to_vec());

        self.feedforward
            .encode(encoder, &self.ff_weights, residual, &ffn_output, temp);
        let hidden_states_after_add3 = temp.get(residual.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &ffn_output], &hidden_states_after_add3);

        let final_output = temp.get(hidden_states_after_add3.shape().to_vec());
        self.ffn_norm.encode(
            encoder,
            &self.ffn_norm_weights,
            &hidden_states_after_add3,
            &final_output,
        );

        Ok((final_output, new_k, new_v))
    }
}
impl TransformerModel for GpuCrossAttentionDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }

    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
}
#[async_trait(?Send)]
impl CrossAttentionDecoderTrait for GpuCrossAttentionDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward<'a>(
        &self,
        decoder_input_ids: &Self::Input,
        encoder_hidden_states: &'a Array3<f32>,
        encoder_attention_mask: Option<&'a Array2<f32>>,
        decoder_attention_mask: Option<&'a Array2<f32>>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let mut encoder =
            self.context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Cross-Attention Decoder Forward"),
                });
        let mut temp = TempStorage::new(self.context.clone());

        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = decoder_input_ids.shape()[1];

        // --- 1. Transfer all CPU inputs to GPU Tensors ---
        let encoder_hidden_states_gpu =
            GpuTensor::from_ndarray::<f32, _>(&self.context, encoder_hidden_states)?;

        let encoder_attn_mask_gpu = if let Some(mask) = encoder_attention_mask {
            Some(GpuTensor::from_ndarray::<f32, _>(&self.context, mask)?)
        } else {
            None
        };

        let decoder_attn_mask_gpu = if let Some(mask) = decoder_attention_mask {
            GpuTensor::from_ndarray::<f32, _>(&self.context, mask)?
        } else {
            // If no mask is provided, we might need a default one (e.g., all ones)
            // For now, we'll rely on the caller providing it.
            // This part might need more robust handling based on model requirements.
            todo!("Handle missing decoder attention mask");
        };

        // --- 2. Perform Decoder Embeddings on GPU ---
        let input_ids_gpu = GpuTensor::from_ndarray::<u32, _>(&self.context, decoder_input_ids)?;
        // Note: Seq2Seq embeddings often have a position offset quirk. Your CPU `embed_decoder_with_offset`
        // handles this. For the GPU, the `embeddings.encode` would need to support this offset.
        // Assuming it does for now.
        let mut hidden_states = self.embeddings.encode(
            &mut encoder,
            &self.embedding_weights,
            &input_ids_gpu,
            None,
            position_offset,
            self.config.as_ref(),
            &mut temp,
        )?;

        let hidden_states_after_norm = temp.get(hidden_states.shape().to_vec());
        self.embed_layer_norm.encode(
            &mut encoder,
            &self.embed_ln_weights,
            &hidden_states,
            &hidden_states_after_norm,
        );
        hidden_states = hidden_states_after_norm;

        // --- 3. Get mutable access to the GPU cache ---
        let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuKVCache>());

        // --- 4. Loop through layers ---
        for (i, layer) in self.layers.iter().enumerate() {
            let cached_kv = gpu_cache.as_ref().and_then(|c| c.get(i));
            let cached_kv_refs = cached_kv.as_ref().map(|(k, v)| (k, v));
            let (output, new_k, new_v) = layer.forward(
                &mut encoder,
                &hidden_states,
                &encoder_hidden_states_gpu,
                &decoder_attn_mask_gpu,
                encoder_attn_mask_gpu.as_ref(),
                cached_kv_refs,
                &mut temp,
            )?;
            hidden_states = output;

            // Update the self-attention cache for the next generation step
            if let Some(cache) = gpu_cache.as_deref_mut() {
                cache.update(&mut encoder, i, &new_k, &new_v, position_offset)?;
            }
        }

        // --- 5. Finalize GPU work and read back to CPU ---
        temp.reclaim();
        self.context.queue.submit(Some(encoder.finish()));
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        if let Some(cache) = gpu_cache {
            cache.increment_len(seq_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: last_hidden_state_cpu,
            past_key_values: None, // Cache is managed externally
        })
    }
}
