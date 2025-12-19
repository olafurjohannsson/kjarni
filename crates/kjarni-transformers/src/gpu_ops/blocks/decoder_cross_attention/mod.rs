use crate::cache::GpuBeamKVCache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
    GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
};
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuFrameContext, GpuTensor, GpuTensorPool, Kernel};
use crate::traits::{
    CrossAttentionDecoderArchitecture, DecoderOutput, Device,
    EncoderDecoderArchitecture, LanguageModelConfig, TransformerModel,
};
use crate::encoder_decoder::traits::{EncoderDecoderLanguageModel};
use crate::weights_old::ModelWeights;
use crate::Cache;
use anyhow::Result;
use async_trait::async_trait;
use ndarray::{s, Array2, Array3};
use std::any::Any;
use std::sync::Arc;
use tokio::sync::Mutex;

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
// pub struct GpuCrossAttentionDecoder {
//     pub layers: Vec<GpuCrossAttentionDecoderLayer>,
//     pub embeddings: GpuEmbeddings,
//     pub embedding_weights: GpuEmbeddingWeights,
//     pub embed_layer_norm: GpuNormalization,
//     pub embed_ln_weights: GpuNormalizationWeights,
//     pub context: Arc<WgpuContext>,
//     pub config: Arc<dyn CrossAttentionDecoderArchitecture + Send + Sync>,
//     pub pool: Mutex<GpuTensorPool>,
// }

// impl GpuCrossAttentionDecoder {
//     fn as_any(&self) -> &dyn Any {
//         self // Simply return a reference to self as a `&dyn Any`
//     }
//     pub fn embedding_weights(&self) -> &GpuEmbeddingWeights {
//         &self.embedding_weights
//     }
//     pub fn new(
//         context: &Arc<WgpuContext>,
//         weights: &ModelWeights,
//         config: Arc<dyn CrossAttentionDecoderArchitecture + Send + Sync>,
//     ) -> Result<Self> {
//         let hidden_size = config.hidden_size() as u32;
//         let num_heads = config.num_attention_heads() as u32;

//         let embedding_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
//         let embeddings = GpuEmbeddings::new(context)?;

//         let (embed_norm_w, embed_norm_b) = config.get_decoder_embedding_ln_names();
//         let embed_layer_norm =
//             GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
//         let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
//             GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_w)?)?,
//             GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_b)?)?,
//         )?);

//         let mut layers = Vec::with_capacity(config.num_decoder_layers());
//         for i in 0..config.num_decoder_layers() {
//             let self_attn_names = config.get_decoder_self_attention_names(i);
//             let self_attn_weights = GpuAttentionWeights::new(
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&self_attn_names.q_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&self_attn_names.q_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&self_attn_names.k_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&self_attn_names.k_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&self_attn_names.v_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&self_attn_names.v_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&self_attn_names.output_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&self_attn_names.output_bias)?,
//                 )?,
//             )?;
//             let self_attn_norm =
//                 GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
//             let self_attn_norm_weights =
//                 GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
//                     GpuTensor::from_ndarray::<f32, _>(
//                         context,
//                         &weights.get_array1(&self_attn_names.norm_weight)?,
//                     )?,
//                     GpuTensor::from_ndarray::<f32, _>(
//                         context,
//                         &weights.get_array1(&self_attn_names.norm_bias)?,
//                     )?,
//                 )?);
//             let self_attn = GpuAttention::new(context, hidden_size, num_heads, num_heads);

//             // --- Load Cross-Attention components for layer `i` ---
//             let cross_attn_names = config.get_decoder_cross_attention_names(i);
//             let cross_attn_weights = GpuAttentionWeights::new(
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&cross_attn_names.q_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&cross_attn_names.q_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&cross_attn_names.k_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&cross_attn_names.k_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&cross_attn_names.v_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&cross_attn_names.v_bias)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_linear_weight(&cross_attn_names.output_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&cross_attn_names.output_bias)?,
//                 )?,
//             )?;
//             let cross_attn_norm =
//                 GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
//             let cross_attn_norm_weights =
//                 GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
//                     GpuTensor::from_ndarray::<f32, _>(
//                         context,
//                         &weights.get_array1(&cross_attn_names.norm_weight)?,
//                     )?,
//                     GpuTensor::from_ndarray::<f32, _>(
//                         context,
//                         &weights.get_array1(&cross_attn_names.norm_bias)?,
//                     )?,
//                 )?);
//             let cross_attn = GpuAttention::new(context, hidden_size, num_heads, num_heads);

//             let ffn_names = config.get_decoder_feed_forward_names(i);
//             let (feedforward, ff_weights) = {
//                 let intermediate_w = weights.get_array2(&ffn_names.intermediate_weight)?;
//                 let fc1_w = if config.transpose_ffn_weights() {
//                     intermediate_w.t().as_standard_layout().to_owned()
//                 } else {
//                     intermediate_w
//                 };
//                 let output_w = weights.get_array2(&ffn_names.output_weight)?;
//                 let fc2_w = if config.transpose_ffn_weights() {
//                     output_w.t().as_standard_layout().to_owned()
//                 } else {
//                     output_w
//                 };

//                 let weights_gpu = GpuFeedForwardWeightsStd::from_ndarrays(
//                     &context,
//                     &fc1_w,
//                     &weights.get_array1(&ffn_names.intermediate_bias)?,
//                     &fc2_w,
//                     &weights.get_array1(&ffn_names.output_bias)?,
//                 )?;
//                 (
//                     GpuFeedForward::Standard(GpuFeedForwardStd::new(
//                         context,
//                         crate::activations::Activation::Gelu,
//                     )?),
//                     GpuFeedForwardWeights::Standard(weights_gpu),
//                 )
//             };
//             let ffn_norm =
//                 GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
//             let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&ffn_names.norm_weight)?,
//                 )?,
//                 GpuTensor::from_ndarray::<f32, _>(
//                     context,
//                     &weights.get_array1(&ffn_names.norm_bias)?,
//                 )?,
//             )?);

//             layers.push(GpuCrossAttentionDecoderLayer {
//                 self_attn,
//                 self_attn_weights,
//                 self_attn_norm,
//                 self_attn_norm_weights,
//                 cross_attn,
//                 cross_attn_weights,
//                 cross_attn_norm,
//                 cross_attn_norm_weights,
//                 feedforward,
//                 ff_weights,
//                 ffn_norm,
//                 ffn_norm_weights,
//                 add: GpuAdd::new(context),
//             });
//         }

//         Ok(Self {
//             layers,
//             embeddings,
//             embedding_weights,
//             embed_layer_norm,
//             embed_ln_weights,
//             context: context.clone(),
//             config,
//             pool: Mutex::new(GpuTensorPool::new(context.clone())),
//         })
//     }
// }

impl GpuCrossAttentionDecoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm: GpuNormalization,
        self_attn_norm_weights: GpuNormalizationWeights,
        cross_attn_weights: GpuAttentionWeights,
        cross_attn_norm: GpuNormalization,
        cross_attn_norm_weights: GpuNormalizationWeights,
        feedforward: GpuFeedForward,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm: GpuNormalization,
        ffn_norm_weights: GpuNormalizationWeights,
        config: &dyn crate::traits::TransformerConfig,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;

        Ok(Self {
            self_attn: GpuAttention::new(context, hidden_size, num_heads, num_heads),
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            cross_attn: GpuAttention::new(context, hidden_size, num_heads, num_heads),
            cross_attn_weights,
            cross_attn_norm,
            cross_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            add: GpuAdd::new(context),
        })
    }
    pub fn precompute_cross_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        encoder_hidden_states: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        self.cross_attn.precompute_cross_kv(
            encoder,
            encoder_hidden_states,
            &self.cross_attn_weights,
            pool,
        )
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        decoder_hidden_states: &GpuTensor,
        encoder_hidden_states: Option<&GpuTensor>,
        decoder_attn_mask: &GpuTensor,
        encoder_attn_mask: Option<&GpuTensor>, // GPU version of the mask
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        precomputed_cross_kv: Option<&(GpuTensor, GpuTensor)>,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> {
        let residual = decoder_hidden_states;

        let (self_attn_output, new_k, new_v) = self.self_attn.forward_seq2seq(
            encoder,
            residual,
            &self.self_attn_weights,
            decoder_attn_mask,
            cached_kv,
            cache_len,
            pool,
        )?;
        let hidden_states_after_add1 = pool.get(residual.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &self_attn_output],
            &hidden_states_after_add1,
        );

        let hidden_states_after_norm1 = pool.get(hidden_states_after_add1.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            &hidden_states_after_add1,
            &hidden_states_after_norm1,
        );

        // Cross-Attention Block (Post-Norm)
        let residual = &hidden_states_after_norm1;
        // let cross_attn_output = self.cross_attn.forward_cross(
        //     encoder,
        //     residual,              // Query (Q)
        //     encoder_hidden_states, // Key & Value (KV)
        //     &self.cross_attn_weights,
        //     encoder_attn_mask,
        //     pool,
        // );
        let cross_attn_output = if let Some((k_pre, v_pre)) = precomputed_cross_kv {
            // FAST PATH: Use precomputed K/V
            self.cross_attn.forward_cross_precomputed(
                encoder,
                residual,
                k_pre,
                v_pre,
                &self.cross_attn_weights,
                encoder_attn_mask,
                pool
            )
        } else {
            // SLOW PATH: Compute K/V on the fly
            self.cross_attn.forward_cross(
                encoder,
                residual,
                encoder_hidden_states.unwrap(), // will crash
                &self.cross_attn_weights,
                encoder_attn_mask,
                pool,
            )
        };
        let hidden_states_after_add2 = pool.get(residual.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &cross_attn_output],
            &hidden_states_after_add2,
        );

        let hidden_states_after_norm2 = pool.get(hidden_states_after_add2.shape().to_vec());
        self.cross_attn_norm.encode(
            encoder,
            &self.cross_attn_norm_weights,
            &hidden_states_after_add2,
            &hidden_states_after_norm2,
        );

        // Feed-Forward Block (Post-Norm)
        let residual = &hidden_states_after_norm2;
        let ffn_output = pool.get(residual.shape().to_vec());

        self.feedforward
            .encode(encoder, &self.ff_weights, residual, &ffn_output, pool);
        let hidden_states_after_add3 = pool.get(residual.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &ffn_output], &hidden_states_after_add3);

        let final_output = pool.get(hidden_states_after_add3.shape().to_vec());
        self.ffn_norm.encode(
            encoder,
            &self.ffn_norm_weights,
            &hidden_states_after_add3,
            &final_output,
        );

        Ok((final_output, new_k, new_v))
    }
}
// impl TransformerModel for GpuCrossAttentionDecoder {
//     fn device(&self) -> Device {
//         Device::Wgpu
//     }
//     fn as_any(&self) -> &dyn std::any::Any {
//         self
//     }
//     fn context(&self) -> Option<Arc<WgpuContext>> {
//         Some(self.context.clone())
//     }
// }

// Ensure this is imported

// #[async_trait(?Send)]
// impl CrossAttentionDecoder for GpuCrossAttentionDecoder {
//     type TokenInput = GpuTensor;
//     type EncoderStateInput = GpuTensor;
//     type MaskInput = GpuTensor;
//     type Output = DecoderOutput;

//     async fn forward<'a>(
//         &self,
//         // encoder: &mut wgpu::CommandEncoder,
//         // pool: &mut GpuTensorPool,
//         decoder_input_ids: &Self::TokenInput,
//         encoder_hidden_states: &'a Self::EncoderStateInput,
//         encoder_attention_mask: Option<&'a Self::MaskInput>,
//         decoder_attention_mask: Option<&'a Self::MaskInput>,
//         cache: Option<&mut dyn Cache>,
//         cross_kv_caches: Option<&Vec<(ndarray::Array4<f32>, ndarray::Array4<f32>)>>,
//     ) -> Result<Self::Output> {
//         let mut pool_guard = self.pool.lock().await;
//         let mut frame = GpuFrameContext::new(&self.context, pool_guard);
//         let (mut encoder, mut pool) = frame.resources();

//         // Get position offset in the cache
//         let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
//         let seq_len = decoder_input_ids.shape()[1];
//         let total_len = position_offset + seq_len;

//         let mut hidden_states = self.embeddings.encode(
//             encoder,
//             &self.embedding_weights,
//             &decoder_input_ids,
//             None,
//             position_offset,
//             self.config.as_ref(),
//             pool,
//         )?;
//         let hidden_states_after_norm = pool.get(hidden_states.shape().to_vec());
//         self.embed_layer_norm.encode(
//             encoder,
//             &self.embed_ln_weights,
//             &hidden_states,
//             &hidden_states_after_norm,
//         );
//         hidden_states = hidden_states_after_norm;

//         // --- 3. Get mutable access to the GPU cache ---
//         let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuBeamKVCache>());

//         // --- 4. Loop through layers ---
//         for (i, layer) in self.layers.iter().enumerate() {
//             // let cached_kv = gpu_cache.as_ref().and_then(|c| c.get(i));
//             let cached_kv = gpu_cache.as_ref().and_then(|c| c.get_layer_tensors(i));
//             // let cached_kv_refs = cached_kv.as_ref().map(|(k, v)| (k, v));

//             let (output, new_k, new_v) = layer.forward(
//                 encoder,
//                 &hidden_states,
//                 Some(&encoder_hidden_states),
//                 &decoder_attention_mask.unwrap(), // this will crash if its not provided TODO: fix
//                 encoder_attention_mask,
//                 cached_kv,
//                 None,
//                 position_offset,
//                 pool,
//             )?;
//             hidden_states = output;

//             // Update the self-attention cache for the next generation step
//             if let Some(cache) = gpu_cache.as_deref_mut() {
//                 // cache.update(encoder, i, &new_k, &new_v, position_offset)?;
//                 cache.update(encoder, i, &new_k, &new_v);
//                 // cache.increment_len(1);
//                 // cache.update_seq2seq(encoder, i, &new_k, &new_v)?;
//             }
//         }

//         if let Some(cache) = gpu_cache {
//             // cache.set_seq_length(total_len);
//             // cache.increment_len(1);
//             // gpu_cache.increment_len(1);
//         }

//         frame.finish();
//         let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

//         Ok(DecoderOutput {
//             last_hidden_state: last_hidden_state_cpu,
//             past_key_values: None, // Cache is managed externally
//         })
//     }
// }


#[cfg(test)]
mod tests;