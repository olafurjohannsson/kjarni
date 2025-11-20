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
    CrossAttentionDecoder, CrossAttentionDecoderArchitecture, DecoderOutput,
    Device, EncoderDecoderArchitecture, LanguageModelConfig, TransformerModel,
};
use crate::weights::ModelWeights;
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
pub struct GpuCrossAttentionDecoder {
    pub layers: Vec<GpuCrossAttentionDecoderLayer>,
    pub embeddings: GpuEmbeddings,
    pub embedding_weights: GpuEmbeddingWeights,
    pub embed_layer_norm: GpuNormalization,
    pub embed_ln_weights: GpuNormalizationWeights,
    pub context: Arc<WgpuContext>,
    pub config: Arc<dyn CrossAttentionDecoderArchitecture + Send + Sync>,
    pub pool: Mutex<GpuTensorPool>,
}

impl GpuCrossAttentionDecoder {
    fn as_any(&self) -> &dyn Any {
        self // Simply return a reference to self as a `&dyn Any`
    }
    pub fn embedding_weights(&self) -> &GpuEmbeddingWeights {
        &self.embedding_weights
    }
    pub fn new(
        context: &Arc<WgpuContext>,
        weights: &ModelWeights,
        config: Arc<dyn CrossAttentionDecoderArchitecture + Send + Sync>,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size() as u32;
        let num_heads = config.num_attention_heads() as u32;

        let embedding_weights = GpuEmbeddingWeights::new(context, weights, config.as_ref())?;
        let embeddings = GpuEmbeddings::new(context)?;

        let (embed_norm_w, embed_norm_b) = config.get_decoder_embedding_ln_names();
        let embed_layer_norm =
            GpuNormalization::LayerNorm(GpuLayerNorm::new(context, config.layer_norm_eps()));
        let embed_ln_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_w)?)?,
            GpuTensor::from_ndarray::<f32, _>(context, &weights.get_array1(embed_norm_b)?)?,
        )?);

        let mut layers = Vec::with_capacity(config.num_decoder_layers());
        for i in 0..config.num_decoder_layers() {
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

                let weights_gpu = GpuFeedForwardWeightsStd::from_ndarrays(
                    &context,
                    &fc1_w,
                    &weights.get_array1(&ffn_names.intermediate_bias)?,
                    &fc2_w,
                    &weights.get_array1(&ffn_names.output_bias)?,
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

        Ok(Self {
            layers,
            embeddings,
            embedding_weights,
            embed_layer_norm,
            embed_ln_weights,
            context: context.clone(),
            config,
            pool: Mutex::new(GpuTensorPool::new(context.clone())),
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
        let cross_attn_output = self.cross_attn.forward_cross(
            encoder,
            residual,              // Query (Q)
            encoder_hidden_states, // Key & Value (KV)
            &self.cross_attn_weights,
            encoder_attn_mask,
            pool,
        );
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
impl TransformerModel for GpuCrossAttentionDecoder {
    fn device(&self) -> Device {
        Device::Wgpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        Some(self.context.clone())
    }
}

// Ensure this is imported

// Helper for the debug assertion (if not already available in this scope)
fn assert_all_close_debug(cpu: &Array3<f32>, gpu: &Array3<f32>, rtol: f32, atol: f32, name: &str) {
    let diff = cpu - gpu;
    let max_diff = diff.mapv(|x| x.abs()).fold(0.0f32, |a, b| a.max(*b));
    let max_val = cpu.mapv(|x| x.abs()).fold(0.0f32, |a, b| a.max(*b));

    if max_diff > atol + rtol * max_val {
        println!("[DEBUG FAIL] {} mismatch!", name);
        println!("  Max Diff: {}", max_diff);
        println!("  CPU First 5: {:?}", cpu.slice(s![0, 0, 0..5]).to_vec());
        println!("  GPU First 5: {:?}", gpu.slice(s![0, 0, 0..5]).to_vec());
        panic!("{} mismatch", name);
    } else {
        println!("[DEBUG PASS] {} matches. Max diff: {}", name, max_diff);
    }
}


#[async_trait(?Send)]
impl CrossAttentionDecoder for GpuCrossAttentionDecoder {
    type TokenInput = GpuTensor;
    type EncoderStateInput = GpuTensor;
    type MaskInput = GpuTensor;
    type Output = DecoderOutput;


    async fn forward<'a>(
        &self,
        // encoder: &mut wgpu::CommandEncoder,
        // pool: &mut GpuTensorPool,
        decoder_input_ids: &Self::TokenInput,
        encoder_hidden_states: &'a Self::EncoderStateInput,
        encoder_attention_mask: Option<&'a Self::MaskInput>,
        decoder_attention_mask: Option<&'a Self::MaskInput>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let mut pool_guard = self.pool.lock().await;
        let mut frame = GpuFrameContext::new(&self.context, pool_guard);
        let (mut encoder, mut pool) = frame.resources();

        // Get position offset in the cache
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = decoder_input_ids.shape()[1];
        let total_len = position_offset + seq_len;


        let mut hidden_states = self.embeddings.encode(
            encoder,
            &self.embedding_weights,
            &decoder_input_ids,
            None,
            position_offset,
            self.config.as_ref(),
            pool,
        )?;
        let hidden_states_after_norm = pool.get(hidden_states.shape().to_vec());
        self.embed_layer_norm.encode(
            encoder,
            &self.embed_ln_weights,
            &hidden_states,
            &hidden_states_after_norm,
        );
        hidden_states = hidden_states_after_norm;

        // --- 3. Get mutable access to the GPU cache ---
        let mut gpu_cache = cache.and_then(|c| c.as_any_mut().downcast_mut::<GpuBeamKVCache>());

        // --- 4. Loop through layers ---
        for (i, layer) in self.layers.iter().enumerate() {
            // let cached_kv = gpu_cache.as_ref().and_then(|c| c.get(i));
            let cached_kv = gpu_cache.as_ref().and_then(|c| c.get_layer_tensors(i));
            // let cached_kv_refs = cached_kv.as_ref().map(|(k, v)| (k, v));

            let (output, new_k, new_v) = layer.forward(
                encoder,
                &hidden_states,
                &encoder_hidden_states,
                &decoder_attention_mask.unwrap(), // this will crash if its not provided TODO: fix
                encoder_attention_mask,
                cached_kv,
                position_offset,
                pool,
            )?;
            hidden_states = output;

            // Update the self-attention cache for the next generation step
            if let Some(cache) = gpu_cache.as_deref_mut() {
                // cache.update(encoder, i, &new_k, &new_v, position_offset)?;
                cache.update(encoder, i, &new_k, &new_v);
                // cache.increment_len(1);
                // cache.update_seq2seq(encoder, i, &new_k, &new_v)?;
            }
        }


        if let Some(cache) = gpu_cache {
            // cache.set_seq_length(total_len);
            // cache.increment_len(1);
            // gpu_cache.increment_len(1);
        }

        frame.finish();
        let last_hidden_state_cpu = hidden_states.to_ndarray_3d().await?;

        Ok(DecoderOutput {
            last_hidden_state: last_hidden_state_cpu,
            past_key_values: None, // Cache is managed externally
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Make sure to adjust these `use` paths to match your project structure
    use crate::attention::MultiHeadAttention as CpuMha;
    use crate::decoder_cross_attn_layer::DecoderCrossAttentionLayer as CpuDecoderLayer;
    use crate::feedforward::{FeedForward as CpuFf, StdFeedForward as CpuStdFf};
    use crate::normalization::LayerNorm as CpuLayerNorm;

    use ndarray::{Array, Array1, Array2, Array3};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] Shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }

        let mut max_abs_diff = 0.0;
        let mut max_rel_diff = 0.0;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let abs_diff = (a_val - b_val).abs();
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }

            // The check: absolute difference must be within the combined tolerance
            let tolerance = atol + rtol * b_val.abs();
            if abs_diff > tolerance {
                panic!(
                    "[{}] Arrays are not close. Failed at values a={}, b={}. \
                 Absolute difference {} is greater than tolerance {}",
                    context, a_val, b_val, abs_diff, tolerance
                );
            }

            if b_val.abs() > 1e-8 {
                // Avoid division by zero
                let rel_diff = abs_diff / b_val.abs();
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
        }
        println!(
            "[{}] Check passed. Max absolute difference: {:.6e}, Max relative difference: {:.6e}",
            context, max_abs_diff, max_rel_diff
        );
    }

    /// Creates a mock CPU DecoderCrossAttentionLayer with deterministic weights.
    fn create_mock_cpu_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> CpuDecoderLayer {
        // Use a simple pattern for weights to make them deterministic
        let gen_weight =
            |shape, scale| Array2::from_shape_fn(shape, |(i, j)| ((i + j) as f32 * scale));
        let gen_bias = |size, val| Array1::from_elem(size, val);

        let self_attn = CpuMha::new(
            hidden_size,
            num_heads,
            gen_weight((hidden_size, hidden_size), 0.001),
            gen_bias(hidden_size, 0.1),
            gen_weight((hidden_size, hidden_size), 0.002),
            gen_bias(hidden_size, 0.0),
            gen_weight((hidden_size, hidden_size), 0.003),
            gen_bias(hidden_size, 0.0),
            gen_weight((hidden_size, hidden_size), -0.001),
            gen_bias(hidden_size, 0.0),
            None,
        );
        let self_attn_layer_norm =
            CpuLayerNorm::new(gen_bias(hidden_size, 1.0), gen_bias(hidden_size, 0.0), 1e-5);

        let cross_attn = CpuMha::new(
            hidden_size,
            num_heads,
            gen_weight((hidden_size, hidden_size), 0.004),
            gen_bias(hidden_size, 0.2),
            gen_weight((hidden_size, hidden_size), 0.005),
            gen_bias(hidden_size, 0.0),
            gen_weight((hidden_size, hidden_size), 0.006),
            gen_bias(hidden_size, 0.0),
            gen_weight((hidden_size, hidden_size), -0.002),
            gen_bias(hidden_size, 0.0),
            None,
        );
        let cross_attn_layer_norm =
            CpuLayerNorm::new(gen_bias(hidden_size, 1.0), gen_bias(hidden_size, 0.0), 1e-5);

        let feedforward = CpuFf::Standard(CpuStdFf::new(
            gen_weight((hidden_size, intermediate_size), 0.01),
            gen_bias(intermediate_size, 0.0),
            gen_weight((intermediate_size, hidden_size), -0.01),
            gen_bias(hidden_size, 0.0),
            crate::activations::Activation::Gelu,
        ));
        let ffn_layer_norm =
            CpuLayerNorm::new(gen_bias(hidden_size, 1.0), gen_bias(hidden_size, 0.0), 1e-5);

        CpuDecoderLayer {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
        }
    }

    /// Creates a GPU GpuCrossAttentionDecoderLayer from a CPU layer's weights.
    fn create_gpu_layer_from_cpu(
        context: &Arc<WgpuContext>,
        cpu_layer: &CpuDecoderLayer,
        hidden_size: u32,
        num_heads: u32,
    ) -> Result<GpuCrossAttentionDecoderLayer> {
        // Self-Attention weights
        let self_attn_weights = GpuAttentionWeights::new(
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.q_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.q_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.k_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.k_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.v_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.v_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.output_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn.output_bias)?,
        )?;
        let self_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn_layer_norm.weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.self_attn_layer_norm.bias)?,
        )?);

        // Cross-Attention weights
        let cross_attn_weights = GpuAttentionWeights::new(
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.q_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.q_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.k_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.k_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.v_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.v_bias)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.output_weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn.output_bias)?,
        )?;
        let cross_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn_layer_norm.weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.cross_attn_layer_norm.bias)?,
        )?);

        // Feed-Forward weights
        let ff_weights = if let CpuFf::Standard(ff) = &cpu_layer.feedforward {
            // --- START MODIFICATION ---

            // Use the "smart" constructor which takes the CPU ndarrays directly
            // and handles all internal transpositions and conversions.
            let weights_std = GpuFeedForwardWeightsStd::from_ndarrays(
                context,
                &ff.dense1_weight,
                &ff.dense1_bias,
                &ff.dense2_weight,
                &ff.dense2_bias,
            )?;

            GpuFeedForwardWeights::Standard(weights_std)

            // --- END MODIFICATION ---
        } else {
            panic!("Expected standard feedforward layer");
        };
        let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(context, &cpu_layer.ffn_layer_norm.weight)?,
            GpuTensor::from_ndarray(context, &cpu_layer.ffn_layer_norm.bias)?,
        )?);

        // Assemble the GPU layer
        Ok(GpuCrossAttentionDecoderLayer {
            self_attn: GpuAttention::new(context, hidden_size, num_heads, num_heads),
            self_attn_weights,
            self_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            self_attn_norm_weights,
            cross_attn: GpuAttention::new(context, hidden_size, num_heads, num_heads),
            cross_attn_weights,
            cross_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            cross_attn_norm_weights,
            feedforward: GpuFeedForward::Standard(GpuFeedForwardStd::new(
                context,
                crate::activations::Activation::Gelu,
            )?),
            ff_weights,
            ffn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            ffn_norm_weights,
            add: GpuAdd::new(context),
        })
    }

    #[tokio::test]
    async fn test_gpu_cpu_layer_consistency() -> Result<()> {
        // 1. SETUP
        let context = Arc::new(WgpuContext::new().await?);
        let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

        // 2. CREATE MODULES with identical weights
        let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
        let gpu_layer =
            create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

        // 3. CREATE IDENTICAL RANDOM INPUTS
        let cpu_decoder_hs = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_decoder_mask = Array2::ones((batch, dec_len));
        let cpu_encoder_mask = Array2::ones((batch, enc_len));

        // 4. RUN CPU FORWARD PASS
        let (cpu_output, (cpu_k, cpu_v)) = cpu_layer.forward(
            &cpu_decoder_hs,
            &cpu_encoder_hs,
            Some(&cpu_decoder_mask),
            Some(&cpu_encoder_mask),
            None, // No cache for first step
        )?;

        // 5. RUN GPU FORWARD PASS
        // --- START CORRECTION ---
        // Create the encoder and pool directly. No FrameContext, no Mutex.
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let gpu_decoder_hs = GpuTensor::from_ndarray(&context, &cpu_decoder_hs)?;
        let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
        let gpu_decoder_mask = GpuTensor::from_ndarray(&context, &cpu_decoder_mask)?;
        let gpu_encoder_mask = GpuTensor::from_ndarray(&context, &cpu_encoder_mask)?;

        // Pass the raw &mut encoder and &mut pool, which is what the layer expects.
        let (gpu_output_t, gpu_k_t, gpu_v_t) = gpu_layer.forward(
            &mut encoder,
            &gpu_decoder_hs,
            &gpu_encoder_hs,
            &gpu_decoder_mask,
            Some(&gpu_encoder_mask),
            None, // No cache
            0,    // Cache len is 0
            &mut pool,
        )?;

        // Submit the work and advance the pool frame.
        context.queue.submit(Some(encoder.finish()));
        pool.next_frame();
        // --- END CORRECTION ---

        let gpu_output = gpu_output_t.to_ndarray_3d().await?;
        let gpu_k = gpu_k_t.to_ndarray_3d().await?;
        let gpu_v = gpu_v_t.to_ndarray_3d().await?;

        // 6. COMPARE RESULTS
        let rtol = 1e-3;
        let atol = 1e-4;
        assert_all_close(&cpu_output, &gpu_output, rtol, atol, "Final Output");
        assert_all_close(&cpu_k, &gpu_k, rtol, atol, "New K Value");
        assert_all_close(&cpu_v, &gpu_v, rtol, atol, "New V Value");

        println!("✅ GPU and CPU decoder layer outputs are consistent!");
        Ok(())
    }
    #[tokio::test]
    async fn test_layer_subcomponent_parity() -> Result<()> {
        // 1. SETUP
        let context = Arc::new(WgpuContext::new().await?);
        let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

        // 2. CREATE MODULES (Identical Weights)
        let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
        let gpu_layer =
            create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

        // 3. CREATE INPUTS
        // Use random inputs to ensure we aren't hitting "lucky" zeros
        let cpu_hidden = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));

        // Masks (Standard generation case: 1x1 decoder mask, 1x93 encoder mask)
        let cpu_dec_mask = Array2::ones((batch, dec_len));
        let cpu_enc_mask = Array2::ones((batch, enc_len));

        // Upload to GPU
        let gpu_hidden = GpuTensor::from_ndarray(&context, &cpu_hidden)?;
        let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
        let gpu_dec_mask = GpuTensor::from_ndarray(&context, &cpu_dec_mask)?;
        let gpu_enc_mask = GpuTensor::from_ndarray(&context, &cpu_enc_mask)?;

        // Resources
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        // ========================================================================
        // STEP 1: SELF-ATTENTION BLOCK PARITY
        // ========================================================================
        println!("--- Testing Self-Attention Block ---");

        // CPU Execution
        let (cpu_sa_out, (cpu_k, cpu_v)) = cpu_layer.self_attention_block(
            &cpu_hidden,
            Some(&cpu_dec_mask),
            None, // No cache for this test
        )?;

        // GPU Execution
        // We have to manually call the sub-components because `gpu_layer.forward` does it all.
        // 1. Self Attn
        let (gpu_sa_attn_out, gpu_k, gpu_v) = gpu_layer.self_attn.forward_seq2seq(
            &mut encoder,
            &gpu_hidden,
            &gpu_layer.self_attn_weights,
            &gpu_dec_mask,
            None, // No cache
            0,    // Cache len
            &mut pool,
        )?;

        // 2. Add (Residual)
        let gpu_sa_add = pool.get(gpu_hidden.shape().to_vec());
        gpu_layer.add.encode(&mut encoder, &[&gpu_hidden, &gpu_sa_attn_out], &gpu_sa_add);

        // 3. Norm
        let gpu_sa_out = pool.get(gpu_sa_add.shape().to_vec());
        gpu_layer.self_attn_norm.encode(
            &mut encoder,
            &gpu_layer.self_attn_norm_weights,
            &gpu_sa_add,
            &gpu_sa_out,
        );

        // Submit & Compare Step 1
        context.queue.submit(Some(encoder.finish()));
        let gpu_sa_out_cpu = gpu_sa_out.to_ndarray_3d().await?;

        assert_all_close(&cpu_sa_out, &gpu_sa_out_cpu, 1e-3, 1e-4, "Self-Attention Block");
        println!("✅ Self-Attention Matches");

        // ========================================================================
        // STEP 2: CROSS-ATTENTION BLOCK PARITY
        // ========================================================================
        println!("--- Testing Cross-Attention Block ---");

        // IMPORTANT: Use the *CPU* output from Step 1 as input to Step 2.
        // This isolates the error to this specific block.
        let input_for_step_2 = cpu_sa_out.clone();
        let gpu_input_for_step_2 = GpuTensor::from_ndarray(&context, &input_for_step_2)?;

        encoder = context.device.create_command_encoder(&Default::default());
        pool.next_frame(); // Reset pool for cleanliness

        // CPU Execution
        let cpu_ca_out = cpu_layer.cross_attention_block(
            &input_for_step_2,
            &cpu_encoder_hs,
            Some(&cpu_enc_mask),
        )?;

        // GPU Execution
        // 1. Cross Attn
        let gpu_ca_attn_out = gpu_layer.cross_attn.forward_cross(
            &mut encoder,
            &gpu_input_for_step_2, // Query
            &gpu_encoder_hs,       // Key/Value
            &gpu_layer.cross_attn_weights,
            Some(&gpu_enc_mask),
            &mut pool,
        );

        // 2. Add (Residual)
        let gpu_ca_add = pool.get(gpu_input_for_step_2.shape().to_vec());
        gpu_layer.add.encode(&mut encoder, &[&gpu_input_for_step_2, &gpu_ca_attn_out], &gpu_ca_add);

        // 3. Norm
        let gpu_ca_out = pool.get(gpu_ca_add.shape().to_vec());
        gpu_layer.cross_attn_norm.encode(
            &mut encoder,
            &gpu_layer.cross_attn_norm_weights,
            &gpu_ca_add,
            &gpu_ca_out,
        );

        // Submit & Compare Step 2
        context.queue.submit(Some(encoder.finish()));
        let gpu_ca_out_cpu = gpu_ca_out.to_ndarray_3d().await?;

        assert_all_close(&cpu_ca_out, &gpu_ca_out_cpu, 1e-3, 1e-4, "Cross-Attention Block");
        println!("✅ Cross-Attention Matches");

        // ========================================================================
        // STEP 3: FEED-FORWARD BLOCK PARITY
        // ========================================================================
        println!("--- Testing Feed-Forward Block ---");

        // Use CPU output from Step 2
        let input_for_step_3 = cpu_ca_out.clone();
        let gpu_input_for_step_3 = GpuTensor::from_ndarray(&context, &input_for_step_3)?;

        encoder = context.device.create_command_encoder(&Default::default());
        pool.next_frame();

        // CPU Execution
        let cpu_ffn_out = cpu_layer.feed_forward_block(&input_for_step_3)?;

        // GPU Execution
        // 1. FFN
        let gpu_ffn_inner_out = pool.get(gpu_input_for_step_3.shape().to_vec());
        gpu_layer.feedforward.encode(
            &mut encoder,
            &gpu_layer.ff_weights,
            &gpu_input_for_step_3,
            &gpu_ffn_inner_out,
            &mut pool,
        );

        // 2. Add (Residual)
        let gpu_ffn_add = pool.get(gpu_input_for_step_3.shape().to_vec());
        gpu_layer.add.encode(&mut encoder, &[&gpu_input_for_step_3, &gpu_ffn_inner_out], &gpu_ffn_add);

        // 3. Norm
        let gpu_ffn_out = pool.get(gpu_ffn_add.shape().to_vec());
        gpu_layer.ffn_norm.encode(
            &mut encoder,
            &gpu_layer.ffn_norm_weights,
            &gpu_ffn_add,
            &gpu_ffn_out,
        );

        // Submit & Compare Step 3
        context.queue.submit(Some(encoder.finish()));
        let gpu_ffn_out_cpu = gpu_ffn_out.to_ndarray_3d().await?;

        assert_all_close(&cpu_ffn_out, &gpu_ffn_out_cpu, 1e-3, 1e-4, "Feed-Forward Block");
        println!("✅ Feed-Forward Matches");

        Ok(())
    }
}
