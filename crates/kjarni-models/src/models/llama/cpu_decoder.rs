// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis, s};

// --- Workspace Crates ---
use kjarni_transformers::{
    TransformerConfig, WgpuContext,
    cache::CpuKVCache,
    decoder::prelude::*,
    embeddings::Embeddings,
    feedforward::SwiGluFeedForward,
    linear_layer::LinearLayer,
    normalization::RMSNorm,
    rope::RoPE,
    tensor::DType,
    traits::{Cache, DecoderArchitecture, Device, LanguageModelConfig, TransformerModel},
    weights::ModelWeights,
};

use crate::models::llama::config::LlamaConfig;

pub struct LlamaCpuDecoder {
    embeddings: Embeddings,
    layers: Vec<LlamaDecoderLayer>,
    final_norm: RMSNorm,
    config: Arc<LlamaConfig>,
}

struct LlamaDecoderLayer {
    attention: DecoderAttention,
    feed_forward: SwiGluFeedForward,
    attention_norm: RMSNorm,
    ffn_norm: RMSNorm,
    rope: Arc<RoPE>,
}

impl LlamaDecoderLayer {
    // Optimized forward pass
    fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        past_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let t_start = std::time::Instant::now();

        // 1. Pre-Norm (RMS)
        let norm_1 = self.attention_norm.forward_3d(hidden_states);

        let t_norm1 = t_start.elapsed();

        // 2. Attention
        let (attn_out, new_k, new_v) =
            self.attention
                .forward(&norm_1, Some(attention_mask), past_kv, Some(&self.rope))?;

        let t_attn = t_start.elapsed() - t_norm1;

        let residual_1 = hidden_states + &attn_out;

        // 3. Pre-Norm (RMS)
        let norm_2 = self.ffn_norm.forward_3d(&residual_1);

        let t_norm2 = t_start.elapsed() - t_attn - t_norm1;

        // 4. FeedForward
        let ffn_out = self.feed_forward.forward(&norm_2)?;

        let t_ffn = t_start.elapsed() - t_norm2 - t_attn - t_norm1;

        let output = residual_1 + ffn_out;

        // Log if slow (> 10ms)
        if t_start.elapsed().as_millis() > 10 {
            log::info!(
                "Layer Perf: Norm1: {:?}, Attn: {:?}, Norm2: {:?}, FFN: {:?}",
                t_norm1,
                t_attn,
                t_norm2,
                t_ffn
            );
        }

        Ok((output, new_k, new_v))
    }
}

impl LlamaCpuDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<LlamaConfig>,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<Self> {
        let (word_w, _, _) = config.get_embedding_weight_names();
        // let word_embeddings = weights.get_array2(word_w)?;
        
        let embeddings = Embeddings::from_weights(
            &weights,
            word_w,
            None,
            None
        )?;
        // let embeddings = Embeddings::new(
        //     kjarni_transformers::embeddings::EmbeddingData::F32(word_embeddings),
        //     None,
        //     None,
        // );

        let (norm_w, _) = config.get_final_layer_norm_names();
        let final_norm = RMSNorm::new(weights.get_array1(norm_w)?, config.layer_norm_eps());

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Self::build_layer(
                weights,
                &config,
                i,
                rope.clone(),
                target_dtype,
            )?);
        }

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            config,
        })
    }

    fn build_layer(
        weights: &ModelWeights,
        config: &LlamaConfig,
        i: usize,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<LlamaDecoderLayer> {
        let layer_names = config.get_layer_attention_names(i);
        let ffn_names = config.get_feed_forward_names(i);
        let strategy = Some(kjarni_transformers::linear_layer::F32MatmulStrategy::Faer);
        // Load Attention Weights (BF16/LinearLayer)
        let q = LinearLayer::from_weights(
            weights,
            &layer_names.q_weight,
            None,
            target_dtype,
            strategy,
        )?;
        let k = LinearLayer::from_weights(
            weights,
            &layer_names.k_weight,
            None,
            target_dtype,
            strategy,
        )?;
        let v = LinearLayer::from_weights(
            weights,
            &layer_names.v_weight,
            None,
            target_dtype,
            strategy,
        )?;
        let o = LinearLayer::from_weights(
            weights,
            &layer_names.output_weight,
            None,
            target_dtype,
            strategy,
        )?;

        let attention = DecoderAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            q,
            k,
            v,
            o,
            // Llama has no biases, passing None is efficient
            Some(config.num_key_value_heads),
        );

        // Load FFN Weights (BF16/LinearLayer)
        let gate = LinearLayer::from_weights(
            weights,
            ffn_names.gate_weight.as_ref().unwrap(),
            None,
            target_dtype,
            strategy,
        )?;
        let up = LinearLayer::from_weights(
            weights,
            &ffn_names.intermediate_weight,
            None,
            target_dtype,
            strategy,
        )?;
        let down = LinearLayer::from_weights(
            weights,
            &ffn_names.output_weight,
            None,
            target_dtype,
            strategy,
        )?;

        let feed_forward = SwiGluFeedForward::new(gate, up, down);

        // Load Norms
        let attention_norm = RMSNorm::new(
            weights.get_array1(&layer_names.norm_weight)?,
            config.rms_norm_eps,
        );
        let ffn_norm = RMSNorm::new(
            weights.get_array1(&ffn_names.norm_weight)?,
            config.rms_norm_eps,
        );

        Ok(LlamaDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        })
    }
}

// --- Trait Impl ---

impl TransformerModel for LlamaCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl CpuDecoder for LlamaCpuDecoder {
    fn embed(&self, input: DecoderInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
        match input {
            DecoderInput::TokensCpu(ids) => {
                let seq_len = ids.len();
                let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;

                Ok(self
                    .embeddings
                    .forward(&input_ids, None, position_offset, false))
            }
            DecoderInput::HiddenCpu(hidden) => Ok(hidden.clone()),
            _ => Err(anyhow!(
                "LlamaCpuDecoder received GPU input. Transfer to CPU first."
            )),
        }
    }

    fn embed_and_normalize(
        &self,
        input: DecoderInput<'_>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        // Llama is Pre-Norm (Norm is inside the layer).
        // No initial LayerNorm exists before the first block.
        self.embed(input, position_offset)
    }

    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        mut cache: Option<&mut dyn Cache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();
        let seq_len = hidden.shape()[1];

        // 1. Downcast to CpuKVCache to access specific get/update methods
        // We use a mutable option so we can borrow it mutably later for updates
        let mut cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());

        // 2. Store new K/V pairs temporarily.
        // We cannot update the cache *inside* the loop because `cpu_cache_opt` is borrowed
        // to get `past_kv` (immutable borrow), preventing a mutable borrow for `update`.
        let mut new_key_values = Vec::with_capacity(end_layer - start_layer);

        for i in start_layer..end_layer {
            if i >= self.layers.len() {
                break;
            }
            let layer = &self.layers[i];

            // 3. Get Past KV View
            // We use .as_ref() to borrow the Option content without moving it
            let past_kv = cpu_cache_opt.as_ref().and_then(|c| c.get(i));

            // Map the Tuple(Array3, Array3) to Tuple(ArrayView3, ArrayView3)
            let past_kv_view = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));

            // 4. Layer Forward
            let (new_hidden, new_k, new_v) =
                layer.forward(&hidden, attention_mask, position_offset, past_kv_view)?;

            hidden = new_hidden;
            new_key_values.push((new_k, new_v));
        }

        // 5. Update Cache (Batch update after the loop)
        if let Some(cache) = cpu_cache_opt {
            for (local_idx, (k, v)) in new_key_values.into_iter().enumerate() {
                let layer_idx = start_layer + local_idx;
                cache.update(layer_idx, &k, &v)?;
            }
            // Important: Increment length so next step knows the offset
            cache.increment_len(seq_len);
        }

        Ok(hidden)
    }

    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn forward(
        &self,
        input: DecoderInput<'_>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Array3<f32>> {
        let hidden = self.embed_and_normalize(input, position_offset)?;

        // 2. Run Layers
        let mut output = self.forward_layers(
            &hidden,
            attention_mask,
            position_offset,
            cache,
            0,
            self.num_layers(),
        )?;

        output = self.final_norm.forward_3d(&output);

        Ok(output)
    }
}
