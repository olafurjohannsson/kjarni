// --- Standard Library ---
use std::sync::Arc;

// --- External Crates ---
use anyhow::{Result, anyhow};
use ndarray::{Array2, Array3, Axis, s};

use crate::models::llama::config::LlamaConfig;

use kjarni_transformers::{
    WgpuContext, cache::CpuKVCache, decoder::prelude::*, embeddings::Embeddings, feedforward::SwiGluFeedForward, linear_layer::LinearLayer, models::base::ModelInput, normalization::RMSNorm, rope::RoPE, tensor::DType, traits::{Cache, Device, InferenceModel, ModelConfig, ModelLayout, ModelMetadata}, weights::ModelWeights
};

pub struct LlamaCpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<LlamaDecoderLayer>,
    pub final_norm: RMSNorm,
    config: Arc<LlamaConfig>,
}

pub struct LlamaDecoderLayer {
    pub attention: DecoderAttention,
    pub feed_forward: SwiGluFeedForward,
    pub attention_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub rope: Arc<RoPE>,
}

impl LlamaDecoderLayer {
    // Optimized forward pass
    pub fn forward(
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
        let metadata = config.metadata();
        let layout = config.layout();
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");

        let embeddings = Embeddings::from_weights(
            weights,
            &layout.token_embedding,
            decoder_layout.position_embedding.as_deref(), // Correctly access nested field
            decoder_layout.token_type_embedding.as_deref(),
        )?;

        let final_norm = RMSNorm::new(
            weights.get_array1(decoder_layout.final_norm_weight.as_ref().unwrap())?,
            metadata.norm_eps,
        );

        let mut layers = Vec::with_capacity(metadata.num_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Self::build_layer(
                weights,
                &metadata,
                &layout,
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
        meta: &ModelMetadata,
        layout: &ModelLayout,
        i: usize,
        rope: Arc<RoPE>,
        target_dtype: Option<DType>,
    ) -> Result<LlamaDecoderLayer> {
        // Get the specific nested layouts for the decoder.
        let decoder_layout = layout
            .decoder
            .as_ref()
            .expect("Llama layout must have a decoder section");
        let layer_layout = &decoder_layout.layer;
        let self_attn_layout = &layer_layout.self_attn;
        let ffn_layout = &layer_layout.ffn;

        let idx = i.to_string();
        let strategy = Some(kjarni_transformers::linear_layer::F32MatmulStrategy::CustomSimd); // Recommended default

        // Helper to replace the index template "{}" with the actual layer index
        let name = |template: &String| template.replace("{}", &idx);

        // --- 1. Load Attention Weights ---
        let q = LinearLayer::from_weights(
            weights,
            &name(&self_attn_layout.q_weight),
            None,
            target_dtype,
            strategy,
        )?;
        let k = LinearLayer::from_weights(
            weights,
            &name(&self_attn_layout.k_weight),
            None,
            target_dtype,
            strategy,
        )?;
        let v = LinearLayer::from_weights(
            weights,
            &name(&self_attn_layout.v_weight),
            None,
            target_dtype,
            strategy,
        )?;
        let o = LinearLayer::from_weights(
            weights,
            &name(&self_attn_layout.o_weight),
            None,
            target_dtype,
            strategy,
        )?;

        let attention = DecoderAttention::new(
            meta.hidden_size,
            meta.num_attention_heads,
            q,
            k,
            v,
            o,
            Some(meta.num_kv_heads),
        );

        // --- 2. Load FFN Weights ---
        let gate_name = ffn_layout
            .gate_weight
            .as_ref()
            .ok_or_else(|| anyhow!("Llama architecture requires ffn_gate for SwiGLU"))?;

        let gate =
            LinearLayer::from_weights(weights, &name(gate_name), None, target_dtype, strategy)?;
        let up = LinearLayer::from_weights(
            weights,
            &name(&ffn_layout.up_weight),
            None,
            target_dtype,
            strategy,
        )?;
        let down = LinearLayer::from_weights(
            weights,
            &name(&ffn_layout.down_weight),
            None,
            target_dtype,
            strategy,
        )?;

        let feed_forward = SwiGluFeedForward::new(gate, up, down);

        // --- 3. Load Norms ---
        let attention_norm = RMSNorm::new(
            weights.get_array1(&name(&self_attn_layout.norm_weight))?,
            meta.norm_eps,
        );
        let ffn_norm = RMSNorm::new(
            weights.get_array1(&name(&ffn_layout.norm_weight))?,
            meta.norm_eps,
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

impl InferenceModel for LlamaCpuDecoder {
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
    fn embed(&self, input: ModelInput<'_>, position_offset: usize) -> Result<Array3<f32>> {
        match input {
            ModelInput::TokensCpu(ids) => {
                let seq_len = ids.len();
                // let input_ids = Array2::from_shape_vec((1, seq_len), ids.to_vec())?;
                
                Ok(self
                    .embeddings
                    .forward(&ids.to_owned(), None, position_offset, false))
            }
            ModelInput::HiddenCpu(hidden) => Ok(hidden.to_owned()),
            _ => Err(anyhow!(
                "LlamaCpuDecoder received GPU input. Transfer to CPU first."
            )),
        }
    }

    fn embed_and_normalize(
        &self,
        input: ModelInput<'_>,
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
        input: ModelInput<'_>,
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
