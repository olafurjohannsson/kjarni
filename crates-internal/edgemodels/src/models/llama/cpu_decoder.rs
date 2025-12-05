use crate::models::llama::config::LlamaConfig;
use edgetransformers::TransformerConfig;
use edgetransformers::decoder_attention::DecoderAttention; // Or DecoderAttention if you renamed it
use edgetransformers::feedforward::SwiGluFeedForward;
use edgetransformers::normalization::{Normalization, RMSNorm};
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::rope::RoPE;
use edgetransformers::weights::{ModelWeights, DType};
use edgetransformers::traits::{Decoder, DecoderOutput, DecoderArchitecture, Cache, Device, TransformerModel};
use edgetransformers::cache::CpuKVCache;
use edgetransformers::traits::LanguageModelConfig;
use edgetransformers::embeddings::Embeddings;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use ndarray::{Array2, Array3, Axis, s};
use std::sync::Arc;

/// A dedicated, highly optimized CPU decoder for Llama architecture.
/// Hardcodes the execution path (RMSNorm -> Attention -> RMSNorm -> SwiGLU) for maximum speed.
pub struct LlamaCpuDecoder {
    embeddings: Embeddings,
    layers: Vec<LlamaDecoderLayer>,
    final_norm: RMSNorm,
    config: Arc<LlamaConfig>,
}

/// A single Llama layer. Concrete types only (No Enums/Box).
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
        let (attn_out, new_k, new_v) = self.attention.forward(
            &norm_1,
            Some(attention_mask),
            past_kv,
            Some(&self.rope),
        )?;
        
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
                t_norm1, t_attn, t_norm2, t_ffn
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
        let word_embeddings = weights.get_array2(word_w)?;
        let embeddings = Embeddings::new(word_embeddings, None, None);

        let (norm_w, _) = config.get_final_layer_norm_names();
        let final_norm = RMSNorm::new(weights.get_array1(norm_w)?, config.layer_norm_eps());

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(Self::build_layer(weights, &config, i, rope.clone(), target_dtype)?);
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

        // Load Attention Weights (BF16/LinearLayer)
        let q = LinearLayer::from_weights(weights, &layer_names.q_weight, target_dtype)?;
        let k = LinearLayer::from_weights(weights, &layer_names.k_weight, target_dtype)?;
        let v = LinearLayer::from_weights(weights, &layer_names.v_weight, target_dtype)?;
        let o = LinearLayer::from_weights(weights, &layer_names.output_weight, target_dtype)?;

        let attention = DecoderAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            q, k, v, o,
            // Llama has no biases, passing None is efficient
            Some(config.num_key_value_heads),
        );

        // Load FFN Weights (BF16/LinearLayer)
        let gate = LinearLayer::from_weights(weights, ffn_names.gate_weight.as_ref().unwrap(), target_dtype)?;
        let up = LinearLayer::from_weights(weights, &ffn_names.intermediate_weight, target_dtype)?;
        let down = LinearLayer::from_weights(weights, &ffn_names.output_weight, target_dtype)?;

        let feed_forward = SwiGluFeedForward::new(gate, up, down);

        // Load Norms
        let attention_norm = RMSNorm::new(
            weights.get_array1(&layer_names.norm_weight)?, 
            config.rms_norm_eps
        );
        let ffn_norm = RMSNorm::new(
            weights.get_array1(&ffn_names.norm_weight)?, 
            config.rms_norm_eps
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
    fn device(&self) -> Device { Device::Cpu }
    fn context(&self) -> Option<Arc<edgetransformers::gpu_context::WgpuContext>> { None }
    fn as_any(&self) -> &dyn std::any::Any { self }
}

#[async_trait(?Send)]
impl Decoder for LlamaCpuDecoder {
    type Input = Array2<u32>;
    type Output = DecoderOutput;

    async fn forward(
        &self,
        input_ids: &Self::Input,
        attention_mask: &Array2<f32>,
        cache: Option<&mut dyn Cache>,
    ) -> Result<Self::Output> {
        let position_offset = cache.as_ref().map_or(0, |c| c.get_seq_length());
        let seq_len = input_ids.shape()[1];

        // 1. Embeddings
        let mut hidden_states = self.embeddings.forward(
            input_ids, None, position_offset, false // scale_embeddings = false for Llama
        );

        // 2. Layers
        let cpu_cache_opt = cache.and_then(|c| c.as_any_mut().downcast_mut::<CpuKVCache>());
        let mut new_key_values = Vec::with_capacity(self.layers.len());

        for (i, layer) in self.layers.iter().enumerate() {
            let past_kv = cpu_cache_opt.as_ref().and_then(|c| c.get(i));
            let past_kv_view = past_kv.as_ref().map(|(k, v)| (k.view(), v.view()));

            let (new_hidden, new_k, new_v) = layer.forward(
                &hidden_states, 
                attention_mask, 
                position_offset, 
                past_kv_view
            )?;
            
            hidden_states = new_hidden;
            new_key_values.push((new_k, new_v));
        }

        // 3. Final Norm
        hidden_states = self.final_norm.forward_3d(&hidden_states);

        // 4. Update Cache
        if let Some(cache) = cpu_cache_opt {
            for (i, (k, v)) in new_key_values.into_iter().enumerate() {
                cache.update(i, &k, &v)?;
            }
            cache.increment_len(seq_len);
        }

        Ok(DecoderOutput {
            last_hidden_state: hidden_states,
            past_key_values: None,
        })
    }

    async fn get_hidden_states(&self, input: &Self::Input, mask: &Array2<f32>) -> Result<Array3<f32>> {
        let out = self.forward(input, mask, None).await?;
        Ok(out.last_hidden_state)
    }
}