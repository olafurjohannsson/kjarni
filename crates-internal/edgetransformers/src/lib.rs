//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! EdgeTransformers: Fast transformer models for Rust

pub mod activations;
pub mod traits;
pub mod models;
pub mod encoder;
pub mod encoder_decoder;
pub mod decoder;
pub mod utils;
pub mod attention;
pub mod feedforward;
pub mod layer_norm;
pub mod embeddings;
pub mod gpu_ops;
pub mod gpu_pipeline;
pub mod gpu_context;
pub mod wgpu_ops;
pub mod weights;
pub mod pooling;
pub mod cache;

// Re-export commonly used items
pub use cache::{Cache, CpuKVCache, GpuKVCache};
pub use traits::{Device, TransformerModel, Encoder, Decoder, TransformerConfig, CrossAttentionDecoder};
pub use gpu_context::WgpuContext;
pub use crate::{
    feedforward::FeedForward,
    attention::MultiHeadAttention,
    layer_norm::LayerNorm,
    weights::ModelWeights,
    embeddings::Embeddings,
    pooling::{PoolingStrategy, cls_pool, last_token_pool, max_pool, mean_pool}
};

use ndarray::{Array2, Array3};
use anyhow::{Result, anyhow};

// Re-export model traits and registry
pub use models::{
    LanguageModel,
    EncoderLanguageModel,
    DecoderLanguageModel,
    Seq2SeqLanguageModel,
    ModelType,
    ModelArchitecture,
    
};

// Prelude for easy imports
pub mod prelude {
    pub use crate::cache::{Cache, CpuKVCache, GpuKVCache};
    pub use crate::traits::{Device, TransformerModel, Encoder, Decoder};
    pub use crate::models::{
        LanguageModel,
        DecoderLanguageModel,
        EncoderLanguageModel,
    };
    pub use crate::gpu_context::WgpuContext;
}


/// A generic transformer layer combining attention and feedforward.
/// This universal struct can represent an encoder layer, a decoder layer,
/// or an encoder-decoder's decoder layer.
pub struct TransformerLayer {
    // Self-Attention Components (always present)
    pub self_attn: MultiHeadAttention,
    pub self_attn_layer_norm: LayerNorm,

    // Cross-Attention Components (only for encoder-decoder models)
    pub cross_attn: Option<MultiHeadAttention>,
    pub cross_attn_layer_norm: Option<LayerNorm>,

    // Feed-Forward Components (always present)
    pub feedforward: FeedForward,
    pub ffn_layer_norm: LayerNorm,
}

impl TransformerLayer {
    /// Forward pass for an encoder or decoder-only layer with KV caching.
    pub fn forward_with_cache(
        &self,
        mut hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
        layer_idx: usize,
        cache: Option<&mut CpuKVCache>,
    ) -> Result<Array3<f32>> {
        let is_prenorm = config.is_prenorm();
        let is_causal = config.is_causal();

        if is_prenorm {
            // === PRE-NORM (e.g., GPT-2) ===
            let residual = hidden.clone();
            let ln1_out = self.self_attn_layer_norm.forward_3d(&hidden);
            
            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));
            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &ln1_out,
                None, // Self-attention
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;
            
            if let Some(cache) = cache {
                cache.update(layer_idx, new_k, new_v)?;
            }
            
            hidden = residual + attn_out;

            let residual = hidden.clone();
            let ln2_out = self.ffn_layer_norm.forward_3d(&hidden);
            let ffn_out = self.feedforward.forward(&ln2_out)?;
            hidden = residual + ffn_out;
            
        } else {
            // === POST-NORM (e.g., BERT, BART) ===
            let residual = hidden.clone();
            let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));

            let (attn_out, new_k, new_v) = self.self_attn.forward_with_cache(
                &hidden,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;
            
            if let Some(cache) = cache {
                cache.update(layer_idx, new_k, new_v)?;
            }
            
            hidden = residual + attn_out;
            hidden = self.self_attn_layer_norm.forward_3d(&hidden);

            let residual = hidden.clone();
            let ffn_out = self.feedforward.forward(&hidden)?;
            hidden = residual + ffn_out;
            hidden = self.ffn_layer_norm.forward_3d(&hidden);
        }

        Ok(hidden)
    }

     pub fn forward_cross_attention(
        &self,
        mut hidden_states: Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: &Array2<f32>,
        encoder_attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
        layer_idx: usize,
        cache: Option<&mut CpuKVCache>,
    ) -> Result<Array3<f32>> {
        let cross_attn = self.cross_attn.as_ref().ok_or_else(|| anyhow!("Layer is not configured for cross-attention (cross_attn is None)"))?;
        let cross_attn_ln = self.cross_attn_layer_norm.as_ref().ok_or_else(|| anyhow!("Layer is not configured for cross-attention (cross_attn_layer_norm is None)"))?;

        anyhow::ensure!(!config.is_prenorm(), "Cross-attention forward pass currently only supports post-norm architectures like BART.");

        // === 1. Self-Attention Block (cached) ===
        let residual = hidden_states.clone();
        let cached_kv = cache.as_ref().and_then(|c| c.get(layer_idx));
        
        let (self_attn_output, new_k, new_v) = self.self_attn.forward_with_cache(
            &hidden_states, None, Some(decoder_attention_mask), true, cached_kv,
        )?;

        if let Some(cache) = cache {
            cache.update(layer_idx, new_k, new_v)?;
        }

        hidden_states = residual + &self_attn_output;
        hidden_states = self.self_attn_layer_norm.forward_3d(&hidden_states);

        // === 2. Cross-Attention Block ===
        let residual = hidden_states.clone();
        
        let (cross_attn_output, _, _) = cross_attn.forward_with_cache(
            &hidden_states, Some(encoder_hidden_states), Some(encoder_attention_mask), false, None,
        )?;

        hidden_states = residual + &cross_attn_output;
        hidden_states = cross_attn_ln.forward_3d(&hidden_states);

        // === 3. Feed-Forward Block ===
        let residual = hidden_states.clone();
        let ffn_output = self.feedforward.forward(&hidden_states)?;
        hidden_states = residual + &ffn_output;
        hidden_states = self.ffn_layer_norm.forward_3d(&hidden_states);

        Ok(hidden_states)
    }

    /// Original forward without cache (for compatibility with encoders).
    pub fn forward(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        self.forward_with_cache(hidden, attention_mask, config, 0, None)
    }
}