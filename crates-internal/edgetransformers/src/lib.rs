//! Core transformer components for building transformer-based models
//!
//! This crate provides the fundamental building blocks for transformer architectures
//! without model-specific implementations.

//! EdgeTransformers: Fast transformer models for Rust

pub mod activations;
pub mod traits;
pub mod models;
pub mod encoder;
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
pub use traits::{Device, TransformerModel, Encoder, Decoder, TransformerConfig};
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


/// A generic transformer layer combining attention and feedforward
pub struct TransformerLayer {
    pub attention: MultiHeadAttention,
    pub feedforward: FeedForward,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
}

impl TransformerLayer {
    /// Forward pass with KV caching support
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
            // === PRE-NORM (GPT-2 style) ===
            
            // Attention block: LayerNorm → Attention → Residual
            let ln1_out = self.layer_norm1.forward_3d(&hidden);
            let cached_kv = cache.as_ref()
                .and_then(|c| c.get(layer_idx));
            let (attn_out, new_k, new_v) = self.attention.forward_with_cache(
                &ln1_out,
                None,  // Self-attention (no separate key/value input)
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;
            
            // Update cache with NEW K, V (not concatenated)
            if let Some(cache) = cache {
                cache.update(layer_idx, new_k, new_v)?;
            }
            
            hidden = hidden + attn_out;

            // FFN block: LayerNorm → FFN → Residual
            let ln2_out = self.layer_norm2.forward_3d(&hidden);
            let ffn_out = self.feedforward.forward(&ln2_out)?;
            hidden = hidden + ffn_out;
            
        } else {
            // === POST-NORM (BERT style) ===
            let cached_kv = cache.as_ref()
                .and_then(|c| c.get(layer_idx));
            // Attention block: Attention → Residual → LayerNorm
            let (attn_out, new_k, new_v) = self.attention.forward_with_cache(
                &hidden,
                None,
                Some(attention_mask),
                is_causal,
                cached_kv,
            )?;
            
            // Update cache
            if let Some(cache) = cache {
                cache.update(layer_idx, new_k, new_v)?;
            }
            
            hidden = hidden + attn_out;
            hidden = self.layer_norm1.forward_3d(&hidden);

            // FFN block: FFN → Residual → LayerNorm
            let ffn_out = self.feedforward.forward(&hidden)?;
            hidden = hidden + ffn_out;
            hidden = self.layer_norm2.forward_3d(&hidden);
        }

        Ok(hidden)
    }

    /// Original forward without cache (for compatibility)
    pub fn forward(
        &self,
        hidden: Array3<f32>,
        attention_mask: &Array2<f32>,
        config: &dyn TransformerConfig,
    ) -> Result<Array3<f32>> {
        // Call forward_with_cache with no cache
        self.forward_with_cache(hidden, attention_mask, config, 0, None)
    }
}
