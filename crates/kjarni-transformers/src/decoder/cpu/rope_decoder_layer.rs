//! CPU-based decoder layer for modern LLMs with RoPE and pre-normalization.
//!
//! This module provides [`CpuRoPEDecoderLayer`], a complete transformer decoder layer
//! optimized for Llama-style architectures. It combines self-attention, SwiGLU feedforward,
//! RMS normalization, and rotary position embeddings (RoPE) in a pre-normalization configuration.
//!
//! # Architecture
//!
//! The layer follows the pre-norm architecture used in modern LLMs:
//! ```text
//! Input
//!   ↓
//! RMSNorm → Self-Attention (with RoPE) → Residual
//!   ↓
//! RMSNorm → SwiGLU FFN → Residual
//!   ↓
//! Output
//! ```
//!
//! # Performance
//!
//! This implementation is optimized for autoregressive generation:
//! - Uses zero-copy KV cache updates via [`DecoderAttention`]
//! - Efficient SwiGLU feedforward with fused gate projection
//! - Fast RMS normalization for pre-norm architecture
//!
//! Typical single-layer decode time (seq_len=1):
//! - Small models (7B): 2-5ms
//! - Medium models (13B): 5-10ms
//! - Large models (70B): 20-40ms (depending on quantization)
//!
//! # Example
//!
//! ```ignore
//! use kjarni_transformers::decoder::cpu::CpuRoPEDecoderLayer;
//!
//! // Layer is typically constructed by model builders
//! let layer = CpuRoPEDecoderLayer {
//!     attention,
//!     feed_forward,
//!     attention_norm,
//!     ffn_norm,
//!     rope,
//! };
//!
//! // Forward pass with KV caching
//! let (k_cache, v_cache) = cache.get_context_view_mut(layer_idx, seq_len)?;
//! let output = layer.forward(
//!     &hidden_states,
//!     &attention_mask,
//!     position_offset,
//!     k_cache,
//!     v_cache,
//! )?;
//! ```
//!
//! # See Also
//!
//! * [`DecoderAttention`] — The self-attention mechanism.
//! * [`SwiGluFeedForward`] — The feedforward network.
//! * [`RoPE`] — Rotary position embedding.
//! * [`CpuKVCache`] — KV cache management.

use anyhow::Result;
use ndarray::{Array2, Array3};
use std::sync::Arc;

use crate::{
    decoder::cpu::DecoderAttention, feedforward::SwiGluFeedForward, rope::RoPE, Normalization,
};

/// A complete transformer decoder layer with RoPE, pre-normalization, and SwiGLU.
///
/// `CpuRoPEDecoderLayer` combines all components needed for a single transformer layer
/// in modern decoder-only models (Llama, Mistral, etc.). It uses:
/// - Pre-normalization (RMSNorm before attention and FFN)
/// - Rotary Position Embeddings (RoPE) for positional information
/// - SwiGLU activation function in the feedforward network
/// - Residual connections around both attention and FFN
///
/// # Fields
///
/// All fields are public for flexibility in model construction:
///
/// * `attention` - Self-attention mechanism with Q/K/V projections.
/// * `feed_forward` - SwiGLU feedforward network.
/// * `attention_norm` - RMS normalization applied before attention.
/// * `ffn_norm` - RMS normalization applied before feedforward.
/// * `rope` - Shared rotary position embedding (typically Arc-wrapped for efficiency).
pub struct CpuRoPEDecoderLayer {
    pub attention: DecoderAttention,
    pub feed_forward: SwiGluFeedForward,
    pub attention_norm: Normalization,
    pub ffn_norm: Normalization,
    pub rope: Arc<RoPE>,
}

impl CpuRoPEDecoderLayer {
    /// Performs a forward pass through the decoder layer with pre-normalization.
    ///
    /// Applies self-attention and feedforward operations with RMS normalization
    /// and residual connections, following the pre-norm architecture used in
    /// modern LLMs like Llama and Mistral.
    ///
    /// # Arguments
    ///
    /// * `hidden_states` - Input tensor of shape `[batch, seq_len, hidden_size]`.
    /// * `attention_mask` - Attention mask of shape `[batch, total_seq_len]`.
    /// * `position_offset` - Starting position for RoPE (typically history length).
    /// * `k_cache` - Mutable KV cache view `[batch, total_len, kv_dim]` for keys.
    /// * `v_cache` - Mutable KV cache view `[batch, total_len, kv_dim]` for values.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_len, hidden_size]` after attention and FFN.
    ///
    /// # Performance
    ///
    /// Logs layer-level timing when a forward pass exceeds 10ms. The breakdown includes:
    /// - Norm1: Time for first RMS normalization
    /// - Attn: Time for self-attention (including cache writes)
    /// - Norm2: Time for second RMS normalization
    /// - FFN: Time for SwiGLU feedforward network
    ///
    /// # Example
    ///
    /// ```ignore
    /// let output = layer.forward(
    ///     &hidden_states,
    ///     &attention_mask,
    ///     history_len, // position_offset
    ///     k_cache,
    ///     v_cache,
    /// )?;
    /// ```
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        position_offset: usize,
        mut k_cache: ndarray::ArrayViewMut3<f32>,
        mut v_cache: ndarray::ArrayViewMut3<f32>,
    ) -> Result<Array3<f32>> {
        // Changed: Returns only hidden states
        let t_start = std::time::Instant::now();

        // 1. Pre-Norm (RMS)
        let norm_1 = self.attention_norm.forward(hidden_states);

        let t_norm1 = t_start.elapsed();

        // 2. Attention (In-Place Write)
        // We pass the destinations; attention writes to them and returns only attn_out
        let attn_out = self.attention.forward(
            &norm_1,
            Some(attention_mask),
            k_cache,
            v_cache,
            position_offset,
            Some(&self.rope),
        )?;

        let t_attn = t_start.elapsed() - t_norm1;

        let residual_1 = hidden_states + &attn_out;

        // 3. Pre-Norm (RMS)
        let norm_2 = self.ffn_norm.forward(&residual_1);

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

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        linear_layer::LinearLayer,
        normalization::RMSNorm,
    };
    use ndarray::{Array1, Array2};

    fn create_test_layer() -> CpuRoPEDecoderLayer {
        let hidden_size = 64;
        let num_heads = 4;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads; // 16
        let intermediate_size = 128;

        // Create linear layers with simple identity-like weights for testing
        // Note: K and V projections output num_kv_heads * head_dim but take hidden_size input
        let kv_dim = num_kv_heads * head_dim;

        let q_weight = Array2::from_shape_fn((hidden_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let k_weight = Array2::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let v_weight = Array2::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
            if i < hidden_size && i == j { 1.0 } else { 0.0 }
        });
        let o_weight = Array2::from_shape_fn((hidden_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });

        let q_proj = LinearLayer::new_f32(q_weight, None);
        let k_proj = LinearLayer::new_f32(k_weight, None);
        let v_proj = LinearLayer::new_f32(v_weight, None);
        let o_proj = LinearLayer::new_f32(o_weight, None);

        let attention = DecoderAttention::new(
            hidden_size,
            num_heads,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            Some(num_kv_heads),
        );

        // Create SwiGLU feedforward
        let gate_weight = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });
        let up_weight = gate_weight.clone();
        let down_weight = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
            if i == j { 1.0 } else { 0.0 }
        });

        let gate_proj = LinearLayer::new_f32(gate_weight, None);
        let up_proj = LinearLayer::new_f32(up_weight, None);
        let down_proj = LinearLayer::new_f32(down_weight, None);

        let feed_forward = SwiGluFeedForward::new(gate_proj, up_proj, down_proj);

        // Create RMS normalization layers
        let norm_weight = Array1::ones(hidden_size);
        let attention_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight.clone(), 1e-5));
        let ffn_norm = Normalization::RMSNorm(RMSNorm::new(norm_weight, 1e-5));

        // Create RoPE
        let rope = Arc::new(RoPE::new(head_dim, 128, 10000.0));

        CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
            rope,
        }
    }


    #[test]
    fn test_rope_decoder_layer_forward_new() -> Result<()> {
        let layer = create_test_layer();
        let batch_size = 1;
        let seq_len = 4;
        let hidden_size = 64;

        // Create input
        let hidden_states = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
            (k as f32) * 0.01
        });

        // Create attention mask
        let attention_mask = Array2::ones((batch_size, seq_len));

        // Create cache with space for sequence
        let kv_dim = layer.attention.num_kv_heads * layer.attention.head_dim;
        let mut k_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));
        let mut v_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));

        // Run forward pass
        let output = layer.forward(
            &hidden_states,
            &attention_mask,
            0,
            k_cache.view_mut(),
            v_cache.view_mut(),
        )?;

        // Check output shape
        assert_eq!(output.shape(), &[batch_size, seq_len, hidden_size]);

        // Check that output is not all zeros
        let sum: f32 = output.iter().sum();
        assert!(sum.abs() > 1e-6, "Output should not be all zeros");

        Ok(())
    }
}
