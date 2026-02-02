//! Causal self-attention for decoder models.
//!
//! This module provides `GpuDecoderSelfAttention`, optimized for autoregressive
//! decoding with KV caching.
//!
//! # Characteristics
//!
//! - **Causal**: Each token can only attend to previous tokens (autoregressive mask).
//! - **KV Cache**: Caches key/value states for efficient token-by-token generation.
//! - **No RoPE**: For models using learned or absolute positions (BART, GPT-2).
//!

use super::ops::AttentionOps;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::gpu::{GpuTensor, GpuTensorPool};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::WgpuContext;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use anyhow::Result;
use std::sync::Arc;

/// Causal self-attention for decoder models with KV caching.
///
/// This attention module is designed for autoregressive decoding where each
/// token can only attend to itself and previous tokens. It supports KV caching
///
pub struct GpuDecoderSelfAttention {
    pub ops: AttentionOps,
    concatenate: GpuConcatenate,
    slice: GpuSlice,
}

/// Output of decoder self-attention forward pass.
pub struct DecoderSelfAttentionOutput {
    /// The attention output tensor `[B, S_new, H]`.
    pub hidden_states: GpuTensor,
    /// New K projection `[B, S_new, H*D]` for cache update.
    pub new_k: GpuTensor,
    /// New V projection `[B, S_new, H*D]` for cache update.
    pub new_v: GpuTensor,
}

impl GpuDecoderSelfAttention {
    /// Creates a new decoder self-attention module.
    ///
    /// # Arguments
    ///
    /// * `context` - The WGPU context for creating GPU resources.
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of attention heads.
    ///s
    pub fn new(context: &Arc<WgpuContext>, hidden_size: u32, num_heads: u32) -> Self {
        Self {
            // Decoder self-attention has equal Q and KV heads (no GQA for BART/GPT-2)
            ops: AttentionOps::new(context, hidden_size, num_heads, num_heads),
            concatenate: GpuConcatenate::new(context),
            slice: GpuSlice::new(context),
        }
    }

    /// Returns the underlying attention operations for advanced usage.
    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    /// Performs the forward pass of decoder self-attention.
    ///
    /// Computes causal self-attention where each token can only attend to
    /// previous tokens. Supports KV caching for efficient autoregressive generation.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `hidden_states` - Input tensor of shape `[B, S_new, H]`.
    /// * `weights` - The Q, K, V, and O projection weights.
    /// * `attention_mask` - Mask of shape `[B, S_total]` for padding.
    /// * `cached_kv` - Optional cached (K, V) tensors in 4D format `[B, H, S_cached, D]`.
    /// * `cache_len` - Number of valid tokens in the cache.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// `DecoderSelfAttentionOutput` containing:
    /// - `hidden_states`: Output tensor `[B, S_new, H]`
    /// - `new_k`: New K projection `[B, S_new, H*D]` for cache update
    /// - `new_v`: New V projection `[B, S_new, H*D]` for cache update
    ///
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<DecoderSelfAttentionOutput> {
        let (_, _, _hidden_size) = hidden_states.dims3();

        // 1. Project Q, K, V from input
        let q_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.q_weight,
            &weights.q_bias,
            pool,
        );
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );

        // 2. Split heads for attention: [B, S, H*D] -> [B, H, S, D]
        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        let k_heads_new = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads_new = self.ops.split_heads(encoder, &v_proj, pool);

        // 3. Concatenate with cache if present
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            if cache_len > 0 {
                self.concat_with_cache(
                    encoder,
                    &k_heads_new,
                    &v_heads_new,
                    cached_k,
                    cached_v,
                    cache_len,
                    pool,
                )?
            } else {
                // Cache exists but is empty, just use new K/V
                (k_heads_new, v_heads_new)
            }
        } else {
            // No cache (prefill step)
            (k_heads_new, v_heads_new)
        };

        // 4. Compute causal attention
        let context = self.ops.attention(
            encoder,
            &q_heads,
            &full_k,
            &full_v,
            Some(attention_mask),
            true,       // is_causal = true for decoder
            cache_len,  // position_offset for causal mask
            pool,
        );

        // 5. Merge heads and output projection
        let context_merged = self.ops.merge_heads(encoder, &context, pool);
        let output = self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        );

        // Return output and NEW K/V projections (3D) for cache update
        Ok(DecoderSelfAttentionOutput {
            hidden_states: output,
            new_k: k_proj,
            new_v: v_proj,
        })
    }

    /// Concatenates new K/V with cached K/V.
    ///
    /// # Arguments
    ///
    /// * `k_new` - New K heads `[B, H, S_new, D]`
    /// * `v_new` - New V heads `[B, H, S_new, D]`
    /// * `cached_k` - Cached K `[B, H, S_cached, D]`
    /// * `cached_v` - Cached V `[B, H, S_cached, D]`
    /// * `cache_len` - Valid length in cache
    ///
    /// # Returns
    ///
    /// (full_k, full_v) each of shape `[B, H, S_total, D]`
    fn concat_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k_new: &GpuTensor,
        v_new: &GpuTensor,
        cached_k: &GpuTensor,
        cached_v: &GpuTensor,
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let (b, h, s_new, d) = k_new.dims4();
        let full_len = cache_len + s_new;

        // Slice valid portion from cache (in case cache tensor is larger than cache_len)
        let slice_shape = &[b, h, cache_len, d];
        let slice_offset = &[0, 0, 0, 0];

        let valid_cache_k = cached_k.slice(encoder, &self.slice, slice_offset, slice_shape)?;
        let valid_cache_v = cached_v.slice(encoder, &self.slice, slice_offset, slice_shape)?;

        // Allocate output tensors
        let full_k = pool.get(vec![b, h, full_len, d]);
        let full_v = pool.get(vec![b, h, full_len, d]);

        // Concatenate along sequence dimension (axis 2)
        self.concatenate.encode(encoder, &[&valid_cache_k, k_new], &full_k, 2);
        self.concatenate.encode(encoder, &[&valid_cache_v, v_new], &full_v, 2);

        Ok((full_k, full_v))
    }

    /// Projects only K and V (useful for cross-attention precomputation pattern).
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder.
    /// * `hidden_states` - Input tensor `[B, S, H]`.
    /// * `weights` - Attention weights.
    /// * `pool` - Tensor pool.
    ///
    /// # Returns
    ///
    /// (K, V) projections, each `[B, S, H*D]`.
    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        let k_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.k_weight,
            &weights.k_bias,
            pool,
        );
        let v_proj = self.ops.project(
            encoder,
            hidden_states,
            &weights.v_weight,
            &weights.v_bias,
            pool,
        );
        (k_proj, v_proj)
    }
}
