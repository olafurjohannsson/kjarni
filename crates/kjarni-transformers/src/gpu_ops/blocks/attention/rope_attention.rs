//! RoPE-based causal attention for modern LLMs.
//!
//! This module provides `GpuRoPEAttention`, optimized for autoregressive
//! models using Rotary Position Embeddings and Grouped Query Attention.
//!
//! # Characteristics
//!
//! - **Causal**: Each token can only attend to previous tokens.
//! - **KV Cache**: Caches key/value states for efficient generation.
//! - **RoPE**: Rotary Position Embeddings for position encoding.
//! - **GQA**: Grouped Query Attention (fewer KV heads than Q heads).
//!
//! # Used By
//!
//! - LLaMA, LLaMA 2, LLaMA 3
//! - Mistral, Mixtral
//! - Phi-2, Phi-3
//! - Gemma, Gemma 2
//! - Qwen, Qwen 2
//! - Most modern open-source LLMs
//!
//! # Example
//!
//! ```rust
//! use kjarni_transformers::gpu_ops::attention::GpuRoPEAttention;
//!
//! // LLaMA 3.2 1B: 2048 hidden, 32 Q heads, 8 KV heads
//! let attn = GpuRoPEAttention::new(&context, 2048, 32, 8);
//!
//! let (output, new_k, new_v) = attn.forward(
//!     &mut encoder,
//!     &hidden_states,
//!     &weights,
//!     &rope,
//!     &mask,
//!     cached_kv,
//!     position_offset,
//!     &mut pool,
//! )?;
//! ```

use super::ops::AttentionOps;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::primitives::repeat_kv::GpuRepeatKV;
use crate::gpu_ops::{blocks::attention::GpuAttentionWeights, GpuTensor, GpuTensorPool};
use crate::WgpuContext;
use anyhow::Result;
use std::sync::Arc;

/// RoPE-based causal attention with GQA support.
///
/// This attention module is designed for modern LLMs that use:
/// - Rotary Position Embeddings (RoPE) for position encoding
/// - Grouped Query Attention (GQA) for efficiency
/// - KV caching for autoregressive generation
///
/// # Architecture
///
/// ```text
/// Input [B, S, H]
///     │
///     ├──► Q Projection ──► Split Heads ──► RoPE ──────────────────┐
///     │                                                             │
///     ├──► K Projection ──► Split Heads ──► RoPE ──► Concat Cache ──┼──► GQA Attention
///     │                                                             │
///     └──► V Projection ──► Split Heads ──────────► Concat Cache ──┘
///                                                                   │
///                                                    Merge Heads ──► O Projection
///                                                                   │
/// Output [B, S, H] ◄────────────────────────────────────────────────┘
///
/// Returns: (output, new_K_rotated, new_V) for cache update
/// ```
///
/// # Grouped Query Attention (GQA)
///
/// In GQA, there are fewer KV heads than Q heads. Each KV head is shared
/// by multiple Q heads, reducing memory bandwidth during inference.
///
/// ```text
/// Q heads:  [h0, h1, h2, h3, h4, h5, h6, h7]  (8 heads)
/// KV heads: [k0,         k1        ]          (2 heads)
/// Mapping:  [k0, k0, k0, k0, k1, k1, k1, k1]  (repeat)
/// ```
///
/// # Thread Safety
///
/// `GpuRoPEAttention` is `Send + Sync` and can be safely shared across threads.
pub struct GpuRoPEAttention {
    ops: AttentionOps,
    concatenate: GpuConcatenate,
    slice: GpuSlice,
    repeat_kv: GpuRepeatKV,
}

/// Output of RoPE attention forward pass.
pub struct RoPEAttentionOutput {
    /// The attention output tensor `[B, S_new, H]`.
    pub hidden_states: GpuTensor,
    /// New K projection with RoPE applied `[B, S_new, KV_H*D]` for cache update.
    /// Note: This is in 3D format for cache storage.
    pub new_k: GpuTensor,
    /// New V projection `[B, S_new, KV_H*D]` for cache update.
    /// Note: This is in 3D format for cache storage.
    pub new_v: GpuTensor,
}

impl GpuRoPEAttention {
    /// Creates a new RoPE attention module.
    ///
    /// # Arguments
    ///
    /// * `context` - The WGPU context for creating GPU resources.
    /// * `hidden_size` - The model's hidden dimension.
    /// * `num_heads` - Number of query attention heads.
    /// * `num_kv_heads` - Number of key/value attention heads (for GQA).
    ///
    /// # Example
    ///
    /// ```rust
    /// // LLaMA 3.2 1B: 2048 hidden, 32 Q heads, 8 KV heads (GQA ratio 4:1)
    /// use kjarni_transformers::gpu_ops::{blocks::attention::GpuRoPEAttention};
    /// use kjarni_transformers::WgpuContext;
    /// let ctx = WgpuContext::new()?;
    /// let attn = GpuRoPEAttention::new(&ctx, 2048, 32, 8);
    ///
    /// // Mistral 7B: 4096 hidden, 32 Q heads, 8 KV heads
    /// let attn = GpuRoPEAttention::new(&ctx, 4096, 32, 8);
    ///
    /// // Phi-3: 3072 hidden, 32 Q heads, 32 KV heads (no GQA)
    /// let attn = GpuRoPEAttention::new(&ctx, 3072, 32, 32);
    /// ```
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        Self {
            ops: AttentionOps::new(context, hidden_size, num_heads, num_kv_heads),
            concatenate: GpuConcatenate::new(context),
            slice: GpuSlice::new(context),
            repeat_kv: GpuRepeatKV::new(context),
        }
    }

    /// Returns the underlying attention operations for advanced usage.
    pub fn ops(&self) -> &AttentionOps {
        &self.ops
    }

    /// Returns the number of query heads.
    pub fn num_heads(&self) -> u32 {
        self.ops.num_heads()
    }

    /// Returns the number of key/value heads.
    pub fn num_kv_heads(&self) -> u32 {
        self.ops.num_kv_heads()
    }

    /// Returns the GQA ratio (num_heads / num_kv_heads).
    pub fn gqa_ratio(&self) -> u32 {
        self.ops.num_heads() / self.ops.num_kv_heads()
    }

    /// Performs the forward pass of RoPE attention.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `hidden_states` - Input tensor of shape `[B, S_new, H]`.
    /// * `weights` - The Q, K, V, and O projection weights.
    /// * `rope` - The RoPE encoder for position embeddings.
    /// * `attention_mask` - Mask of shape `[B, S_total]` for padding/causal.
    /// * `cached_kv` - Optional cached (K, V) tensors in 4D format `[B, KV_H, S_cached, D]`.
    /// * `position_offset` - Position offset for RoPE and causal mask (= cache_len).
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// `RoPEAttentionOutput` containing:
    /// - `hidden_states`: Output tensor `[B, S_new, H]`
    /// - `new_k`: New K projection with RoPE `[B, S_new, KV_H*D]` for cache
    /// - `new_v`: New V projection `[B, S_new, KV_H*D]` for cache
    ///
    /// # Example
    ///
    /// ```rust
    /// use kjarni_transformers::gpu_ops::{blocks::attention::{GpuRoPEAttention, GpuAttentionWeights}, GpuTensor, GpuTensorPool};
    /// use kjarni_transformers::WgpuContext;
    /// let context = WgpuContext::new()?;
    /// let mut pool = GpuTensorPool::new();
    /// let mut enc = context.create_command_encoder();
    /// let attn = GpuRoPEAttention::new(&context, 2048, 32, 8);
    /// 
    /// // Prefill (no cache)
    /// let out = attn.forward(
    ///     &mut enc, &prompt_hidden, &weights, &rope, &mask,
    ///     None, 0, &mut pool
    /// )?;
    ///
    /// // Decode (with cache)
    /// let out = attn.forward(
    ///     &mut enc, &token_hidden, &weights, &rope, &mask,
    ///     Some((&cache_k, &cache_v)), cache_len, &mut pool
    /// )?;
    /// ```
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        attention_mask: &GpuTensor,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<RoPEAttentionOutput> {
        let (batch_size, query_len, _hidden_size) = hidden_states.dims3();
        let num_heads = self.ops.num_heads() as usize;
        let num_kv_heads = self.ops.num_kv_heads() as usize;
        let head_dim = self.ops.head_dim() as usize;

        // 1. Project Q, K, V
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

        // 2. Split heads
        // Q: [B, S, H] -> [B, num_heads, S, head_dim]
        // K, V: [B, S, KV_H] -> [B, num_kv_heads, S, head_dim]
        let q_heads = self.ops.split_heads(encoder, &q_proj, pool);
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);
        let v_heads = self.ops.split_heads(encoder, &v_proj, pool);

        // 3. Apply RoPE to Q and K
        let q_rotated = pool.get(q_heads.shape().to_vec());
        let k_rotated = pool.get(k_heads.shape().to_vec());

        rope.encode(encoder, &q_heads, &q_rotated, position_offset);
        rope.encode(encoder, &k_heads, &k_rotated, position_offset);

        // 4. Prepare K/V for cache return (3D format)
        // Merge heads back: [B, KV_H, S, D] -> [B, S, KV_H*D]
        let new_k_3d = self.ops.merge_heads(encoder, &k_rotated, pool);
        let new_v_3d = self.ops.merge_heads(encoder, &v_heads, pool);

        // 5. Concatenate with cache if present
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            let cache_len = cached_k.shape()[2]; // [B, H, S_cache, D]
            if cache_len > 0 {
                self.concat_with_cache(
                    encoder,
                    &k_rotated,
                    &v_heads,
                    cached_k,
                    cached_v,
                    cache_len,
                    pool,
                )?
            } else {
                (k_rotated.clone(), v_heads.clone())
            }
        } else {
            (k_rotated.clone(), v_heads.clone())
        };

        // 6. Apply GQA expansion if needed
        let (k_for_attn, v_for_attn) = if num_heads != num_kv_heads {
            self.expand_kv_for_gqa(encoder, &full_k, &full_v, pool)
        } else {
            (full_k, full_v)
        };

        // 7. Compute attention
        let context = self.compute_attention(
            encoder,
            &q_rotated,
            &k_for_attn,
            &v_for_attn,
            attention_mask,
            position_offset,
            pool,
        );

        // 8. Merge heads and output projection
        let context_merged = self.ops.merge_heads(encoder, &context, pool);
        let output = self.ops.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            pool,
        );

        Ok(RoPEAttentionOutput {
            hidden_states: output,
            new_k: new_k_3d,
            new_v: new_v_3d,
        })
    }

    /// Concatenates new K/V with cached K/V.
    fn concat_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k_new: &GpuTensor,    // [B, KV_H, S_new, D]
        v_new: &GpuTensor,    // [B, KV_H, S_new, D]
        cached_k: &GpuTensor, // [B, KV_H, S_cache, D]
        cached_v: &GpuTensor, // [B, KV_H, S_cache, D]
        cache_len: usize,
        pool: &mut GpuTensorPool,
    ) -> Result<(GpuTensor, GpuTensor)> {
        let (b, h, s_new, d) = k_new.dims4();
        let full_len = cache_len + s_new;

        // Slice valid portion from cache
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

    /// Expands K/V tensors for Grouped Query Attention.
    ///
    /// Repeats each KV head to match the number of Q heads.
    fn expand_kv_for_gqa(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        k: &GpuTensor, // [B, KV_H, S, D]
        v: &GpuTensor, // [B, KV_H, S, D]
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        let (batch, _kv_heads, seq_len, head_dim) = k.dims4();
        let num_heads = self.ops.num_heads() as usize;

        let expanded_shape = vec![batch, num_heads, seq_len, head_dim];

        let k_expanded = pool.get(expanded_shape.clone());
        let v_expanded = pool.get(expanded_shape);

        self.repeat_kv.encode(encoder, k, &k_expanded);
        self.repeat_kv.encode(encoder, v, &v_expanded);

        (k_expanded, v_expanded)
    }

    /// Computes the core attention: Q @ K^T, mask, softmax, @ V.
    fn compute_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuTensor,    // [B, Q_H, S_q, D]
        k: &GpuTensor,    // [B, Q_H, S_k, D] (already expanded for GQA)
        v: &GpuTensor,    // [B, Q_H, S_k, D] (already expanded for GQA)
        mask: &GpuTensor,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        // Q @ K^T
        let k_transposed = k.permute(encoder, self.ops.permute_kernel(), &[0, 1, 3, 2]);
        let scores = self.ops.bmm_4d(encoder, q, &k_transposed, pool);

        // Apply causal mask and softmax
        self.ops.apply_mask_and_softmax(
            encoder,
            &scores,
            Some(mask),
            true, // is_causal = true
            position_offset,
        );

        // Scores @ V
        self.ops.bmm_4d(encoder, &scores, v, pool)
    }

    /// Projects K and V only (useful for cache warming).
    ///
    /// Returns rotated K and raw V in 3D format for cache storage.
    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        weights: &GpuAttentionWeights,
        rope: &GpuRoPE,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> (GpuTensor, GpuTensor) {
        // Project
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

        // Split heads for K (to apply RoPE)
        let k_heads = self.ops.split_heads(encoder, &k_proj, pool);

        // Apply RoPE to K
        let k_rotated = pool.get(k_heads.shape().to_vec());
        rope.encode(encoder, &k_heads, &k_rotated, position_offset);

        // Merge back to 3D for cache storage
        let k_3d = self.ops.merge_heads(encoder, &k_rotated, pool);

        (k_3d, v_proj)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rope_attention_dimensions() {
        // LLaMA 3.2 1B
        let hidden_size = 2048u32;
        let num_heads = 32u32;
        let num_kv_heads = 8u32;
        let head_dim = hidden_size / num_heads;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(head_dim, 64);
        assert_eq!(gqa_ratio, 4);
    }

    #[test]
    fn test_no_gqa() {
        // Phi-3 style (no GQA)
        let num_heads = 32u32;
        let num_kv_heads = 32u32;
        let gqa_ratio = num_heads / num_kv_heads;

        assert_eq!(gqa_ratio, 1);
    }
}