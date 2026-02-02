//! Shared attention primitives.
//!
//! This module contains the low-level building blocks used by all attention
//! implementations. These operations are stateless and operate on tensors directly.

use crate::gpu_ops::primitives::{
    add_bias::GpuAddBias,
    bmm::GpuBatchedMatMul,
    linear::GpuLinearLayer,
    layout::permute::GpuPermute,
    layout::reshape::GpuReshape,
    layout::unreshape::GpuUnreshape,
    apply_mask::GpuApplyMask,
    softmax::GpuSoftmax,
};
use crate::gpu::{GpuTensor, GpuTensorPool, Kernel};
use crate::WgpuContext;
use std::sync::Arc;

/// Shared attention operations used by all attention variants.
pub struct AttentionOps {
    // Compute kernels
    linear: GpuLinearLayer,
    add_bias: GpuAddBias,
    bmm: GpuBatchedMatMul,
    softmax: GpuSoftmax,
    apply_mask: GpuApplyMask,

    // Layout kernels
    reshape: GpuReshape,
    unreshape: GpuUnreshape,
    permute: GpuPermute,

    // Configuration
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    scale_factor: f32,
}

impl AttentionOps {
    /// Creates a new `AttentionOps` instance.
    ///
    /// # Arguments
    ///
    /// * `context` - The WGPU context for creating GPU resources.
    /// * `hidden_size` - The model's hidden dimension (e.g., 1024 for BART-large).
    /// * `num_heads` - Number of attention heads for queries.
    /// * `num_kv_heads` - Number of attention heads for keys/values (same as num_heads for MHA).
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            linear: GpuLinearLayer::new(context),
            add_bias: GpuAddBias::new(context),
            bmm: GpuBatchedMatMul::new(context),
            softmax: GpuSoftmax::new(context),
            apply_mask: GpuApplyMask::new(context),
            reshape: GpuReshape::new(context),
            unreshape: GpuUnreshape::new(context),
            permute: GpuPermute::new(context),
            num_heads,
            num_kv_heads,
            head_dim,
            scale_factor,
        }
    }

    /// Returns the number of query attention heads.
    #[inline]
    pub fn num_heads(&self) -> u32 {
        self.num_heads
    }

    /// Returns the number of key/value attention heads.
    #[inline]
    pub fn num_kv_heads(&self) -> u32 {
        self.num_kv_heads
    }

    /// Returns the dimension of each attention head.
    #[inline]
    pub fn head_dim(&self) -> u32 {
        self.head_dim
    }

    /// Returns the attention scale factor (1 / sqrt(head_dim)).
    #[inline]
    pub fn scale_factor(&self) -> f32 {
        self.scale_factor
    }

    /// Returns a reference to the permute kernel for external use.
    #[inline]
    pub fn permute_kernel(&self) -> &GpuPermute {
        &self.permute
    }

    /// Projects input through a linear layer with bias.
    ///
    /// Computes: `output = input @ weight + bias`
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `input` - Input tensor of shape `[B, S, H]`.
    /// * `weight` - Weight matrix of shape `[H, O]` or `[O, H]` depending on layout.
    /// * `bias` - Bias vector of shape `[O]`.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, S, O]`.
    pub fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, s, h) = input.dims3();
        let (_, out_features) = weight.linear_layer_dims();

        // Flatten to 2D for matmul
        let input_2d = input.view(vec![b * s, h]);
        let proj_2d = pool.get(vec![b * s, out_features]);

        self.linear.encode(encoder, &input_2d, weight, &proj_2d);

        // Add bias if present
        let output_2d = if bias.shape()[0] > 0 {
            let biased_2d = pool.get(vec![b * s, out_features]);
            self.add_bias.encode(encoder, &[&proj_2d, bias], &biased_2d);
            biased_2d
        } else {
            proj_2d
        };

        // Reshape back to 3D
        output_2d.view(vec![b, s, out_features])
    }

    /// Splits heads from `[B, S, H*D]` into `[B, H, S, D]`.
    ///
    /// This reshapes the hidden dimension into separate attention heads,
    /// then permutes to put the head dimension before sequence length
    /// for efficient batched matrix multiplication.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `input` - Input tensor of shape `[B, S, H*D]`.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, H, S, D]`.
    pub fn split_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, s, total_dim) = input.dims3();

        // Determine number of heads based on dimension
        let num_heads = if total_dim == (self.num_heads * self.head_dim) as usize {
            self.num_heads as usize
        } else {
            // For K/V in GQA
            self.num_kv_heads as usize
        };

        let head_dim = total_dim / num_heads;
        let output = pool.get(vec![b, num_heads, s, head_dim]);

        self.reshape.encode(encoder, input, &output, false);
        output
    }

    /// Merges heads from `[B, H, S, D]` back to `[B, S, H*D]`.
    ///
    /// This is the inverse of `split_heads`, combining the attention head
    /// outputs back into a single hidden dimension.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `input` - Input tensor of shape `[B, H, S, D]`.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, S, H*D]`.
    pub fn merge_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b, h, s, d) = input.dims4();
        let output = pool.get(vec![b, s, h * d]);
        self.unreshape.encode(encoder, input, &output);
        output
    }

    /// Performs 4D batched matrix multiplication.
    ///
    /// Computes: `C[b,h,:,:] = A[b,h,:,:] @ B[b,h,:,:]`
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `a` - First input tensor of shape `[B, H, M, K]`.
    /// * `b` - Second input tensor of shape `[B, H, K, N]`.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[B, H, M, N]`.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions don't match (`a.shape[3] != b.shape[2]`).
    pub fn bmm_4d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuTensor,
        b: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        let (b_size, h_size, m, k1) = a.dims4();
        let (_, _, k2, n) = b.dims4();
        assert_eq!(k1, k2, "Matrix dimensions incompatible for BMM: {} vs {}", k1, k2);

        // Flatten batch and head dimensions
        let a_3d = a.view(vec![b_size * h_size, m, k1]);
        let b_3d = b.view(vec![b_size * h_size, k1, n]);

        let c_3d = pool.get(vec![b_size * h_size, m, n]);
        self.bmm.encode(encoder, &[&a_3d, &b_3d], &c_3d);

        // Unflatten back to 4D
        c_3d.view(vec![b_size, h_size, m, n])
    }

    /// Applies attention mask and softmax to attention scores.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `scores` - Attention scores of shape `[B, H, S_q, S_k]`.
    /// * `mask` - Attention mask of shape `[B, S_k]` or broadcastable.
    /// * `is_causal` - Whether to apply causal (autoregressive) masking.
    /// * `position_offset` - Offset for causal mask (used with KV cache).
    pub fn apply_mask_and_softmax(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scores: &GpuTensor,
        mask: Option<&GpuTensor>,
        is_causal: bool,
        position_offset: usize,
    ) {
        if is_causal || mask.is_some() {
            let mask_tensor = mask.unwrap_or(scores);
            let logical_key_len = (scores.shape()[3] + position_offset) as u32;

            self.apply_mask.encode(
                encoder,
                scores,
                mask_tensor,
                is_causal,
                position_offset as u32,
                logical_key_len,
            );
        }

        self.softmax.encode(encoder, scores, self.scale_factor);
    }

    /// Computes attention output from Q, K, V tensors.
    ///
    /// This is the core attention computation:
    /// 1. Compute scores: `Q @ K^T`
    /// 2. Apply mask and softmax
    /// 3. Compute context: `scores @ V`
    ///
    /// # Arguments
    ///
    /// * `encoder` - The command encoder for recording GPU operations.
    /// * `q` - Query tensor of shape `[B, H, S_q, D]`.
    /// * `k` - Key tensor of shape `[B, H, S_k, D]`.
    /// * `v` - Value tensor of shape `[B, H, S_k, D]`.
    /// * `mask` - Optional attention mask.
    /// * `is_causal` - Whether to apply causal masking.
    /// * `position_offset` - Offset for causal mask.
    /// * `pool` - Tensor pool for allocating intermediate tensors.
    ///
    /// # Returns
    ///
    /// Context tensor of shape `[B, H, S_q, D]`.
    pub fn attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q: &GpuTensor,
        k: &GpuTensor,
        v: &GpuTensor,
        mask: Option<&GpuTensor>,
        is_causal: bool,
        position_offset: usize,
        pool: &mut GpuTensorPool,
    ) -> GpuTensor {
        // Q @ K^T
        let k_transposed = k.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, q, &k_transposed, pool);

        // Mask and softmax
        self.apply_mask_and_softmax(encoder, &scores, mask, is_causal, position_offset);

        // Scores @ V
        self.bmm_4d(encoder, &scores, v, pool)
    }
}