use crate::gpu_context::WgpuContext;
use crate::gpu_ops::primitives::{
    add_bias::GpuAddBias,
    apply_mask::GpuApplyMask,
    matmul::GpuMatmul,
    bmm::GpuBatchedMatmul,
    softmax::GpuSoftmax,
    // You will also need reshape/transpose primitives
    // transpose::GpuTranspose,
    // reshape::GpuReshape,
};
use crate::gpu_ops::GpuTensor;
use std::sync::Arc;

// NOTE: You will need GPU primitives for reshape/transpose.
// For now, we will assume they exist and have an `encode` method.

/// GPU buffers for attention weights, mirroring the CPU struct.
pub struct GpuAttentionWeights {
    pub q_weight: GpuTensor,
    pub q_bias: GpuTensor,
    pub k_weight: GpuTensor,
    pub k_bias: GpuTensor,
    pub v_weight: GpuTensor,
    pub v_bias: GpuTensor,
    pub output_weight: GpuTensor,
    pub output_bias: GpuTensor,
}

/// A reusable, configurable GPU Multi-Head Attention block.
pub struct GpuAttention {
    // GPU compute primitives
    matmul: GpuMatmul,
    bmm: GpuBatchedMatmul,
    add_bias: GpuAddBias,
    apply_mask: GpuApplyMask,
    softmax: GpuSoftmax,
    // You will need to add your reshape/transpose primitives here
    // reshape: GpuReshape,
    // transpose: GpuTranspose,

    // Configuration
    num_heads: u32,
    head_dim: u32,
    scale_factor: f32,
    context: Arc<WgpuContext>,
}

impl GpuAttention {
    pub fn new(context: Arc<WgpuContext>, hidden_size: u32, num_heads: u32) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            matmul: GpuMatmul::new(&context),
            bmm: GpuBatchedMatmul::new(&context),
            add_bias: GpuAddBias::new(&context),
            apply_mask: GpuApplyMask::new(&context),
            softmax: GpuSoftmax::new(&context),
            // Initialize other primitives here
            num_heads,
            head_dim,
            scale_factor,
            context,
        }
    }

    /// GPU equivalent of the CPU `forward_with_cache` method.
    ///
    /// This function is stateless and relies on a caller to manage temporary buffers
    /// and the state of the KV cache.
    ///
    /// # Returns
    /// A tuple of `(output, new_k, new_v)`, where `new_k` and `new_v` are the
    /// computed key/value tensors for the current input only.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,                      // Input hidden states [B, S_q, H]
        key_value: &GpuTensor,                  // For cross-attention or self-attention [B, S_k, H]
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,             // Padding mask [B, S_total]
        is_causal: bool,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>, // Past K,V from GpuKVCache
        position_offset: usize,                 // The length of the sequence in the cache
        // A helper to manage allocation of temporary tensors for intermediate results
        temp_storage: &mut TempStorage,
    ) -> (GpuTensor, GpuTensor, GpuTensor) {
        let (batch_size, query_len, hidden_size) = query.dims3();

        // === 1. Linear Projections ===
        // Q, K, V are calculated for the *new inputs only*.
        
        // Q = query @ Wq + bq
        let q_proj = self.matmul.run(encoder, query, &weights.q_weight, temp_storage);
        let q_biased = self.add_bias.run(encoder, &q_proj, &weights.q_bias, temp_storage);

        // K = key_value @ Wk + bk
        let k_proj = self.matmul.run(encoder, key_value, &weights.k_weight, temp_storage);
        let new_k = self.add_bias.run(encoder, &k_proj, &weights.k_bias, temp_storage);

        // V = key_value @ Wv + bv
        let v_proj = self.matmul.run(encoder, key_value, &weights.v_weight, temp_storage);
        let new_v = self.add_bias.run(encoder, &v_proj, &weights.v_bias, temp_storage);

        // === 2. Combine with Cached K, V for Attention Calculation ===
        // This is where the logic diverges from the simple encoder.
        // We do NOT concatenate. We use the provided full cache.
        let (k_for_attn, v_for_attn) = if let Some((cache_k, cache_v)) = cached_kv {
            // For decoding, we attend to the full cache.
            // The caller must have already copied `new_k` and `new_v` into the cache.
            (cache_k, cache_v)
        } else {
            // For encoders, the "cache" is just the current keys/values.
            (&new_k, &new_v)
        };
        let key_len_for_attn = k_for_attn.shape()[1];

        // === 3. Reshape and Transpose for Multi-Head Attention ===
        // Q: [B, S_q, H] -> [B, N, S_q, D]
        let q_reshaped = self.reshape_and_transpose_for_bmm(encoder, &q_biased, temp_storage);
        // K: [B, S_k, H] -> [B, N, S_k, D] -> Transposed for BMM -> [B, N, D, S_k]
        let k_reshaped_t = self.reshape_and_transpose_for_bmm_T(encoder, k_for_attn, temp_storage);
        // V: [B, S_k, H] -> [B, N, S_k, D]
        let v_reshaped = self.reshape_and_transpose_for_bmm(encoder, v_for_attn, temp_storage);

        // === 4. Compute Attention Scores: Q @ K^T ===
        let scores = self.bmm.run(encoder, &q_reshaped, &k_reshaped_t, temp_storage);
        // Note: Scaling can be done in a dedicated kernel or inside softmax.
        
        // === 5. Apply Masks ===
        // This uses our perfected, universal masking kernel.
        self.apply_mask.encode(encoder, &scores, attention_mask, is_causal, position_offset as u32);

        // === 6. Softmax ===
        self.softmax.encode(encoder, &scores, self.scale_factor);
        let attention_weights = scores; // The scores tensor is now the weights tensor

        // === 7. Apply Attention to Values ===
        let context = self.bmm.run(encoder, &attention_weights, &v_reshaped, temp_storage);
        
        // === 8. Reshape Back & 9. Output Projection ===
        let context_unreshaped = self.unreshape(encoder, &context, temp_storage);
        let output_proj = self.matmul.run(encoder, &context_unreshaped, &weights.output_weight, temp_storage);
        let output = self.add_bias.run(encoder, &output_proj, &weights.output_bias, temp_storage);

        // === 10. Return (output, new_k, new_v) ===
        // The caller is responsible for copying new_k and new_v into the cache.
        (output, new_k, new_v)
    }

    // Placeholder for reshape/transpose helpers you will need
    fn reshape_and_transpose_for_bmm(&self, /*...*/) -> GpuTensor { /* ... */ }
    fn reshape_and_transpose_for_bmm_T(&self, /*...*/) -> GpuTensor { /* ... */ }
    fn unreshape(&self, /*...*/) -> GpuTensor { /* ... */ }
}

// Dummy struct for managing temporary tensor allocations
pub struct TempStorage {}
impl TempStorage {
    // In a real implementation, this would manage a pool of reusable buffers
    fn get_temp_tensor(&mut self, shape: Vec<usize>) -> GpuTensor {
        // ...
    }
}