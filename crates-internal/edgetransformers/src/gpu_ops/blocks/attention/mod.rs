mod fused;

use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::kernel::Kernel;
use crate::gpu_ops::primitives::layout::permute::GpuPermute;
use crate::gpu_ops::primitives::repeat_kv::GpuRepeatKV;
use crate::gpu_ops::primitives::{
    add_bias::GpuAddBias,
    apply_mask::GpuApplyMask,
    bmm::{BStrides, GpuBatchedMatMul},
    layout::reshape::GpuReshape,
    layout::unreshape::GpuUnreshape,
    matmul::GpuMatMul,
    softmax::GpuSoftmax,
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;

/// A simple memory pool for managing temporary, intermediate GPU tensors.
pub struct TempStorage {
    context: Arc<WgpuContext>,
    pool: HashMap<Vec<usize>, Vec<GpuTensor>>,
    active: Vec<GpuTensor>,
}

impl TempStorage {
    pub fn new(context: Arc<WgpuContext>) -> Self {
        Self {
            context,
            pool: HashMap::new(),
            active: Vec::new(),
        }
    }

    pub fn get(&mut self, shape: Vec<usize>) -> GpuTensor {
        let tensor = if let Some(tensors) = self.pool.get_mut(&shape) {
            if let Some(tensor) = tensors.pop() {
                tensor
            } else {
                GpuTensor::uninitialized(
                    &self.context,
                    shape,
                    crate::gpu_ops::DType::F32,
                    "TempTensor",
                )
            }
        } else {
            GpuTensor::uninitialized(
                &self.context,
                shape,
                crate::gpu_ops::DType::F32,
                "TempTensor",
            )
        };
        self.active.push(tensor.clone());
        tensor
    }

    pub fn reclaim(&mut self) {
        for tensor in self.active.drain(..) {
            self.pool
                .entry(tensor.shape().to_vec())
                .or_default()
                .push(tensor);
        }
    }
}

/// GPU tensors for attention weights.
pub struct GpuAttentionWeights {
    pub(crate) q_weight: GpuTensor,
    pub(crate) q_bias: GpuTensor,
    pub(crate) k_weight: GpuTensor,
    pub(crate) k_bias: GpuTensor,
    pub(crate) v_weight: GpuTensor,
    pub(crate) v_bias: GpuTensor,
    pub(crate) output_weight: GpuTensor,
    pub(crate) output_bias: GpuTensor,
}

impl GpuAttentionWeights {
    pub fn new(
        q_weight: GpuTensor,
        q_bias: GpuTensor,
        k_weight: GpuTensor,
        k_bias: GpuTensor,
        v_weight: GpuTensor,
        v_bias: GpuTensor,
        output_weight: GpuTensor,
        output_bias: GpuTensor,
    ) -> Result<Self> {
        // --- Assert Q dimensions ---
        assert_eq!(q_weight.rank(), 2, "Q weight must be 2D");
        assert_eq!(q_bias.rank(), 1, "Q bias must be 1D");
        assert_eq!(
            q_weight.shape()[1],
            q_bias.shape()[0],
            "Q weight's output dim must match its bias size"
        );

        // --- Assert K dimensions (GQA-aware) ---
        assert_eq!(k_weight.rank(), 2, "K weight must be 2D");
        assert_eq!(k_bias.rank(), 1, "K bias must be 1D");
        assert_eq!(
            k_weight.shape()[1],
            k_bias.shape()[0],
            "K weight's output dim must match its bias size"
        );

        // --- Assert V dimensions (GQA-aware) ---
        assert_eq!(v_weight.rank(), 2, "V weight must be 2D");
        assert_eq!(v_bias.rank(), 1, "V bias must be 1D");
        assert_eq!(
            v_weight.shape()[1],
            v_bias.shape()[0],
            "V weight's output dim must match its bias size"
        );

        // --- Assert Output dimensions ---
        assert_eq!(output_weight.rank(), 2, "Output weight must be 2D");
        assert_eq!(output_bias.rank(), 1, "Output bias must be 1D");
        assert_eq!(
            output_weight.shape()[1],
            output_bias.shape()[0],
            "Output weight's output dim must match its bias size"
        );

        // --- Assert consistency between weights ---

        // Input hidden dimension must be consistent across all input projections.
        let hidden_size = q_weight.shape()[0];
        assert_eq!(
            k_weight.shape()[0],
            hidden_size,
            "K weight input dim must match Q weight input dim"
        );
        assert_eq!(
            v_weight.shape()[0],
            hidden_size,
            "V weight input dim must match Q weight input dim"
        );

        // Output projection's input dimension must match the Q projection's output dimension.
        // In MHA, this is hidden_size. In GQA, it's still hidden_size because V heads are concatenated.
        assert_eq!(
            output_weight.shape()[0],
            q_weight.shape()[1],
            "Output projection input dim must match Q projection output dim"
        );

        // Output projection's output dimension must match the original hidden size.
        assert_eq!(
            output_weight.shape()[1],
            hidden_size,
            "Output projection output dim must match the model's hidden size"
        );

        Ok(Self {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            output_weight,
            output_bias,
        })
    }
}

pub struct GpuAttention {
    matmul: GpuMatMul,
    bmm: GpuBatchedMatMul,
    add_bias: GpuAddBias,
    reshape: GpuReshape,
    unreshape: GpuUnreshape,
    apply_mask: GpuApplyMask,
    softmax: GpuSoftmax,
    permute: GpuPermute,
    repeat_kv: GpuRepeatKV,
    gpu_concatenate: GpuConcatenate,
    slice_kernel: GpuSlice,
    num_heads: u32,
    num_kv_heads: u32,
    scale_factor: f32,
    head_dim: u32
}

impl GpuAttention {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: u32,
        num_heads: u32,
        num_kv_heads: u32,
    ) -> Self {
        let head_dim = hidden_size / num_heads;
        let scale_factor = 1.0 / (head_dim as f32).sqrt();

        Self {
            matmul: GpuMatMul::new(context),
            bmm: GpuBatchedMatMul::new(context),
            add_bias: GpuAddBias::new(context),
            reshape: GpuReshape::new(context),
            unreshape: GpuUnreshape::new(context),
            apply_mask: GpuApplyMask::new(context),
            softmax: GpuSoftmax::new(context),
            permute: GpuPermute::new(context),
            repeat_kv: GpuRepeatKV::new(context),
            gpu_concatenate: GpuConcatenate::new(context),
            slice_kernel: GpuSlice::new(context),
            num_heads,
            num_kv_heads,
            scale_factor,
            head_dim,
        }
    }
    pub fn llama_attention(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_heads: &GpuTensor,  // [B, H, S_q, D] - already rotated
        k_heads: &GpuTensor,  // [B, H, S_k, D] - already rotated
        v_heads: &GpuTensor,  // [B, H, S_v, D] - not rotated
        attention_mask: &GpuTensor,
        position_offset: usize,
        temp: &mut TempStorage,
        weights: &GpuAttentionWeights,
    ) -> GpuTensor {
        let (batch, _, seq_q, _) = q_heads.dims4();
        let seq_k = k_heads.shape()[2];

        // Handle GQA if needed
        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![
                batch,
                self.num_heads as usize,
                seq_k,
                self.head_dim as usize,
            ];

            let k_repeated = temp.get(expanded_shape.clone());
            let v_repeated = temp.get(expanded_shape);

            self.repeat_kv.encode(encoder, k_heads, &k_repeated);
            self.repeat_kv.encode(encoder, v_heads, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (k_heads.clone(), v_heads.clone())
        };

        // Standard attention computation
        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, q_heads, &k_transposed, temp);

        // Apply causal mask
        self.apply_mask.encode(encoder, &scores, attention_mask, true, position_offset as u32);

        // Softmax
        self.softmax.encode(encoder, &scores, self.scale_factor);

        // Context
        let context = self.bmm_4d(encoder, &scores, &v_expanded, temp);

        // Merge heads and output projection
        let context_merged = self.merge_heads(encoder, &context, temp);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        )
    }
    pub fn forward_with_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: Option<&GpuTensor>,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        is_causal: bool,
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        rope: Option<&GpuRoPE>,
        temp: &mut TempStorage,
    ) -> (GpuTensor, GpuTensor, GpuTensor) {
        let kv_source = key_value.unwrap_or(query);
        let (batch_size, query_len, hidden_size) = query.dims3();

        // Position offset from cache length
        let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[1]);

        // === 1. Project Q, K, V ===
        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
        let k_proj = self.project(encoder, kv_source, &weights.k_weight, &weights.k_bias, temp);
        let v_proj = self.project(encoder, kv_source, &weights.v_weight, &weights.v_bias, temp);

        // === 2. Apply RoPE to Q and K ===
        let (q_rotated, k_rotated) = if let Some(rope_encoder) = rope {
            // Split heads for RoPE
            let q_split = self.split_heads(encoder, &q_proj, temp);
            let k_split = self.split_heads(encoder, &k_proj, temp);

            let q_rope = temp.get(q_split.shape().to_vec());
            let k_rope = temp.get(k_split.shape().to_vec());

            // Apply RoPE with cache_len as position offset
            rope_encoder.encode(encoder, &q_split, &q_rope, cache_len);
            rope_encoder.encode(encoder, &k_split, &k_rope, cache_len);

            // Merge back to 3D
            let q_3d = self.merge_heads(encoder, &q_rope, temp);
            let k_3d = self.merge_heads(encoder, &k_rope, temp);
            (q_3d, k_3d)
        } else {
            (q_proj, k_proj)
        };

        // === 3. Concatenate with cache ===
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            let full_len = cache_len + query_len;
            let kv_hidden = k_rotated.shape()[2];

            let full_k_tensor = temp.get(vec![batch_size, full_len, kv_hidden]);
            let full_v_tensor = temp.get(vec![batch_size, full_len, kv_hidden]);

            self.gpu_concatenate.encode(encoder, &[cached_k, &k_rotated], &full_k_tensor, 1);
            self.gpu_concatenate.encode(encoder, &[cached_v, &v_proj], &full_v_tensor, 1);

            (full_k_tensor, full_v_tensor)
        } else {
            (k_rotated.clone(), v_proj.clone())
        };

        // === 4. Perform attention ===
        // Split heads for attention computation
        let q_heads = self.split_heads(encoder, &q_rotated, temp);
        let k_heads = self.split_heads(encoder, &full_k, temp);
        let v_heads = self.split_heads(encoder, &full_v, temp);

        // Handle GQA if needed
        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![
                batch_size,
                self.num_heads as usize,
                full_k.shape()[1],
                self.head_dim as usize,
            ];

            let k_repeated = temp.get(expanded_shape.clone());
            let v_repeated = temp.get(expanded_shape);

            self.repeat_kv.encode(encoder, &k_heads, &k_repeated);
            self.repeat_kv.encode(encoder, &v_heads, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (k_heads, v_heads)
        };

        // Attention scores
        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_heads, &k_transposed, temp);

        // Apply masks
        // if let Some(mask) = attention_mask {
        //     self.apply_mask.encode(encoder, &scores, mask, false, 0);
        // }
        // if is_causal {
        //     self.apply_mask.encode(encoder, &scores, &scores, true, cache_len as u32);
        // }
        if is_causal || attention_mask.is_some() {
            // If a specific padding mask is provided, use it.
            // Otherwise, for a purely causal mask, we can just pass the scores tensor
            // as a dummy, since the kernel only needs its shape to know the key_stride.
            let padding_mask = attention_mask.unwrap_or(&scores);

            self.apply_mask.encode(
                encoder,
                &scores,
                padding_mask,
                is_causal,
                cache_len as u32,
            );
        }

        // Softmax
        self.softmax.encode(encoder, &scores, self.scale_factor);

        // Context
        let context = self.bmm_4d(encoder, &scores, &v_expanded, temp);

        // Merge heads and output projection
        let context_merged = self.merge_heads(encoder, &context, temp);
        let output = self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        );

        // Return output and NEW K/V for cache
        // K is rotated (has RoPE), V is not
        (output, k_rotated, v_proj)
    }

    // Add this method to GpuAttention
    pub fn attend_cached(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        q_heads: &GpuTensor,     // Already split and rotated [B, H, S_q, D]
        cache_k: &GpuTensor,      // Cached K [B, H, S_full, D]
        cache_v: &GpuTensor,      // Cached V [B, H, S_full, D]
        attention_mask: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (batch, num_heads, _, head_dim) = q_heads.dims4();
        let key_len = cache_k.shape()[2];

        // Handle GQA if needed
        let (k_expanded, v_expanded) = if self.num_kv_heads != self.num_heads {
            let expanded_shape = vec![batch, self.num_heads as usize, key_len, head_dim];
            let k_repeated = temp.get(expanded_shape.clone());
            let v_repeated = temp.get(expanded_shape);

            self.repeat_kv.encode(encoder, cache_k, &k_repeated);
            self.repeat_kv.encode(encoder, cache_v, &v_repeated);

            (k_repeated, v_repeated)
        } else {
            (cache_k.clone(), cache_v.clone())
        };

        // Attention computation
        let k_transposed = k_expanded.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, q_heads, &k_transposed, temp);

        self.apply_mask.encode(encoder, &scores, attention_mask, true, position_offset as u32);
        self.softmax.encode(encoder, &scores, self.scale_factor);

        let context = self.bmm_4d(encoder, &scores, &v_expanded, temp);
        let context_merged = self.merge_heads(encoder, &context, temp);

        self.project(encoder, &context_merged, &weights.output_weight, &weights.output_bias, temp)
    }
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: Option<&GpuTensor>,
        weights: &GpuAttentionWeights,
        attention_mask: Option<&GpuTensor>,
        is_causal: bool,
        rope: Option<&GpuRoPE>,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (output, _, _) = self.forward_with_cache(
            encoder,
            query,
            key_value,
            weights,
            attention_mask,
            is_causal,
            None,
            rope,
            temp,
        );
        output
    }
    pub fn forward_seq2seq(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor, // The 3D input from the previous layer [B, S_q, H]
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor, // The full attention mask [B, S_total]
        cached_kv: Option<(&GpuTensor, &GpuTensor)>, // The 4D cache [B, H, S_cache, D]
        cache_len: usize,
        temp: &mut TempStorage,
    ) -> Result<(GpuTensor, GpuTensor, GpuTensor)> { // Returns (output, new_k_3d, new_v_3d)

        let (batch_size, query_len, _hidden_size) = query.dims3();
        // let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[2]);
        println!("!!! INSIDE forward_seq2seq. Query shape: {:?}", query.shape());
        println!("cache_len {}", cache_len);
        // === 1. Project Q, K, V (produces 3D tensors) ===
        // For self-attention, Q, K, and V are all projected from the same input `query`.
        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
        let k_proj = self.project(encoder, query, &weights.k_weight, &weights.k_bias, temp);
        let v_proj = self.project(encoder, query, &weights.v_weight, &weights.v_bias, temp);
        // Note: No RoPE for Seq2Seq models like BART.

        // === 2. Prepare Tensors for Attention (Split Heads) ===
        let q_heads = self.split_heads(encoder, &q_proj, temp); // [B, H, S_q, D]
        let k_heads_new = self.split_heads(encoder, &k_proj, temp); // [B, H, S_new, D]
        let v_heads_new = self.split_heads(encoder, &v_proj, temp); // [B, H, S_new, D]

        // === 3. Concatenate with cache (4D + 4D) ===
        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            // Only concatenate if the cache is not empty
            if cache_len > 0 {
                let (b, h, _, d) = k_heads_new.dims4();
                let full_len = cache_len + query_len;

                let slice_shape = &[b, h, cache_len, d];
                let slice_offset = &[0, 0, 0, 0];
                let valid_cache_k = cached_k.slice(encoder, &self.slice_kernel, slice_offset, slice_shape)?;
                let valid_cache_v = cached_v.slice(encoder, &self.slice_kernel, slice_offset, slice_shape)?;

                let full_k_tensor = temp.get(vec![b, h, full_len, d]);
                let full_v_tensor = temp.get(vec![b, h, full_len, d]);

                self.gpu_concatenate.encode(encoder, &[&valid_cache_k, &k_heads_new], &full_k_tensor, 2);
                self.gpu_concatenate.encode(encoder, &[&valid_cache_v, &v_heads_new], &full_v_tensor, 2);

                (full_k_tensor, full_v_tensor)
            } else {
                // Cache exists but is logically empty, so just use the new K/V
                (k_heads_new, v_heads_new)
            }
        } else {
            // No cache exists (prefill step)
            (k_heads_new, v_heads_new)
        };

        // === 4. Perform Attention Calculation ===
        // `full_k` and `full_v` are now guaranteed to be the complete 4D KV history.
        let k_transposed = full_k.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_heads, &k_transposed, temp);
        let expected_scores_shape = vec![
            q_heads.shape()[0],
            q_heads.shape()[1],
            q_heads.shape()[2],
            k_transposed.shape()[3],
        ];
        assert_eq!(scores.shape(), &expected_scores_shape, "Scores tensor has unexpected shape!");
        println!(
            "!!! INSIDE forward_seq2seq. Scores shape: {:?}, Mask shape: {:?}",
            scores.shape(),
            attention_mask.shape()
        );
        // Apply combined causal and padding mask.
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            true, // Self-attention is always causal
            cache_len as u32,
        );

        self.softmax.encode(encoder, &scores, self.scale_factor);
        let context = self.bmm_4d(encoder, &scores, &full_v, temp);

        // === 5. Final Projection ===
        let context_merged = self.merge_heads(encoder, &context, temp);
        let output = self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        );

        // Return the final 3D output, and the NEW 3D K/V tensors for the cache manager.
        Ok((output, k_proj, v_proj))
    }
    ///
    /// This function is now a stateless calculator. It assumes the caller (e.g., a TransformerLayer
    /// or a test) has already updated the KV cache with the results from the previous step.
    // pub fn forward(
    //     &self,
    //     encoder: &mut wgpu::CommandEncoder,
    //     query: &GpuTensor,
    //     key_value: &GpuTensor,
    //     weights: &GpuAttentionWeights,
    //     attention_mask: &GpuTensor,
    //     is_causal: bool,
    //     // The cache now contains the FULL, up-to-date K and V states for this step.
    //     cached_kv: Option<(&GpuTensor, &GpuTensor)>,
    //     position_offset: usize,
    //     temp: &mut TempStorage,
    // ) -> (GpuTensor, GpuTensor, GpuTensor) {
    //     let (batch_size, query_len, hidden_size) = query.dims3();
    //     let head_dim = hidden_size / self.num_heads as usize;
    //
    //     // === 1. Linear Projections ===
    //     // These are the NEW K/V values that will be returned for the *next* step's cache update.
    //     let q_biased = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
    //     let new_k = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, temp);
    //     let new_v = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, temp);
    //
    //     // === 2. Split Heads of the NEW projections ===
    //     // Note: We only split `q_biased` here. K/V are handled by the cache update kernel.
    //     let q_split = self.split_heads(encoder, &q_biased, temp);
    //
    //     // === 3. Determine which K/V tensors to use for the calculation ===
    //     let (k_for_attn, v_for_attn);
    //     let (new_k_split, new_v_split);
    //     if let Some((cache_k, cache_v)) = cached_kv {
    //         // If cache exists, point our attention tensors to the cache.
    //         k_for_attn = cache_k;
    //         v_for_attn = cache_v;
    //     } else {
    //         // If no cache, split the heads and point our attention tensors to the new, owned results.
    //         new_k_split = self.split_heads(encoder, &new_k, temp);
    //         new_v_split = self.split_heads(encoder, &new_v, temp);
    //         k_for_attn = &new_k_split;
    //         v_for_attn = &new_v_split;
    //     };
    //
    //     // Transpose the K tensor for BMM.
    //     let k_transposed = k_for_attn.permute(encoder, &self.permute, &[0, 1, 3, 2]);
    //
    //     // === 4. Compute Attention Scores (QKᵀ) ===
    //     let scores = self.bmm_4d(encoder, &q_split, &k_transposed, temp);
    //
    //     // === 5. Apply Masks ===
    //     self.apply_mask.encode(
    //         encoder,
    //         &scores,
    //         attention_mask,
    //         is_causal,
    //         position_offset as u32,
    //     );
    //
    //     // === 6. Softmax ===
    //     self.softmax.encode(encoder, &scores, self.scale_factor);
    //     let attention_weights = scores;
    //
    //     // === 7. Compute Context (Score-V) ===
    //     let context = self.bmm_4d(encoder, &attention_weights, v_for_attn, temp);
    //
    //     // === 8. Merge Heads & 9. Output Projection ===
    //     let context_merged = self.merge_heads(encoder, &context, temp);
    //     let output = self.project(
    //         encoder,
    //         &context_merged,
    //         &weights.output_weight,
    //         &weights.output_bias,
    //         temp,
    //     );
    //
    //     // === 10. Return ===
    //     // We return the raw, 3D `new_k` and `new_v` for the orchestrator to update the cache with.
    //     (output, new_k, new_v)
    // }

    // Fix the project_kv method in GpuAttention
    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        key_value: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        temp: &mut TempStorage,
        rope: Option<&GpuRoPE>,
    ) -> (GpuTensor, GpuTensor) {
        // Project K and V
        let new_k = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, temp);
        let new_v = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, temp);

        // Apply RoPE to K if provided (for LLaMA)
        let new_k_rotated = if let Some(rope_encoder) = rope {
            // Split heads for RoPE
            let k_split = self.split_heads(encoder, &new_k, temp);
            let k_rotated_4d = temp.get(k_split.shape().to_vec());
            let dummy_q_out = temp.get(k_split.shape().to_vec());
            rope_encoder.encode(encoder, &k_split, &k_rotated_4d, position_offset);
            // Merge back to 3D for cache storage
            self.merge_heads(encoder, &k_rotated_4d, temp)
        } else {
            new_k
        };

        // Return rotated K and original V
        (new_k_rotated, new_v)
    }

    pub fn attend_with_physical_cache(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        physical_k: &GpuTensor,
        physical_v: &GpuTensor,
        attention_mask: &GpuTensor,
        weights: &GpuAttentionWeights,
        position_offset: usize,
        temp: &mut TempStorage,
        rope: Option<&GpuRoPE>,
    ) -> GpuTensor {
        let (batch, query_len, _) = query.dims3();
        let (_, max_seq_len, head_dim) = physical_k.dims3();

        // --- 1. Prepare Q ---
        // 1a. Project Q
        let q_proj = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
        // 1b. Split heads of Q: [B, S_q, H] -> [B, H_q, S_q, D]
        let q_split = self.split_heads(encoder, &q_proj, temp);
        // 1c. Apply RoPE to the 4D split Q tensor
        let q_for_attn = if let Some(r) = rope {
            let out = temp.get(q_split.shape().to_vec());
            // The new encode function requires both Q and K. Since we only care about Q here,
            // we can pass q_split as the input for K and a temporary tensor as the output for K,
            // which we will then ignore.
            let dummy_k_out = temp.get(q_split.shape().to_vec());
            r.encode(encoder, &q_split, &out, position_offset);
            out
        } else {
            q_split
        };

        // --- 2. Prepare K and V (Handle GQA) ---
        let num_q_heads = self.num_heads as usize;
        let num_kv_heads = self.num_kv_heads as usize;

        let (k_for_attn, v_for_attn) = if num_q_heads != num_kv_heads {
            // Reshape physical K/V from 3D [B*H_kv, S_kv, D] to 4D [B, H_kv, S_kv, D]
            let physical_k_4d = physical_k.view(vec![batch, num_kv_heads, max_seq_len, head_dim]);
            let physical_v_4d = physical_v.view(vec![batch, num_kv_heads, max_seq_len, head_dim]);

            let k_repeated_4d = temp.get(vec![batch, num_q_heads, max_seq_len, head_dim]);
            let v_repeated_4d = temp.get(vec![batch, num_q_heads, max_seq_len, head_dim]);

            self.repeat_kv.encode(encoder, &physical_k_4d, &k_repeated_4d);
            self.repeat_kv.encode(encoder, &physical_v_4d, &v_repeated_4d);

            (k_repeated_4d, v_repeated_4d)
        } else {
            // No repeat needed, just reshape to 4D for consistency
            let physical_k_4d = physical_k.view(vec![batch, num_q_heads, max_seq_len, head_dim]);
            let physical_v_4d = physical_v.view(vec![batch, num_q_heads, max_seq_len, head_dim]);
            (physical_k_4d, physical_v_4d)
        };

        // --- 3. Attention Calculation ---
        // 3a. Q @ K^T
        let k_t = k_for_attn.permute(encoder, &self.permute, &[0, 1, 3, 2]);
        let scores = self.bmm_4d(encoder, &q_for_attn, &k_t, temp);

        // 3b. Apply mask
        self.apply_mask.encode(encoder, &scores, attention_mask, true, position_offset as u32);

        // 3c. Softmax
        self.softmax.encode(encoder, &scores, self.scale_factor);

        // 3d. Scores @ V
        let context = self.bmm_4d(encoder, &scores, &v_for_attn, temp);

        // --- 4. Output ---
        let context_merged = self.merge_heads(encoder, &context, temp);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        )
    }

    /// Performs the core attention calculation using the pre-updated KV cache.
    ///
    /// # Returns
    /// The final 3D output tensor of the attention block.
    pub fn attend(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor, // Input hidden states for this step [B, S_q, H]
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,
        is_causal: bool,
        // The cache now contains the FULL, up-to-date K and V states.
        // Shape: [B, H, S_total, D]
        kv_cache: (&GpuTensor, &GpuTensor),
        position_offset: usize,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (cache_k, cache_v) = kv_cache;

        // === 1. Project and Split Heads for Q ===
        // This is the only projection and split needed inside `attend`.
        let q_biased = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
        let q_split = self.split_heads(encoder, &q_biased, temp);

        // === 2. Transpose K for BMM (The Correct Way) ===
        // The K cache is already in the correct 4D head-split format [B, H, S, D].
        // To prepare it for batched matrix multiplication, we simply need to permute
        // the last two dimensions to get [B, H, D, S].
        // `GpuPermute` is the correct, efficient tool for this operation.
        let k_transposed = cache_k.permute(encoder, &self.permute, &[0, 1, 3, 2]);

        // === 3. Compute Attention Scores (QKᵀ) ===
        let scores = self.bmm_4d(encoder, &q_split, &k_transposed, temp);

        // === 4. Apply Masks & Softmax ===
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            is_causal,
            position_offset as u32,
        );
        self.softmax.encode(encoder, &scores, self.scale_factor);
        let attention_weights = scores;

        // === 5. Compute Context (Score-V) ===
        // The V cache is already in the correct [B, H, S, D] format.
        let context = self.bmm_4d(encoder, &attention_weights, cache_v, temp);

        // === 6. Merge Heads & Output Projection ===
        let context_merged = self.merge_heads(encoder, &context, temp);
        self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        )
    }

    /// A helper for the common `view -> matmul -> add_bias -> view` pattern.
    pub fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (b, s, h) = input.dims3();
        let out_features = weight.shape()[1];

        // Flatten for matmul
        let input_2d = input.view(vec![b * s, h]);
        let proj_2d = temp.get(vec![b * s, out_features]);

        self.matmul.encode(encoder, &[&input_2d, weight], &proj_2d);

        // Add bias if it exists (check for non-zero length)
        let output_2d = if bias.shape()[0] > 0 {
            let biased_2d = temp.get(vec![b * s, out_features]);
            self.add_bias.encode(encoder, &[&proj_2d, bias], &biased_2d);
            biased_2d
        } else {
            proj_2d
        };

        // Reshape back to 3D
        output_2d.view(vec![b, s, out_features])
    }

    /// Splits heads from [B, S, H*D] into [B, H, S, D] (or [B, H, D, S] if transposed).
    pub fn split_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        temp: &mut TempStorage,
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
        let output = temp.get(vec![b, num_heads, s, head_dim]);

        self.reshape.encode(encoder, input, &output, false);
        output
    }

    /// Merges heads from [B, H, S, D] back to [B, S, H*D].
    pub fn merge_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (b, h, s, d) = input.dims4();
        let output = temp.get(vec![b, s, h * d]);
        self.unreshape.encode(encoder, input, &output);
        output
    }

    /// A helper for the `view -> bmm -> view` pattern needed for 4D attention.
    fn bmm_4d(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        a: &GpuTensor, // [B, H, M, K]
        b: &GpuTensor, // [B, H, K, N]
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (b_size, h_size, m, k1) = a.dims4();
        let (_, _, k2, n) = b.dims4();
        assert_eq!(k1, k2, "Matrix dimensions are incompatible for BMM");

        let a_3d = a.view(vec![b_size * h_size, m, k1]);
        let b_3d = b.view(vec![b_size * h_size, k1, n]);

        let c_3d = temp.get(vec![b_size * h_size, m, n]);
        self.bmm.encode(encoder, &[&a_3d, &b_3d], &c_3d);

        c_3d.view(vec![b_size, h_size, m, n])
    }
    fn apply_padding_mask(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        scores: &GpuTensor,  // [B, H, S_q, S_k]
        mask: &GpuTensor,     // [B, S_k]
        temp: &mut TempStorage,
    ) {
        // The apply_mask kernel should handle broadcasting
        self.apply_mask.encode(encoder, scores, mask, false, 0);
    }
}
