use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::kernel::Kernel;
use crate::gpu_ops::primitives::layout::permute::GpuPermute;
use crate::gpu_ops::primitives::{
    add_bias::GpuAddBias, apply_mask::GpuApplyMask, bmm::GpuBatchedMatMul,
    layout::reshape::GpuReshape, layout::unreshape::GpuUnreshape, matmul::GpuMatMul,
    softmax::GpuSoftmax,
};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;

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
        // --- dimensional checks ---
        for (name, w, b) in [
            ("Q", &q_weight, &q_bias),
            ("K", &k_weight, &k_bias),
            ("V", &v_weight, &v_bias),
            ("Output", &output_weight, &output_bias),
        ] {
            assert_eq!(w.rank(), 2, "{name} weight must be 2D");
            assert_eq!(b.rank(), 1, "{name} bias must be 1D");
            assert_eq!(
                w.shape()[1],
                b.shape()[0],
                "{name} weight's output dim must match its bias size"
            );
        }

        // Check that Q, K, V weights all share the same input dimension
        assert_eq!(
            q_weight.shape()[0],
            k_weight.shape()[0],
            "Q and K must have same input dimension"
        );
        assert_eq!(
            q_weight.shape()[0],
            v_weight.shape()[0],
            "Q and V must have same input dimension"
        );

        // Ensure output_weight input dim matches Q/K/V output dim (projected dimension)
        assert_eq!(
            output_weight.shape()[0],
            q_weight.shape()[1] + k_weight.shape()[1] + v_weight.shape()[1],
            "Output projection input dim must equal concatenated QKV dim"
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
    num_heads: u32,
    scale_factor: f32,
}

impl GpuAttention {
    pub fn new(context: &Arc<WgpuContext>, hidden_size: u32, num_heads: u32) -> Self {
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
            num_heads,
            scale_factor,
        }
    }

    ///
    /// This function is now a stateless calculator. It assumes the caller (e.g., a TransformerLayer
    /// or a test) has already updated the KV cache with the results from the previous step.
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        query: &GpuTensor,
        key_value: &GpuTensor,
        weights: &GpuAttentionWeights,
        attention_mask: &GpuTensor,
        is_causal: bool,
        // The cache now contains the FULL, up-to-date K and V states for this step.
        cached_kv: Option<(&GpuTensor, &GpuTensor)>,
        position_offset: usize,
        temp: &mut TempStorage,
    ) -> (GpuTensor, GpuTensor, GpuTensor) {
        let (batch_size, query_len, hidden_size) = query.dims3();
        let head_dim = hidden_size / self.num_heads as usize;

        // === 1. Linear Projections ===
        // These are the NEW K/V values that will be returned for the *next* step's cache update.
        let q_biased = self.project(encoder, query, &weights.q_weight, &weights.q_bias, temp);
        let new_k = self.project(encoder, key_value, &weights.k_weight, &weights.k_bias, temp);
        let new_v = self.project(encoder, key_value, &weights.v_weight, &weights.v_bias, temp);

        // === 2. Split Heads of the NEW projections ===
        // Note: We only split `q_biased` here. K/V are handled by the cache update kernel.
        let q_split = self.split_heads(encoder, &q_biased, temp);

        // === 3. Determine which K/V tensors to use for the calculation ===
        let (k_for_attn, v_for_attn);
        let (new_k_split, new_v_split);
        if let Some((cache_k, cache_v)) = cached_kv {
            // If cache exists, point our attention tensors to the cache.
            k_for_attn = cache_k;
            v_for_attn = cache_v;
        } else {
            // If no cache, split the heads and point our attention tensors to the new, owned results.
            new_k_split = self.split_heads(encoder, &new_k, temp);
            new_v_split = self.split_heads(encoder, &new_v, temp);
            k_for_attn = &new_k_split;
            v_for_attn = &new_v_split;
        };

        // Transpose the K tensor for BMM.
        let k_transposed = k_for_attn.permute(encoder, &self.permute, &[0, 1, 3, 2]);

        // === 4. Compute Attention Scores (QKᵀ) ===
        let scores = self.bmm_4d(encoder, &q_split, &k_transposed, temp);

        // === 5. Apply Masks ===
        self.apply_mask.encode(
            encoder,
            &scores,
            attention_mask,
            is_causal,
            position_offset as u32,
        );

        // === 6. Softmax ===
        self.softmax.encode(encoder, &scores, self.scale_factor);
        let attention_weights = scores;

        // === 7. Compute Context (Score-V) ===
        let context = self.bmm_4d(encoder, &attention_weights, v_for_attn, temp);

        // === 8. Merge Heads & 9. Output Projection ===
        let context_merged = self.merge_heads(encoder, &context, temp);
        let output = self.project(
            encoder,
            &context_merged,
            &weights.output_weight,
            &weights.output_bias,
            temp,
        );

        // === 10. Return ===
        // We return the raw, 3D `new_k` and `new_v` for the orchestrator to update the cache with.
        (output, new_k, new_v)
    }

    /// Computes the new K and V tensors from the input hidden states.
    /// This is the first part of the attention mechanism, separated to allow for
    /// the "update-then-calculate" cache pattern.
    ///
    /// # Returns
    /// A tuple of `(new_k, new_v)`, which are the raw, 3D projection outputs.
    pub fn project_kv(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        key_value: &GpuTensor, // Source for K/V [B, S, H]
        weights: &GpuAttentionWeights,
        temp: &mut TempStorage,
    ) -> (GpuTensor, GpuTensor) {
        let (b, s, h) = key_value.dims3();
        let input_2d = key_value.view(vec![b * s, h]);

        // Project K
        let k_proj_2d = temp.get(vec![b * s, h]);
        self.matmul
            .encode(encoder, &[&input_2d, &weights.k_weight], &k_proj_2d);
        let new_k_2d = temp.get(vec![b * s, h]);
        self.add_bias
            .encode(encoder, &[&k_proj_2d, &weights.k_bias], &new_k_2d);
        let new_k = new_k_2d.view(vec![b, s, h]);

        // Project V
        let v_proj_2d = temp.get(vec![b * s, h]);
        self.matmul
            .encode(encoder, &[&input_2d, &weights.v_weight], &v_proj_2d);
        let new_v_2d = temp.get(vec![b * s, h]);
        self.add_bias
            .encode(encoder, &[&v_proj_2d, &weights.v_bias], &new_v_2d);
        let new_v = new_v_2d.view(vec![b, s, h]);

        (new_k, new_v)
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
    fn project(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        weight: &GpuTensor,
        bias: &GpuTensor,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (b, s, h) = input.dims3();
        let input_2d = input.view(vec![b * s, h]);

        let proj_2d = temp.get(vec![b * s, h]);
        self.matmul.encode(encoder, &[&input_2d, weight], &proj_2d);

        let biased_2d = temp.get(vec![b * s, h]);
        self.add_bias.encode(encoder, &[&proj_2d, bias], &biased_2d);

        biased_2d.view(vec![b, s, h])
    }

    /// Splits heads from [B, S, H*D] into [B, H, S, D] (or [B, H, D, S] if transposed).
    pub fn split_heads(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuTensor,
        temp: &mut TempStorage,
    ) -> GpuTensor {
        let (b, s, _) = input.dims3();
        let h = self.num_heads as usize;
        let d = input.shape()[2] / h;
        let output = temp.get(vec![b, h, s, d]);
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
}
