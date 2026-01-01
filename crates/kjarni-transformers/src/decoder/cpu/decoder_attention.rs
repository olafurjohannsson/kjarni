use crate::linear_layer::LinearLayer;
use crate::rope::RoPE;
use crate::utils::linear_algebra::softmax_inplace;
use anyhow::Result;
use ndarray::{Array2, Array3, Array4, Axis, s};
use std::time::{Duration, Instant};

/// Optimized Attention for modern Decoders (Llama, Phi, Qwen, Mistral).
/// Features: Separate QKV, RoPE, GQA, BF16 support via LinearLayer.
pub struct DecoderAttention {
    pub q_proj: LinearLayer,
    pub k_proj: LinearLayer,
    pub v_proj: LinearLayer,
    pub o_proj: LinearLayer,

    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub scale_factor: f32,
}

impl DecoderAttention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        q: LinearLayer,
        k: LinearLayer,
        v: LinearLayer,
        o: LinearLayer,
        num_kv_heads: Option<usize>,
    ) -> Self {
        let num_kv_heads = num_kv_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        Self {
            q_proj: q,
            k_proj: k,
            v_proj: v,
            o_proj: o,
            num_heads,
            num_kv_heads,
            head_dim,
            scale_factor: 1.0 / (head_dim as f32).sqrt(),
        }
    }

    
pub fn forward_2(
    &self,
    hidden_states: &Array3<f32>,
    attention_mask: Option<&Array2<f32>>,
    past_k: ndarray::ArrayView3<f32>,
    past_v: ndarray::ArrayView3<f32>,
    mut dest_k: ndarray::ArrayViewMut3<f32>,
    mut dest_v: ndarray::ArrayViewMut3<f32>,
    rope: Option<&RoPE>,
) -> Result<Array3<f32>> {
    let (batch_size, seq_len, _) = hidden_states.dim();
    let is_decode_step = seq_len == 1;

    // 1. QKV Projection
    let hidden_2d = hidden_states
        .view()
        .into_shape_with_order((batch_size * seq_len, self.num_heads * self.head_dim))?;

    let q = self.q_proj.matmul(&hidden_2d);
    let k = self.k_proj.matmul(&hidden_2d);
    let v = self.v_proj.matmul(&hidden_2d);

    let q_3d = q.into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?;
    let k_3d = k.into_shape_with_order((batch_size, seq_len, self.num_kv_heads * self.head_dim))?;
    let v_3d = v.into_shape_with_order((batch_size, seq_len, self.num_kv_heads * self.head_dim))?;

    // 2. Apply Rotary Positional Embedding (RoPE)
    let cache_len = past_k.shape()[1];
    let (q_rope, k_rope) = if let Some(r) = rope {
        r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_kv_heads, cache_len)?
    } else {
        (q_3d, k_3d)
    };

    // 3. ZERO-COPY CACHE WRITE
    // Write the newly computed K and V tensors directly into their destination slices in the cache.
    // This is the primary optimization, as it avoids allocating and returning these tensors.
    dest_k.assign(&k_rope);
    dest_v.assign(&v_3d);

    // 4. PREPARE UNIFIED K/V FOR ATTENTION
    // **CRITICAL FIX**: Instead of computing on separate history/current views, we create
    // temporary, contiguous arrays representing the *full* sequence (history + current).
    // This ensures the memory layout is correct for the high-performance matmul operations.
    // While `concatenate!` involves a copy, it's a smaller, localized copy than what the
    // original working code did, thus still providing a significant performance gain.
    let full_k = ndarray::concatenate![Axis(1), past_k, k_rope.view()];
    let full_v = ndarray::concatenate![Axis(1), past_v, v_3d.view()];

    // 5. Prepare Q for Attention Heads
    let q_heads = q_rope
        .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
        .permuted_axes([0, 2, 1, 3])
        .to_owned(); // .to_owned() is cheap for seq_len=1 and ensures standard layout

    let n_rep = self.num_heads / self.num_kv_heads;

    // 6. Compute Attention Scores (Q @ K^T)
    // The logic is now simplified to use the unified `full_k` and `full_v`,
    // just like in the original working code.
    let mut scores = if is_decode_step {
        // --- FAST PATH (Decoding) ---
        let k_view = full_k
            .view()
            .into_shape_with_order((
                batch_size,
                full_k.shape()[1],
                self.num_kv_heads,
                self.head_dim,
            ))?
            .permuted_axes([0, 2, 3, 1]); // Permute for matmul: [B, KV_H, D, T]

        crate::utils::linear_algebra::matmul_4d_decode_gqa(&q_heads, &k_view, n_rep)
    } else {
        // --- SLOW PATH (Prefill) ---
        let k_heads = self.prepare_kv_heads(&full_k, batch_size)?;
        let k_t = k_heads.permuted_axes([0, 1, 3, 2]);
        crate::utils::linear_algebra::matmul_4d(&q_heads, &k_t)
    };

    // 7. Apply Scaling, Masking, and Softmax
    scores.mapv_inplace(|x| x * self.scale_factor);
    if let Some(mask) = attention_mask {
        scores = crate::utils::linear_algebra::apply_attention_mask(scores, mask);
    }
    // The causal mask offset `cache_len` ensures we don't attend to future tokens
    self.apply_causal_mask(&mut scores, cache_len);
    crate::utils::linear_algebra::softmax_inplace(&mut scores);

    // 8. Compute Context Vector (Scores @ V)
    let context = if is_decode_step {
        // --- FAST PATH (Decoding) ---
        let v_view = full_v
            .view()
            .into_shape_with_order((
                batch_size,
                full_v.shape()[1],
                self.num_kv_heads,
                self.head_dim,
            ))?
            .permuted_axes([0, 2, 1, 3]); // [B, KV_H, T, D]

        crate::utils::linear_algebra::matmul_4d_context_gqa(&scores, &v_view, n_rep)
    } else {
        // --- SLOW PATH (Prefill) ---
        let v_heads = self.prepare_kv_heads(&full_v, batch_size)?;
        crate::utils::linear_algebra::matmul_4d(&scores, &v_heads)
    };

    // 9. Final Output Projection
    let context_flat = context
        .permuted_axes([0, 2, 1, 3])
        .as_standard_layout()
        .into_shape_with_order((batch_size * seq_len, self.num_heads * self.head_dim))?
        .to_owned();

    let output = self.o_proj.matmul(&context_flat.view());
    let output_3d =
        output.into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?;

    Ok(output_3d)
}
    pub fn forward(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: Option<&Array2<f32>>,
        cached_kv: Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)>,
        rope: Option<&RoPE>,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let is_decode_step = hidden_states.shape()[1] == 1;
        let t_total = Instant::now();
        let (batch_size, seq_len, _) = hidden_states.dim();

        // 1. Flatten Input for LinearLayer [Batch*Seq, Hidden]
        let hidden_2d = hidden_states
            .view()
            .into_shape_with_order((batch_size * seq_len, self.num_heads * self.head_dim))?;

        let t_qkv = Instant::now();
        // 2. Project Q, K, V (BF16 OPTIMIZED KERNEL)
        let q = self.q_proj.matmul(&hidden_2d);
        let k = self.k_proj.matmul(&hidden_2d);
        let v = self.v_proj.matmul(&hidden_2d);

        // 3. Reshape [Batch, Seq, Dim]
        let q_3d =
            q.into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?;
        let k_3d =
            k.into_shape_with_order((batch_size, seq_len, self.num_kv_heads * self.head_dim))?;
        let v_3d =
            v.into_shape_with_order((batch_size, seq_len, self.num_kv_heads * self.head_dim))?;
        let d_qkv = t_qkv.elapsed();

        let t_rope = Instant::now();
        // 4. RoPE
        let cache_len = cached_kv.map_or(0, |(k, _)| k.shape()[1]);
        let (q_rope, k_rope) = if let Some(r) = rope {
            r.apply_3d(&q_3d, &k_3d, self.num_heads, self.num_kv_heads, cache_len)?
        } else {
            (q_3d, k_3d)
        };

        let d_rope = t_rope.elapsed();

        let t_cache = Instant::now();

        let (full_k, full_v) = if let Some((cached_k, cached_v)) = cached_kv {
            // Note: Standard layout required here for cache update efficiency usually
            let full_k = ndarray::concatenate![Axis(1), cached_k, k_rope.view()]
                .as_standard_layout()
                .to_owned();
            let full_v = ndarray::concatenate![Axis(1), cached_v, v_3d.view()]
                .as_standard_layout()
                .to_owned();
            (full_k, full_v)
        } else {
            (k_rope.clone(), v_3d.clone())
        };

        // 6. Reshape Q for Attention [Batch, Heads, Seq, Dim]
        // This is small (seq=1 usually), so .to_owned() is cheap.
        let q_heads = q_rope
            .into_shape_with_order((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        let d_cache = t_cache.elapsed();
        let t_scores = Instant::now();
        // ---------------------------------------------------------------------
        // CRITICAL OPTIMIZATION: GQA WITHOUT COPYING
        // ---------------------------------------------------------------------
        let n_rep = self.num_heads / self.num_kv_heads;

        let mut scores = if seq_len == 1 {
            // --- FAST PATH (Decoding) ---
            // 1. Create a View of K [Batch, Seq, KV_Heads, Dim]
            // 2. Permute to [Batch, KV_Heads, Dim, Seq] (Transposed for Dot Product)
            // NO MEMORY COPY HERE.
            let k_view = full_k
                .view()
                .into_shape_with_order((
                    batch_size,
                    full_k.shape()[1],
                    self.num_kv_heads,
                    self.head_dim,
                ))?
                .permuted_axes([0, 2, 3, 1]);

            // Call GQA-aware matmul
            crate::utils::linear_algebra::matmul_4d_decode_gqa(&q_heads, &k_view, n_rep)
        } else {
            // --- SLOW PATH (Prefill) ---
            // Fallback to expanding heads physically in memory
            let k_heads = self.prepare_kv_heads(&full_k, batch_size)?;
            let k_t = k_heads.permuted_axes([0, 1, 3, 2]);
            crate::utils::linear_algebra::matmul_4d(&q_heads, &k_t)
        };

        let d_scores = t_scores.elapsed();

        let t_softmax = Instant::now();
        // Scaling
        scores.mapv_inplace(|x| x * self.scale_factor);

        if let Some(mask) = attention_mask {
            scores = crate::utils::linear_algebra::apply_attention_mask(scores, mask);
        }

        self.apply_causal_mask(&mut scores, cache_len);

        // Softmax
        softmax_inplace(&mut scores);

        let d_softmax = t_softmax.elapsed();
        let t_context = Instant::now();

        // 8. Output Context
        let context = if seq_len == 1 {
            // --- FAST PATH (Decoding) ---
            // View V: [Batch, Seq, KV_Heads, Dim] -> [Batch, KV_Heads, Seq, Dim]
            // NO MEMORY COPY.
            let v_view = full_v
                .view()
                .into_shape_with_order((
                    batch_size,
                    full_v.shape()[1],
                    self.num_kv_heads,
                    self.head_dim,
                ))?
                .permuted_axes([0, 2, 1, 3]);

            crate::utils::linear_algebra::matmul_4d_context_gqa(&scores, &v_view, n_rep)
        } else {
            // --- SLOW PATH (Prefill) ---
            let v_heads = self.prepare_kv_heads(&full_v, batch_size)?;
            crate::utils::linear_algebra::matmul_4d(&scores, &v_heads)
        };
        let d_context = t_context.elapsed();
        let t_o_proj = Instant::now();
        let context_flat = context
            .permuted_axes([0, 2, 1, 3])
            .as_standard_layout()
            .into_shape_with_order((batch_size * seq_len, self.num_heads * self.head_dim))?
            .to_owned();

        // 9. Output Projection
        let output = self.o_proj.matmul(&context_flat.view());
        let output_3d =
            output.into_shape_with_order((batch_size, seq_len, self.num_heads * self.head_dim))?;
        let d_o_proj = t_o_proj.elapsed();

        let d_total = t_total.elapsed();
        if is_decode_step {
            log::info!(
                "[Attention Perf] Total: {:?}, QKV Proj: {:?}, RoPE: {:?}, Cache: {:?}, Scores (Q@K): {:?}, Softmax: {:?}, Context (S@V): {:?}, Out Proj: {:?}",
                d_total,
                d_qkv,
                d_rope,
                d_cache,
                d_scores,
                d_softmax,
                d_context,
                d_o_proj
            );
        } else {
            log::info!(
                "[PREFILL Attention Perf] Total: {:?}, QKV Proj: {:?}, RoPE: {:?}, Cache: {:?}, Scores (Q@K): {:?}, Softmax: {:?}, Context (S@V): {:?}, Out Proj: {:?}",
                d_total,
                d_qkv,
                d_rope,
                d_cache,
                d_scores,
                d_softmax,
                d_context,
                d_o_proj
            );
        }

        Ok((output_3d, k_rope, v_3d))
    }

    // --- Helpers ---

    fn prepare_kv_heads(&self, kv: &Array3<f32>, batch: usize) -> Result<Array4<f32>> {
        let total_seq = kv.shape()[1];
        let kv_heads = kv
            .view()
            .into_shape_with_order((batch, total_seq, self.num_kv_heads, self.head_dim))?
            .permuted_axes([0, 2, 1, 3]); // [B, KV_H, T, D]

        if self.num_heads == self.num_kv_heads {
            Ok(kv_heads.to_owned())
        } else {
            // Repeat Logic for GQA
            let groups = self.num_heads / self.num_kv_heads;
            let mut out = Array4::zeros((batch, self.num_heads, total_seq, self.head_dim));
            for i in 0..self.num_kv_heads {
                for g in 0..groups {
                    out.slice_mut(s![.., i * groups + g, .., ..])
                        .assign(&kv_heads.slice(s![.., i, .., ..]));
                }
            }
            Ok(out)
        }
    }

    fn apply_causal_mask(&self, scores: &mut Array4<f32>, cache_len: usize) {
        let (_, _, seq_q, _) = scores.dim();
        for i in 0..seq_q {
            let query_pos = cache_len + i;
            for j in 0..scores.shape()[3] {
                if j > query_pos {
                    scores.slice_mut(s![.., .., i, j]).fill(f32::NEG_INFINITY);
                }
            }
        }
    }
}
