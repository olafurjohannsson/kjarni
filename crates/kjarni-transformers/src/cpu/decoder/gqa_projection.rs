//! Grouped Query Attention (GQA) Projection
//!
//! For GQA models (Llama 2/3, Qwen, Mistral), Q has different output size than K/V:
//! - Q: [hidden_size] -> [num_heads * head_dim]  
//! - K: [hidden_size] -> [num_kv_heads * head_dim]
//! - V: [hidden_size] -> [num_kv_heads * head_dim]
//!
//! Since K and V have the same dimensions, we fuse them into a single
//! matmul, reducing 3 matmuls to 2.
//!
//! # Performance
//!
//! | Approach     | Matmuls | Kernel Launches |
//! |--------------|---------|-----------------|
//! | Separate Q/K/V | 3       | 3               |
//! | Q + fused KV   | 2       | 2               |
//!
//! Expected speedup: ~15-25% on QKV projection

use anyhow::Result;
use ndarray::{s, Array1, Array2, ArrayView2, ArrayViewMut2};

use crate::linear_layer::LinearLayer;

/// GQA-aware projection that keeps Q separate and fuses K+V.
///
/// This is optimal for GQA where:
/// - Q output: num_heads * head_dim (typically 4096)
/// - K output: num_kv_heads * head_dim (typically 1024 for 8 KV heads)
/// - V output: num_kv_heads * head_dim (same as K)
pub struct GQAProjection {
    /// Query projection: [hidden_size] -> [num_heads * head_dim]
    q_proj: LinearLayer,
    /// Fused Key+Value projection: [hidden_size] -> [2 * num_kv_heads * head_dim]
    kv_proj: LinearLayer,
    /// Output dimension for K (and V): num_kv_heads * head_dim
    kv_dim: usize,
    /// Output dimension for Q: num_heads * head_dim
    q_dim: usize,
}

impl GQAProjection {
    /// Creates a new GQA projection by fusing K and V weights.
    ///
    /// # Arguments
    ///
    /// * `q` - Query projection `[q_dim, hidden_size]`
    /// * `k` - Key projection `[kv_dim, hidden_size]`
    /// * `v` - Value projection `[kv_dim, hidden_size]`
    ///
    /// # Panics
    ///
    /// If K and V have different dimensions.
    pub fn new(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        assert_eq!(
            k.out_features(),
            v.out_features(),
            "K and V must have same output dimension, got {} vs {}",
            k.out_features(),
            v.out_features()
        );
        assert_eq!(
            k.in_features(),
            v.in_features(),
            "K and V must have same input dimension"
        );
        assert_eq!(
            q.in_features(),
            k.in_features(),
            "Q, K, V must have same input dimension (hidden_size)"
        );

        let kv_dim = k.out_features();
        let q_dim = q.out_features();

        let kv_proj = Self::fuse_kv_weights(&k, &v);

        Self {
            q_proj: q,
            kv_proj,
            kv_dim,
            q_dim,
        }
    }

    /// Forward pass returning (Q, K, V).
    ///
    /// # Arguments
    ///
    /// * `hidden` - Input tensor `[tokens, hidden_size]`
    ///
    /// # Returns
    ///
    /// Tuple of (Q, K, V):
    /// - Q: `[tokens, q_dim]` = `[tokens, num_heads * head_dim]`
    /// - K: `[tokens, kv_dim]` = `[tokens, num_kv_heads * head_dim]`
    /// - V: `[tokens, kv_dim]` = `[tokens, num_kv_heads * head_dim]`
    pub fn forward(&self, hidden: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        // Q: separate projection
        let q = self.q_proj.matmul(hidden);

        // KV: fused projection then split
        let kv = self.kv_proj.matmul(hidden);
        let k = kv.slice(s![.., ..self.kv_dim]).to_owned();
        let v = kv.slice(s![.., self.kv_dim..]).to_owned();

        (q, k, v)
    }

    /// Forward pass writing K/V directly to cache slices.
    ///
    /// This is the optimal path for decoder attention - avoids intermediate
    /// K/V allocations by writing directly to the cache.
    ///
    /// # Arguments
    ///
    /// * `hidden` - Input tensor `[tokens, hidden_size]`
    /// * `k_cache_slice` - Mutable slice of K cache to write to `[tokens, kv_dim]`
    /// * `v_cache_slice` - Mutable slice of V cache to write to `[tokens, kv_dim]`
    /// * `kv_scratch` - Scratch buffer for fused KV output `[tokens, 2 * kv_dim]`
    ///
    /// # Returns
    ///
    /// Q tensor `[tokens, q_dim]`
    pub fn forward_to_cache(
        &self,
        hidden: &ArrayView2<f32>,
        k_cache_slice: &mut ArrayViewMut2<f32>,
        v_cache_slice: &mut ArrayViewMut2<f32>,
        kv_scratch: &mut Array2<f32>,
    ) -> Array2<f32> {
        let tokens = hidden.shape()[0];

        // Q projection (returns owned array)
        let q = self.q_proj.matmul(hidden);

        // Fused KV projection into scratch
        self.kv_proj.matmul_noalloc(hidden, kv_scratch);

        // Split KV and write directly to cache (cache-friendly sequential copy)
        let kv_slice = kv_scratch.as_slice().expect("kv_scratch must be contiguous");
        let k_slice = k_cache_slice
            .as_slice_mut()
            .expect("k_cache must be contiguous");
        let v_slice = v_cache_slice
            .as_slice_mut()
            .expect("v_cache must be contiguous");

        let kv_dim = self.kv_dim;
        for t in 0..tokens {
            let src_offset = t * 2 * kv_dim;
            let dst_offset = t * kv_dim;

            // Copy K
            k_slice[dst_offset..dst_offset + kv_dim]
                .copy_from_slice(&kv_slice[src_offset..src_offset + kv_dim]);

            // Copy V
            v_slice[dst_offset..dst_offset + kv_dim]
                .copy_from_slice(&kv_slice[src_offset + kv_dim..src_offset + 2 * kv_dim]);
        }

        q
    }

    /// Forward pass with pre-allocated output buffers (zero allocation).
    ///
    /// For use with buffer pools.
    pub fn forward_noalloc(
        &self,
        hidden: &ArrayView2<f32>,
        q_out: &mut Array2<f32>,
        k_out: &mut Array2<f32>,
        v_out: &mut Array2<f32>,
        kv_scratch: &mut Array2<f32>,
    ) {
        let tokens = hidden.shape()[0];

        // Q projection
        self.q_proj.matmul_noalloc(hidden, q_out);

        // Fused KV projection
        self.kv_proj.matmul_noalloc(hidden, kv_scratch);

        // Split KV
        let kv_slice = kv_scratch.as_slice().expect("kv_scratch must be contiguous");
        let k_slice = k_out.as_slice_mut().expect("k_out must be contiguous");
        let v_slice = v_out.as_slice_mut().expect("v_out must be contiguous");

        let kv_dim = self.kv_dim;
        for t in 0..tokens {
            let src_offset = t * 2 * kv_dim;
            let dst_offset = t * kv_dim;

            k_slice[dst_offset..dst_offset + kv_dim]
                .copy_from_slice(&kv_slice[src_offset..src_offset + kv_dim]);

            v_slice[dst_offset..dst_offset + kv_dim]
                .copy_from_slice(&kv_slice[src_offset + kv_dim..src_offset + 2 * kv_dim]);
        }
    }

    /// Fuses K and V weight matrices into a single [2*kv_dim, hidden_size] matrix.
    fn fuse_kv_weights(k: &LinearLayer, v: &LinearLayer) -> LinearLayer {
        let kv_dim = k.out_features();
        let hidden_size = k.in_features();

        // Stack K and V weights: [K_weights; V_weights]
        let mut fused_weights = Array2::<f32>::zeros((2 * kv_dim, hidden_size));
        fused_weights
            .slice_mut(s![..kv_dim, ..])
            .assign(&k.weights_view());
        fused_weights
            .slice_mut(s![kv_dim.., ..])
            .assign(&v.weights_view());

        // Fuse biases if present
        let fused_bias = match (k.bias(), v.bias()) {
            (Some(kb), Some(vb)) => {
                let mut bias = Array1::<f32>::zeros(2 * kv_dim);
                bias.slice_mut(s![..kv_dim]).assign(kb);
                bias.slice_mut(s![kv_dim..]).assign(vb);
                Some(bias)
            }
            (None, None) => None,
            _ => {
                log::warn!("K and V have inconsistent bias state, ignoring biases");
                None
            }
        };

        LinearLayer::new_f32(fused_weights, fused_bias)
    }

    /// Returns the KV dimension (num_kv_heads * head_dim).
    #[inline]
    pub fn kv_dim(&self) -> usize {
        self.kv_dim
    }

    /// Returns the Q dimension (num_heads * head_dim).
    #[inline]
    pub fn q_dim(&self) -> usize {
        self.q_dim
    }

    /// Returns the hidden size (input dimension).
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.q_proj.in_features()
    }

    /// Returns reference to Q projection layer.
    pub fn q_proj(&self) -> &LinearLayer {
        &self.q_proj
    }

    /// Returns reference to fused KV projection layer.
    pub fn kv_proj(&self) -> &LinearLayer {
        &self.kv_proj
    }
}

impl std::fmt::Debug for GQAProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GQAProjection")
            .field("q_dim", &self.q_dim)
            .field("kv_dim", &self.kv_dim)
            .field("hidden_size", &self.hidden_size())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_layer(out_features: usize, in_features: usize, seed: usize) -> LinearLayer {
        let weights = Array2::from_shape_fn((out_features, in_features), |(i, j)| {
            ((i * 17 + j * 13 + seed) % 1000) as f32 * 0.002 - 1.0
        });
        LinearLayer::new_f32(weights, None)
    }

    #[test]
    fn test_gqa_matches_separate_projections() {
        // Llama 3.2 3B dimensions
        let hidden_size = 3072;
        let num_heads = 24;
        let num_kv_heads = 8;
        let head_dim = 128;

        let q_dim = num_heads * head_dim; // 3072
        let kv_dim = num_kv_heads * head_dim; // 1024

        let q = create_test_layer(q_dim, hidden_size, 42);
        let k = create_test_layer(kv_dim, hidden_size, 123);
        let v = create_test_layer(kv_dim, hidden_size, 456);

        // Input
        let tokens = 4;
        let hidden = Array2::from_shape_fn((tokens, hidden_size), |(t, h)| {
            ((t * 31 + h * 7) % 1000) as f32 * 0.002 - 1.0
        });

        // Reference: separate matmuls
        let q_ref = q.matmul(&hidden.view());
        let k_ref = k.matmul(&hidden.view());
        let v_ref = v.matmul(&hidden.view());

        // Fused projection
        let gqa = GQAProjection::new(q.clone(), k.clone(), v.clone());
        let (q_fused, k_fused, v_fused) = gqa.forward(&hidden.view());

        // Compare
        assert_eq!(q_ref.shape(), q_fused.shape());
        assert_eq!(k_ref.shape(), k_fused.shape());
        assert_eq!(v_ref.shape(), v_fused.shape());

        for (i, (&exp, &got)) in q_ref.iter().zip(q_fused.iter()).enumerate() {
            let diff = (exp - got).abs();
            assert!(diff < 1e-5, "Q mismatch at {}: {} vs {}", i, exp, got);
        }

        for (i, (&exp, &got)) in k_ref.iter().zip(k_fused.iter()).enumerate() {
            let diff = (exp - got).abs();
            assert!(diff < 1e-5, "K mismatch at {}: {} vs {}", i, exp, got);
        }

        for (i, (&exp, &got)) in v_ref.iter().zip(v_fused.iter()).enumerate() {
            let diff = (exp - got).abs();
            assert!(diff < 1e-5, "V mismatch at {}: {} vs {}", i, exp, got);
        }
    }

    #[test]
    fn test_gqa_forward_to_cache() {
        let hidden_size = 256;
        let num_heads = 8;
        let num_kv_heads = 2;
        let head_dim = hidden_size / num_heads;

        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let tokens = 4;

        let q = create_test_layer(q_dim, hidden_size, 1);
        let k = create_test_layer(kv_dim, hidden_size, 2);
        let v = create_test_layer(kv_dim, hidden_size, 3);

        let hidden = Array2::from_shape_fn((tokens, hidden_size), |(t, h)| (t + h) as f32 * 0.01);

        // Reference
        let k_ref = k.matmul(&hidden.view());
        let v_ref = v.matmul(&hidden.view());

        // Forward to cache
        let gqa = GQAProjection::new(q, k, v);

        let mut k_cache = Array2::zeros((tokens, kv_dim));
        let mut v_cache = Array2::zeros((tokens, kv_dim));
        let mut kv_scratch = Array2::zeros((tokens, 2 * kv_dim));

        let _q = gqa.forward_to_cache(
            &hidden.view(),
            &mut k_cache.view_mut(),
            &mut v_cache.view_mut(),
            &mut kv_scratch,
        );

        // Compare K
        for (i, (&exp, &got)) in k_ref.iter().zip(k_cache.iter()).enumerate() {
            let diff = (exp - got).abs();
            assert!(diff < 1e-5, "K cache mismatch at {}: {} vs {}", i, exp, got);
        }

        // Compare V
        for (i, (&exp, &got)) in v_ref.iter().zip(v_cache.iter()).enumerate() {
            let diff = (exp - got).abs();
            assert!(diff < 1e-5, "V cache mismatch at {}: {} vs {}", i, exp, got);
        }
    }

    #[test]
    fn test_gqa_dimensions() {
        // Llama 3.1 70B dimensions
        let hidden_size = 8192;
        let num_heads = 64;
        let num_kv_heads = 8;
        let head_dim = 128;

        let q_dim = num_heads * head_dim; // 8192
        let kv_dim = num_kv_heads * head_dim; // 1024

        let q = create_test_layer(q_dim, hidden_size, 1);
        let k = create_test_layer(kv_dim, hidden_size, 2);
        let v = create_test_layer(kv_dim, hidden_size, 3);

        let gqa = GQAProjection::new(q, k, v);

        assert_eq!(gqa.q_dim(), 8192);
        assert_eq!(gqa.kv_dim(), 1024);
        assert_eq!(gqa.hidden_size(), 8192);
    }
}