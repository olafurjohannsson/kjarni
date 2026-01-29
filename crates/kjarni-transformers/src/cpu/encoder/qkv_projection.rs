//! QKV Projection with adaptive fused/separate strategy.
//!
//! Supports both fused (1 matmul) and separate (3 matmuls) strategies,
//! with automatic selection based on hidden size.

use ndarray::{Array1, Array2, ArrayView2, s};

use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::linear_layer::LinearLayer;

/// QKV Projection supporting both fused (1 matmul) and separate (3 matmul) strategies.
///
/// # Strategy Selection
///
/// - **Fused** (hidden <= 512): Single matmul producing [tokens, 3*hidden], then split.
///   Better for small models due to reduced kernel launch overhead.
/// - **Separate** (hidden > 512): Three independent matmuls.
///   Better for large models where cache efficiency matters more.
///
/// # Example
///
/// ```ignore
/// // Automatic strategy selection
/// let qkv = QKVProjection::new(q_proj, k_proj, v_proj);
///
/// // Allocating forward
/// let (q, k, v) = qkv.forward(&hidden_states);
///
/// // No-alloc forward (uses pre-allocated buffers)
/// qkv.forward_noalloc(&hidden_states, &mut buffers);
/// // Results in buffers.q, buffers.k, buffers.v
/// ```
pub struct QKVProjection {
    storage: QKVStorage,
    hidden_size: usize,
}

enum QKVStorage {
    Separate {
        q_proj: LinearLayer,
        k_proj: LinearLayer,
        v_proj: LinearLayer,
    },
    Fused {
        qkv_proj: LinearLayer, // [3*hidden, hidden] weights
    },
}

impl QKVProjection {
    /// Creates projection with automatic strategy selection based on hidden size.
    ///
    /// - hidden <= 512: Uses fused (1 matmul, ~1.5-3x faster)
    /// - hidden > 512: Uses separate (3 matmuls, better cache efficiency)
    ///
    /// # Arguments
    ///
    /// * `q` - Query projection layer [hidden, hidden]
    /// * `k` - Key projection layer [hidden, hidden]
    /// * `v` - Value projection layer [hidden, hidden]
    pub fn new(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        let hidden_size = q.out_features();

        // Heuristic based on benchmarks:
        // - Fused wins for hidden <= 512 across most token counts
        // - Separate wins for hidden >= 768 with large batches
        if hidden_size <= 512 {
            Self::new_fused(q, k, v)
        } else {
            Self::new_separate(q, k, v)
        }
    }

    /// Creates fused projection (single matmul, ~1.5-3x faster for small models).
    ///
    /// Fuses Q, K, V weights into a single [3*hidden, hidden] matrix.
    /// Forward pass does 1 matmul then splits the result.
    pub fn new_fused(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        let hidden_size = q.out_features();
        let qkv_proj = Self::fuse_layers(&q, &k, &v);

        Self {
            storage: QKVStorage::Fused { qkv_proj },
            hidden_size,
        }
    }

    /// Creates separate projection (three independent matmuls).
    ///
    /// Better for large models (hidden >= 768) with large batch sizes.
    pub fn new_separate(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        let hidden_size = q.out_features();
        Self {
            storage: QKVStorage::Separate {
                q_proj: q,
                k_proj: k,
                v_proj: v,
            },
            hidden_size,
        }
    }

    /// Forward pass - returns (Q, K, V) arrays.
    ///
    /// Allocates new arrays for the output. For hot paths, use `forward_noalloc`.
    ///
    /// # Arguments
    ///
    /// * `hidden` - Input tensor [tokens, hidden]
    ///
    /// # Returns
    ///
    /// Tuple of (Q, K, V) each with shape [tokens, hidden]
    pub fn forward(&self, hidden: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        match &self.storage {
            QKVStorage::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => (
                q_proj.matmul(hidden),
                k_proj.matmul(hidden),
                v_proj.matmul(hidden),
            ),
            QKVStorage::Fused { qkv_proj } => {
                let qkv = qkv_proj.matmul(hidden);
                let h = self.hidden_size;

                // Slice into Q, K, V
                let q = qkv.slice(s![.., 0..h]).to_owned();
                let k = qkv.slice(s![.., h..2 * h]).to_owned();
                let v = qkv.slice(s![.., 2 * h..3 * h]).to_owned();
                (q, k, v)
            }
        }
    }

    /// Forward pass writing to pre-allocated EncoderBuffers (no allocation).
    ///
    /// Writes Q, K, V to `buffers.q`, `buffers.k`, `buffers.v` respectively.
    /// For fused storage, uses `buffers.qkv_scratch` as intermediate.
    ///
    /// # Arguments
    ///
    /// * `hidden` - Input tensor [tokens, hidden]
    /// * `buffers` - Pre-allocated encoder buffers
    ///
    /// # Panics
    ///
    /// - In debug mode: panics if buffer capacity is insufficient
    /// - Panics if fused storage but `buffers.qkv_scratch` is None
    pub fn forward_noalloc(&self, hidden: &ArrayView2<f32>, buffers: &mut EncoderBuffers) {
        let tokens = hidden.shape()[0];

        #[cfg(debug_assertions)]
        buffers.ensure_capacity_tokens(tokens);

        match &self.storage {
            QKVStorage::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => {
                q_proj.matmul_noalloc(hidden, &mut buffers.q);
                k_proj.matmul_noalloc(hidden, &mut buffers.k);
                v_proj.matmul_noalloc(hidden, &mut buffers.v);
            }
            QKVStorage::Fused { qkv_proj } => {
                // Scoped mutable borrow for matmul - released before split
                {
                    let scratch = buffers.qkv_scratch.as_mut().expect(
                        "qkv_scratch not allocated - buffers created with use_fused_qkv=false",
                    );
                    qkv_proj.matmul_noalloc(hidden, scratch);
                } // scratch borrow ends here

                // Now split - Rust allows split borrows of different struct fields
                let h = self.hidden_size;
                let qkv_slice = buffers.qkv_scratch.as_ref().unwrap().as_slice().unwrap();
                let q_slice = buffers.q.as_slice_mut().unwrap();
                let k_slice = buffers.k.as_slice_mut().unwrap();
                let v_slice = buffers.v.as_slice_mut().unwrap();

                // Sequential read from qkv, sequential write to q/k/v - cache friendly
                for t in 0..tokens {
                    let src_offset = t * 3 * h;
                    let dst_offset = t * h;

                    q_slice[dst_offset..dst_offset + h]
                        .copy_from_slice(&qkv_slice[src_offset..src_offset + h]);

                    k_slice[dst_offset..dst_offset + h]
                        .copy_from_slice(&qkv_slice[src_offset + h..src_offset + 2 * h]);

                    v_slice[dst_offset..dst_offset + h]
                        .copy_from_slice(&qkv_slice[src_offset + 2 * h..src_offset + 3 * h]);
                }
            }
        }
    }

    /// Returns a reference to the internal projections for testing/debugging.
    /// Returns (q_proj, k_proj, v_proj) if separate, or None if fused.
    pub fn get_separate_projections(&self) -> Option<(&LinearLayer, &LinearLayer, &LinearLayer)> {
        match &self.storage {
            QKVStorage::Separate {
                q_proj,
                k_proj,
                v_proj,
            } => Some((q_proj, k_proj, v_proj)),
            QKVStorage::Fused { .. } => None,
        }
    }

    /// Forward pass returning separate Q, K, V (for testing compatibility).
    /// Same as `forward()` but more explicit name.
    pub fn forward_qkv(&self, hidden: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        self.forward(hidden)
    }

    /// Efficient split of fused QKV buffer into separate Q, K, V buffers.
    ///
    /// Uses sequential memory access pattern for cache efficiency.
    #[inline]
    fn split_qkv_buffer(
        qkv: &Array2<f32>,   // [tokens, 3*hidden]
        q: &mut Array2<f32>, // [tokens, hidden]
        k: &mut Array2<f32>, // [tokens, hidden]
        v: &mut Array2<f32>, // [tokens, hidden]
        tokens: usize,
        hidden: usize,
    ) {
        let qkv_slice = qkv.as_slice().unwrap();
        let q_slice = q.as_slice_mut().unwrap();
        let k_slice = k.as_slice_mut().unwrap();
        let v_slice = v.as_slice_mut().unwrap();

        // Sequential read from qkv, sequential write to q/k/v
        // This is cache-friendly: reads each cache line once
        for t in 0..tokens {
            let src_offset = t * 3 * hidden;
            let dst_offset = t * hidden;

            q_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset..src_offset + hidden]);

            k_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset + hidden..src_offset + 2 * hidden]);

            v_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset + 2 * hidden..src_offset + 3 * hidden]);
        }
    }

    /// Returns whether this projection uses fused storage.
    #[inline]
    pub fn is_fused(&self) -> bool {
        matches!(self.storage, QKVStorage::Fused { .. })
    }

    /// Returns the hidden size.
    #[inline]
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Fuses three separate linear layers into one.
    fn fuse_layers(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> LinearLayer {
        let fused_weights = Self::fuse_qkv_weights(q, k, v);
        let fused_bias = Self::fuse_qkv_bias(q, k, v);
        LinearLayer::new_f32(fused_weights, fused_bias)
    }

    /// Fuse Q, K, V weight matrices into single [3*hidden, hidden] matrix.
    fn fuse_qkv_weights(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> Array2<f32> {
        let hidden = q.in_features();
        let out = q.out_features();

        let mut fused = Array2::zeros((3 * out, hidden));

        fused.slice_mut(s![0..out, ..]).assign(&q.weights_view());
        fused
            .slice_mut(s![out..2 * out, ..])
            .assign(&k.weights_view());
        fused
            .slice_mut(s![2 * out..3 * out, ..])
            .assign(&v.weights_view());

        fused
    }

    /// Fuse Q, K, V biases into single [3*hidden] vector.
    fn fuse_qkv_bias(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> Option<Array1<f32>> {
        match (q.bias(), k.bias(), v.bias()) {
            (Some(qb), Some(kb), Some(vb)) => {
                let out = qb.len();
                let mut fused = Array1::zeros(3 * out);
                fused.slice_mut(s![0..out]).assign(qb);
                fused.slice_mut(s![out..2 * out]).assign(kb);
                fused.slice_mut(s![2 * out..3 * out]).assign(vb);
                Some(fused)
            }
            (None, None, None) => None,
            _ => {
                log::warn!("QKV layers have inconsistent bias state, ignoring biases");
                None
            }
        }
    }
}

impl std::fmt::Debug for QKVProjection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QKVProjection")
            .field("hidden_size", &self.hidden_size)
            .field("is_fused", &self.is_fused())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn create_test_layer(in_features: usize, out_features: usize) -> LinearLayer {
        let weights = Array2::from_elem((out_features, in_features), 0.01f32);
        LinearLayer::new_f32(weights, None)
    }

    #[test]
    fn test_auto_strategy_selection() {
        // Small model -> fused
        let small = QKVProjection::new(
            create_test_layer(384, 384),
            create_test_layer(384, 384),
            create_test_layer(384, 384),
        );
        assert!(small.is_fused());

        // Large model -> separate
        let large = QKVProjection::new(
            create_test_layer(768, 768),
            create_test_layer(768, 768),
            create_test_layer(768, 768),
        );
        assert!(!large.is_fused());
    }

    #[test]
    fn test_forward_parity() {
        let hidden = 384;
        let tokens = 32;

        let q = create_test_layer(hidden, hidden);
        let k = create_test_layer(hidden, hidden);
        let v = create_test_layer(hidden, hidden);

        let fused = QKVProjection::new_fused(
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
        );
        let separate = QKVProjection::new_separate(q, k, v);

        let input = Array2::from_elem((tokens, hidden), 0.5f32);

        let (q_fused, k_fused, v_fused) = fused.forward(&input.view());
        let (q_sep, k_sep, v_sep) = separate.forward(&input.view());

        // Should produce same shapes
        assert_eq!(q_fused.dim(), q_sep.dim());
        assert_eq!(k_fused.dim(), k_sep.dim());
        assert_eq!(v_fused.dim(), v_sep.dim());
    }

    #[test]
    fn test_forward_noalloc() {
        let hidden = 384;
        let tokens = 32;
        let intermediate = 1536;
        let num_heads = 6;

        let qkv = QKVProjection::new_fused(
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
        );
        let max_batch = 1;
        let max_seq = tokens;
        let mut buffers =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, true);
        let input = Array2::from_elem((tokens, hidden), 0.5f32);

        // Forward with allocation
        let (q_alloc, k_alloc, v_alloc) = qkv.forward(&input.view());

        // Forward without allocation
        qkv.forward_noalloc(&input.view(), &mut buffers);

        // Compare results - use max difference instead of sum
        let q_diff = (&q_alloc - &buffers.q.slice(s![..tokens, ..]))
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));
        let k_diff = (&k_alloc - &buffers.k.slice(s![..tokens, ..]))
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));
        let v_diff = (&v_alloc - &buffers.v.slice(s![..tokens, ..]))
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));

        assert!(q_diff < 1e-5, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-5, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-5, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_forward_noalloc_separate() {
        let hidden = 768;
        let tokens = 32;
        let intermediate = 3072;
        let max_batch = 1;
        let num_heads = 12;
        let max_seq = tokens;
        let qkv = QKVProjection::new(
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
            create_test_layer(hidden, hidden),
        );
        assert!(!qkv.is_fused());

        let mut buffers =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, false);
        let input = Array2::from_elem((tokens, hidden), 0.5f32);

        // Forward with allocation
        let (q_alloc, k_alloc, v_alloc) = qkv.forward(&input.view());

        // Forward without allocation
        qkv.forward_noalloc(&input.view(), &mut buffers);

        // Compare results - use max difference
        let q_diff = (&q_alloc - &buffers.q.slice(s![..tokens, ..]))
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));

        assert!(q_diff < 1e-5, "Q mismatch: {}", q_diff);
    }
}
// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod qkv_projection_tests {
    use super::*;
    use std::time::Instant;

    // =========================================================================
    // Test Utilities
    // =========================================================================

    fn make_linear_layer(
        out_features: usize,
        in_features: usize,
        seed: usize,
        with_bias: bool,
    ) -> LinearLayer {
        let data: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i + seed) % 10000) as f32 * 0.0001 - 0.5)
            .collect();
        let weights = Array2::from_shape_vec((out_features, in_features), data).unwrap();

        let bias = if with_bias {
            Some(Array1::from_vec(
                (0..out_features)
                    .map(|i| (i + seed) as f32 * 0.001)
                    .collect(),
            ))
        } else {
            None
        };

        LinearLayer::new_f32(weights, bias)
    }

    fn make_input(tokens: usize, hidden: usize, seed: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..tokens * hidden)
            .map(|i| ((i + seed) % 1000) as f32 * 0.001 - 0.5)
            .collect();
        Array2::from_shape_vec((tokens, hidden), data).unwrap()
    }

    fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    // =========================================================================
    // Correctness Tests
    // =========================================================================

    #[test]
    fn test_separate_forward_works() {
        let hidden = 64;
        let tokens = 16;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let proj = QKVProjection::new_separate(q, k, v);
        let input = make_input(tokens, hidden, 42);

        let (q_out, k_out, v_out) = proj.forward(&input.view());

        assert_eq!(q_out.dim(), (tokens, hidden));
        assert_eq!(k_out.dim(), (tokens, hidden));
        assert_eq!(v_out.dim(), (tokens, hidden));
        assert!(!proj.is_fused());

        println!("\n=== test_separate_forward_works ===");
        println!("Q output shape: {:?}", q_out.dim());
    }

    #[test]
    fn test_fused_forward_works() {
        let hidden = 64;
        let tokens = 16;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let proj = QKVProjection::new_fused(q, k, v);
        let input = make_input(tokens, hidden, 42);

        let (q_out, k_out, v_out) = proj.forward(&input.view());

        assert_eq!(q_out.dim(), (tokens, hidden));
        assert_eq!(k_out.dim(), (tokens, hidden));
        assert_eq!(v_out.dim(), (tokens, hidden));
        assert!(proj.is_fused());

        println!("\n=== test_fused_forward_works ===");
        println!("Q output shape: {:?}", q_out.dim());
    }

    #[test]
    fn test_fused_matches_separate_small() {
        let hidden = 64;
        let tokens = 16;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let q2 = make_linear_layer(hidden, hidden, 0, true);
        let k2 = make_linear_layer(hidden, hidden, 1000, true);
        let v2 = make_linear_layer(hidden, hidden, 2000, true);

        let fused = QKVProjection::new_fused(q, k, v);
        let separate = QKVProjection::new_separate(q2, k2, v2);

        let input = make_input(tokens, hidden, 42);

        let (q_fused, k_fused, v_fused) = fused.forward(&input.view());
        let (q_sep, k_sep, v_sep) = separate.forward(&input.view());

        let q_diff = max_diff(&q_fused, &q_sep);
        let k_diff = max_diff(&k_fused, &k_sep);
        let v_diff = max_diff(&v_fused, &v_sep);

        println!("\n=== test_fused_matches_separate_small ===");
        println!("Config: tokens={}, hidden={}", tokens, hidden);
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_fused_matches_separate_minilm() {
        let hidden = 384;
        let tokens = 2880;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let q2 = make_linear_layer(hidden, hidden, 0, true);
        let k2 = make_linear_layer(hidden, hidden, 1000, true);
        let v2 = make_linear_layer(hidden, hidden, 2000, true);

        let fused = QKVProjection::new_fused(q, k, v);
        let separate = QKVProjection::new_separate(q2, k2, v2);

        let input = make_input(tokens, hidden, 42);

        let (q_fused, k_fused, v_fused) = fused.forward(&input.view());
        let (q_sep, k_sep, v_sep) = separate.forward(&input.view());

        let q_diff = max_diff(&q_fused, &q_sep);
        let k_diff = max_diff(&k_fused, &k_sep);
        let v_diff = max_diff(&v_fused, &v_sep);

        println!("\n=== test_fused_matches_separate_minilm ===");
        println!("Config: tokens={}, hidden={}", tokens, hidden);
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_fused_matches_separate_no_bias() {
        let hidden = 384;
        let tokens = 256;

        let q = make_linear_layer(hidden, hidden, 0, false);
        let k = make_linear_layer(hidden, hidden, 1000, false);
        let v = make_linear_layer(hidden, hidden, 2000, false);

        let q2 = make_linear_layer(hidden, hidden, 0, false);
        let k2 = make_linear_layer(hidden, hidden, 1000, false);
        let v2 = make_linear_layer(hidden, hidden, 2000, false);

        let fused = QKVProjection::new_fused(q, k, v);
        let separate = QKVProjection::new_separate(q2, k2, v2);

        let input = make_input(tokens, hidden, 42);

        let (q_fused, k_fused, v_fused) = fused.forward(&input.view());
        let (q_sep, k_sep, v_sep) = separate.forward(&input.view());

        let q_diff = max_diff(&q_fused, &q_sep);
        let k_diff = max_diff(&k_fused, &k_sep);
        let v_diff = max_diff(&v_fused, &v_sep);

        println!("\n=== test_fused_matches_separate_no_bias ===");
        println!("Config: tokens={}, hidden={}, bias=false", tokens, hidden);
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-5, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-5, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-5, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_fused_matches_separate_bert() {
        let hidden = 768;
        let tokens = 2048;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let q2 = make_linear_layer(hidden, hidden, 0, true);
        let k2 = make_linear_layer(hidden, hidden, 1000, true);
        let v2 = make_linear_layer(hidden, hidden, 2000, true);

        let fused = QKVProjection::new_fused(q, k, v);
        let separate = QKVProjection::new_separate(q2, k2, v2);

        let input = make_input(tokens, hidden, 42);

        let (q_fused, k_fused, v_fused) = fused.forward(&input.view());
        let (q_sep, k_sep, v_sep) = separate.forward(&input.view());

        let q_diff = max_diff(&q_fused, &q_sep);
        let k_diff = max_diff(&k_fused, &k_sep);
        let v_diff = max_diff(&v_fused, &v_sep);

        println!("\n=== test_fused_matches_separate_bert ===");
        println!("Config: tokens={}, hidden={}", tokens, hidden);
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }

    // =========================================================================
    // No-Alloc Correctness Tests
    // =========================================================================
    #[test]
    fn test_noalloc_separate_matches_alloc() {
        let hidden = 384;
        let tokens = 256;
        let intermediate = 1536;
        let num_heads = 6;

        // max_batch * max_seq must be >= tokens
        let max_batch = 1;
        let max_seq = tokens; // 256

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let proj = QKVProjection::new_separate(q, k, v);
        let input = make_input(tokens, hidden, 42);

        let (q_alloc, k_alloc, v_alloc) = proj.forward(&input.view());

        // Correct order: max_batch, max_seq, hidden, num_heads, intermediate, use_fused
        let mut buffers =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, false);
        proj.forward_noalloc(&input.view(), &mut buffers);

        let q_diff = max_diff(&q_alloc, &buffers.q.slice(s![..tokens, ..]).to_owned());
        let k_diff = max_diff(&k_alloc, &buffers.k.slice(s![..tokens, ..]).to_owned());
        let v_diff = max_diff(&v_alloc, &buffers.v.slice(s![..tokens, ..]).to_owned());

        println!("\n=== test_noalloc_separate_matches_alloc ===");
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_noalloc_fused_matches_alloc() {
        let hidden = 384;
        let tokens = 256;
        let intermediate = 1536;
        let max_batch = 1;
        let num_heads = 6;
        let max_seq = tokens;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let proj = QKVProjection::new_fused(q, k, v);
        let input = make_input(tokens, hidden, 42);

        // Allocating version
        let (q_alloc, k_alloc, v_alloc) = proj.forward(&input.view());

        // No-alloc version using EncoderBuffers (fused needs qkv_scratch)
        let mut buffers =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, true);
        proj.forward_noalloc(&input.view(), &mut buffers);

        let q_diff = max_diff(&q_alloc, &buffers.q.slice(s![..tokens, ..]).to_owned());
        let k_diff = max_diff(&k_alloc, &buffers.k.slice(s![..tokens, ..]).to_owned());
        let v_diff = max_diff(&v_alloc, &buffers.v.slice(s![..tokens, ..]).to_owned());

        println!("\n=== test_noalloc_fused_matches_alloc ===");
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }

    #[test]
    fn test_noalloc_fused_matches_noalloc_separate() {
        let hidden = 384;
        let tokens = 2880;
        let intermediate = 1536;
        let max_batch = 1;
        let max_seq = tokens;
        let num_heads = 6;

        let q = make_linear_layer(hidden, hidden, 0, true);
        let k = make_linear_layer(hidden, hidden, 1000, true);
        let v = make_linear_layer(hidden, hidden, 2000, true);

        let q2 = make_linear_layer(hidden, hidden, 0, true);
        let k2 = make_linear_layer(hidden, hidden, 1000, true);
        let v2 = make_linear_layer(hidden, hidden, 2000, true);

        let fused = QKVProjection::new_fused(q, k, v);
        let separate = QKVProjection::new_separate(q2, k2, v2);

        let input = make_input(tokens, hidden, 42);

        // Fused needs qkv_scratch=true, separate doesn't need it but can have it
        let mut buffers_fused =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, true);
        let mut buffers_sep =
            EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, false);

        fused.forward_noalloc(&input.view(), &mut buffers_fused);
        separate.forward_noalloc(&input.view(), &mut buffers_sep);

        let q_diff = max_diff(
            &buffers_fused.q.slice(s![..tokens, ..]).to_owned(),
            &buffers_sep.q.slice(s![..tokens, ..]).to_owned(),
        );
        let k_diff = max_diff(
            &buffers_fused.k.slice(s![..tokens, ..]).to_owned(),
            &buffers_sep.k.slice(s![..tokens, ..]).to_owned(),
        );
        let v_diff = max_diff(
            &buffers_fused.v.slice(s![..tokens, ..]).to_owned(),
            &buffers_sep.v.slice(s![..tokens, ..]).to_owned(),
        );

        println!("\n=== test_noalloc_fused_matches_noalloc_separate ===");
        println!("Config: tokens={}, hidden={}", tokens, hidden);
        println!("Q diff: {:.2e}", q_diff);
        println!("K diff: {:.2e}", k_diff);
        println!("V diff: {:.2e}", v_diff);

        assert!(q_diff < 1e-4, "Q mismatch: {}", q_diff);
        assert!(k_diff < 1e-4, "K mismatch: {}", k_diff);
        assert!(v_diff < 1e-4, "V mismatch: {}", v_diff);
    }
}

// =============================================================================
// COMPREHENSIVE STRATEGY BENCHMARK
// =============================================================================

#[cfg(test)]
mod qkv_strategy_benchmark {
    use super::*;
    use std::time::Instant;

    // =========================================================================
    // Test Utilities
    // =========================================================================

    fn make_linear_layer(
        out_features: usize,
        in_features: usize,
        seed: usize,
        with_bias: bool,
    ) -> LinearLayer {
        let data: Vec<f32> = (0..out_features * in_features)
            .map(|i| ((i + seed) % 10000) as f32 * 0.0001 - 0.5)
            .collect();
        let weights = Array2::from_shape_vec((out_features, in_features), data).unwrap();
        let bias = if with_bias {
            Some(Array1::from_vec(
                (0..out_features)
                    .map(|i| (i + seed) as f32 * 0.001)
                    .collect(),
            ))
        } else {
            None
        };
        LinearLayer::new_f32(weights, bias)
    }

    fn make_input(tokens: usize, hidden: usize, seed: usize) -> Array2<f32> {
        let data: Vec<f32> = (0..tokens * hidden)
            .map(|i| ((i + seed) % 1000) as f32 * 0.001 - 0.5)
            .collect();
        Array2::from_shape_vec((tokens, hidden), data).unwrap()
    }

    // =========================================================================
    // STRATEGY A: 3 Separate Matmuls (allocating)
    // =========================================================================
    fn strategy_a_separate_alloc(
        input: &ArrayView2<f32>,
        q_proj: &LinearLayer,
        k_proj: &LinearLayer,
        v_proj: &LinearLayer,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let q = q_proj.matmul(input);
        let k = k_proj.matmul(input);
        let v = v_proj.matmul(input);
        (q, k, v)
    }

    // =========================================================================
    // STRATEGY B: 1 Fused Matmul + Slice Copy (allocating)
    // =========================================================================
    fn strategy_b_fused_alloc(
        input: &ArrayView2<f32>,
        qkv_proj: &LinearLayer,
        hidden: usize,
    ) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        let qkv = qkv_proj.matmul(input);
        let q = qkv.slice(s![.., 0..hidden]).to_owned();
        let k = qkv.slice(s![.., hidden..2 * hidden]).to_owned();
        let v = qkv.slice(s![.., 2 * hidden..3 * hidden]).to_owned();
        (q, k, v)
    }

    // =========================================================================
    // STRATEGY C: 3 Separate Matmuls (no-alloc)
    // =========================================================================
    fn strategy_c_separate_noalloc(
        input: &ArrayView2<f32>,
        q_proj: &LinearLayer,
        k_proj: &LinearLayer,
        v_proj: &LinearLayer,
        q_out: &mut Array2<f32>,
        k_out: &mut Array2<f32>,
        v_out: &mut Array2<f32>,
    ) {
        q_proj.matmul_noalloc(input, q_out);
        k_proj.matmul_noalloc(input, k_out);
        v_proj.matmul_noalloc(input, v_out);
    }

    // =========================================================================
    // STRATEGY D: 1 Fused Matmul + Split (no-alloc)
    // =========================================================================
    fn strategy_d_fused_noalloc(
        input: &ArrayView2<f32>,
        qkv_proj: &LinearLayer,
        qkv_scratch: &mut Array2<f32>,
        q_out: &mut Array2<f32>,
        k_out: &mut Array2<f32>,
        v_out: &mut Array2<f32>,
        hidden: usize,
    ) {
        // 1 fused matmul
        qkv_proj.matmul_noalloc(input, qkv_scratch);

        // Split into separate buffers
        let tokens = input.shape()[0];
        let qkv_slice = qkv_scratch.as_slice().unwrap();
        let q_slice = q_out.as_slice_mut().unwrap();
        let k_slice = k_out.as_slice_mut().unwrap();
        let v_slice = v_out.as_slice_mut().unwrap();

        for t in 0..tokens {
            let src_offset = t * 3 * hidden;
            let dst_offset = t * hidden;

            q_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset..src_offset + hidden]);

            k_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset + hidden..src_offset + 2 * hidden]);

            v_slice[dst_offset..dst_offset + hidden]
                .copy_from_slice(&qkv_slice[src_offset + 2 * hidden..src_offset + 3 * hidden]);
        }
    }

    // =========================================================================
    // STRATEGY E: 1 Fused Matmul, Return Views (zero copy but non-contiguous)
    // =========================================================================
    fn strategy_e_fused_views<'a>(
        input: &ArrayView2<f32>,
        qkv_proj: &LinearLayer,
        qkv_scratch: &'a mut Array2<f32>,
        hidden: usize,
    ) -> (
        ndarray::ArrayView2<'a, f32>,
        ndarray::ArrayView2<'a, f32>,
        ndarray::ArrayView2<'a, f32>,
    ) {
        qkv_proj.matmul_noalloc(input, qkv_scratch);

        let q = qkv_scratch.slice(s![.., 0..hidden]);
        let k = qkv_scratch.slice(s![.., hidden..2 * hidden]);
        let v = qkv_scratch.slice(s![.., 2 * hidden..3 * hidden]);
        (q, k, v)
    }

    // =========================================================================
    // BENCHMARK: All Strategies
    // =========================================================================

    // #[test]
    // fn bench_all_qkv_strategies() {
    //     println!("\n");
    //     println!("╔══════════════════════════════════════════════════════════════════════╗");
    //     println!("║           QKV PROJECTION STRATEGY BENCHMARK                          ║");
    //     println!("║           Comparing: Fused vs Separate, Alloc vs No-Alloc            ║");
    //     println!("╚══════════════════════════════════════════════════════════════════════╝");

    //     let configs = [
    //         // (name, tokens, hidden, iterations, warmup)
    //         ("Decode (1 token)", 1, 384, 1000, 100),
    //         ("Small batch (16)", 16, 384, 500, 50),
    //         ("Medium batch (256)", 256, 384, 100, 10),
    //         ("MiniLM batch (2880)", 2880, 384, 50, 5),
    //         ("BERT-base (2048)", 2048, 768, 30, 3),
    //         ("Large hidden (1024)", 1024, 1024, 30, 3),
    //     ];

    //     for (name, tokens, hidden, iterations, warmup) in configs {
    //         println!("=== {} (tokens={}, hidden={}) ===", name, tokens, hidden);

    //         // Create layers
    //         let q_proj = make_linear_layer(hidden, hidden, 0, true);
    //         let k_proj = make_linear_layer(hidden, hidden, 1000, true);
    //         let v_proj = make_linear_layer(hidden, hidden, 2000, true);

    //         // Create fused layer
    //         let fused_weights = {
    //             let mut fused = Array2::zeros((3 * hidden, hidden));
    //             fused
    //                 .slice_mut(s![0..hidden, ..])
    //                 .assign(&q_proj.weights_view());
    //             fused
    //                 .slice_mut(s![hidden..2 * hidden, ..])
    //                 .assign(&k_proj.weights_view());
    //             fused
    //                 .slice_mut(s![2 * hidden..3 * hidden, ..])
    //                 .assign(&v_proj.weights_view());
    //             fused
    //         };
    //         let fused_bias = {
    //             match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
    //                 (Some(qb), Some(kb), Some(vb)) => {
    //                     let mut fused = Array1::zeros(3 * hidden);
    //                     fused.slice_mut(s![0..hidden]).assign(qb);
    //                     fused.slice_mut(s![hidden..2 * hidden]).assign(kb);
    //                     fused.slice_mut(s![2 * hidden..3 * hidden]).assign(vb);
    //                     Some(fused)
    //                 }
    //                 _ => None,
    //             }
    //         };
    //         let qkv_proj = LinearLayer::new_f32(fused_weights, fused_bias);

    //         // Create input
    //         let input = make_input(tokens, hidden, 42);

    //         // Create scratch buffers
    //         let mut q_scratch = Array2::zeros((tokens, hidden));
    //         let mut k_scratch = Array2::zeros((tokens, hidden));
    //         let mut v_scratch = Array2::zeros((tokens, hidden));
    //         let mut qkv_scratch = Array2::zeros((tokens, 3 * hidden));

    //         // =====================================================
    //         // WARMUP
    //         // =====================================================
    //         for _ in 0..warmup {
    //             let _ = strategy_a_separate_alloc(&input.view(), &q_proj, &k_proj, &v_proj);
    //             let _ = strategy_b_fused_alloc(&input.view(), &qkv_proj, hidden);
    //             strategy_c_separate_noalloc(
    //                 &input.view(),
    //                 &q_proj,
    //                 &k_proj,
    //                 &v_proj,
    //                 &mut q_scratch,
    //                 &mut k_scratch,
    //                 &mut v_scratch,
    //             );
    //             strategy_d_fused_noalloc(
    //                 &input.view(),
    //                 &qkv_proj,
    //                 &mut qkv_scratch,
    //                 &mut q_scratch,
    //                 &mut k_scratch,
    //                 &mut v_scratch,
    //                 hidden,
    //             );
    //             let _ = strategy_e_fused_views(&input.view(), &qkv_proj, &mut qkv_scratch, hidden);
    //         }

    //         // =====================================================
    //         // STRATEGY A: 3 Separate Matmuls (allocating)
    //         // =====================================================
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             let result = strategy_a_separate_alloc(&input.view(), &q_proj, &k_proj, &v_proj);
    //             std::hint::black_box(result);
    //         }
    //         let time_a = start.elapsed();

    //         // =====================================================
    //         // STRATEGY B: 1 Fused Matmul + Copy (allocating)
    //         // =====================================================
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             let result = strategy_b_fused_alloc(&input.view(), &qkv_proj, hidden);
    //             std::hint::black_box(result);
    //         }
    //         let time_b = start.elapsed();

    //         // =====================================================
    //         // STRATEGY C: 3 Separate Matmuls (no-alloc)
    //         // =====================================================
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             strategy_c_separate_noalloc(
    //                 &input.view(),
    //                 &q_proj,
    //                 &k_proj,
    //                 &v_proj,
    //                 &mut q_scratch,
    //                 &mut k_scratch,
    //                 &mut v_scratch,
    //             );
    //             std::hint::black_box(&q_scratch);
    //         }
    //         let time_c = start.elapsed();

    //         // =====================================================
    //         // STRATEGY D: 1 Fused Matmul + Split (no-alloc)
    //         // =====================================================
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             strategy_d_fused_noalloc(
    //                 &input.view(),
    //                 &qkv_proj,
    //                 &mut qkv_scratch,
    //                 &mut q_scratch,
    //                 &mut k_scratch,
    //                 &mut v_scratch,
    //                 hidden,
    //             );
    //             std::hint::black_box(&q_scratch);
    //         }
    //         let time_d = start.elapsed();

    //         // =====================================================
    //         // STRATEGY E: 1 Fused Matmul, Return Views (zero copy)
    //         // =====================================================
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             let result =
    //                 strategy_e_fused_views(&input.view(), &qkv_proj, &mut qkv_scratch, hidden);
    //             std::hint::black_box(result);
    //         }
    //         let time_e = start.elapsed();

    //         // =====================================================
    //         // REPORT
    //         // =====================================================
    //         let per_a = time_a / iterations as u32;
    //         let per_b = time_b / iterations as u32;
    //         let per_c = time_c / iterations as u32;
    //         let per_d = time_d / iterations as u32;
    //         let per_e = time_e / iterations as u32;

    //         let baseline = per_a.as_secs_f64();

    //         println!();
    //         println!("Strategy                              Time/iter    vs Baseline");
    //         println!("─────────────────────────────────────────────────────────────────");
    //         println!(
    //             "A) 3 separate matmuls (alloc)         {:>9.2?}    1.00x (baseline)",
    //             per_a
    //         );
    //         println!(
    //             "B) 1 fused + slice copy (alloc)       {:>9.2?}    {:.2}x",
    //             per_b,
    //             baseline / per_b.as_secs_f64()
    //         );
    //         println!(
    //             "C) 3 separate matmuls (no-alloc)      {:>9.2?}    {:.2}x",
    //             per_c,
    //             baseline / per_c.as_secs_f64()
    //         );
    //         println!(
    //             "D) 1 fused + split (no-alloc)         {:>9.2?}    {:.2}x",
    //             per_d,
    //             baseline / per_d.as_secs_f64()
    //         );
    //         println!(
    //             "E) 1 fused, views (no-alloc, no-copy) {:>9.2?}    {:.2}x",
    //             per_e,
    //             baseline / per_e.as_secs_f64()
    //         );

    //         // Find winner
    //         let times = [
    //             (per_a, "A) 3 separate (alloc)"),
    //             (per_b, "B) 1 fused + copy (alloc)"),
    //             (per_c, "C) 3 separate (no-alloc)"),
    //             (per_d, "D) 1 fused + split (no-alloc)"),
    //             (per_e, "E) 1 fused views (no-alloc)"),
    //         ];
    //         let winner = times.iter().min_by_key(|(t, _)| *t).unwrap();
    //         println!();
    //         println!("🏆 WINNER: {} ({:.2?})", winner.1, winner.0);

    //         // Analysis
    //         let fused_vs_separate_alloc = per_a.as_secs_f64() / per_b.as_secs_f64();
    //         let fused_vs_separate_noalloc = per_c.as_secs_f64() / per_d.as_secs_f64();
    //         let noalloc_vs_alloc_separate = per_a.as_secs_f64() / per_c.as_secs_f64();
    //         let noalloc_vs_alloc_fused = per_b.as_secs_f64() / per_e.as_secs_f64();

    //         println!();
    //         println!("Analysis:");
    //         println!(
    //             "  Fused vs Separate (alloc):    {:.2}x {}",
    //             fused_vs_separate_alloc,
    //             if fused_vs_separate_alloc > 1.0 {
    //                 "(fused wins)"
    //             } else {
    //                 "(separate wins)"
    //             }
    //         );
    //         println!(
    //             "  Fused vs Separate (no-alloc): {:.2}x {}",
    //             fused_vs_separate_noalloc,
    //             if fused_vs_separate_noalloc > 1.0 {
    //                 "(fused wins)"
    //             } else {
    //                 "(separate wins)"
    //             }
    //         );
    //         println!(
    //             "  No-alloc vs Alloc (separate): {:.2}x {}",
    //             noalloc_vs_alloc_separate,
    //             if noalloc_vs_alloc_separate > 1.0 {
    //                 "(no-alloc wins)"
    //             } else {
    //                 "(alloc wins)"
    //             }
    //         );
    //         println!(
    //             "  No-alloc vs Alloc (fused):    {:.2}x {}",
    //             noalloc_vs_alloc_fused,
    //             if noalloc_vs_alloc_fused > 1.0 {
    //                 "(no-alloc wins)"
    //             } else {
    //                 "(alloc wins)"
    //             }
    //         );
    //     }

    //     println!("\n");
    //     println!("╔══════════════════════════════════════════════════════════════════════╗");
    //     println!("║                           SUMMARY                                    ║");
    //     println!("╚══════════════════════════════════════════════════════════════════════╝");
    //     println!();
    //     println!("LEGEND:");
    //     println!("  A = Current EncoderSelfAttention (3 separate, allocating)");
    //     println!("  B = Fused QKV with slice copy (allocating)");
    //     println!("  C = 3 separate matmuls with scratch buffers (no-alloc)");
    //     println!("  D = Fused QKV with scratch + split (no-alloc)");
    //     println!("  E = Fused QKV returning views (no-alloc, zero copy)");
    //     println!();
    //     println!("KEY INSIGHTS:");
    //     println!("  - For DECODE (1 token): Allocation overhead dominates, no-alloc wins");
    //     println!("  - For SMALL batches: Separate matmuls may win due to split overhead");
    //     println!("  - For LARGE batches: Fused matmul wins (better memory bandwidth)");
    //     println!("  - Strategy E (views) is fastest when downstream can use non-contiguous data");
    // }

    // =========================================================================
    // BENCHMARK: Specific Recommendations
    // =========================================================================

    // #[test]
    // #[ignore] // Ignored by default due to long runtime
    // fn bench_qkv_recommendation_matrix() {
    //     println!("\n");
    //     println!("╔══════════════════════════════════════════════════════════════════════╗");
    //     println!("║           QKV STRATEGY RECOMMENDATION MATRIX                         ║");
    //     println!("║           Finding optimal strategy per workload                      ║");
    //     println!("╚══════════════════════════════════════════════════════════════════════╝");

    //     // Test matrix: tokens × hidden
    //     let token_counts = [1, 8, 32, 128, 512, 1024, 2048, 2880];
    //     let hidden_sizes = [256, 384, 512, 768, 1024];

    //     println!();
    //     println!("Winner Matrix (C=3 separate no-alloc, D=fused no-alloc, E=fused views):");
    //     println!();
    //     print!("{:>8} |", "tokens");
    //     for h in &hidden_sizes {
    //         print!(" h={:<4} |", h);
    //     }
    //     println!();
    //     println!("{}", "-".repeat(8 + hidden_sizes.len() * 9));

    //     for &tokens in &token_counts {
    //         print!("{:>8} |", tokens);

    //         for &hidden in &hidden_sizes {
    //             // Create layers
    //             let q_proj = make_linear_layer(hidden, hidden, 0, true);
    //             let k_proj = make_linear_layer(hidden, hidden, 1000, true);
    //             let v_proj = make_linear_layer(hidden, hidden, 2000, true);

    //             let fused_weights = {
    //                 let mut fused = Array2::zeros((3 * hidden, hidden));
    //                 fused
    //                     .slice_mut(s![0..hidden, ..])
    //                     .assign(&q_proj.weights_view());
    //                 fused
    //                     .slice_mut(s![hidden..2 * hidden, ..])
    //                     .assign(&k_proj.weights_view());
    //                 fused
    //                     .slice_mut(s![2 * hidden..3 * hidden, ..])
    //                     .assign(&v_proj.weights_view());
    //                 fused
    //             };
    //             let fused_bias = match (q_proj.bias(), k_proj.bias(), v_proj.bias()) {
    //                 (Some(qb), Some(kb), Some(vb)) => {
    //                     let mut fused = Array1::zeros(3 * hidden);
    //                     fused.slice_mut(s![0..hidden]).assign(qb);
    //                     fused.slice_mut(s![hidden..2 * hidden]).assign(kb);
    //                     fused.slice_mut(s![2 * hidden..3 * hidden]).assign(vb);
    //                     Some(fused)
    //                 }
    //                 _ => None,
    //             };
    //             let qkv_proj = LinearLayer::new_f32(fused_weights, fused_bias);

    //             let input = make_input(tokens, hidden, 42);

    //             let mut q_scratch = Array2::zeros((tokens, hidden));
    //             let mut k_scratch = Array2::zeros((tokens, hidden));
    //             let mut v_scratch = Array2::zeros((tokens, hidden));
    //             let mut qkv_scratch = Array2::zeros((tokens, 3 * hidden));

    //             let iterations = if tokens <= 32 { 200 } else { 50 };
    //             let warmup = iterations / 10;

    //             // Warmup
    //             for _ in 0..warmup {
    //                 strategy_c_separate_noalloc(
    //                     &input.view(),
    //                     &q_proj,
    //                     &k_proj,
    //                     &v_proj,
    //                     &mut q_scratch,
    //                     &mut k_scratch,
    //                     &mut v_scratch,
    //                 );
    //                 strategy_d_fused_noalloc(
    //                     &input.view(),
    //                     &qkv_proj,
    //                     &mut qkv_scratch,
    //                     &mut q_scratch,
    //                     &mut k_scratch,
    //                     &mut v_scratch,
    //                     hidden,
    //                 );
    //                 let _ =
    //                     strategy_e_fused_views(&input.view(), &qkv_proj, &mut qkv_scratch, hidden);
    //             }

    //             // Benchmark C (3 separate no-alloc)
    //             let start = Instant::now();
    //             for _ in 0..iterations {
    //                 strategy_c_separate_noalloc(
    //                     &input.view(),
    //                     &q_proj,
    //                     &k_proj,
    //                     &v_proj,
    //                     &mut q_scratch,
    //                     &mut k_scratch,
    //                     &mut v_scratch,
    //                 );
    //                 std::hint::black_box(&q_scratch);
    //             }
    //             let time_c = start.elapsed();

    //             // Benchmark D (fused no-alloc)
    //             let start = Instant::now();
    //             for _ in 0..iterations {
    //                 strategy_d_fused_noalloc(
    //                     &input.view(),
    //                     &qkv_proj,
    //                     &mut qkv_scratch,
    //                     &mut q_scratch,
    //                     &mut k_scratch,
    //                     &mut v_scratch,
    //                     hidden,
    //                 );
    //                 std::hint::black_box(&q_scratch);
    //             }
    //             let time_d = start.elapsed();

    //             // Benchmark E (fused views)
    //             let start = Instant::now();
    //             for _ in 0..iterations {
    //                 let result =
    //                     strategy_e_fused_views(&input.view(), &qkv_proj, &mut qkv_scratch, hidden);
    //                 std::hint::black_box(result);
    //             }
    //             let time_e = start.elapsed();

    //             // Determine winner
    //             let winner = if time_c <= time_d && time_c <= time_e {
    //                 "C"
    //             } else if time_d <= time_e {
    //                 "D"
    //             } else {
    //                 "E"
    //             };

    //             let speedup = time_c.as_secs_f64() / time_d.min(time_e).as_secs_f64();

    //             if winner == "C" {
    //                 print!("  {:>3}    |", winner);
    //             } else {
    //                 print!(" {:>3}{:.1}x |", winner, speedup);
    //             }
    //         }
    //         println!();
    //     }

    //     println!();
    //     println!("Legend:");
    //     println!("  C = 3 separate matmuls (no-alloc) - best when split overhead > matmul savings");
    //     println!("  D = 1 fused matmul + split (no-alloc) - best when you need contiguous Q,K,V");
    //     println!(
    //         "  E = 1 fused matmul, return views - best when downstream handles non-contiguous"
    //     );
    //     println!("  Speedup shown is vs strategy C (separate)");
    // }

    // =========================================================================
    // BENCHMARK: Using QKVProjection API
    // =========================================================================

    // #[test]
    // fn bench_qkv_projection_api() {
    //     println!("\n");
    //     println!("╔══════════════════════════════════════════════════════════════════════╗");
    //     println!("║           QKVProjection API BENCHMARK                                ║");
    //     println!("║           Testing the high-level API                                 ║");
    //     println!("╚══════════════════════════════════════════════════════════════════════╝");

    //     let configs = [
    //         // (name, tokens, hidden, intermediate, num_heads, iterations, warmup)
    //         ("Decode", 1, 384, 1536, 6, 1000, 100),
    //         ("Small batch", 64, 384, 1536, 6, 200, 20),
    //         ("MiniLM batch", 2880, 384, 1536, 6, 50, 5),
    //         ("BERT-base", 2048, 768, 3072, 12, 30, 3),
    //     ];

    //     for (name, tokens, hidden, intermediate, num_heads, iterations, warmup) in configs {
    //         let q = make_linear_layer(hidden, hidden, 0, true);
    //         let k = make_linear_layer(hidden, hidden, 1000, true);
    //         let v = make_linear_layer(hidden, hidden, 2000, true);

    //         let q2 = make_linear_layer(hidden, hidden, 0, true);
    //         let k2 = make_linear_layer(hidden, hidden, 1000, true);
    //         let v2 = make_linear_layer(hidden, hidden, 2000, true);

    //         let fused_proj = QKVProjection::new_fused(q, k, v);
    //         let separate_proj = QKVProjection::new_separate(q2, k2, v2);

    //         let input = make_input(tokens, hidden, 42);
    //         // Calculate max_batch and max_seq such that max_batch * max_seq >= tokens
    //         let max_batch = 1;
    //         let max_seq = tokens;

    //         // CORRECT argument order: max_batch, max_seq, hidden, num_heads, intermediate, use_fused
    //         let mut buffers_fused =
    //             EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, true);
    //         let mut buffers_sep =
    //             EncoderBuffers::new(max_batch, max_seq, hidden, num_heads, intermediate, false);

    //         // Warmup
    //         for _ in 0..warmup {
    //             let _ = separate_proj.forward(&input.view());
    //             let _ = fused_proj.forward(&input.view());
    //             separate_proj.forward_noalloc(&input.view(), &mut buffers_sep);
    //             fused_proj.forward_noalloc(&input.view(), &mut buffers_fused);
    //         }

    //         // Separate (alloc)
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             let result = separate_proj.forward(&input.view());
    //             std::hint::black_box(result);
    //         }
    //         let time_sep_alloc = start.elapsed();

    //         // Fused (alloc)
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             let result = fused_proj.forward(&input.view());
    //             std::hint::black_box(result);
    //         }
    //         let time_fused_alloc = start.elapsed();

    //         // Separate (no-alloc)
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             separate_proj.forward_noalloc(&input.view(), &mut buffers_sep);
    //             std::hint::black_box(&buffers_sep.q);
    //         }
    //         let time_sep_noalloc = start.elapsed();

    //         // Fused (no-alloc)
    //         let start = Instant::now();
    //         for _ in 0..iterations {
    //             fused_proj.forward_noalloc(&input.view(), &mut buffers_fused);
    //             std::hint::black_box(&buffers_fused.q);
    //         }
    //         let time_fused_noalloc = start.elapsed();

    //         let per_sep_alloc = time_sep_alloc / iterations as u32;
    //         let per_fused_alloc = time_fused_alloc / iterations as u32;
    //         let per_sep_noalloc = time_sep_noalloc / iterations as u32;
    //         let per_fused_noalloc = time_fused_noalloc / iterations as u32;

    //         println!("\n=== {} (tokens={}, hidden={}) ===", name, tokens, hidden);
    //         println!("Separate (alloc):    {:>9.2?}", per_sep_alloc);
    //         println!(
    //             "Fused (alloc):       {:>9.2?}  ({:.2}x vs separate)",
    //             per_fused_alloc,
    //             per_sep_alloc.as_secs_f64() / per_fused_alloc.as_secs_f64()
    //         );
    //         println!(
    //             "Separate (no-alloc): {:>9.2?}  ({:.2}x vs alloc)",
    //             per_sep_noalloc,
    //             per_sep_alloc.as_secs_f64() / per_sep_noalloc.as_secs_f64()
    //         );
    //         println!(
    //             "Fused (no-alloc):    {:>9.2?}  ({:.2}x vs alloc)",
    //             per_fused_noalloc,
    //             per_fused_alloc.as_secs_f64() / per_fused_noalloc.as_secs_f64()
    //         );

    //         // Best overall
    //         let times = [
    //             (per_sep_alloc, "Separate (alloc)"),
    //             (per_fused_alloc, "Fused (alloc)"),
    //             (per_sep_noalloc, "Separate (no-alloc)"),
    //             (per_fused_noalloc, "Fused (no-alloc)"),
    //         ];
    //         let winner = times.iter().min_by_key(|(t, _)| *t).unwrap();
    //         println!("🏆 Best: {} ({:.2?})", winner.1, winner.0);
    //     }
    // }
}
