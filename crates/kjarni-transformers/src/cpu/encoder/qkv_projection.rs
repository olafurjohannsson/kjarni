//! QKV Projection with adaptive fused/separate strategy.
//!
//! Supports both fused (1 matmul) and separate (3 matmuls) strategies,
//! with automatic selection based on hidden size.

use ndarray::{Array1, Array2, ArrayView2, s};

use crate::cpu::encoder::buffers::EncoderBuffers;
use crate::linear_layer::LinearLayer;

/// QKV Projection supporting both fused (1 matmul) and separate (3 matmul) strategies.
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
    /// Creates projection
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
    pub fn new_fused(q: LinearLayer, k: LinearLayer, v: LinearLayer) -> Self {
        let hidden_size = q.out_features();
        let qkv_proj = Self::fuse_layers(&q, &k, &v);

        Self {
            storage: QKVStorage::Fused { qkv_proj },
            hidden_size,
        }
    }

    /// Creates separate projection (three independent matmuls).
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

    /// Forward pass writing to pre-allocated EncoderBuffers
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
                {
                    let scratch = buffers.qkv_scratch.as_mut().expect(
                        "qkv_scratch not allocated - buffers created with use_fused_qkv=false",
                    );
                    qkv_proj.matmul_noalloc(hidden, scratch);
                }

                let h = self.hidden_size;
                let qkv_slice = buffers.qkv_scratch.as_ref().unwrap().as_slice().unwrap();
                let q_slice = buffers.q.as_slice_mut().unwrap();
                let k_slice = buffers.k.as_slice_mut().unwrap();
                let v_slice = buffers.v.as_slice_mut().unwrap();

                // Split across threads, a block of tokens each: serially this
                // was half the cost of the projection on a large batch.
                const ROWS: usize = 64;
                let split =
                    |(((qkv, q), k), v): (((&[f32], &mut [f32]), &mut [f32]), &mut [f32])| {
                        for (t, row) in qkv.chunks_exact(3 * h).enumerate() {
                            q[t * h..(t + 1) * h].copy_from_slice(&row[..h]);
                            k[t * h..(t + 1) * h].copy_from_slice(&row[h..2 * h]);
                            v[t * h..(t + 1) * h].copy_from_slice(&row[2 * h..]);
                        }
                    };
                let (qkv, q, k, v) = (
                    &qkv_slice[..tokens * 3 * h],
                    &mut q_slice[..tokens * h],
                    &mut k_slice[..tokens * h],
                    &mut v_slice[..tokens * h],
                );
                if tokens > ROWS {
                    use rayon::prelude::*;
                    qkv.par_chunks(ROWS * 3 * h)
                        .zip(q.par_chunks_mut(ROWS * h))
                        .zip(k.par_chunks_mut(ROWS * h))
                        .zip(v.par_chunks_mut(ROWS * h))
                        .for_each(split);
                } else {
                    split((((qkv, q), k), v));
                }
            }
        }
    }

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

    /// Forward pass returning separate Q, K, V
    pub fn forward_qkv(&self, hidden: &ArrayView2<f32>) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
        self.forward(hidden)
    }

    #[expect(
        dead_code,
        reason = "not referenced yet; kept until the path that needs it lands"
    )]
    /// split of fused QKV buffer
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

    /// Fuse Q, K, V biases into single [3*hidden] vector. A projection
    /// without a bias, such as Whisper's `k_proj`, contributes zeros.
    fn fuse_qkv_bias(q: &LinearLayer, k: &LinearLayer, v: &LinearLayer) -> Option<Array1<f32>> {
        if q.bias().is_none() && k.bias().is_none() && v.bias().is_none() {
            return None;
        }
        let out = q.out_features();
        let mut fused = Array1::zeros(3 * out);
        for (i, layer) in [q, k, v].into_iter().enumerate() {
            if let Some(b) = layer.bias() {
                fused.slice_mut(s![i * out..(i + 1) * out]).assign(b);
            }
        }
        Some(fused)
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
    use ndarray::{Array1, Array2};

    fn create_test_layer(in_features: usize, out_features: usize) -> LinearLayer {
        let weights = Array2::from_elem((out_features, in_features), 0.01f32);
        LinearLayer::new_f32(weights, None)
    }

    /// A layer of pseudo-random weights, with or without a bias.
    fn layer(in_f: usize, out_f: usize, seed: f32, bias: bool) -> LinearLayer {
        let w = Array2::from_shape_fn((out_f, in_f), |(o, i)| {
            ((o * 7 + i * 3) as f32 * seed).sin()
        });
        let b = bias.then(|| Array1::from_shape_fn(out_f, |o| (o as f32 * seed).cos()));
        LinearLayer::new_f32(w, b)
    }

    /// Whisper's q_proj and v_proj have a bias and its k_proj does not. Fused,
    /// the projections lost every bias, which put the CPU encoder of
    /// whisper-tiny and whisper-base off from PyTorch from the first layer.
    #[test]
    fn fused_projection_keeps_q_and_v_biases_when_k_has_none() {
        let (hidden, tokens) = (384, 5);
        let x = Array2::from_shape_fn((tokens, hidden), |(t, i)| {
            ((t * hidden + i) as f32 * 0.01).sin()
        });
        let (q, k, v) = (
            layer(hidden, hidden, 0.1, true),
            layer(hidden, hidden, 0.2, false),
            layer(hidden, hidden, 0.3, true),
        );
        let want = (
            q.matmul(&x.view()),
            k.matmul(&x.view()),
            v.matmul(&x.view()),
        );
        let fused = QKVProjection::new_fused(q, k, v);
        assert!(fused.is_fused());
        let (gq, gk, gv) = fused.forward(&x.view());
        for (got, want) in [(gq, want.0), (gk, want.1), (gv, want.2)] {
            let diff = (&got - &want).mapv(f32::abs).fold(0.0f32, |m, &d| m.max(d));
            assert!(diff < 1e-4, "max difference {diff}");
        }
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
        let (q_alloc, _k_alloc, _v_alloc) = qkv.forward(&input.view());

        // Forward without allocation
        qkv.forward_noalloc(&input.view(), &mut buffers);

        // Compare results - use max difference
        let q_diff = (&q_alloc - &buffers.q.slice(s![..tokens, ..]))
            .mapv(|x| x.abs())
            .fold(0.0f32, |a, &b| a.max(b));

        assert!(q_diff < 1e-5, "Q mismatch: {}", q_diff);
    }
}

// TESTS

#[cfg(test)]
mod qkv_projection_tests {
    use super::*;

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

        // Fused and separate take different tiles, so sums round differently per CPU:
        // 8e-6 on AVX2 and 1.2e-5 on NEON.
        assert!(q_diff < 2e-5, "Q mismatch: {}", q_diff);
        assert!(k_diff < 2e-5, "K mismatch: {}", k_diff);
        assert!(v_diff < 2e-5, "V mismatch: {}", v_diff);
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
