//! Rotary Position Embeddings (RoPE)
//!
//! RoPE is used in LLaMA and other modern transformers instead of learned position embeddings.
//! It encodes positional information by rotating query and key vectors in the complex plane.
//!
//! Key properties:
//! - Relative position encoding (distance between tokens matters)
//! - No learned parameters (purely geometric)
//! - Applied to Q and K, not V
//! - Different rotation frequencies for different dimensions

use ndarray::{Array1, Array2, Array3, Array4, s};
use std::f32::consts::PI;

/// Rotary Position Embeddings
pub struct RoPE {
    /// Cosine cache: [max_seq_len, head_dim]
    cos_cache: Array2<f32>,
    /// Sine cache: [max_seq_len, head_dim]
    sin_cache: Array2<f32>,
    /// Dimension of each attention head
    head_dim: usize,
    /// Base value for computing rotation frequencies (typically 10000.0)
    theta: f32,
}

impl RoPE {
    /// Create a new RoPE module
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head (typically hidden_size / num_heads)
    /// * `max_seq_len` - Maximum sequence length to precompute
    /// * `theta` - Base for computing frequencies (10000.0 for LLaMA, 500000.0 for LLaMA-3)
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let (cos_cache, sin_cache) = Self::precompute_freqs(head_dim, max_seq_len, theta);
        Self {
            cos_cache,
            sin_cache,
            head_dim,
            theta,
        }
    }

    /// Precompute rotation frequencies for all positions
    fn precompute_freqs(head_dim: usize, max_seq_len: usize, theta: f32) -> (Array2<f32>, Array2<f32>) {
        // Compute inverse frequencies for each dimension pair
        // freq_i = 1 / (theta ^ (2i / head_dim)) for i in 0..head_dim/2
        let mut inv_freq = Array1::<f32>::zeros(head_dim / 2);
        for i in 0..head_dim / 2 {
            let exponent = (2 * i) as f32 / head_dim as f32;
            inv_freq[i] = 1.0 / theta.powf(exponent);
        }

        let mut cos_cache = Array2::<f32>::zeros((max_seq_len, head_dim));
        let mut sin_cache = Array2::<f32>::zeros((max_seq_len, head_dim));

        // For each position, compute cos and sin of (position * inv_freq)
        for pos in 0..max_seq_len {
            for i in 0..head_dim / 2 {
                let angle = pos as f32 * inv_freq[i];
                // Store cos and sin for each dimension pair
                cos_cache[[pos, 2 * i]] = angle.cos();
                cos_cache[[pos, 2 * i + 1]] = angle.cos();
                sin_cache[[pos, 2 * i]] = angle.sin();
                sin_cache[[pos, 2 * i + 1]] = angle.sin();
            }
        }

        (cos_cache, sin_cache)
    }

    /// Apply rotary embeddings to query and key tensors (4D version)
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `k` - Key tensor [batch, num_heads, seq_len, head_dim]
    /// * `position_offset` - Starting position (for incremental decoding with cache)
    ///
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn apply_4d(
        &self,
        q: &Array4<f32>,
        k: &Array4<f32>,
        position_offset: usize,
    ) -> (Array4<f32>, Array4<f32>) {
        let (batch, num_heads, seq_len, head_dim) = q.dim();
        assert_eq!(head_dim, self.head_dim, "Head dimension mismatch");

        let rotated_q = self.rotate_4d(q, position_offset);
        let rotated_k = self.rotate_4d(k, position_offset);

        (rotated_q, rotated_k)
    }

    /// Apply rotary embeddings to query and key tensors (3D version)
    ///
    /// # Arguments
    /// * `q` - Query tensor [batch, seq_len, hidden_size]
    /// * `k` - Key tensor [batch, seq_len, hidden_size]
    /// * `num_heads` - Number of attention heads
    /// * `position_offset` - Starting position (for incremental decoding with cache)
    ///
    /// # Returns
    /// Tuple of (rotated_q, rotated_k)
    pub fn apply_3d(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        num_heads: usize,
        position_offset: usize,
    ) -> (Array3<f32>, Array3<f32>) {
        let (batch, seq_len, hidden_size) = q.dim();
        assert_eq!(hidden_size, num_heads * self.head_dim, "Hidden size mismatch");

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let q_reshaped = q.to_shape((batch, seq_len, num_heads, self.head_dim)).unwrap();
        let k_reshaped = k.to_shape((batch, seq_len, num_heads, self.head_dim)).unwrap();

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q_transposed = q_reshaped.permuted_axes([0, 2, 1, 3]);
        let k_transposed = k_reshaped.permuted_axes([0, 2, 1, 3]);

        // Apply rotation
        let (rotated_q, rotated_k) = self.apply_4d(
            &q_transposed.to_owned(),
            &k_transposed.to_owned(),
            position_offset,
        );

        // Transpose back to [batch, seq_len, num_heads, head_dim]
        let q_back = rotated_q.permuted_axes([0, 2, 1, 3]);
        let k_back = rotated_k.permuted_axes([0, 2, 1, 3]);

        // Reshape back to [batch, seq_len, hidden_size]
        let q_final = q_back.to_shape((batch, seq_len, hidden_size)).unwrap().to_owned();
        let k_final = k_back.to_shape((batch, seq_len, hidden_size)).unwrap().to_owned();

        (q_final, k_final)
    }

    /// Apply rotation to a 4D tensor
    fn rotate_4d(&self, x: &Array4<f32>, position_offset: usize) -> Array4<f32> {
        let (batch, num_heads, seq_len, head_dim) = x.dim();
        let mut rotated = Array4::<f32>::zeros((batch, num_heads, seq_len, head_dim));

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let pos = position_offset + s;
                    
                    // Get cos and sin for this position
                    let cos = self.cos_cache.slice(s![pos, ..]);
                    let sin = self.sin_cache.slice(s![pos, ..]);
                    
                    // Apply rotation to pairs of dimensions
                    // For each pair (i, i+1), apply rotation matrix:
                    // [cos  -sin] [x_i  ]
                    // [sin   cos] [x_i+1]
                    for i in (0..head_dim).step_by(2) {
                        let x0 = x[[b, h, s, i]];
                        let x1 = x[[b, h, s, i + 1]];
                        
                        rotated[[b, h, s, i]] = x0 * cos[i] - x1 * sin[i];
                        rotated[[b, h, s, i + 1]] = x0 * sin[i + 1] + x1 * cos[i + 1];
                    }
                }
            }
        }

        rotated
    }

    /// Get maximum sequence length supported
    pub fn max_seq_len(&self) -> usize {
        self.cos_cache.shape()[0]
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, Array4};

    #[test]
    fn test_rope_precompute() {
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        // Check cache shapes
        assert_eq!(rope.cos_cache.shape(), &[8, 4]);
        assert_eq!(rope.sin_cache.shape(), &[8, 4]);
        
        // Check that values are finite
        for i in 0..max_seq_len {
            for j in 0..head_dim {
                assert!(rope.cos_cache[[i, j]].is_finite());
                assert!(rope.sin_cache[[i, j]].is_finite());
            }
        }
    }

    #[test]
    fn test_rope_rotation_basic() {
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        // Simple query and key: [1, 1, 1, 4]
        let q = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![1.0, 0.0, 1.0, 0.0],
        ).unwrap();
        let k = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![0.0, 1.0, 0.0, 1.0],
        ).unwrap();
        
        let (rotated_q, rotated_k) = rope.apply_4d(&q, &k, 0);
        
        // Check output shapes
        assert_eq!(rotated_q.shape(), &[1, 1, 1, 4]);
        assert_eq!(rotated_k.shape(), &[1, 1, 1, 4]);
        
        // Check values are finite
        assert!(rotated_q[[0, 0, 0, 0]].is_finite());
        assert!(rotated_k[[0, 0, 0, 0]].is_finite());
    }

    #[test]
    fn test_rope_rotation_identity() {
        // At position 0, rotation should be minimal (close to identity)
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        let q = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let k = q.clone();
        
        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);
        
        // At position 0, first dimension pair should be very close to original
        // cos(0) = 1, sin(0) = 0, so rotation is identity
        assert!((rotated_q[[0, 0, 0, 0]] - 1.0).abs() < 1e-5);
        assert!((rotated_q[[0, 0, 0, 1]] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_position_offset() {
        let head_dim = 4;
        let max_seq_len = 16;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        let q = Array4::from_shape_vec(
            (1, 1, 2, 4),
            vec![
                1.0, 0.0, 1.0, 0.0,  // position 0 (or offset)
                1.0, 0.0, 1.0, 0.0,  // position 1 (or offset+1)
            ],
        ).unwrap();
        let k = q.clone();
        
        // Apply with offset 0
        let (rotated_q_0, _) = rope.apply_4d(&q, &k, 0);
        
        // Apply with offset 5
        let (rotated_q_5, _) = rope.apply_4d(&q, &k, 5);
        
        // Values should be different because positions are different
        assert!((rotated_q_0[[0, 0, 0, 0]] - rotated_q_5[[0, 0, 0, 0]]).abs() > 1e-3);
    }

    #[test]
    fn test_rope_3d_interface() {
        let head_dim = 4;
        let num_heads = 2;
        let hidden_size = head_dim * num_heads;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        // [batch=1, seq_len=2, hidden_size=8]
        let q = Array3::from_shape_vec(
            (1, 2, hidden_size),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,   // pos 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,   // pos 1
            ],
        ).unwrap();
        let k = q.clone();
        
        let (rotated_q, rotated_k) = rope.apply_3d(&q, &k, num_heads, 0);
        
        // Check output shape preserved
        assert_eq!(rotated_q.shape(), &[1, 2, hidden_size]);
        assert_eq!(rotated_k.shape(), &[1, 2, hidden_size]);
    }

    #[test]
    fn test_rope_preserves_norm() {
        // RoPE is a rotation, so it should preserve L2 norm
        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        let q = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![3.0, 4.0, 0.0, 0.0],
        ).unwrap();
        let k = q.clone();
        
        // Original norm: sqrt(9 + 16) = 5.0
        let original_norm: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);
        
        let rotated_norm: f32 = rotated_q.iter().map(|x| x * x).sum::<f32>().sqrt();
        
        // Norm should be preserved (within floating point tolerance)
        assert!((original_norm - rotated_norm).abs() < 1e-4);
    }

    #[test]
    fn test_rope_pytorch_parity() {
        // Test against known PyTorch output
        // PyTorch code:
        // ```python
        // import torch
        // def precompute_freqs_cis(dim, end, theta=10000.0):
        //     freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        //     t = torch.arange(end)
        //     freqs = torch.outer(t, freqs).float()
        //     freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        //     return freqs_cis
        //
        // def apply_rotary_emb(xq, xk, freqs_cis):
        //     xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        //     xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        //     freqs_cis = freqs_cis[:xq_.shape[1]]
        //     xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
        //     xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
        //     return xq_out, xk_out
        //
        // freqs = precompute_freqs_cis(4, 8)
        // q = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])
        // k = torch.tensor([[[[0.0, 1.0, 0.0, 1.0]]]])
        // rq, rk = apply_rotary_emb(q, k, freqs)
        // print(rq, rk)
        // ```

        let head_dim = 4;
        let max_seq_len = 8;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        let q = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![1.0, 0.0, 1.0, 0.0],
        ).unwrap();
        let k = Array4::from_shape_vec(
            (1, 1, 1, 4),
            vec![0.0, 1.0, 0.0, 1.0],
        ).unwrap();
        
        let (rotated_q, rotated_k) = rope.apply_4d(&q, &k, 0);
        
        // At position 0:
        // freq[0] = 1.0 / (10000^(0/4)) = 1.0, angle = 0 * 1.0 = 0
        // freq[1] = 1.0 / (10000^(2/4)) = 0.01, angle = 0 * 0.01 = 0
        // So cos ≈ [1.0, 1.0, 1.0, 1.0], sin ≈ [0.0, 0.0, 0.0, 0.0]
        
        // For q = [1, 0, 1, 0]:
        // Pair 0: [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
        // Pair 1: [1*1 - 0*0, 1*0 + 0*1] = [1, 0]
        assert!((rotated_q[[0, 0, 0, 0]] - 1.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 1]] - 0.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 2]] - 1.0).abs() < 1e-3);
        assert!((rotated_q[[0, 0, 0, 3]] - 0.0).abs() < 1e-3);
    }

    #[test]
    fn test_rope_multiple_positions() {
        let head_dim = 8;
        let max_seq_len = 16;
        let theta = 10000.0;
        
        let rope = RoPE::new(head_dim, max_seq_len, theta);
        
        // Test with a sequence of length 3
        let q = Array4::from_shape_vec(
            (1, 1, 3, 8),
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  // pos 0
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  // pos 1
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,  // pos 2
            ],
        ).unwrap();
        let k = q.clone();
        
        let (rotated_q, _) = rope.apply_4d(&q, &k, 0);
        
        // Each position should have different rotations
        let pos0_val = rotated_q[[0, 0, 0, 0]];
        let pos1_val = rotated_q[[0, 0, 1, 0]];
        let pos2_val = rotated_q[[0, 0, 2, 0]];
        
        assert!((pos0_val - pos1_val).abs() > 1e-5);
        assert!((pos1_val - pos2_val).abs() > 1e-5);
    }
}