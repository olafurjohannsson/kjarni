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
pub mod loader;
use crate::models::base::RopeScalingConfig;
use ndarray::{s, Array1, Array2, Array3, Array4};
use std::f32::consts::PI;

/// Rotary Position Embeddings
pub struct RoPE {
    /// Cosine cache: [max_seq_len, head_dim]
    pub cos_cache: Array2<f32>,
    /// Sine cache: [max_seq_len, head_dim]
    pub sin_cache: Array2<f32>,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Base value for computing rotation frequencies (typically 10000.0)
    pub theta: f32,
}

impl RoPE {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self::new_with_scaling(head_dim, max_seq_len, theta, None)
    }

    pub fn new_with_scaling(
        head_dim: usize,
        max_seq_len: usize,
        theta: f32,
        rope_scaling: Option<&RopeScalingConfig>,
    ) -> Self {
        // This is now the single source of truth for frequency calculation.
        let inv_freq = if let Some(scaling) = rope_scaling {
            if scaling.rope_type == "llama3" {
                Self::calculate_inv_freq_llama3(
                    head_dim,
                    theta,
                    scaling.factor,
                    scaling.low_freq_factor,
                    scaling.high_freq_factor,
                    scaling.original_max_position_embeddings,
                )
            } else {
                Self::calculate_inv_freq_base(head_dim, theta)
            }
        } else {
            Self::calculate_inv_freq_base(head_dim, theta)
        };

        // This is now the single source of truth for cache generation.
        let (cos_cache, sin_cache) = Self::build_cache(max_seq_len, &inv_freq);

        Self {
            cos_cache,
            sin_cache,
            head_dim,
            theta,
        }
    }

    /// Base inverse frequency calculation (non-scaled).
    fn calculate_inv_freq_base(head_dim: usize, theta: f32) -> Array1<f32> {
        Array1::from_iter((0..head_dim / 2).map(|i| {
            let exponent = (2 * i) as f32 / head_dim as f32;
            1.0 / theta.powf(exponent)
        }))
    }

    /// LLaMA 3 scaled inverse frequency calculation.
    fn calculate_inv_freq_llama3(
        head_dim: usize,
        theta: f32,
        factor: f32,
        low_freq_factor: f32,
        high_freq_factor: f32,
        original_max_position_embeddings: usize,
    ) -> Array1<f32> {
        let half_dim = head_dim / 2;
        let mut inv_freq = Array1::<f32>::zeros(half_dim);

        let low_freq_wavelen = original_max_position_embeddings as f32 / low_freq_factor;
        let high_freq_wavelen = original_max_position_embeddings as f32 / high_freq_factor;

        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / head_dim as f32;
            let base_freq = 1.0 / theta.powf(exponent);
            let wavelen = 2.0 * PI / base_freq;

            let scaled_freq = if wavelen < high_freq_wavelen {
                base_freq
            } else if wavelen > low_freq_wavelen {
                base_freq / factor
            } else {
                let smooth = (original_max_position_embeddings as f32 / wavelen - low_freq_factor)
                    / (high_freq_factor - low_freq_factor);
                base_freq / ((1.0 - smooth) * factor + smooth)
            };
            inv_freq[i] = scaled_freq;
        }
        inv_freq
    }

    fn build_cache(max_seq_len: usize, inv_freq: &Array1<f32>) -> (Array2<f32>, Array2<f32>) {
        let half_dim = inv_freq.len();
        let head_dim = half_dim * 2; // The full head dimension
        //[128000, 791, 2115, 315, 59294, 22107, 706, 3970, 264, 2763, 315, 5208, 304, 279, 1566, 2478, 1667]
        // Create caches with the full head_dim
        let mut cos_cache = Array2::<f32>::zeros((max_seq_len, head_dim));
        let mut sin_cache = Array2::<f32>::zeros((max_seq_len, head_dim));

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                let cos_val = angle.cos();
                let sin_val = angle.sin();

                // Apply the same rotation to both halves of the dimension
                cos_cache[[pos, i]] = cos_val;
                sin_cache[[pos, i]] = sin_val;
                cos_cache[[pos, i + half_dim]] = cos_val;
                sin_cache[[pos, i + half_dim]] = sin_val;
            }
        }
        (cos_cache, sin_cache)
    }
    fn rotate_4d_in_place(&self, x: &mut Array4<f32>, position_offset: usize) {
        let (batch, num_heads, seq_len, head_dim) = x.dim();
        let half_dim = head_dim / 2;

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let pos = position_offset + s;

                    for i in 0..half_dim {
                        let cos = self.cos_cache[[pos, i]];
                        let sin = self.sin_cache[[pos, i]];

                        // CRITICAL: We must read both values from the tensor *before*
                        // we write either of them back. Otherwise, the second calculation
                        // would use the already-modified first value.
                        let x0 = x[[b, h, s, i]];
                        let x1 = x[[b, h, s, i + half_dim]];

                        x[[b, h, s, i]] = x0 * cos - x1 * sin;
                        x[[b, h, s, i + half_dim]] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
        // No return value is needed as `x` was modified directly.
    }
    /// This is the correct "Rotate Half" implementation.
    pub fn rotate_4d(&self, x: &Array4<f32>, position_offset: usize) -> Array4<f32> {
        let mut rotated = x.to_owned();
        let (batch, num_heads, seq_len, head_dim) = rotated.dim();
        let half_dim = head_dim / 2;

        for b in 0..batch {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    let pos = position_offset + s;

                    for i in 0..half_dim {
                        let cos = self.cos_cache[[pos, i]];
                        let sin = self.sin_cache[[pos, i]];

                        let x0 = x[[b, h, s, i]];
                        let x1 = x[[b, h, s, i + half_dim]];

                        rotated[[b, h, s, i]] = x0 * cos - x1 * sin;
                        rotated[[b, h, s, i + half_dim]] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
        rotated
    }

    // --- The rest of your methods (apply_4d, apply_3d, etc.) remain unchanged ---
    pub fn apply_4d(
        &self,
        q: &Array4<f32>,
        k: &Array4<f32>,
        position_offset: usize,
    ) -> (Array4<f32>, Array4<f32>) {
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
        num_q_heads: usize,
        num_kv_heads: usize, // ✅ New parameter
        position_offset: usize,
    ) -> anyhow::Result<(Array3<f32>, Array3<f32>)> {
        // Make sure it returns a Result
        let (batch, seq_len, hidden_size_q) = q.dim();
        let (_, _, hidden_size_k) = k.dim();

        // Reshape Q to [batch, seq_len, num_q_heads, head_dim]
        let q_reshaped = q
            .to_shape((batch, seq_len, num_q_heads, self.head_dim))
            .map_err(|e| anyhow::anyhow!("Failed to reshape Q in RoPE: {}", e))?; // Return proper error
        let k_reshaped = k
            .to_shape((batch, seq_len, num_kv_heads, self.head_dim))
            .map_err(|e| anyhow::anyhow!("Failed to reshape K in RoPE: {}", e))?;

        // Transpose to [batch, heads, seq_len, head_dim]
        let mut q_transposed = q_reshaped.permuted_axes([0, 2, 1, 3]).to_owned();
        let mut k_transposed = k_reshaped.permuted_axes([0, 2, 1, 3]).to_owned();

        self.rotate_4d_in_place(&mut q_transposed, position_offset);
        self.rotate_4d_in_place(&mut k_transposed, position_offset);

        // Transpose and reshape back
        let q_final = q_transposed
            .permuted_axes([0, 2, 1, 3])
            .into_shape_with_order((batch, seq_len, hidden_size_q))?
            .to_owned();
        let k_final = k_transposed
            .permuted_axes([0, 2, 1, 3])
            .into_shape_with_order((batch, seq_len, hidden_size_k))?
            .to_owned();
        Ok((q_final, k_final))
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
mod tests;
