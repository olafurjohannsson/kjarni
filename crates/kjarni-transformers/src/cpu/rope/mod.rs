//! Rotary Position Embeddings (RoPE) for LLaMA-style transformers.

use std::f32::consts::PI;

use ndarray::{Array1, Array2, Array3, Array4};

use crate::cpu::kernels::x86::rope_avx2;
use crate::models::base::RopeScalingConfig;

pub struct RoPE {
    pub cos_cache: Array2<f32>,
    pub sin_cache: Array2<f32>,
    pub head_dim: usize,
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

        let (cos_cache, sin_cache) = Self::build_cache(max_seq_len, &inv_freq);

        Self {
            cos_cache,
            sin_cache,
            head_dim,
            theta,
        }
    }

    fn calculate_inv_freq_base(head_dim: usize, theta: f32) -> Array1<f32> {
        Array1::from_iter((0..head_dim / 2).map(|i| {
            let exponent = (2 * i) as f32 / head_dim as f32;
            1.0 / theta.powf(exponent)
        }))
    }

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
        let head_dim = half_dim * 2;

        let mut cos_cache = Array2::<f32>::zeros((max_seq_len, head_dim));
        let mut sin_cache = Array2::<f32>::zeros((max_seq_len, head_dim));

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                let cos_val = angle.cos();
                let sin_val = angle.sin();

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

        if x.is_standard_layout() {
            let x_slice = x.as_slice_mut().expect("array should be contiguous");
            let cos_slice = self.cos_cache.as_slice().expect("cache should be contiguous");
            let sin_slice = self.sin_cache.as_slice().expect("cache should be contiguous");

            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe {
                    rope_avx2::rotate_4d_avx2(
                        x_slice,
                        cos_slice,
                        sin_slice,
                        batch,
                        num_heads,
                        seq_len,
                        head_dim,
                        self.cos_cache.ncols(),
                        position_offset,
                    );
                }
                return;
            }
        }

        self.rotate_4d_in_place_scalar(x, position_offset);
    }

    fn rotate_4d_in_place_scalar(&self, x: &mut Array4<f32>, position_offset: usize) {
        let (batch, num_heads, seq_len, head_dim) = x.dim();
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

                        x[[b, h, s, i]] = x0 * cos - x1 * sin;
                        x[[b, h, s, i + half_dim]] = x0 * sin + x1 * cos;
                    }
                }
            }
        }
    }

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

    pub fn apply_3d(
        &self,
        q: &Array3<f32>,
        k: &Array3<f32>,
        num_q_heads: usize,
        num_kv_heads: usize,
        position_offset: usize,
    ) -> anyhow::Result<(Array3<f32>, Array3<f32>)> {
        let (batch, seq_len, hidden_size_q) = q.dim();
        let (_, _, hidden_size_k) = k.dim();

        let q_reshaped = q
            .to_shape((batch, seq_len, num_q_heads, self.head_dim))
            .map_err(|e| anyhow::anyhow!("failed to reshape Q in RoPE: {}", e))?;
        let k_reshaped = k
            .to_shape((batch, seq_len, num_kv_heads, self.head_dim))
            .map_err(|e| anyhow::anyhow!("failed to reshape K in RoPE: {}", e))?;

        let mut q_transposed = q_reshaped.permuted_axes([0, 2, 1, 3]).to_owned();
        let mut k_transposed = k_reshaped.permuted_axes([0, 2, 1, 3]).to_owned();

        self.rotate_4d_in_place(&mut q_transposed, position_offset);
        self.rotate_4d_in_place(&mut k_transposed, position_offset);

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

    pub fn max_seq_len(&self) -> usize {
        self.cos_cache.shape()[0]
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}

#[cfg(test)]
mod tests;