//! Pooling traits and implementations for encoder outputs

use super::config::PoolingStrategy;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use anyhow::Result;
use ndarray::{Array2, Array3};

/// CPU pooling head - reduces sequence to single vector.
pub trait CpuPooler: Send + Sync {
    /// Pool hidden states to a single vector per batch item.
    ///
    /// # Arguments
    /// * `hidden_states` - `[batch_size, seq_len, hidden_size]`
    /// * `attention_mask` - `[batch_size, seq_len]` (for mean pooling)
    /// * `strategy` - Pooling strategy to use
    ///
    /// # Returns
    /// Pooled output `[batch_size, hidden_size]`
    fn pool(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        strategy: &PoolingStrategy,
    ) -> Result<Array2<f32>>;
}

/// GPU pooling head - reduces sequence to single vector.
pub trait GpuPooler: Send + Sync {
    /// Pool hidden states to a single vector per batch item.
    fn pool(
        &self,
        cmd: &mut wgpu::CommandEncoder,
        pool: &mut GpuTensorPool,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        strategy: &PoolingStrategy,
    ) -> Result<GpuTensor>;
}

// ============================================================================
// IMPLEMENTATIONS
// ============================================================================

/// Standard CPU pooler implementation.
pub struct StandardCpuPooler;

impl CpuPooler for StandardCpuPooler {
    fn pool(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        strategy: &PoolingStrategy,
    ) -> Result<Array2<f32>> {
        match strategy {
            PoolingStrategy::Cls => {
                // Take first token: [batch, hidden]
                Ok(hidden_states.slice(ndarray::s![.., 0, ..]).to_owned())
            }
            PoolingStrategy::Mean => {
                crate::pooling::mean_pool(hidden_states, attention_mask)
            }
            PoolingStrategy::Max => {
                // Max over sequence dimension
                let (batch, _seq, hidden) = hidden_states.dim();
                let mut result = Array2::zeros((batch, hidden));
                for b in 0..batch {
                    for h in 0..hidden {
                        let mut max_val = f32::NEG_INFINITY;
                        for s in 0..hidden_states.dim().1 {
                            if attention_mask[[b, s]] > 0.0 {
                                max_val = max_val.max(hidden_states[[b, s, h]]);
                            }
                        }
                        result[[b, h]] = max_val;
                    }
                }
                Ok(result)
            }
            PoolingStrategy::LastToken => {
                // Find last non-padding token for each batch
                let (batch, seq, hidden) = hidden_states.dim();
                let mut result = Array2::zeros((batch, hidden));
                for b in 0..batch {
                    let last_idx = (0..seq)
                        .rev()
                        .find(|&s| attention_mask[[b, s]] > 0.0)
                        .unwrap_or(0);
                    for h in 0..hidden {
                        result[[b, h]] = hidden_states[[b, last_idx, h]];
                    }
                }
                Ok(result)
            }
        }
    }
}