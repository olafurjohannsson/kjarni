pub use crate::traits::Cache;
use ndarray::Array3;
use std::any::Any;
use std::sync::Arc;

/// CPU-based KV cache storing key and value tensors in ndarray format
pub struct CpuKVCache {
    /// Cached key states per layer: Vec[num_layers] of (batch, seq_len, hidden_size)
    pub keys: Vec<Option<Array3<f32>>>,

    /// Cached value states per layer: Vec[num_layers] of (batch, seq_len, hidden_size)
    pub values: Vec<Option<Array3<f32>>>,

    /// Current sequence length
    seq_length: usize,
}

impl CpuKVCache {
    /// Create a new empty cache with the specified number of layers
    pub fn new(num_layers: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            seq_length: 0,
        }
    }

    /// Update the cache for a specific layer by appending new K, V states
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_keys: Array3<f32>,
        new_values: Array3<f32>,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.keys.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }

        // Concatenate with existing cache along sequence dimension
        self.keys[layer_idx] = Some(match &self.keys[layer_idx] {
            None => new_keys.clone(),
            Some(cached) => {
                ndarray::concatenate(ndarray::Axis(1), &[cached.view(), new_keys.view()])?
            }
        });

        self.values[layer_idx] = Some(match &self.values[layer_idx] {
            None => new_values.clone(),
            Some(cached) => {
                ndarray::concatenate(ndarray::Axis(1), &[cached.view(), new_values.view()])?
            }
        });

        // Update sequence length from first layer
        if layer_idx == 0 {
            self.seq_length = self.keys[0].as_ref().unwrap().shape()[1];
        }

        Ok(())
    }

    /// Get cached K, V for a specific layer (returns None if not cached yet)
    pub fn get(&self, layer_idx: usize) -> Option<(&Array3<f32>, &Array3<f32>)> {
        if layer_idx >= self.keys.len() {
            return None;
        }

        match (&self.keys[layer_idx], &self.values[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }
}

impl Cache for CpuKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    fn clear(&mut self) {
        for k in &mut self.keys {
            *k = None;
        }
        for v in &mut self.values {
            *v = None;
        }
        self.seq_length = 0;
    }
}