pub use crate::traits::Cache;
use ndarray::Array3;
use std::any::Any;

/// CPU-based KV cache storing key and value tensors in ndarray format
/// 
#[derive(Clone)]
pub struct CpuKVCache {
    pub keys: Vec<Option<Array3<f32>>>,
    pub values: Vec<Option<Array3<f32>>>,
    seq_length: usize,
    batch_size: usize,
}

impl CpuKVCache {
    /// Create a new empty cache for a given number of layers and batch size.
    pub fn new(num_layers: usize, batch_size: usize) -> Self {
        Self {
            keys: vec![None; num_layers],
            values: vec![None; num_layers],
            seq_length: 0,
            batch_size,
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

        self.keys[layer_idx] = Some(match self.keys[layer_idx].take() {
            None => new_keys,
            Some(cached) => {
                ndarray::concatenate(ndarray::Axis(1), &[cached.view(), new_keys.view()])?
            }
        });

        self.values[layer_idx] = Some(match self.values[layer_idx].take() {
            None => new_values,
            Some(cached) => {
                ndarray::concatenate(ndarray::Axis(1), &[cached.view(), new_values.view()])?
            }
        });

        Ok(())
    }

    // Add a new public method to CpuKVCache to set the length
    pub fn set_seq_length(&mut self, new_length: usize) {
        self.seq_length = new_length;
    }

    /// Get cached K, V for a specific layer
    pub fn get(&self, layer_idx: usize) -> Option<(&Array3<f32>, &Array3<f32>)> {
        match (self.keys.get(layer_idx), self.values.get(layer_idx)) {
            (Some(Some(k)), Some(Some(v))) => Some((k, v)),
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
        self.keys.iter_mut().for_each(|k| *k = None);
        self.values.iter_mut().for_each(|v| *v = None);
        self.seq_length = 0;
    }
}