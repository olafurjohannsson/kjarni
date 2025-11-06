use crate::traits::Cache;
use ndarray::{Array3, ArrayView3, s};
use std::any::Any;

/// A high-performance, pre-allocated Key-Value cache for CPU decoding.
///
/// This implementation avoids the overhead of re-allocating and concatenating on every step
/// by pre-allocating memory to the maximum required capacity at creation.
/// The `update` operation becomes a highly efficient copy into a slice of the buffer.
pub struct CpuKVCache {
    /// A Vec of (Key, Value) tuples, one for each decoder layer.
    /// The Arrays are pre-allocated to the maximum sequence length.
    layers: Vec<(Array3<f32>, Array3<f32>)>,
    /// The current number of tokens stored in the cache. This is the main state variable.
    current_len: usize,
}

impl CpuKVCache {
    /// Creates a new, empty cache pre-allocated to the maximum size.
    ///
    /// # Arguments
    /// * `num_layers` - The number of decoder layers in the model.
    /// * `batch_size` - The batch size for the generation.
    /// * `max_len` - The maximum sequence length the model supports (e.g., `max_position_embeddings`).
    /// * `hidden_size` - The hidden dimensionality of the model.
    pub fn new(num_layers: usize, batch_size: usize, max_len: usize, hidden_size: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let k_cache = Array3::zeros((batch_size, max_len, hidden_size));
            let v_cache = Array3::zeros((batch_size, max_len, hidden_size));
            layers.push((k_cache, v_cache));
        }
        Self {
            layers,
            current_len: 0,
        }
    }

    /// Appends the new key and value tensors to the cache for a specific layer.
    ///
    /// This is a highly efficient operation that copies the new data into a
    /// slice of the pre-allocated buffer without triggering any new heap allocations.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Array3<f32>,
        new_v: &Array3<f32>,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.layers.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }

        let new_tokens_len = new_k.shape()[1];
        let target_slice = s![.., self.current_len..self.current_len + new_tokens_len, ..];

        let (cache_k, cache_v) = &mut self.layers[layer_idx];
        cache_k.slice_mut(target_slice).assign(new_k);
        cache_v.slice_mut(target_slice).assign(new_v);

        Ok(())
    }

    /// Retrieves a view of the cached keys and values for a specific layer.
    ///
    /// Returns a tuple of views, `(Key, Value)`, containing all tokens
    /// processed so far. This is a zero-copy operation.
    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<f32>, ArrayView3<f32>)> {
        if layer_idx >= self.layers.len() {
            return None;
        }
        let (cache_k, cache_v) = &self.layers[layer_idx];
        let active_slice = s![.., 0..self.current_len, ..];
        Some((cache_k.slice(active_slice), cache_v.slice(active_slice)))
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
        self.current_len
    }
    fn clear(&mut self) {
        // Clearing the cache is now a very cheap operation.
        // We just reset the length counter. The data in the buffers becomes
        // garbage and will be overwritten on the next run.
        self.current_len = 0;
    }
    /// Increments the internal length counter.
    ///
    /// This should be called once per generation step by the orchestrator (e.g., `TextGenerator`)
    /// after all layers have been updated for that step.
    fn increment_len(&mut self, new_tokens_len: usize) {
        self.current_len += new_tokens_len;
    }
}
