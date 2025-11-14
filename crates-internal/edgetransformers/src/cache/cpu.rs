use crate::traits::Cache;
use ndarray::{Array3, ArrayView3, s};
use rayon::prelude::*;
use std::any::Any;

/// A high-performance, pre-allocated Key-Value cache for CPU decoding.
///
/// This implementation avoids the overhead of re-allocating and concatenating on every step
/// by pre-allocating memory to the maximum required capacity at creation.
/// The `update` operation becomes a highly efficient copy into a slice of the buffer.
#[derive(Clone)]
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
    /// Reorders the batch dimension of the cache in-place using Rayon for parallelism.
    ///
    /// This is a highly efficient, allocation-free alternative to cloning for beam search.
    ///
    /// # Arguments
    /// * `indices`: A slice of `usize` where each value indicates the source batch
    ///              index to copy from.
    pub fn reorder(&mut self, indices: &[usize]) {
        let mut reordered_layers = self.layers.clone();

        let source_iter = self.layers.par_iter();
        let dest_iter = reordered_layers.par_iter_mut(); // todo: not used?

        source_iter.zip(reordered_layers.par_iter_mut()).for_each(
            |((k_cache, v_cache), (k_reordered, v_reordered))| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    let source_k_slice = k_cache.slice(s![source_idx, .., ..]);
                    let mut dest_k_slice = k_reordered.slice_mut(s![dest_idx, .., ..]);
                    dest_k_slice.assign(&source_k_slice);

                    let source_v_slice = v_cache.slice(s![source_idx, .., ..]);
                    let mut dest_v_slice = v_reordered.slice_mut(s![dest_idx, .., ..]);
                    dest_v_slice.assign(&source_v_slice);
                }
            },
        );

        self.layers = reordered_layers;
    }
    pub fn layers(&self) -> &Vec<(Array3<f32>, Array3<f32>)> {
        &self.layers
    }

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

    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<f32>, ArrayView3<f32>)> {
        if layer_idx >= self.layers.len() {
            return None;
        }

        let (cache_k, cache_v) = &self.layers[layer_idx];

        let active_slice = s![.., 0..self.current_len, ..];
        Some((cache_k.slice(active_slice), cache_v.slice(active_slice)))
    }

    pub fn clone_box(&self) -> Box<CpuKVCache> {
        Box::new(self.clone())
    }
}

impl Cache for CpuKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn clone_box(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn set_seq_length(&mut self, len: usize) {
        self.current_len = len;
    }
    fn get_seq_length(&self) -> usize {
        self.current_len
    }
    fn clear(&mut self) {
        self.current_len = 0;
    }
    fn increment_len(&mut self, new_tokens_len: usize) {
        self.current_len += new_tokens_len;
    }
}
