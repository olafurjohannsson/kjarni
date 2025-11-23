// Create a new file, e.g., `src/cache/cpu_beam_cache.rs` or add it alongside the other caches.

use crate::traits::Cache;
use ndarray::{s, Array3};
use rayon::prelude::*;
use std::any::Any;

/// A high-performance, pre-allocated Key-Value cache for CPU beam search decoding.
///
/// This implementation avoids the overhead of re-allocating on every step by using
/// temporary buffers for the reorder operation and then swapping pointers. This makes
/// the reorder step an O(1) operation in terms of memory allocation.
#[derive(Clone)]
pub struct CpuBeamKVCache {
    // Main K/V tensors. Layout: [NumBeams, Capacity, HiddenSize]
    layers_k: Vec<Array3<f32>>,
    layers_v: Vec<Array3<f32>>,

    // Temporary tensors used as the destination for the reorder operation.
    temp_layers_k: Vec<Array3<f32>>,
    temp_layers_v: Vec<Array3<f32>>,

    // The current number of tokens stored in the cache.
    seq_length: usize,
    capacity: usize,
}

impl CpuBeamKVCache {
    /// Creates a new, empty cache pre-allocated to the maximum size for beam search.
    pub fn new(
        num_layers: usize,
        num_beams: usize,
        max_len: usize,
        hidden_size: usize,
    ) -> Self {
        let mut layers_k = Vec::with_capacity(num_layers);
        let mut layers_v = Vec::with_capacity(num_layers);
        let mut temp_layers_k = Vec::with_capacity(num_layers);
        let mut temp_layers_v = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            layers_k.push(Array3::zeros((num_beams, max_len, hidden_size)));
            layers_v.push(Array3::zeros((num_beams, max_len, hidden_size)));
            temp_layers_k.push(Array3::zeros((num_beams, max_len, hidden_size)));
            temp_layers_v.push(Array3::zeros((num_beams, max_len, hidden_size)));
        }
        Self {
            layers_k,
            layers_v,
            temp_layers_k,
            temp_layers_v,
            seq_length: 0,
            capacity: max_len,
        }
    }

    /// Reorders the beam histories based on parent indices.
    /// This is a highly efficient operation that writes to a temporary buffer and then
    /// swaps pointers, avoiding any new memory allocations.
    pub fn reorder(&mut self, indices: &[usize]) {
        assert!(self.seq_length > 0, "Cannot reorder an empty cache.");
        assert_eq!(
            indices.len(),
            self.layers_k[0].shape()[0],
            "Number of indices must match the number of beams."
        );

        // Parallel reordering from main buffers to temporary buffers
        self.layers_k
            .par_iter()
            .zip(self.layers_v.par_iter())
            .zip(self.temp_layers_k.par_iter_mut())
            .zip(self.temp_layers_v.par_iter_mut())
            .for_each(|((((source_k, source_v), dest_k), dest_v))| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    let source_k_slice = source_k.slice(s![source_idx, .., ..]);
                    let mut dest_k_slice = dest_k.slice_mut(s![dest_idx, .., ..]);
                    dest_k_slice.assign(&source_k_slice);

                    let source_v_slice = source_v.slice(s![source_idx, .., ..]);
                    let mut dest_v_slice = dest_v.slice_mut(s![dest_idx, .., ..]);
                    dest_v_slice.assign(&source_v_slice);
                }
            });

        // Swap the pointers. The temp buffers are now the main buffers.
        std::mem::swap(&mut self.layers_k, &mut self.temp_layers_k);
        std::mem::swap(&mut self.layers_v, &mut self.temp_layers_v);
    }

    /// Updates a layer with new key-value states.
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Array3<f32>,
        new_v: &Array3<f32>,
    ) -> anyhow::Result<()> {
        let new_tokens_len = new_k.shape()[1];
        let target_slice = s![.., self.seq_length..self.seq_length + new_tokens_len, ..];

        let (cache_k, cache_v) = (&mut self.layers_k[layer_idx], &mut self.layers_v[layer_idx]);
        cache_k.slice_mut(target_slice).assign(new_k);
        cache_v.slice_mut(target_slice).assign(new_v);

        Ok(())
    }
    pub fn get(&self, layer_idx: usize) -> Option<(ndarray::ArrayView3<f32>, ndarray::ArrayView3<f32>)> {
        if layer_idx >= self.layers_k.len() {
            return None;
        }
        let k_cache = &self.layers_k[layer_idx];
        let v_cache = &self.layers_v[layer_idx];

        // Slice the tensors to return only the part that has been filled.
        let active_slice = s![.., 0..self.seq_length, ..];
        Some((k_cache.slice(active_slice), v_cache.slice(active_slice)))
    }
}

impl Cache for CpuBeamKVCache {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn get_seq_length(&self) -> usize { self.seq_length }
    fn set_seq_length(&mut self, len: usize) { self.seq_length = len; }
    fn clear(&mut self) { self.seq_length = 0; }

    fn increment_len(&mut self, new_tokens_len: usize) {
        self.seq_length += new_tokens_len;
        assert!(
            self.seq_length <= self.capacity,
            "Cache overflow! New length {} exceeds capacity {}",
            self.seq_length, self.capacity
        );
    }

    fn clone_box(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
}