//! Double-buffered CPU Key-Value cache optimized for beam search decoding.

use std::any::Any;

use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;

use crate::traits::Cache;

#[derive(Clone)]
pub struct CpuBeamKVCache {
    layers_k: Vec<Array3<f32>>,
    layers_v: Vec<Array3<f32>>,
    temp_layers_k: Vec<Array3<f32>>,
    temp_layers_v: Vec<Array3<f32>>,
    seq_length: usize,
    capacity: usize,
}

impl CpuBeamKVCache {
    pub fn new(num_layers: usize, num_beams: usize, max_len: usize, hidden_size: usize) -> Self {
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

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn num_beams(&self) -> usize {
        self.layers_k
            .first()
            .map(|k| k.shape()[0])
            .unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.layers_k.len()
    }

    pub fn hidden_size(&self) -> usize {
        self.layers_k
            .first()
            .map(|k| k.shape()[2])
            .unwrap_or(0)
    }

    pub fn reorder(&mut self, indices: &[usize]) {
        assert!(
            self.seq_length > 0,
            "cannot reorder an empty cache (seq_length=0)"
        );
        assert_eq!(
            indices.len(),
            self.num_beams(),
            "number of indices ({}) must match number of beams ({})",
            indices.len(),
            self.num_beams()
        );

        let valid_len = self.seq_length;

        self.layers_k
            .par_iter()
            .zip(self.layers_v.par_iter())
            .zip(self.temp_layers_k.par_iter_mut())
            .zip(self.temp_layers_v.par_iter_mut())
            .for_each(|(((source_k, source_v), dest_k), dest_v)| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    let source_k_slice = source_k.slice(s![source_idx, ..valid_len, ..]);
                    let mut dest_k_slice = dest_k.slice_mut(s![dest_idx, ..valid_len, ..]);
                    dest_k_slice.assign(&source_k_slice);

                    let source_v_slice = source_v.slice(s![source_idx, ..valid_len, ..]);
                    let mut dest_v_slice = dest_v.slice_mut(s![dest_idx, ..valid_len, ..]);
                    dest_v_slice.assign(&source_v_slice);
                }
            });

        std::mem::swap(&mut self.layers_k, &mut self.temp_layers_k);
        std::mem::swap(&mut self.layers_v, &mut self.temp_layers_v);
    }

    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Array3<f32>,
        new_v: &Array3<f32>,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.layers_k.len() {
            anyhow::bail!(
                "layer index {} out of bounds (num_layers={})",
                layer_idx,
                self.layers_k.len()
            );
        }

        let new_tokens_len = new_k.shape()[1];
        let end_pos = self.seq_length + new_tokens_len;

        if end_pos > self.capacity {
            anyhow::bail!(
                "cache overflow: seq_length={}, new_tokens={}, capacity={}",
                self.seq_length,
                new_tokens_len,
                self.capacity
            );
        }

        let target_slice = s![.., self.seq_length..end_pos, ..];

        let cache_k = &mut self.layers_k[layer_idx];
        let cache_v = &mut self.layers_v[layer_idx];

        cache_k.slice_mut(target_slice).assign(new_k);
        cache_v.slice_mut(target_slice).assign(new_v);

        Ok(())
    }

    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<f32>, ArrayView3<f32>)> {
        if layer_idx >= self.layers_k.len() {
            return None;
        }

        let k_cache = &self.layers_k[layer_idx];
        let v_cache = &self.layers_v[layer_idx];

        let active_slice = s![.., 0..self.seq_length, ..];
        Some((k_cache.slice(active_slice), v_cache.slice(active_slice)))
    }

    pub fn clone_box(&self) -> Box<CpuBeamKVCache> {
        Box::new(self.clone())
    }
}

impl Cache for CpuBeamKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    fn set_seq_length(&mut self, len: usize) {
        debug_assert!(len <= self.capacity, "set_seq_length exceeds capacity");
        self.seq_length = len;
    }

    fn clear(&mut self) {
        self.seq_length = 0;
    }

    fn increment_len(&mut self, new_tokens_len: usize) {
        let new_len = self.seq_length + new_tokens_len;
        assert!(
            new_len <= self.capacity,
            "cache overflow: seq_length={} + new_tokens={} = {} exceeds capacity={}",
            self.seq_length,
            new_tokens_len,
            new_len,
            self.capacity
        );
        self.seq_length = new_len;
    }

    fn clone_box(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache_dimensions() {
        let cache = CpuBeamKVCache::new(12, 4, 2048, 768);

        assert_eq!(cache.num_layers(), 12);
        assert_eq!(cache.num_beams(), 4);
        assert_eq!(cache.capacity(), 2048);
        assert_eq!(cache.hidden_size(), 768);
        assert_eq!(cache.get_seq_length(), 0);
    }

    #[test]
    fn test_update_and_get() {
        let mut cache = CpuBeamKVCache::new(2, 4, 100, 64);

        let k = Array3::ones((4, 10, 64));
        let v = Array3::ones((4, 10, 64)) * 2.0;

        cache.update(0, &k, &v).unwrap();
        cache.update(1, &k, &v).unwrap();
        cache.increment_len(10);

        assert_eq!(cache.get_seq_length(), 10);

        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[4, 10, 64]);
        assert_eq!(cached_v[[0, 0, 0]], 2.0);
    }

    #[test]
    fn test_reorder_basic() {
        let mut cache = CpuBeamKVCache::new(1, 4, 10, 2);

        let mut k = Array3::zeros((4, 5, 2));
        let mut v = Array3::zeros((4, 5, 2));
        for beam in 0..4 {
            k[[beam, 0, 0]] = beam as f32;
            v[[beam, 0, 0]] = (beam * 10) as f32;
        }

        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        cache.reorder(&[2, 2, 0, 1]);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k[[0, 0, 0]], 2.0);
        assert_eq!(cached_k[[1, 0, 0]], 2.0);
        assert_eq!(cached_k[[2, 0, 0]], 0.0);
        assert_eq!(cached_k[[3, 0, 0]], 1.0);
    }

    #[test]
    fn test_double_buffer_swap() {
        let mut cache = CpuBeamKVCache::new(1, 2, 10, 2);

        let k = Array3::ones((2, 3, 2));
        let v = Array3::ones((2, 3, 2));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(3);

        let ptr_before = cache.layers_k[0].as_ptr();

        cache.reorder(&[1, 0]);

        let ptr_after = cache.layers_k[0].as_ptr();

        assert_ne!(ptr_before, ptr_after, "buffers should have been swapped");
    }

    #[test]
    fn test_incremental_updates() {
        let mut cache = CpuBeamKVCache::new(1, 2, 100, 4);

        let k1 = Array3::ones((2, 5, 4)) * 1.0;
        let v1 = Array3::ones((2, 5, 4)) * 1.0;
        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(5);

        let k2 = Array3::ones((2, 1, 4)) * 2.0;
        let v2 = Array3::ones((2, 1, 4)) * 2.0;
        cache.update(0, &k2, &v2).unwrap();
        cache.increment_len(1);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[2, 6, 4]);
        assert_eq!(cached_k[[0, 4, 0]], 1.0);
        assert_eq!(cached_k[[0, 5, 0]], 2.0);
    }

    #[test]
    #[should_panic(expected = "cache overflow")]
    fn test_overflow_panics() {
        let mut cache = CpuBeamKVCache::new(1, 1, 10, 4);

        let k = Array3::ones((1, 5, 4));
        let v = Array3::ones((1, 5, 4));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        cache.increment_len(10);
    }

    #[test]
    #[should_panic(expected = "cannot reorder an empty cache")]
    fn test_reorder_empty_panics() {
        let mut cache = CpuBeamKVCache::new(1, 2, 10, 4);
        cache.reorder(&[1, 0]);
    }

    #[test]
    fn test_clear_preserves_capacity() {
        let mut cache = CpuBeamKVCache::new(1, 2, 100, 4);

        let k = Array3::ones((2, 50, 4));
        let v = Array3::ones((2, 50, 4));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(50);

        assert_eq!(cache.get_seq_length(), 50);

        cache.clear();

        assert_eq!(cache.get_seq_length(), 0);
        assert_eq!(cache.capacity(), 100);
    }
}