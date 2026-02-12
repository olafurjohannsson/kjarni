//! CPU Key-Value cache for autoregressive decoder models.

use std::any::Any;

use ndarray::{s, Array3, ArrayView3, ArrayViewMut3};
use rayon::prelude::*;

use crate::traits::Cache;

#[derive(Clone)]
pub struct CpuKVCache {
    layers: Vec<(Array3<f32>, Array3<f32>)>,
    current_len: usize,
}

impl CpuKVCache {
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

    pub fn max_len(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[1]).unwrap_or(0)
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn batch_size(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[0]).unwrap_or(0)
    }

    pub fn hidden_size(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[2]).unwrap_or(0)
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
            anyhow::bail!(
                "layer index {} out of bounds (num_layers={})",
                layer_idx,
                self.layers.len()
            );
        }

        let new_tokens_len = new_k.shape()[1];
        let end_pos = self.current_len + new_tokens_len;

        if end_pos > self.max_len() {
            anyhow::bail!(
                "cache overflow: current_len={}, new_tokens={}, max_len={}",
                self.current_len,
                new_tokens_len,
                self.max_len()
            );
        }

        let target_slice = s![.., self.current_len..end_pos, ..];

        let (cache_k, cache_v) = &mut self.layers[layer_idx];
        cache_k.slice_mut(target_slice).assign(new_k);
        cache_v.slice_mut(target_slice).assign(new_v);

        Ok(())
    }

    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<'_, f32>, ArrayView3<'_, f32>)> {
        if layer_idx >= self.layers.len() {
            return None;
        }

        let (cache_k, cache_v) = &self.layers[layer_idx];
        let active_slice = s![.., 0..self.current_len, ..];

        Some((cache_k.slice(active_slice), cache_v.slice(active_slice)))
    }

    pub fn reorder(&mut self, indices: &[usize]) {
        let mut reordered_layers = self.layers.clone();

        self.layers
            .par_iter()
            .zip(reordered_layers.par_iter_mut())
            .for_each(|((k_cache, v_cache), (k_reordered, v_reordered))| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    let source_k = k_cache.slice(s![source_idx, .., ..]);
                    let mut dest_k = k_reordered.slice_mut(s![dest_idx, .., ..]);
                    dest_k.assign(&source_k);

                    let source_v = v_cache.slice(s![source_idx, .., ..]);
                    let mut dest_v = v_reordered.slice_mut(s![dest_idx, .., ..]);
                    dest_v.assign(&source_v);
                }
            });

        self.layers = reordered_layers;
    }

    pub fn get_context_view_mut(
        &mut self,
        layer_idx: usize,
        new_tokens: usize,
    ) -> anyhow::Result<(ArrayViewMut3<'_, f32>, ArrayViewMut3<'_, f32>)> {
        if layer_idx >= self.layers.len() {
            anyhow::bail!("layer index out of bounds");
        }

        let (k_cache, v_cache) = &mut self.layers[layer_idx];
        let total_len = self.current_len + new_tokens;

        if total_len > k_cache.shape()[1] {
            anyhow::bail!("cache overflow: {} > capacity", total_len);
        }

        let k_view = k_cache.slice_mut(s![.., 0..total_len, ..]);
        let v_view = v_cache.slice_mut(s![.., 0..total_len, ..]);

        Ok((k_view, v_view))
    }

    pub fn clone_box(&self) -> Box<CpuKVCache> {
        Box::new(self.clone())
    }
}

impl Cache for CpuKVCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Cache> {
        Box::new(self.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_cache_dimensions() {
        let cache = CpuKVCache::new(12, 1, 2048, 768);

        assert_eq!(cache.num_layers(), 12);
        assert_eq!(cache.batch_size(), 1);
        assert_eq!(cache.max_len(), 2048);
        assert_eq!(cache.hidden_size(), 768);
        assert_eq!(cache.get_seq_length(), 0);
    }

    #[test]
    fn test_update_and_get() {
        let mut cache = CpuKVCache::new(2, 1, 100, 64);

        let k = Array3::ones((1, 10, 64));
        let v = Array3::ones((1, 10, 64)) * 2.0;

        cache.update(0, &k, &v).unwrap();
        cache.update(1, &k, &v).unwrap();
        cache.increment_len(10);

        assert_eq!(cache.get_seq_length(), 10);

        let (cached_k, cached_v) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[1, 10, 64]);
        assert_eq!(cached_v[[0, 0, 0]], 2.0);
    }

    #[test]
    fn test_incremental_update() {
        let mut cache = CpuKVCache::new(1, 1, 100, 64);

        let k1 = Array3::ones((1, 5, 64));
        let v1 = Array3::ones((1, 5, 64));
        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(5);

        let k2 = Array3::ones((1, 1, 64)) * 2.0;
        let v2 = Array3::ones((1, 1, 64)) * 2.0;
        cache.update(0, &k2, &v2).unwrap();
        cache.increment_len(1);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[1, 6, 64]);
        assert_eq!(cached_k[[0, 4, 0]], 1.0);
        assert_eq!(cached_k[[0, 5, 0]], 2.0);
    }

    #[test]
    fn test_clear_resets_length() {
        let mut cache = CpuKVCache::new(1, 1, 100, 64);

        let k = Array3::ones((1, 10, 64));
        let v = Array3::ones((1, 10, 64));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(10);

        assert_eq!(cache.get_seq_length(), 10);

        cache.clear();

        assert_eq!(cache.get_seq_length(), 0);
        assert_eq!(cache.max_len(), 100);
    }

    #[test]
    fn test_overflow_error() {
        let mut cache = CpuKVCache::new(1, 1, 10, 64);

        let k = Array3::ones((1, 15, 64));
        let v = Array3::ones((1, 15, 64));

        let result = cache.update(0, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_reorder() {
        let mut cache = CpuKVCache::new(1, 4, 10, 2);

        let mut k = Array3::zeros((4, 5, 2));
        let mut v = Array3::zeros((4, 5, 2));
        for b in 0..4 {
            k[[b, 0, 0]] = b as f32;
            v[[b, 0, 0]] = b as f32;
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
}