//! Double-buffered CPU Key-Value cache optimized for beam search decoding.
//!
//! This module provides an optimized KV cache implementation that uses double-buffering
//! to make beam reordering allocation-free. This is critical for beam search performance
//! where reordering happens on every decode step.
//!
//! # Double-Buffering Strategy
//!
//! ```text
//! Step N: Main buffers active
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │   Main K/V Buffers  │     │   Temp K/V Buffers  │
//! │   [active data]     │     │   [stale/unused]    │
//! └─────────────────────┘     └─────────────────────┘
//!            │                          ▲
//!            │    reorder(indices)      │
//!            └──────────────────────────┘
//!                    copy + swap
//!
//! Step N+1: Temp buffers are now main (swapped)
//! ┌─────────────────────┐     ┌─────────────────────┐
//! │   Temp K/V Buffers  │     │   Main K/V Buffers  │
//! │   [reordered data]  │     │   [now temp/stale]  │
//! └─────────────────────┘     └─────────────────────┘
//! ```
//!
//! # Comparison with CpuKVCache
//!
//! | Operation | CpuKVCache | CpuBeamKVCache |
//! |-----------|------------|----------------|
//! | Memory usage | 1× | 2× (double-buffered) |
//! | `update()` | O(new_tokens) | O(new_tokens) |
//! | `get()` | O(1) view | O(1) view |
//! | `reorder()` | O(n) + clone | O(n) copy + O(1) swap |
//! | Allocation in reorder | Yes (full clone) | **No** |
//!
//! # When to Use
//!
//! - **CpuKVCache**: Single-sequence greedy/sampling generation
//! - **CpuBeamKVCache**: Beam search, speculative decoding, any workflow
//!   requiring frequent reordering
//!
//! # Memory Layout
//!
//! ```text
//! layers_k[layer_idx]: [num_beams, max_len, hidden_size]
//!                       │          │        │
//!                       │          │        └─ num_heads × head_dim
//!                       │          └─ sequence dimension (pre-allocated)
//!                       └─ beam/batch dimension
//!
//! Example for Llama 3.2 1B with 4 beams:
//!   Shape: [4, 2048, 2048]
//!   Size per tensor: 4 × 2048 × 2048 × 4 bytes = 64 MB
//!   Total (K+V, main+temp, 16 layers): 64 × 4 × 16 = 4 GB
//! ```
//!
//! # Example
//!
//! ```ignore
//! // Create cache for 4-beam search
//! let mut cache = CpuBeamKVCache::new(
//!     16,     // num_layers
//!     4,      // num_beams
//!     2048,   // max_seq_len
//!     2048,   // hidden_size
//! );
//!
//! // During generation:
//! for step in 0..max_steps {
//!     // 1. Run forward pass, get new K/V for each layer
//!     for layer_idx in 0..16 {
//!         cache.update(layer_idx, &new_k, &new_v)?;
//!     }
//!     cache.increment_len(1);
//!
//!     // 2. Score beams and select best continuations
//!     let beam_indices = select_top_beams(&scores);
//!
//!     // 3. Reorder cache to match new beam arrangement
//!     //    This is O(copy) + O(1) swap - no allocation!
//!     cache.reorder(&beam_indices);
//! }
//! ```

use crate::traits::Cache;
use ndarray::{s, Array3, ArrayView3};
use rayon::prelude::*;
use std::any::Any;

/// A double-buffered Key-Value cache optimized for beam search decoding.
///
/// This cache maintains two sets of K/V buffers and swaps between them during
/// reordering operations. This eliminates allocation overhead in the critical
/// path of beam search, where reordering happens on every decode step.
///
/// # Memory Trade-off
///
/// Uses 2× the memory of `CpuKVCache` but provides allocation-free reordering.
/// For beam search workloads, this trade-off is almost always worthwhile since:
/// - Beam search already uses N× memory for N beams
/// - Allocation jitter causes unpredictable latency
/// - The reorder operation is on the critical path
///
/// # Thread Safety
///
/// `CpuBeamKVCache` is `Send` but not `Sync`. The `reorder()` method uses
/// Rayon for internal parallelism across layers.
///
/// # Panics
///
/// - `reorder()` panics if called on an empty cache
/// - `increment_len()` panics if the cache would overflow
#[derive(Clone)]
pub struct CpuBeamKVCache {
    /// Primary key buffers, one per layer.
    /// Shape: `[num_beams, max_len, hidden_size]`
    layers_k: Vec<Array3<f32>>,

    /// Primary value buffers, one per layer.
    /// Shape: `[num_beams, max_len, hidden_size]`
    layers_v: Vec<Array3<f32>>,

    /// Secondary key buffers for double-buffering.
    /// Used as destination during `reorder()`, then swapped with primary.
    temp_layers_k: Vec<Array3<f32>>,

    /// Secondary value buffers for double-buffering.
    temp_layers_v: Vec<Array3<f32>>,

    /// Number of tokens currently stored in the cache.
    /// Invariant: `seq_length <= capacity`
    seq_length: usize,

    /// Maximum sequence length (pre-allocated capacity).
    capacity: usize,
}

impl CpuBeamKVCache {
    /// Creates a new double-buffered cache for beam search.
    ///
    /// Allocates 4 tensors per layer (K main, K temp, V main, V temp),
    /// all pre-sized to the maximum sequence length.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of decoder layers in the model
    /// * `num_beams` - Number of beams in beam search
    /// * `max_len` - Maximum sequence length to support
    /// * `hidden_size` - Hidden dimension (`num_heads × head_dim`)
    ///
    /// # Memory Usage
    ///
    /// Total = `4 × num_layers × num_beams × max_len × hidden_size × sizeof(f32)`
    ///
    /// Example: 16 layers, 4 beams, 2048 max_len, 2048 hidden:
    /// = 4 × 16 × 4 × 2048 × 2048 × 4 bytes = 4.3 GB
    ///
    /// # Example
    ///
    /// ```ignore
    /// let cache = CpuBeamKVCache::new(
    ///     16,     // Llama 3.2 1B has 16 layers
    ///     4,      // 4-beam search
    ///     2048,   // max sequence length
    ///     2048,   // hidden_size = 32 heads × 64 head_dim
    /// );
    /// ```
    pub fn new(num_layers: usize, num_beams: usize, max_len: usize, hidden_size: usize) -> Self {
        let mut layers_k = Vec::with_capacity(num_layers);
        let mut layers_v = Vec::with_capacity(num_layers);
        let mut temp_layers_k = Vec::with_capacity(num_layers);
        let mut temp_layers_v = Vec::with_capacity(num_layers);

        for _ in 0..num_layers {
            // Main buffers
            layers_k.push(Array3::zeros((num_beams, max_len, hidden_size)));
            layers_v.push(Array3::zeros((num_beams, max_len, hidden_size)));
            // Temp buffers (for double-buffering)
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

    /// Returns the maximum sequence length this cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of beams.
    pub fn num_beams(&self) -> usize {
        self.layers_k
            .first()
            .map(|k| k.shape()[0])
            .unwrap_or(0)
    }

    /// Returns the number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers_k.len()
    }

    /// Returns the hidden size.
    pub fn hidden_size(&self) -> usize {
        self.layers_k
            .first()
            .map(|k| k.shape()[2])
            .unwrap_or(0)
    }

    /// Reorders beam histories according to parent indices.
    ///
    /// This is the key operation that makes beam search efficient. When beams
    /// are reranked (e.g., beam 2 becomes the new beam 0), the corresponding
    /// KV cache entries must be rearranged to match.
    ///
    /// # Algorithm
    ///
    /// 1. Copy from main buffers to temp buffers according to indices
    /// 2. Swap main ↔ temp pointers (O(1), no data movement)
    ///
    /// # Arguments
    ///
    /// * `indices` - Reordering map where `indices[i]` is the source beam
    ///               index for destination beam `i`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Before: beams are [A, B, C, D] with scores [0.1, 0.4, 0.3, 0.2]
    /// // After selection: we want [B, C, B, A] (B is best, duplicate it)
    /// cache.reorder(&[1, 2, 1, 0]);
    /// // Now beam 0 has B's history, beam 1 has C's, etc.
    /// ```
    ///
    /// # Performance
    ///
    /// - **Copy**: O(num_layers × num_beams × seq_length × hidden_size)
    /// - **Swap**: O(1) pointer swap
    /// - **Allocation**: None (uses pre-allocated temp buffers)
    ///
    /// Parallelized across layers using Rayon.
    ///
    /// # Panics
    ///
    /// - If `seq_length == 0` (nothing to reorder)
    /// - If `indices.len() != num_beams`
    pub fn reorder(&mut self, indices: &[usize]) {
        assert!(
            self.seq_length > 0,
            "Cannot reorder an empty cache (seq_length=0)"
        );
        assert_eq!(
            indices.len(),
            self.num_beams(),
            "Number of indices ({}) must match number of beams ({})",
            indices.len(),
            self.num_beams()
        );

        // Parallel copy from main → temp according to indices
        // OPTIMIZATION: Only copy valid history (0..seq_length), ignore tail capacity.
        let valid_len = self.seq_length;

        self.layers_k
            .par_iter()
            .zip(self.layers_v.par_iter())
            .zip(self.temp_layers_k.par_iter_mut())
            .zip(self.temp_layers_v.par_iter_mut())
            .for_each(|(((source_k, source_v), dest_k), dest_v)| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    // Copy only active history
                    let source_k_slice = source_k.slice(s![source_idx, ..valid_len, ..]);
                    let mut dest_k_slice = dest_k.slice_mut(s![dest_idx, ..valid_len, ..]);
                    dest_k_slice.assign(&source_k_slice);

                    let source_v_slice = source_v.slice(s![source_idx, ..valid_len, ..]);
                    let mut dest_v_slice = dest_v.slice_mut(s![dest_idx, ..valid_len, ..]);
                    dest_v_slice.assign(&source_v_slice);
                }
            });

        // Swap pointers: temp becomes main, main becomes temp
        std::mem::swap(&mut self.layers_k, &mut self.temp_layers_k);
        std::mem::swap(&mut self.layers_v, &mut self.temp_layers_v);
    }
    // pub fn reorder(&mut self, indices: &[usize]) {
    //     assert!(
    //         self.seq_length > 0,
    //         "Cannot reorder an empty cache (seq_length=0)"
    //     );
    //     assert_eq!(
    //         indices.len(),
    //         self.num_beams(),
    //         "Number of indices ({}) must match number of beams ({})",
    //         indices.len(),
    //         self.num_beams()
    //     );

    //     // Parallel copy from main → temp according to indices
    //     self.layers_k
    //         .par_iter()
    //         .zip(self.layers_v.par_iter())
    //         .zip(self.temp_layers_k.par_iter_mut())
    //         .zip(self.temp_layers_v.par_iter_mut())
    //         .for_each(|(((source_k, source_v), dest_k), dest_v)| {
    //             for (dest_idx, &source_idx) in indices.iter().enumerate() {
    //                 // Copy entire beam's history (all positions, all hidden dims)
    //                 let source_k_slice = source_k.slice(s![source_idx, .., ..]);
    //                 let mut dest_k_slice = dest_k.slice_mut(s![dest_idx, .., ..]);
    //                 dest_k_slice.assign(&source_k_slice);

    //                 let source_v_slice = source_v.slice(s![source_idx, .., ..]);
    //                 let mut dest_v_slice = dest_v.slice_mut(s![dest_idx, .., ..]);
    //                 dest_v_slice.assign(&source_v_slice);
    //             }
    //         });

    //     // Swap pointers: temp becomes main, main becomes temp
    //     // This is O(1) - just swapping Vec pointers
    //     std::mem::swap(&mut self.layers_k, &mut self.temp_layers_k);
    //     std::mem::swap(&mut self.layers_v, &mut self.temp_layers_v);
    // }

    /// Appends new key/value tensors to a layer's cache.
    ///
    /// The new tensors are copied into the pre-allocated buffer starting
    /// at position `seq_length`. Call `increment_len()` after updating
    /// all layers.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the decoder layer (0-indexed)
    /// * `new_k` - New keys, shape `[num_beams, new_tokens, hidden_size]`
    /// * `new_v` - New values, shape `[num_beams, new_tokens, hidden_size]`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `layer_idx` is out of bounds
    /// - The update would exceed capacity (checked in `increment_len`)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // During decode, each step adds 1 token per beam
    /// let new_k = /* shape [num_beams, 1, hidden_size] */;
    /// let new_v = /* shape [num_beams, 1, hidden_size] */;
    ///
    /// for layer_idx in 0..num_layers {
    ///     cache.update(layer_idx, &new_k, &new_v)?;
    /// }
    /// cache.increment_len(1);
    /// ```
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: &Array3<f32>,
        new_v: &Array3<f32>,
    ) -> anyhow::Result<()> {
        if layer_idx >= self.layers_k.len() {
            anyhow::bail!(
                "Layer index {} out of bounds (num_layers={})",
                layer_idx,
                self.layers_k.len()
            );
        }

        let new_tokens_len = new_k.shape()[1];
        let end_pos = self.seq_length + new_tokens_len;

        if end_pos > self.capacity {
            anyhow::bail!(
                "Cache overflow: seq_length={}, new_tokens={}, capacity={}",
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

    /// Returns views of the cached keys and values for a layer.
    ///
    /// Only returns the "active" portion (positions 0..seq_length).
    /// Zero-copy operation.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the decoder layer (0-indexed)
    ///
    /// # Returns
    ///
    /// `Some((keys, values))` where each has shape `[num_beams, seq_length, hidden_size]`,
    /// or `None` if `layer_idx` is out of bounds.
    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<f32>, ArrayView3<f32>)> {
        if layer_idx >= self.layers_k.len() {
            return None;
        }

        let k_cache = &self.layers_k[layer_idx];
        let v_cache = &self.layers_v[layer_idx];

        let active_slice = s![.., 0..self.seq_length, ..];
        Some((k_cache.slice(active_slice), v_cache.slice(active_slice)))
    }

    /// Returns a boxed clone of this cache.
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

    /// Resets the cache to empty state without deallocating.
    ///
    /// The pre-allocated buffers remain available for reuse.
    fn clear(&mut self) {
        self.seq_length = 0;
    }

    /// Advances the sequence length after updating all layers.
    ///
    /// # Panics
    ///
    /// Panics if the new length would exceed capacity. This is intentional -
    /// overflow is a serious bug that should fail fast.
    fn increment_len(&mut self, new_tokens_len: usize) {
        let new_len = self.seq_length + new_tokens_len;
        assert!(
            new_len <= self.capacity,
            "Cache overflow! seq_length={} + new_tokens={} = {} exceeds capacity={}",
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

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

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

        // Add 10 tokens worth of KV for all beams
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

        // Create distinct values for each beam
        let mut k = Array3::zeros((4, 5, 2));
        let mut v = Array3::zeros((4, 5, 2));
        for beam in 0..4 {
            k[[beam, 0, 0]] = beam as f32;
            v[[beam, 0, 0]] = (beam * 10) as f32;
        }

        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        // Before: beams have values [0, 1, 2, 3]
        // Reorder: [2, 2, 0, 1] means:
        //   - new beam 0 gets old beam 2's data
        //   - new beam 1 gets old beam 2's data (duplicate)
        //   - new beam 2 gets old beam 0's data
        //   - new beam 3 gets old beam 1's data
        cache.reorder(&[2, 2, 0, 1]);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k[[0, 0, 0]], 2.0); // Was beam 2
        assert_eq!(cached_k[[1, 0, 0]], 2.0); // Was beam 2 (duplicate)
        assert_eq!(cached_k[[2, 0, 0]], 0.0); // Was beam 0
        assert_eq!(cached_k[[3, 0, 0]], 1.0); // Was beam 1
    }

    #[test]
    fn test_double_buffer_swap() {
        let mut cache = CpuBeamKVCache::new(1, 2, 10, 2);

        // Add initial data
        let k = Array3::ones((2, 3, 2));
        let v = Array3::ones((2, 3, 2));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(3);

        // Get pointers before reorder
        let ptr_before = cache.layers_k[0].as_ptr();

        // Reorder (just swap beam 0 and 1)
        cache.reorder(&[1, 0]);

        // Get pointers after reorder
        let ptr_after = cache.layers_k[0].as_ptr();

        // Pointers should be different (we swapped buffers)
        assert_ne!(ptr_before, ptr_after, "Buffers should have been swapped");
    }

    #[test]
    fn test_incremental_updates() {
        let mut cache = CpuBeamKVCache::new(1, 2, 100, 4);

        // Prefill with 5 tokens
        let k1 = Array3::ones((2, 5, 4)) * 1.0;
        let v1 = Array3::ones((2, 5, 4)) * 1.0;
        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(5);

        // Decode step: add 1 token
        let k2 = Array3::ones((2, 1, 4)) * 2.0;
        let v2 = Array3::ones((2, 1, 4)) * 2.0;
        cache.update(0, &k2, &v2).unwrap();
        cache.increment_len(1);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[2, 6, 4]);
        assert_eq!(cached_k[[0, 4, 0]], 1.0); // Original token
        assert_eq!(cached_k[[0, 5, 0]], 2.0); // New token
    }

    #[test]
    #[should_panic(expected = "Cache overflow")]
    fn test_overflow_panics() {
        let mut cache = CpuBeamKVCache::new(1, 1, 10, 4);

        // Try to add more than capacity
        let k = Array3::ones((1, 5, 4));
        let v = Array3::ones((1, 5, 4));
        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        // This should panic
        cache.increment_len(10);
    }

    #[test]
    #[should_panic(expected = "Cannot reorder an empty cache")]
    fn test_reorder_empty_panics() {
        let mut cache = CpuBeamKVCache::new(1, 2, 10, 4);
        cache.reorder(&[1, 0]); // Should panic - nothing to reorder
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
        assert_eq!(cache.capacity(), 100); // Capacity unchanged
    }
}