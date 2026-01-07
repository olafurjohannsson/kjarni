//! CPU Key-Value cache for autoregressive decoder models.
//!
//! This module provides a high-performance, pre-allocated KV cache implementation
//! optimized for CPU-based text generation. The cache stores key and value tensors
//! from each decoder layer's self-attention, enabling efficient autoregressive
//! generation without recomputing attention over the full sequence.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                          CpuKVCache                                 │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │  Layer 0: ┌─────────────────────┐  ┌─────────────────────┐         │
//! │           │    K [B, S, H]      │  │    V [B, S, H]      │         │
//! │           │ ████████░░░░░░░░░░░ │  │ ████████░░░░░░░░░░░ │         │
//! │           └─────────────────────┘  └─────────────────────┘         │
//! │                  ▲                                                  │
//! │                  │ current_len = 8                                  │
//! │                                                                     │
//! │  Layer 1: ┌─────────────────────┐  ┌─────────────────────┐         │
//! │           │    K [B, S, H]      │  │    V [B, S, H]      │         │
//! │           │ ████████░░░░░░░░░░░ │  │ ████████░░░░░░░░░░░ │         │
//! │           └─────────────────────┘  └─────────────────────┘         │
//! │                                                                     │
//! │  ...                                                                │
//! │                                                                     │
//! │  Layer N: ┌─────────────────────┐  ┌─────────────────────┐         │
//! │           │    K [B, S, H]      │  │    V [B, S, H]      │         │
//! │           │ ████████░░░░░░░░░░░ │  │ ████████░░░░░░░░░░░ │         │
//! │           └─────────────────────┘  └─────────────────────┘         │
//! │                                                                     │
//! │  Legend: ████ = filled, ░░░░ = pre-allocated but unused            │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Design Decisions
//!
//! ## Pre-allocation Strategy
//!
//! The cache pre-allocates memory for the maximum sequence length at creation.
//! This trades memory for performance:
//!
//! | Approach | Memory | Update Cost | Allocation During Generation |
//! |----------|--------|-------------|------------------------------|
//! | Dynamic (Vec::push) | O(current) | O(n) amortized | Yes |
//! | **Pre-allocated** | O(max) | O(new_tokens) | **No** |
//!
//! For text generation where we know the max length upfront, pre-allocation
//! eliminates allocation jitter and enables predictable latency.
//!
//! ## Memory Layout
//!
//! Tensors use shape `[batch, seq_len, hidden_size]` where:
//! - `batch` - Batch dimension (typically 1 for single-sequence generation)
//! - `seq_len` - Sequence dimension (pre-allocated to max, only `current_len` used)
//! - `hidden_size` - Combined dimension for all attention heads (`num_heads * head_dim`)
//!
//! This layout is cache-friendly for the common access pattern of reading
//! all past keys/values for a given batch element.
//!
//! # Performance Characteristics
//!
//! | Operation | Complexity | Notes |
//! |-----------|------------|-------|
//! | `new()` | O(layers × batch × max_len × hidden) | One-time allocation |
//! | `update()` | O(batch × new_tokens × hidden) | Memcpy into slice |
//! | `get()` | O(1) | Returns view (no copy) |
//! | `reorder()` | O(layers × batch × current_len × hidden) | Parallel over layers |
//! | `clear()` | O(1) | Just resets length counter |
//!
//! # Example
//!
//! ```ignore
//! // Create cache for a 12-layer model
//! let mut cache = CpuKVCache::new(
//!     12,     // num_layers
//!     1,      // batch_size
//!     2048,   // max_seq_len
//!     768,    // hidden_size (num_heads * head_dim)
//! );
//!
//! // During prefill: store keys/values for all prompt tokens
//! for layer_idx in 0..12 {
//!     cache.update(layer_idx, &new_keys, &new_values)?;
//! }
//! cache.increment_len(prompt_len);
//!
//! // During decode: append single token's keys/values
//! for layer_idx in 0..12 {
//!     cache.update(layer_idx, &single_k, &single_v)?;
//! }
//! cache.increment_len(1);
//!
//! // In attention: read all cached keys/values
//! let (past_k, past_v) = cache.get(layer_idx).unwrap();
//! // past_k shape: [batch, current_len, hidden]
//! ```

use crate::traits::Cache;
use ndarray::{Array3, ArrayView3, ArrayViewMut3, s};
use rayon::prelude::*;
use std::any::Any;

/// A high-performance, pre-allocated Key-Value cache for CPU decoding.
///
/// This cache stores the key and value tensors from each decoder layer's
/// self-attention mechanism. By pre-allocating to the maximum sequence length,
/// it avoids allocation overhead during the generation loop.
///
/// # Memory Usage
///
/// Total memory = `2 × num_layers × batch_size × max_len × hidden_size × sizeof(f32)`
///
/// Example for Llama 3.2 1B (16 layers, 2048 hidden, 2048 max length):
/// - Per layer: 2 × 1 × 2048 × 2048 × 4 = 32 MB
/// - Total: 16 × 32 MB = 512 MB
///
/// # Thread Safety
///
/// `CpuKVCache` is `Send` but not `Sync`. It should be owned by a single
/// generation task. The `reorder()` method uses Rayon for internal parallelism.
#[derive(Clone)]
pub struct CpuKVCache {
    /// Storage for each decoder layer's key and value tensors.
    ///
    /// Each tuple contains:
    /// - Key tensor: `[batch_size, max_len, hidden_size]`
    /// - Value tensor: `[batch_size, max_len, hidden_size]`
    ///
    /// Only positions `0..current_len` contain valid data.
    layers: Vec<(Array3<f32>, Array3<f32>)>,

    /// The number of tokens currently stored in the cache.
    ///
    /// This is the primary state variable. All operations that modify
    /// the cache (update, clear, increment_len) update this value.
    ///
    /// Invariant: `current_len <= max_len`
    current_len: usize,
}

impl CpuKVCache {
    /// Creates a new cache with pre-allocated storage.
    ///
    /// # Arguments
    ///
    /// * `num_layers` - Number of decoder layers in the model
    /// * `batch_size` - Batch size for generation (typically 1)
    /// * `max_len` - Maximum sequence length (tokens) to support
    /// * `hidden_size` - Hidden dimension (`num_heads × head_dim`)
    ///
    /// # Panics
    ///
    /// May panic if the requested allocation exceeds available memory.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // For Llama 3.2 1B:
    /// let cache = CpuKVCache::new(
    ///     16,     // 16 decoder layers
    ///     1,      // batch size 1
    ///     2048,   // max 2048 tokens
    ///     2048,   // hidden_size = 32 heads × 64 head_dim
    /// );
    /// ```
    pub fn new(num_layers: usize, batch_size: usize, max_len: usize, hidden_size: usize) -> Self {
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            // Pre-allocate with zeros
            // Note: zeros() is slightly slower than uninit but safer
            let k_cache = Array3::zeros((batch_size, max_len, hidden_size));
            let v_cache = Array3::zeros((batch_size, max_len, hidden_size));
            layers.push((k_cache, v_cache));
        }

        Self {
            layers,
            current_len: 0,
        }
    }

    /// Returns the maximum sequence length this cache can hold.
    pub fn max_len(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[1]).unwrap_or(0)
    }

    /// Returns the number of decoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[0]).unwrap_or(0)
    }

    /// Returns the hidden size (num_heads × head_dim).
    pub fn hidden_size(&self) -> usize {
        self.layers.first().map(|(k, _)| k.shape()[2]).unwrap_or(0)
    }

    /// Returns a reference to the underlying layer storage.
    ///
    /// Useful for advanced use cases like serialization or debugging.
    /// Prefer `get()` for normal access.
    pub fn layers(&self) -> &Vec<(Array3<f32>, Array3<f32>)> {
        &self.layers
    }

    /// Appends new key/value tensors to a specific layer's cache.
    ///
    /// The new tensors are copied into the pre-allocated buffer at
    /// position `current_len`. Call `increment_len()` after updating
    /// all layers.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the decoder layer (0-indexed)
    /// * `new_k` - New key tensor, shape `[batch, new_tokens, hidden]`
    /// * `new_v` - New value tensor, shape `[batch, new_tokens, hidden]`
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `layer_idx` is out of bounds
    /// - The update would exceed `max_len`
    ///
    /// # Performance
    ///
    /// This is a memory copy operation: O(batch × new_tokens × hidden).
    /// No allocation occurs.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // During decode step, append single token's KV
    /// let new_k = /* shape [1, 1, hidden] */;
    /// let new_v = /* shape [1, 1, hidden] */;
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
        if layer_idx >= self.layers.len() {
            anyhow::bail!(
                "Layer index {} out of bounds (num_layers={})",
                layer_idx,
                self.layers.len()
            );
        }

        let new_tokens_len = new_k.shape()[1];
        let end_pos = self.current_len + new_tokens_len;

        if end_pos > self.max_len() {
            anyhow::bail!(
                "Cache overflow: current_len={}, new_tokens={}, max_len={}",
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

    /// Returns views of the cached keys and values for a layer.
    ///
    /// Only returns the "active" portion of the cache (positions 0..current_len).
    /// The returned views are zero-copy slices into the underlying storage.
    ///
    /// # Arguments
    ///
    /// * `layer_idx` - Index of the decoder layer (0-indexed)
    ///
    /// # Returns
    ///
    /// `Some((key_view, value_view))` where each view has shape
    /// `[batch, current_len, hidden]`, or `None` if layer_idx is out of bounds.
    ///
    /// # Performance
    ///
    /// O(1) - returns a view without copying data.
    ///
    /// # Example
    ///
    /// ```ignore
    /// if let Some((past_k, past_v)) = cache.get(layer_idx) {
    ///     // past_k.shape() == [batch, current_len, hidden]
    ///     let attention_scores = query.dot(&past_k.t());
    ///     // ...
    /// }
    /// ```
    pub fn get(&self, layer_idx: usize) -> Option<(ArrayView3<'_, f32>, ArrayView3<'_, f32>)> {
        if layer_idx >= self.layers.len() {
            return None;
        }

        let (cache_k, cache_v) = &self.layers[layer_idx];
        let active_slice = s![.., 0..self.current_len, ..];

        Some((cache_k.slice(active_slice), cache_v.slice(active_slice)))
    }

    /// Reorders the batch dimension according to beam search indices.
    ///
    /// Used during beam search to rearrange cache entries when beams are
    /// reordered (e.g., when a beam is pruned and replaced by a copy of
    /// a higher-scoring beam).
    ///
    /// # Arguments
    ///
    /// * `indices` - Reordering indices where `indices[i]` is the source
    ///               batch index for destination position `i`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Before: batch contains beams [A, B, C, D]
    /// // We want: [B, B, D, A] (beam B was best, duplicate it)
    /// cache.reorder(&[1, 1, 3, 0]);
    /// // After: batch contains [B, B, D, A]
    /// ```
    ///
    /// # Performance
    ///
    /// O(layers × batch × current_len × hidden) with Rayon parallelization
    /// across layers. This is an expensive operation - minimize beam search
    /// reordering frequency if possible.
    ///
    /// # Implementation Note
    ///
    /// Currently clones the entire cache before reordering to avoid
    /// aliasing issues. A future optimization could use double-buffering
    /// to reduce allocation.
    pub fn reorder(&mut self, indices: &[usize]) {
        // Clone to avoid aliasing (source and dest would overlap)
        // TODO: Consider double-buffering for better performance
        let mut reordered_layers = self.layers.clone();

        self.layers
            .par_iter()
            .zip(reordered_layers.par_iter_mut())
            .for_each(|((k_cache, v_cache), (k_reordered, v_reordered))| {
                for (dest_idx, &source_idx) in indices.iter().enumerate() {
                    // Copy source batch slice to destination
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
    /// Returns a mutable view of the cache spanning (0 .. current_len + new_tokens).
    ///
    /// This allows the caller to:
    /// 1. Slice the end to write new tokens.
    /// 2. Read the *entire* sequence as a contiguous block for attention.
    pub fn get_context_view_mut(
        &mut self,
        layer_idx: usize,
        new_tokens: usize,
    ) -> anyhow::Result<(ArrayViewMut3<'_, f32>, ArrayViewMut3<'_, f32>)> {
        if layer_idx >= self.layers.len() {
            anyhow::bail!("Layer index out of bounds");
        }

        let (k_cache, v_cache) = &mut self.layers[layer_idx];
        let total_len = self.current_len + new_tokens;

        if total_len > k_cache.shape()[1] {
            anyhow::bail!("Cache overflow: {} > capacity", total_len);
        }

        // Return a view of the ENTIRE active region [Batch, 0..Total, Hidden]
        let k_view = k_cache.slice_mut(s![.., 0..total_len, ..]);
        let v_view = v_cache.slice_mut(s![.., 0..total_len, ..]);

        Ok((k_view, v_view))
    }
    
    /// Returns a boxed clone of this cache.
    ///
    /// Useful for beam search where each beam may need its own cache copy.
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

    /// Sets the current sequence length.
    ///
    /// # Warning
    ///
    /// Setting this to a value greater than what was actually written
    /// will cause `get()` to return uninitialized data.
    fn set_seq_length(&mut self, len: usize) {
        self.current_len = len;
    }

    fn get_seq_length(&self) -> usize {
        self.current_len
    }

    /// Resets the cache to empty state.
    ///
    /// This does NOT deallocate memory - it just resets the length counter.
    /// The pre-allocated buffers remain available for reuse.
    fn clear(&mut self) {
        self.current_len = 0;
    }

    /// Advances the sequence length by the given amount.
    ///
    /// Call this after updating all layers with new tokens.
    ///
    /// # Arguments
    ///
    /// * `new_tokens_len` - Number of new tokens that were added
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

        // Add 10 tokens worth of KV
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

        // Prefill with 5 tokens
        let k1 = Array3::ones((1, 5, 64));
        let v1 = Array3::ones((1, 5, 64));
        cache.update(0, &k1, &v1).unwrap();
        cache.increment_len(5);

        // Add 1 more token
        let k2 = Array3::ones((1, 1, 64)) * 2.0;
        let v2 = Array3::ones((1, 1, 64)) * 2.0;
        cache.update(0, &k2, &v2).unwrap();
        cache.increment_len(1);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k.shape(), &[1, 6, 64]);
        assert_eq!(cached_k[[0, 4, 0]], 1.0); // Original token
        assert_eq!(cached_k[[0, 5, 0]], 2.0); // New token
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
        // Memory is still allocated, just not "used"
        assert_eq!(cache.max_len(), 100);
    }

    #[test]
    fn test_overflow_error() {
        let mut cache = CpuKVCache::new(1, 1, 10, 64);

        let k = Array3::ones((1, 15, 64)); // More than max_len
        let v = Array3::ones((1, 15, 64));

        let result = cache.update(0, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_reorder() {
        let mut cache = CpuKVCache::new(1, 4, 10, 2);

        // Create distinct values for each batch element
        let mut k = Array3::zeros((4, 5, 2));
        let mut v = Array3::zeros((4, 5, 2));
        for b in 0..4 {
            k[[b, 0, 0]] = b as f32;
            v[[b, 0, 0]] = b as f32;
        }

        cache.update(0, &k, &v).unwrap();
        cache.increment_len(5);

        // Reorder: [0,1,2,3] -> [2,2,0,1]
        cache.reorder(&[2, 2, 0, 1]);

        let (cached_k, _) = cache.get(0).unwrap();
        assert_eq!(cached_k[[0, 0, 0]], 2.0); // Was batch 2
        assert_eq!(cached_k[[1, 0, 0]], 2.0); // Was batch 2 (duplicated)
        assert_eq!(cached_k[[2, 0, 0]], 0.0); // Was batch 0
        assert_eq!(cached_k[[3, 0, 0]], 1.0); // Was batch 1
    }
}
