use crate::cache::Cache;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, blocks::cache::GpuUpdateCache};
use anyhow::{Result, anyhow};
use std::any::Any;
use std::sync::Arc;
use wgpu::CommandEncoder;

/// A high-performance, GPU-resident KV cache for transformer decoders.
///
/// This cache pre-allocates memory and uses a specialized compute kernel
/// for efficient, in-place updates, mirroring the design of the `CpuKVCache`.

pub struct GpuKVCache {
    // A vector of K-cache tensors, one for each decoder layer.
    // Layout: [Batch, Heads, MaxSequenceLength, HeadDim] for efficient attention.
    k_tensors: Vec<GpuTensor>,

    // A vector of V-cache tensors, one for each decoder layer.
    // Layout: [Batch, Heads, MaxSequenceLength, HeadDim]
    v_tensors: Vec<GpuTensor>,

    // The current number of tokens stored in the cache.
    seq_length: usize,

    // The specialized kernel for performing in-place updates.
    update_kernel: GpuUpdateCache,
    context: Arc<WgpuContext>,
}

impl GpuKVCache {
    /// Creates a new, pre-allocated GPU KV cache.
    pub fn new(
        context: &Arc<WgpuContext>,
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(anyhow!("Cache capacity cannot be zero."));
        }

        let cache_shape = vec![batch_size, num_heads, capacity, head_dim];
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            k_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu_ops::DType::F32,
                &format!("Layer {} K-Cache", i),
            ));
            v_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu_ops::DType::F32,
                &format!("Layer {} V-Cache", i),
            ));
        }

        Ok(Self {
            k_tensors,
            v_tensors,
            seq_length: 0,
            update_kernel: GpuUpdateCache::new(context),
            context: context.clone(),
        })
    }

    /// Encodes a GPU command to append new key and value tensors to the cache.
    ///
    /// This method is the architectural equivalent of `CpuKVCache::update`. It uses a
    /// specialized kernel to perform a fused "split-heads and copy" operation.
    pub fn update(
        &self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        new_k: &GpuTensor,
        new_v: &GpuTensor,
        position_offset: usize,
    ) -> Result<()> {
        if layer_idx >= self.k_tensors.len() {
            anyhow::bail!("Layer index {} out of bounds", layer_idx);
        }
        if new_k.rank() != 3 || new_v.rank() != 3 {
            return Err(anyhow::anyhow!(
                "Input tensors for cache update must be rank 3"
            ));
        }
        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];

        self.update_kernel
            .encode(encoder, new_k, new_v, cache_k, cache_v, position_offset); //self.seq_length);

        Ok(())
    }

    /// Retrieves a view of the cached keys and values for a specific layer.
    pub fn get(&self, layer_idx: usize) -> Option<(GpuTensor, GpuTensor)> {
        // if self.seq_length == 0 {
        //     return None;
        // }
        if layer_idx >= self.k_tensors.len() {
            return None;
        }
        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];
        Some((cache_k.clone(), cache_v.clone()))
    }

    /// Increments the internal length counter after a generation step.
    pub fn increment_len(&mut self, new_tokens_len: usize) {
        self.seq_length += new_tokens_len;
    }
}

impl Cache for GpuKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    fn clear(&mut self) {
        self.seq_length = 0;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn increment_len(&mut self, new_tokens_len: usize) {
        self.increment_len(new_tokens_len);
    }
}
