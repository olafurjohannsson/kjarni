use crate::cache::Cache;
use crate::gpu_ops::{blocks::cache::GpuUpdateCache};
use crate::WgpuContext;
use crate::gpu::GpuTensor;
use anyhow::{anyhow, Result};
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
        log::info!(
            "[GpuKVCache] Starting new()... batch={}, heads={}, capacity={}",
            batch_size, num_heads, capacity
        );
        let cache_shape = vec![batch_size, num_heads, capacity, head_dim];
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);

        for i in 0..num_layers {

            k_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu::DType::F32,
                &format!("Layer {} K-Cache", i),
            ));
            v_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu::DType::F32,
                &format!("Layer {} V-Cache", i),
            ));
        }

        Ok(Self {
            k_tensors,
            v_tensors,
            seq_length: 0,
            update_kernel: GpuUpdateCache::new(context),
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
        // let position_offset = self.seq_length;
        self.update_kernel
            .encode(encoder, new_k, new_v, cache_k, cache_v, position_offset); //self.seq_length);

        Ok(())
    }

    /// Retrieves a view of the cached keys and values for a specific layer.
    pub fn get(&self, layer_idx: usize) -> Option<(GpuTensor, GpuTensor)> {
        if layer_idx >= self.k_tensors.len() {
            return None;
        }
        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];
        Some((cache_k.clone(), cache_v.clone()))
    }

    pub fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    pub fn max_seq_len(&self) -> usize {
        // The cache is consistent, so we can check the shape of the first k_tensor.
        // The shape is [Batch, Heads, Capacity, HeadDim]. We want the 3rd dimension (index 2).
        if self.k_tensors.is_empty() {
            0
        } else {
            self.k_tensors[0].shape()[2]
        }
    }
    /// Increments the internal length counter after a generation step.
    pub fn increment_len(&mut self, new_tokens_len: usize) {
        let new_len = self.seq_length + new_tokens_len;
        let max = self.max_seq_len();

        if new_len > max {
            panic!(
                "GpuKVCache overflow: tried to set seq_length to {} but capacity is {}. \
             Current: {}, adding: {}. This indicates a bug in the generation loop.",
                new_len, max, self.seq_length, new_tokens_len
            );
        }

        self.seq_length = new_len;
    }
    // pub fn increment_len(&mut self, new_tokens_len: usize) {
    //     self.seq_length += new_tokens_len;
    // }
}

impl Cache for GpuKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    fn clone_box(&self) -> Box<dyn Cache> {
        // This creates a new GpuKVCache struct on the heap.
        // It clones the Vecs (which clones the GpuTensors inside).
        // If GpuTensor is an Arc wrapper, this is a cheap reference count bump.
        // The `seq_length` is copied by value.
        // The `update_kernel` is also cheaply cloned (if it wraps an Arc).
        // let new_cache = GpuKVCache {
        //     k_tensors: self.k_tensors.clone(),
        //     v_tensors: self.v_tensors.clone(),
        //     seq_length: self.seq_length,
        //     update_kernel: self.update_kernel.clone(), // Assuming GpuUpdateCache is Clone
        // };
        // Box::new(new_cache)
        let new_k_tensors = self
            .k_tensors
            .iter()
            .enumerate()
            .map(|(i, t)| t.deep_clone(&format!("Cloned Layer {} K-Cache", i)))
            .collect();

        let new_v_tensors = self
            .v_tensors
            .iter()
            .enumerate()
            .map(|(i, t)| t.deep_clone(&format!("Cloned Layer {} V-Cache", i)))
            .collect();

        let new_cache = GpuKVCache {
            k_tensors: new_k_tensors,
            v_tensors: new_v_tensors,
            seq_length: self.seq_length,
            update_kernel: self.update_kernel.clone(),
        };
        Box::new(new_cache)
    }
    fn clear(&mut self) {
        self.seq_length = 0;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
    fn set_seq_length(&mut self, len: usize) {
        self.seq_length = len;
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn increment_len(&mut self, new_tokens_len: usize) {
        self.increment_len(new_tokens_len);
    }
}
