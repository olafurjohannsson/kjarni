use crate::cache::Cache;
use crate::gpu_ops::{
    blocks::cache::reorder::GpuReorderCache, blocks::cache::GpuUpdateCache, GpuTensor,
};
use crate::WgpuContext;
use anyhow::{anyhow, Result};
use std::any::Any;
use std::sync::Arc;
use wgpu::CommandEncoder;

/// A specialized, high-performance, GPU-resident KV cache for beam search decoding.
///
/// This cache pre-allocates memory for all beams and uses a specialized `reorder`
/// kernel to efficiently shuffle beam histories, avoiding expensive deep copies.
pub struct GpuBeamKVCache {
    // Main K/V tensors. Layout: [NumBeams, NumHeads, Capacity, HeadDim]
    k_tensors: Vec<GpuTensor>,
    v_tensors: Vec<GpuTensor>,

    // Temporary tensors used as the destination for the reorder operation.
    temp_k_tensors: Vec<GpuTensor>,
    temp_v_tensors: Vec<GpuTensor>,

    // The current number of tokens stored in the cache.
    seq_length: usize,

    // Kernels for cache operations.
    update_kernel: GpuUpdateCache,
    reorder_kernel: GpuReorderCache,
}

impl GpuBeamKVCache {
    /// Creates a new, pre-allocated GPU KV cache optimized for beam search.
    pub fn new(
        context: &Arc<WgpuContext>,
        num_layers: usize,
        num_beams: usize, // Now takes num_beams instead of batch_size
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self> {
        if capacity == 0 {
            return Err(anyhow!("Cache capacity cannot be zero."));
        }

        let cache_shape = vec![num_beams, num_heads, capacity, head_dim];
        let mut k_tensors = Vec::with_capacity(num_layers);
        let mut v_tensors = Vec::with_capacity(num_layers);
        let mut temp_k_tensors = Vec::with_capacity(num_layers);
        let mut temp_v_tensors = Vec::with_capacity(num_layers);

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
            // Allocate the temporary buffers for reordering
            temp_k_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu_ops::DType::F32,
                &format!("Temp Layer {} K-Cache", i),
            ));
            temp_v_tensors.push(GpuTensor::uninitialized(
                context,
                cache_shape.clone(),
                crate::gpu_ops::DType::F32,
                &format!("Temp Layer {} V-Cache", i),
            ));
        }

        Ok(Self {
            k_tensors,
            v_tensors,
            temp_k_tensors,
            temp_v_tensors,
            seq_length: 0,
            update_kernel: GpuUpdateCache::new(context),
            reorder_kernel: GpuReorderCache::new(context),
        })
    }

    /// Updates all beams with their new KV values. This method is stateful and
    /// updates the internal sequence length.
    pub fn update(
        &mut self,
        encoder: &mut CommandEncoder,
        layer_idx: usize,
        new_k: &GpuTensor, // Expected shape: [NumBeams, 1, HiddenSize]
        new_v: &GpuTensor, // Expected shape: [NumBeams, 1, HiddenSize]
    ) -> Result<()> {
        // ASSERTION 1: Valid layer index
        assert!(
            layer_idx < self.k_tensors.len(),
            "Layer index {} out of bounds (num_layers: {})",
            layer_idx,
            self.k_tensors.len()
        );

        // ASSERTION 2: Check input shapes
        assert_eq!(new_k.shape().len(), 3, "new_k must be 3D");
        assert_eq!(new_v.shape().len(), 3, "new_v must be 3D");
        assert_eq!(new_k.shape()[1], 1, "new_k must have seq_len=1");
        assert_eq!(new_v.shape()[1], 1, "new_v must have seq_len=1");

        // ASSERTION 3: Check we're not exceeding capacity
        let capacity = self.k_tensors[0].shape()[2];
        assert!(
            self.seq_length < capacity,
            "Cache is full! seq_length={}, capacity={}",
            self.seq_length,
            capacity
        );

        // ASSERTION 4: Verify tensor dimensions match
        let expected_beams = self.k_tensors[0].shape()[0];
        assert_eq!(
            new_k.shape()[0],
            expected_beams,
            "Beam count mismatch: got {}, expected {}",
            new_k.shape()[0],
            expected_beams
        );

        let cache_k = &self.k_tensors[layer_idx];
        let cache_v = &self.v_tensors[layer_idx];
        let position_offset = self.seq_length;

        self.update_kernel
            .encode(encoder, new_k, new_v, cache_k, cache_v, position_offset);
        Ok(())
    }

    /// Reorders the beam histories based on the parent indices from the beam search algorithm.
    pub fn reorder(&mut self, encoder: &mut CommandEncoder, parent_indices: &GpuTensor) {
        // ASSERTION 5: Check indices shape
        assert_eq!(parent_indices.shape().len(), 1, "parent_indices must be 1D");

        let num_beams = self.k_tensors[0].shape()[0];
        assert_eq!(
            parent_indices.shape()[0],
            num_beams,
            "parent_indices length {} doesn't match num_beams {}",
            parent_indices.shape()[0],
            num_beams
        );

        // ASSERTION 6: Can't reorder empty cache
        assert!(self.seq_length > 0, "Cannot reorder empty cache!");

        // Original reorder code...
        for i in 0..self.k_tensors.len() {
            self.reorder_kernel.encode(
                encoder,
                &self.k_tensors[i],
                &self.temp_k_tensors[i],
                parent_indices,
                self.seq_length,
            );
        }
        // Reorder V tensors
        for i in 0..self.v_tensors.len() {
            self.reorder_kernel.encode(
                encoder,
                &self.v_tensors[i],
                &self.temp_v_tensors[i],
                parent_indices,
                self.seq_length,
            );
        }

        // Swap the pointers: the temporary buffers are now the main buffers.
        std::mem::swap(&mut self.k_tensors, &mut self.temp_k_tensors);
        std::mem::swap(&mut self.v_tensors, &mut self.temp_v_tensors);
    }

    // We still need a way to get the tensors for the forward pass.
    pub fn get_layer_tensors(&self, layer_idx: usize) -> Option<(&GpuTensor, &GpuTensor)> {
        if layer_idx >= self.k_tensors.len() {
            None
        } else {
            Some((&self.k_tensors[layer_idx], &self.v_tensors[layer_idx]))
        }
    }
}

// Implement the base Cache trait
impl Cache for GpuBeamKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }
    fn set_seq_length(&mut self, len: usize) {
        self.seq_length = len;
    }
    // fn increment_len(&mut self, new_tokens_len: usize) { self.seq_length += new_tokens_len; }
    fn increment_len(&mut self, new_tokens_len: usize) {
        self.seq_length += new_tokens_len;

        let capacity = self.k_tensors[0].shape()[2];
        assert!(
            self.seq_length <= capacity,
            "Cache overflow! Tried to set length {} but capacity is {}",
            self.seq_length,
            capacity
        );
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

    // This is an expensive operation and should not be used in a hot loop for beam search.
    // The `reorder` method is the performant alternative.
    fn clone_box(&self) -> Box<dyn Cache> {
        // Your deep-clone logic is correct for creating a truly independent copy.
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

        Box::new(Self {
            k_tensors: new_k_tensors,
            v_tensors: new_v_tensors,
            temp_k_tensors: vec![], // Temp buffers don't need to be cloned
            temp_v_tensors: vec![], // They are recreated if this new cache is reordered
            seq_length: self.seq_length,
            update_kernel: self.update_kernel.clone(),
            reorder_kernel: self.reorder_kernel.clone(),
        })
    }
}
