use crate::cache::Cache;
use crate::gpu_context::WgpuContext;
use anyhow::{Result, anyhow};
use std::any::Any;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// A high-performance, GPU-resident KV cache for transformer decoders.
///
/// This cache pre-allocates the full required memory on the GPU at creation time.
/// During generation, it is updated entirely on the GPU via compute shaders,
/// avoiding any costly data transfers between the CPU and GPU.
pub struct GpuKVCache {
    /// A vector of GPU buffers for the K-cache, one for each decoder layer.
    /// Layout: [Batch, Heads, HeadDim, MaxSequenceLength]
    k_buffers: Vec<Arc<wgpu::Buffer>>,

    /// A vector of GPU buffers for the V-cache, one for each decoder layer.
    /// Layout: [Batch, Heads, MaxSequenceLength, HeadDim]
    v_buffers: Vec<Arc<wgpu::Buffer>>,

    /// The current number of tokens stored in the cache.
    seq_length: usize,

    /// The maximum number of tokens this cache can hold.
    capacity: usize,

    batch_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl GpuKVCache {
    /// Creates a new, pre-allocated GPU KV cache.
    ///
    /// # Arguments
    /// * `context` - The WGPU context used to create buffers.
    /// * `num_layers` - The number of decoder layers in the model.
    /// * `batch_size` - The batch size for generation (typically 1).
    /// * `num_heads` - The number of attention heads.
    /// * `head_dim` - The dimension of each attention head.
    /// * `capacity` - The maximum sequence length the cache can hold.
    pub fn new(
        context: &Arc<WgpuContext>,
        num_layers: usize,
        batch_size: usize,
        num_heads: usize,
        head_dim: usize,
        capacity: usize,
    ) -> Result<Self> {
        let device = &context.device;
        let layer_cache_size =
            (batch_size * num_heads * capacity * head_dim * std::mem::size_of::<f32>()) as u64;

        if layer_cache_size == 0 {
            return Err(anyhow!("Cache size cannot be zero."));
        }

        let mut k_buffers = Vec::with_capacity(num_layers);
        let mut v_buffers = Vec::with_capacity(num_layers);

        // CREATE ZERO BUFFER
        let zeros = vec![0.0f32; (layer_cache_size / 4) as usize];

        for i in 0..num_layers {
            let usage = wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC;

            // INITIALIZE WITH ZEROS
            let k_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} K-Cache", i)),
                contents: bytemuck::cast_slice(&zeros),
                usage,
            });
            k_buffers.push(Arc::new(k_buffer));

            let v_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} V-Cache", i)),
                contents: bytemuck::cast_slice(&zeros),
                usage,
            });
            v_buffers.push(Arc::new(v_buffer));
        }

        Ok(Self {
            k_buffers,
            v_buffers,
            seq_length: 0,
            capacity,
            batch_size,
            num_heads,
            head_dim,
        })
    }
    pub fn batch_size(&self) -> usize {
        // This is a bit of a workaround since we don't store it directly.
        // We infer it from the buffer sizes. A better way would be to store it.
        if self.k_buffers.is_empty() {
            return 0;
        }
        let layer_cache_size = self.k_buffers[0].size() as usize;
        let num_elements = layer_cache_size / std::mem::size_of::<f32>();
        // num_elements = B * H * D * S_capacity. We need to get B.
        // Let's add the fields to the struct instead. It's cleaner.

        // I will modify the struct and `new` function below.
        self.batch_size
    }

    /// Returns the number of attention heads the cache was configured with.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    // This is the one you need for the printout
    pub fn get_buffers(
        &self,
        layer_idx: usize,
    ) -> Option<(&Arc<wgpu::Buffer>, &Arc<wgpu::Buffer>)> {
        if layer_idx < self.k_buffers.len() {
            Some((&self.k_buffers[layer_idx], &self.v_buffers[layer_idx]))
        } else {
            None
        }
    }
    /// Returns the dimension of each attention head.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
    /// Returns a slice of the K-cache buffers for all layers.
    pub fn k_buffers(&self) -> &[Arc<wgpu::Buffer>] {
        &self.k_buffers
    }

    /// Returns a slice of the V-cache buffers for all layers.
    pub fn v_buffers(&self) -> &[Arc<wgpu::Buffer>] {
        &self.v_buffers
    }

    /// Returns the maximum number of tokens the cache can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn set_seq_length(&mut self, new_length: usize) {
        // Ensure we don't exceed the pre-allocated capacity.
        if new_length > self.capacity {
            // In a real application, you might want to handle this more gracefully,
            // e.g., by reallocating or using a sliding window.
            panic!(
                "KV Cache capacity exceeded: tried to set seq_length to {}, but capacity is {}.",
                new_length, self.capacity
            );
        }
        self.seq_length = new_length;
    }
    pub fn get_seq_length(&self) -> usize {
        self.seq_length
    }
}

impl Cache for GpuKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    fn clear(&mut self) {
        // Clearing the cache only requires resetting the sequence length.
        // The data in the buffers becomes garbage and will be overwritten on the next run.
        self.seq_length = 0;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}