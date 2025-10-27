//! GPU-based KV cache for transformer decoders

use anyhow::{Result, anyhow};
use ndarray::Array3;
use std::any::Any;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::cache::Cache;
use crate::gpu_context::WgpuContext;

/// GPU-side KV cache for decoder generation
///
/// Stores key and value tensors on GPU memory for each layer.
/// Used for efficient autoregressive generation by avoiding recomputation.
pub struct GpuKVCache {
    context: Arc<WgpuContext>,
    /// Cached K tensors per layer: Vec<Option<Buffer>>
    /// Each buffer stores [batch, cached_seq_len, hidden_size]
    cached_k: Vec<Option<Arc<wgpu::Buffer>>>,
    /// Cached V tensors per layer
    cached_v: Vec<Option<Arc<wgpu::Buffer>>>,
    /// Current sequence length (same for all layers)
    seq_length: usize,
    /// Batch size
    batch_size: usize,
    /// Hidden size
    hidden_size: usize,
    /// Number of layers
    num_layers: usize,
}

impl GpuKVCache {
    /// Create a new GPU KV cache
    ///
    /// # Arguments
    /// * `context` - WGPU context for buffer creation
    /// * `num_layers` - Number of transformer layers
    /// * `batch_size` - Batch size (typically 1 for generation)
    /// * `hidden_size` - Hidden dimension size
    pub fn new(
        context: Arc<WgpuContext>,
        num_layers: usize,
        batch_size: usize,
        hidden_size: usize,
    ) -> Self {
        Self {
            context,
            cached_k: vec![None; num_layers],
            cached_v: vec![None; num_layers],
            seq_length: 0,
            batch_size,
            hidden_size,
            num_layers,
        }
    }

    /// Update cache for a specific layer with new K, V tensors
    ///
    /// This concatenates new K, V with existing cached K, V on GPU.
    ///
    /// # Arguments
    /// * `layer_idx` - Layer index to update
    /// * `new_k` - New K tensor [batch, new_seq_len, hidden]
    /// * `new_v` - New V tensor [batch, new_seq_len, hidden]
    pub fn update(
        &mut self,
        layer_idx: usize,
        new_k: Array3<f32>,
        new_v: Array3<f32>,
    ) -> Result<()> {
        if layer_idx >= self.num_layers {
            return Err(anyhow!(
                "Layer index {} out of bounds (max {})",
                layer_idx,
                self.num_layers - 1
            ));
        }

        let (batch, new_seq_len, hidden) = new_k.dim();
        
        if batch != self.batch_size || hidden != self.hidden_size {
            return Err(anyhow!(
                "Tensor shape mismatch: expected [{}, ?, {}], got [{}, {}, {}]",
                self.batch_size,
                self.hidden_size,
                batch,
                new_seq_len,
                hidden
            ));
        }

        // Concatenate with existing cache (if any)
        let full_k = if let Some(cached_k_buf) = &self.cached_k[layer_idx] {
            // Download cached K from GPU
            let cached_k = self.download_buffer_as_array3(
                cached_k_buf,
                self.batch_size,
                self.seq_length,
                self.hidden_size,
            )?;
            
            // Concatenate along sequence dimension
            ndarray::concatenate(ndarray::Axis(1), &[cached_k.view(), new_k.view()])?
        } else {
            new_k.clone()
        };

        let full_v = if let Some(cached_v_buf) = &self.cached_v[layer_idx] {
            let cached_v = self.download_buffer_as_array3(
                cached_v_buf,
                self.batch_size,
                self.seq_length,
                self.hidden_size,
            )?;
            ndarray::concatenate(ndarray::Axis(1), &[cached_v.view(), new_v.view()])?
        } else {
            new_v.clone()
        };

        // Upload concatenated tensors back to GPU
        self.cached_k[layer_idx] = Some(self.upload_array3_to_buffer(&full_k, layer_idx, "K")?);
        self.cached_v[layer_idx] = Some(self.upload_array3_to_buffer(&full_v, layer_idx, "V")?);

        // Update sequence length (only on first layer to avoid redundancy)
        if layer_idx == 0 {
            self.seq_length = full_k.shape()[1];
        }

        Ok(())
    }

    /// Get cached K, V buffers for a specific layer
    ///
    /// Returns None if this layer hasn't been cached yet.
    pub fn get_buffers(
        &self,
        layer_idx: usize,
    ) -> Option<(&Arc<wgpu::Buffer>, &Arc<wgpu::Buffer>)> {
        if layer_idx >= self.num_layers {
            return None;
        }

        match (&self.cached_k[layer_idx], &self.cached_v[layer_idx]) {
            (Some(k), Some(v)) => Some((k, v)),
            _ => None,
        }
    }

    /// Clear the cache
    pub fn clear(&mut self) {
        self.cached_k = vec![None; self.num_layers];
        self.cached_v = vec![None; self.num_layers];
        self.seq_length = 0;
    }

    /// Helper: Upload Array3 to GPU buffer
    fn upload_array3_to_buffer(
        &self,
        array: &Array3<f32>,
        layer_idx: usize,
        name: &str,
    ) -> Result<Arc<wgpu::Buffer>> {
        let data = array.as_standard_layout();
        let slice = data
            .as_slice()
            .ok_or_else(|| anyhow!("Array must be contiguous"))?;

        let buffer = self.context.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some(&format!("Layer {} Cached {}", layer_idx, name)),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            },
        );

        Ok(Arc::new(buffer))
    }

    /// Helper: Download GPU buffer as Array3
    fn download_buffer_as_array3(
        &self,
        buffer: &wgpu::Buffer,
        batch: usize,
        seq_len: usize,
        hidden: usize,
    ) -> Result<Array3<f32>> {
        let size = (batch * seq_len * hidden * std::mem::size_of::<f32>()) as u64;
        
        // Create staging buffer
        let staging_buffer = self.context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("KV Cache Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from storage to staging
        let mut encoder = self.context.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("KV Cache Download"),
            },
        );
        encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
        self.context.queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        self.context.device.poll(wgpu::PollType::wait_indefinitely());
        futures::executor::block_on(rx)
            .map_err(|_| anyhow!("Failed to receive buffer mapping result"))??;

        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let array = Array3::from_shape_vec((batch, seq_len, hidden), floats.to_vec())?;

        drop(data);
        staging_buffer.unmap();

        Ok(array)
    }
}

impl Cache for GpuKVCache {
    fn get_seq_length(&self) -> usize {
        self.seq_length
    }

    fn clear(&mut self) {
        GpuKVCache::clear(self);
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}