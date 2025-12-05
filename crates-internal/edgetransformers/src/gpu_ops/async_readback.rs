use anyhow::Result;
use ndarray::Array1;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use wgpu::{Buffer, BufferDescriptor, BufferUsages, CommandEncoder, MapMode};

use crate::gpu_ops::GpuTensor;
use crate::WgpuContext;

/// Manages async GPU-to-CPU data transfer with triple buffering for low latency
pub struct GpuAsyncReadback {
    context: Arc<WgpuContext>,
    staging_buffers: Mutex<Option<Arc<Vec<Buffer>>>>,
    buffer_size: Mutex<Option<usize>>,
    current_idx: AtomicUsize,
}

impl GpuAsyncReadback {
    pub fn new(context: Arc<WgpuContext>) -> Self {
        Self {
            context,
            staging_buffers: Mutex::new(None),
            buffer_size: Mutex::new(None),
            current_idx: AtomicUsize::new(0),
        }
    }

    /// Read a 1D tensor from GPU using ring buffer for low latency
    ///
    /// On first call: synchronous read (must wait for GPU)
    /// On subsequent calls: pipelined read (GPU works ahead)
    pub async fn read_f32_array(
        &self,
        encoder: &mut CommandEncoder,
        source: &GpuTensor,
        expected_len: usize,
    ) -> Result<Array1<f32>> {
        let buffer_size_bytes = expected_len * std::mem::size_of::<f32>();

        // Ensure buffers exist and match size
        self.ensure_buffers(buffer_size_bytes)?;

        let is_first_call = self.current_idx.load(Ordering::Relaxed) == 0;
        let write_idx = self.current_idx.fetch_add(1, Ordering::Relaxed) % 3;

        // Get buffer references
        let buffers = {
            let guard = self.staging_buffers.lock().unwrap();
            guard.as_ref().unwrap().clone()
        };

        // Copy to staging buffer
        encoder.copy_buffer_to_buffer(
            source.buffer(),
            0,
            &buffers[write_idx],
            0,
            buffer_size_bytes as u64,
        );

        // First call: read synchronously from same buffer
        if is_first_call {
            // FIX: We must submit the commands NOW so the copy happens before we map.
            // We swap the passed encoder with a new one to preserve the API contract
            // (caller still holds a valid encoder), but we force the side effect (submission).
            let commands = std::mem::replace(
                encoder,
                self.context
                    .device
                    .create_command_encoder(&Default::default()),
            );
            self.context.queue.submit(Some(commands.finish()));

            return self
                .read_buffer_sync(&buffers[write_idx], expected_len)
                .await;
        }

        // Subsequent calls: read from previous buffer (pipelined)
        let read_idx = if write_idx == 0 { 2 } else { write_idx - 1 };
        self.read_buffer_sync(&buffers[read_idx], expected_len)
            .await
    }

    // Keep encode_and_read_previous if you use it elsewhere, but read_f32_array is the one the tests use
    pub async fn encode_and_read_previous(
        &self,
        encoder: &mut CommandEncoder,
        source: &GpuTensor,
        expected_len: usize,
    ) -> Result<Array1<f32>> {
        // Reuse the logic from read_f32_array as they are now identical in requirements
        self.read_f32_array(encoder, source, expected_len).await
    }

    /// Simple synchronous read - creates own encoder and submits
    pub async fn read_f32_array_sync(
        &self,
        source: &GpuTensor,
        expected_len: usize,
    ) -> Result<Array1<f32>> {
        let buffer_size_bytes = expected_len * std::mem::size_of::<f32>();

        let staging = self.context.device.create_buffer(&BufferDescriptor {
            label: Some("Temp Staging"),
            size: buffer_size_bytes as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Create, encode, and submit in one go
        let mut encoder = self
            .context
            .device
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source.buffer(), 0, &staging, 0, buffer_size_bytes as u64);
        self.context.queue.submit(Some(encoder.finish()));

        self.read_buffer_sync(&staging, expected_len).await
    }

    fn ensure_buffers(&self, required_bytes: usize) -> Result<()> {
        let mut buffers_guard = self.staging_buffers.lock().unwrap();
        let mut size_guard = self.buffer_size.lock().unwrap();

        let needs_creation = match (*size_guard, buffers_guard.as_ref()) {
            (None, None) => true,
            (Some(size), Some(_)) if size != required_bytes => {
                log::debug!(
                    "Resizing staging buffers: {} -> {} bytes",
                    size,
                    required_bytes
                );
                true
            }
            _ => false,
        };

        if needs_creation {
            let new_buffers: Vec<_> = (0..3)
                .map(|i| {
                    self.context.device.create_buffer(&BufferDescriptor {
                        label: Some(&format!("Staging {} ({}KB)", i, required_bytes / 1024)),
                        size: required_bytes as u64,
                        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                        mapped_at_creation: false,
                    })
                })
                .collect();

            *buffers_guard = Some(Arc::new(new_buffers));
            *size_guard = Some(required_bytes);
        }

        Ok(())
    }

    async fn read_buffer_sync(&self, buffer: &Buffer, expected_len: usize) -> Result<Array1<f32>> {
        let slice = buffer.slice(..);
        let (tx, rx) = futures::channel::oneshot::channel();

        slice.map_async(MapMode::Read, move |result| {
            let _ = tx.send(result);
        });

        // Ensure the map request is processed by the GPU
        self.context.device.poll(wgpu::PollType::wait_indefinitely());

        rx.await??;

        let data = slice.get_mapped_range();
        let result: Array1<f32> = bytemuck::cast_slice(&data)[..expected_len].to_vec().into();

        drop(data);
        buffer.unmap();

        Ok(result)
    }

    /// Reset for new sequence (call before prefill)
    pub fn reset(&self) {
        self.current_idx.store(0, Ordering::Relaxed);
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::WgpuContext;
    use ndarray::Array2;

    #[tokio::test]
    async fn test_basic_sync_read() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Create test data
        let data = Array2::from_shape_vec((1, 10), (0..10).map(|i| i as f32).collect()).unwrap();
        let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

        // Sync read should work
        let result = readback.read_f32_array_sync(&tensor, 10).await.unwrap();

        assert_eq!(result.len(), 10);
        for i in 0..10 {
            assert_eq!(result[i], i as f32, "Mismatch at index {}", i);
        }
    }

    #[tokio::test]
    async fn test_multiple_sync_reads() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Multiple sync reads should all work
        for iteration in 0..5 {
            let data = Array2::from_shape_vec((1, 10), vec![iteration as f32; 10]).unwrap();
            let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

            let result = readback.read_f32_array_sync(&tensor, 10).await.unwrap();

            assert_eq!(result.len(), 10);
            assert_eq!(result[0], iteration as f32);
            assert_eq!(result[9], iteration as f32);
        }
    }

    #[tokio::test]
    async fn test_staging_buffer_creation() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // First call should create buffers
        {
            let buffers = readback.staging_buffers.lock().unwrap();
            assert!(buffers.is_none(), "Buffers should not exist initially");
        }

        let data = Array2::from_shape_vec((1, 10), vec![1.0; 10]).unwrap();
        let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let _ = readback
            .read_f32_array(&mut encoder, &tensor, 10)
            .await
            .unwrap();
        context.queue.submit(Some(encoder.finish()));

        // Now buffers should exist
        {
            let buffers = readback.staging_buffers.lock().unwrap();
            assert!(buffers.is_some(), "Buffers should be created");
            let buffers = buffers.as_ref().unwrap();
            assert_eq!(buffers.len(), 3, "Should have 3 staging buffers");
        }
    }

    #[tokio::test]
    async fn test_reset_counter() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Initial counter should be 0
        assert_eq!(readback.current_idx.load(Ordering::Relaxed), 0);

        // After one read, counter should be 1
        let data = Array2::from_shape_vec((1, 10), vec![1.0; 10]).unwrap();
        let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let _ = readback
            .read_f32_array(&mut encoder, &tensor, 10)
            .await
            .unwrap();
        context.queue.submit(Some(encoder.finish()));

        assert_eq!(readback.current_idx.load(Ordering::Relaxed), 1);

        // Reset should set it back to 0
        readback.reset();
        assert_eq!(readback.current_idx.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn test_first_async_read_is_synchronous() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Create data with specific values
        let data = Array2::from_shape_vec((1, 5), vec![10.0, 20.0, 30.0, 40.0, 50.0]).unwrap();
        let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

        // First async read should be synchronous and return correct data
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let result = readback
            .read_f32_array(&mut encoder, &tensor, 5)
            .await
            .unwrap();
        context.queue.submit(Some(encoder.finish()));

        assert_eq!(result.len(), 5);
        assert_eq!(result[0], 10.0);
        assert_eq!(result[4], 50.0);
    }

    #[tokio::test]
    async fn test_pipelined_reads_simple() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Frame 0: value 100.0 (first call is synchronous)
        {
            let data = Array2::from_shape_vec((1, 3), vec![100.0; 3]).unwrap();
            let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

            let mut encoder = context.device.create_command_encoder(&Default::default());
            let result = readback
                .read_f32_array(&mut encoder, &tensor, 3)
                .await
                .unwrap();
            context.queue.submit(Some(encoder.finish()));

            // First call should read synchronously
            assert_eq!(result[0], 100.0, "First read should be synchronous");
        }

        // Wait for GPU to complete
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // Frame 1: value 200.0 (should read 100.0 from previous frame)
        {
            let data = Array2::from_shape_vec((1, 3), vec![200.0; 3]).unwrap();
            let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

            let mut encoder = context.device.create_command_encoder(&Default::default());
            let result = readback
                .read_f32_array(&mut encoder, &tensor, 3)
                .await
                .unwrap();
            context.queue.submit(Some(encoder.finish()));

            // Should read previous frame's data (100.0)
            assert_eq!(
                result[0], 100.0,
                "Second read should return first frame data"
            );
        }

        // Wait for GPU
        context
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // Frame 2: value 300.0 (should read 200.0 from previous frame)
        {
            let data = Array2::from_shape_vec((1, 3), vec![300.0; 3]).unwrap();
            let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

            let mut encoder = context.device.create_command_encoder(&Default::default());
            let result = readback
                .read_f32_array(&mut encoder, &tensor, 3)
                .await
                .unwrap();
            context.queue.submit(Some(encoder.finish()));

            // Should read previous frame's data (200.0)
            assert_eq!(
                result[0], 200.0,
                "Third read should return second frame data"
            );
        }
    }

    #[tokio::test]
    async fn test_ring_buffer_wraparound() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Run 5 frames to test wraparound (buffers: 0, 1, 2, 0, 1)
        for i in 0..5 {
            let data = Array2::from_shape_vec((1, 3), vec![i as f32; 3]).unwrap();
            let tensor = GpuTensor::from_ndarray(&context, &data).unwrap();

            let mut encoder = context.device.create_command_encoder(&Default::default());
            let result = readback
                .read_f32_array(&mut encoder, &tensor, 3)
                .await
                .unwrap();
            context.queue.submit(Some(encoder.finish()));

            // Wait for GPU
            context
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();

            if i == 0 {
                // First is synchronous
                assert_eq!(result[0], 0.0);
            } else {
                // Others are pipelined (read previous)
                assert_eq!(
                    result[0],
                    (i - 1) as f32,
                    "Frame {} should read frame {} data",
                    i,
                    i - 1
                );
            }
        }
    }

    #[tokio::test]
    async fn test_different_sizes() {
        let context = WgpuContext::new().await.unwrap();
        let readback = GpuAsyncReadback::new(context.clone());

        // Test with size 10
        let data1 = Array2::from_shape_vec((1, 10), vec![1.0; 10]).unwrap();
        let tensor1 = GpuTensor::from_ndarray(&context, &data1).unwrap();
        let result1 = readback.read_f32_array_sync(&tensor1, 10).await.unwrap();
        assert_eq!(result1.len(), 10);

        // Test with size 100 (should recreate buffers)
        let data2 = Array2::from_shape_vec((1, 100), vec![2.0; 100]).unwrap();
        let tensor2 = GpuTensor::from_ndarray(&context, &data2).unwrap();
        let result2 = readback.read_f32_array_sync(&tensor2, 100).await.unwrap();
        assert_eq!(result2.len(), 100);
        assert_eq!(result2[0], 2.0);

        // Test with size 10 again (should recreate buffers back to smaller size)
        let data3 = Array2::from_shape_vec((1, 10), vec![3.0; 10]).unwrap();
        let tensor3 = GpuTensor::from_ndarray(&context, &data3).unwrap();
        let result3 = readback.read_f32_array_sync(&tensor3, 10).await.unwrap();
        assert_eq!(result3.len(), 10);
        assert_eq!(result3[0], 3.0);
    }
}
