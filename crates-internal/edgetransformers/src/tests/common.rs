// Allow this module to be used by other tests, but not compiled into the final library.
#![allow(dead_code)]

use crate::gpu_ops::{GpuTensor}; // Adjust path as needed
use anyhow::Result;
use crate::WgpuContext;
use ndarray::{Array, Dimension};
use std::sync::Arc;

/// Creates a WGPU context for all tests.
// pub async fn setup_test_context() -> Arc<WgpuContext> {
//     // ... (implementation is the same as before)
// }

/// A test utility function to read any GpuTensor back to the CPU for verification.
pub async fn read_gpu_tensor_to_vec<A>(tensor: &GpuTensor) -> Result<(Vec<A>, Vec<usize>)>
where
    A: bytemuck::Pod + Copy,
{
    let context = tensor.context();
    let buffer = tensor.buffer();
    let size = buffer.size();

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Readback Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    
    context.device.poll(wgpu::PollType::wait_indefinitely());
    rx.receive().await.unwrap()?;

    let data = buffer_slice.get_mapped_range();
    let result_slice: &[A] = bytemuck::cast_slice(&data);
    let result_vec = result_slice.to_vec(); // Convert to an owned Vec
    
    drop(data);
    staging_buffer.unmap();
    
    // Return the flat data and the shape
    Ok((result_vec, tensor.shape().to_vec()))
}