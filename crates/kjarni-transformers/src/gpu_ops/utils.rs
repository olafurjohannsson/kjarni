//! GPU buffer utilities for testing and debugging.
//!
//! Buffer readback functions are slow (CPU/GPU sync) - use only for tests.

use anyhow::Result;
use ndarray::{Array2, Array3};
use wgpu::util::DeviceExt;

use crate::WgpuContext;

/// Reads a GPU buffer as a 2D array. Blocks until GPU completes.
pub async fn read_buffer_2d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize),
) -> Result<Array2<f32>> {
    let (rows, cols) = dims;
    let buffer_size = (rows * cols * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging buffer 2d"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder 2d"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => log::debug!("GPU poll ok: {:?}", status),
        Err(e) => panic!("GPU poll failed: {:?}", e),
    }

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array2::from_shape_vec((rows, cols), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("failed to read back 2D array from GPU")
    }
}

/// Reads a GPU buffer as a 3D array. Blocks until GPU completes.
pub async fn read_buffer_3d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize, usize),
) -> Result<Array3<f32>> {
    let (batch, seq, hidden) = dims;
    let buffer_size = (batch * seq * hidden * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback staging buffer 3d"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback encoder 3d"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => log::debug!("GPU poll ok: {:?}", status),
        Err(e) => panic!("GPU poll failed: {:?}", e),
    }

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array3::from_shape_vec((batch, seq, hidden), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("failed to read back 3D array from GPU")
    }
}

/// Asserts two float slices are approximately equal within tolerance.
pub fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "vectors have different lengths: {} vs {}",
        vec1.len(),
        vec2.len()
    );

    for (i, (&a, &b)) in vec1.iter().zip(vec2.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > tolerance {
            let start = i.saturating_sub(2);
            let end = (i + 3).min(vec1.len());

            panic!(
                "vectors differ at index {}:\n\
                 CPU value: {:.8}\n\
                 GPU value: {:.8}\n\
                 difference: {:.8} (tolerance: {:.8})\n\
                 context [{}-{}]:\n\
                   CPU: {:?}\n\
                   GPU: {:?}",
                i, a, b, diff, tolerance, start, end, &vec1[start..end], &vec2[start..end]
            );
        }
    }
}

/// Creates a GPU uniform buffer from a Pod type.
pub fn create_uniform_buffer<T: bytemuck::Pod>(
    device: &wgpu::Device,
    data: &T,
    label: &str,
) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&[*data]),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_vecs_close_pass() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.00001, 2.00001, 3.00001];
        assert_vecs_are_close(&a, &b, 1e-4);
    }

    #[test]
    #[should_panic(expected = "vectors differ at index")]
    fn test_assert_vecs_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.1, 3.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }

    #[test]
    #[should_panic(expected = "different lengths")]
    fn test_assert_vecs_close_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_vecs_are_close(&a, &b, 1e-5);
    }
}