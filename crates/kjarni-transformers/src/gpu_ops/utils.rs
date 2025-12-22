//! GPU Utilities
//!
//! Helper functions for GPU buffer operations, testing, and debugging.
//!
//! # Performance Note
//! Buffer readback functions (`read_buffer_*`) are **SLOW** - they cause CPU/GPU sync.
//! Only use for testing and debugging, never in production hot paths.

use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array2, Array3};
use wgpu::util::DeviceExt;

/// Read a GPU buffer as a 2D array.
///
/// # Warning
/// This function is **synchronous** and blocks until GPU completes all work.
/// Use only for testing/debugging - causes significant performance overhead.
///
/// # Arguments
/// * `context` - WGPU context with device and queue
/// * `buffer` - GPU buffer to read from
/// * `dims` - Shape as (rows, cols)
///
/// # Returns
/// 2D ndarray containing the buffer data in row-major order
///
/// # Example
/// ```ignore
/// let buffer = /* some GPU buffer */;
/// let array = read_buffer_2d(&context, &buffer, (128, 384)).await?;
/// assert_eq!(array.dim(), (128, 384));
/// ```
///
/// # Implementation Details
/// 1. Creates a staging buffer with MAP_READ capability
/// 2. Copies GPU buffer to staging buffer
/// 3. Submits command and waits for GPU
/// 4. Maps staging buffer to CPU memory
/// 5. Copies data to ndarray
///
/// This causes a full GPU flush and CPU/GPU synchronization point.
pub async fn read_buffer_2d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize),
) -> Result<Array2<f32>> {
    let (rows, cols) = dims;
    let buffer_size = (rows * cols * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    // Create staging buffer accessible from CPU
    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Staging Buffer 2D"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Copy GPU buffer to staging buffer
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder 2D"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    // Map staging buffer to CPU (async operation)
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Wait for GPU to finish
    context.device.poll(wgpu::PollType::wait_indefinitely());

    // Read mapped data
    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array2::from_shape_vec((rows, cols), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("Failed to read back 2D array from GPU")
    }
}

/// Read a GPU buffer as a 3D array.
///
/// # Warning
/// This function is **synchronous** and blocks until GPU completes all work.
/// Use only for testing/debugging - causes significant performance overhead.
///
/// # Arguments
/// * `context` - WGPU context with device and queue
/// * `buffer` - GPU buffer to read from
/// * `dims` - Shape as (batch, sequence, hidden)
///
/// # Returns
/// 3D ndarray containing the buffer data in row-major order
///
/// # Example
/// ```ignore
/// let buffer = /* some GPU buffer */;
/// let array = read_buffer_3d(&context, &buffer, (32, 128, 384)).await?;
/// assert_eq!(array.dim(), (32, 128, 384));
/// ```
///
/// # Performance
/// Same caveats as `read_buffer_2d` - very expensive operation.
/// Typical cost: ~5-10ms regardless of buffer size due to sync overhead.
pub async fn read_buffer_3d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    dims: (usize, usize, usize),
) -> Result<Array3<f32>> {
    let (batch, seq, hidden) = dims;
    let buffer_size = (batch * seq * hidden * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback Staging Buffer 3D"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback Encoder 3D"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, buffer_size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    context.device.poll(wgpu::PollType::wait_indefinitely());

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result_vec: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        let result_array = Array3::from_shape_vec((batch, seq, hidden), result_vec)?;
        Ok(result_array)
    } else {
        anyhow::bail!("Failed to read back 3D array from GPU")
    }
}

/// Assert two float vectors are approximately equal within tolerance.
///
/// This is essential for comparing CPU vs GPU results because floating-point
/// operations are not deterministic across platforms. Tiny differences in
/// rounding and FMA (fused multiply-add) usage are expected.
///
/// # Arguments
/// * `vec1` - First vector (typically CPU result)
/// * `vec2` - Second vector (typically GPU result)
/// * `tolerance` - Maximum absolute difference allowed (e.g., 1e-5)
///
/// # Panics
/// Panics with detailed message if:
/// - Vectors have different lengths
/// - Any element pair differs by more than tolerance
///
/// # Example
/// ```ignore
/// let cpu_result = vec![1.0, 2.0, 3.0];
/// let gpu_result = vec![1.00001, 2.00001, 3.00001];
/// assert_vecs_are_close(&cpu_result, &gpu_result, 1e-4); // ✅ Pass
/// ```
///
/// # Typical Tolerances
/// - `1e-6` - Very strict, for simple operations (add, copy)
/// - `1e-5` - Normal, for most operations (matmul, layer norm)
/// - `1e-4` - Relaxed, for complex chains (full transformer layer)
/// - `1e-3` - Loose, for long accumulations (softmax on large sequences)
pub fn assert_vecs_are_close(vec1: &[f32], vec2: &[f32], tolerance: f32) {
    assert_eq!(
        vec1.len(),
        vec2.len(),
        "Vectors have different lengths: {} vs {}",
        vec1.len(),
        vec2.len()
    );

    for (i, (&a, &b)) in vec1.iter().zip(vec2.iter()).enumerate() {
        let diff = (a - b).abs();
        if diff > tolerance {
            // Find context around error
            let start = i.saturating_sub(2);
            let end = (i + 3).min(vec1.len());

            panic!(
                "Vectors differ at index {}:\n\
                 CPU value: {:.8}\n\
                 GPU value: {:.8}\n\
                 Difference: {:.8} (tolerance: {:.8})\n\
                 Context [{}-{}]:\n\
                   CPU: {:?}\n\
                   GPU: {:?}",
                i,
                a,
                b,
                diff,
                tolerance,
                start,
                end,
                &vec1[start..end],
                &vec2[start..end]
            );
        }
    }
}

/// Create a GPU uniform buffer from a Pod type.
///
/// Uniform buffers hold small constant data (< 64KB) accessible by shaders.
/// Common uses: matrix dimensions, scaling factors, configuration flags.
///
/// # Arguments
/// * `device` - WGPU device
/// * `data` - Data to upload (must implement `bytemuck::Pod`)
/// * `label` - Debug label for buffer
///
/// # Returns
/// GPU buffer with UNIFORM usage, suitable for binding to shader group 0 binding 0
///
/// # Example
/// ```ignore
/// #[repr(C)]
/// #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
/// struct MatmulUniforms {
///     m: u32,
///     k: u32,
///     n: u32,
/// }
///
/// let uniforms = MatmulUniforms { m: 128, k: 384, n: 384 };
/// let buffer = create_uniform_buffer(&device, &uniforms, "Matmul Uniforms");
/// ```
///
/// # Size Limits
/// Uniform buffers are typically limited to 64KB by hardware.
/// For larger data, use storage buffers instead.
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
    #[should_panic(expected = "Vectors differ at index")]
    fn test_assert_vecs_close_fail() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.1, 3.0]; // Too different
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
