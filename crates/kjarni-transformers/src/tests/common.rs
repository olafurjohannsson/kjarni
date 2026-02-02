// Allow this module to be used by other tests, but not compiled into the final library.
#![allow(dead_code)]

use crate::WgpuContext;
use crate::gpu::GpuTensor; // Adjust path as needed
use anyhow::Result;
use ndarray::{Array, Array2, Array3, Array4, Dimension, Ix2, Ix3, Ix4};
use std::sync::Arc;

pub async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

pub async fn read_gpu_tensor_4d(tensor: &GpuTensor) -> Result<Array4<f32>> {
    let shape = tensor.shape();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        data_slice.to_vec(),
    )?)
}

pub async fn read_gpu_tensor<D: Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?.into_dimensionality::<D>()?)
}

/// A crucial helper function to compare CPU and GPU tensors with a tolerance.
pub async fn assert_tensors_are_close_4d(
    cpu_tensor: &Array4<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix4>(gpu_tensor).await.unwrap();
    let close = cpu_tensor
        .iter()
        .zip(gpu_as_cpu.iter())
        .all(|(a, b)| (a - b).abs() < tolerance);

    if !close {
        // For smaller tensors, you can print the whole thing
        println!("CPU tensor: \n{:?}", cpu_tensor);
        println!("GPU tensor: \n{:?}", gpu_as_cpu);
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

/// A crucial helper function to compare CPU and GPU tensors with a tolerance.
pub async fn assert_tensors_are_close_2d(
    cpu_tensor: &Array2<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix2>(gpu_tensor).await.unwrap();

    // Calculate the absolute differences for all elements
    let diffs = (cpu_tensor - &gpu_as_cpu).mapv(f32::abs);

    // Find the maximum difference and its index
    let mut max_diff = 0.0;
    let mut max_diff_index = 0;
    
    for (i, &d) in diffs.iter().enumerate() {
        if d > max_diff {
            max_diff = d;
            max_diff_index = i;
        }
    }

    if max_diff > tolerance {
        println!("Mismatch in tensor '{}'", label);
        println!("Max difference: {} at flat index {}", max_diff, max_diff_index);
        
        // Only print full tensors if they aren't massive to avoid spamming the logs
        if cpu_tensor.len() < 1000 {
            println!("CPU tensor: \n{:?}", cpu_tensor);
            println!("GPU tensor: \n{:?}", gpu_as_cpu);
        }
        
        panic!(
            "Tensor '{}' mismatch. Max diff: {} > tolerance {}", 
            label, max_diff, tolerance
        );
    }
}

pub async fn assert_tensors_are_close(
    cpu_tensor: &Array3<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix3>(gpu_tensor).await.unwrap();
    let close = cpu_tensor
        .iter()
        .zip(gpu_as_cpu.iter())
        .all(|(a, b)| (a - b).abs() < tolerance);

    if !close {
        println!("Mismatch in tensor '{}'", label);
        println!("CPU tensor: \n{:?}", cpu_tensor);
        println!("GPU tensor: \n{:?}", gpu_as_cpu);
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

pub fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays not close. Max diff: {}",
        max_diff
    );
}

pub fn assert_arrays_are_close_2d(a: &Array2<f32>, b: &Array2<f32>, epsilon: f32) {
    assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
    for (val_a, val_b) in a.iter().zip(b.iter()) {
        assert!(
            (val_a - val_b).abs() < epsilon,
            "Values differ: {} vs {}",
            val_a,
            val_b
        );
    }
}

pub fn assert_all_close_4d(a: &Array4<f32>, b: &Array4<f32>, tolerance: f32) {
    assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays are not close. Max difference: {}, Tolerance: {}",
        max_diff,
        tolerance
    );
}

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

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => log::debug!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }

    rx.receive().await.unwrap()?;

    let data = buffer_slice.get_mapped_range();
    let result_slice: &[A] = bytemuck::cast_slice(&data);
    let result_vec = result_slice.to_vec(); // Convert to an owned Vec

    drop(data);
    staging_buffer.unmap();

    // Return the flat data and the shape
    Ok((result_vec, tensor.shape().to_vec()))
}