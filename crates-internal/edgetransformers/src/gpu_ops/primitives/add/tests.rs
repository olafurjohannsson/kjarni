use crate::gpu_context::WgpuContext;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuTensor, Kernel};
use anyhow::Result;
use ndarray::{Array, Array3, arr3, arr2, Ix3};
use std::sync::Arc;

// --- You will need these helper functions in your test module ---

// Helper to read a GpuTensor back to the CPU for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?.into_dimensionality::<D>()?)
}

/// A crucial helper function to compare CPU and GPU tensors with a tolerance.
async fn assert_tensors_are_close(
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
        println!(
            "CPU tensor (shape {:?}): \n{:?}",
            cpu_tensor.shape(),
            cpu_tensor
        );
        println!(
            "GPU tensor (shape {:?}): \n{:?}",
            gpu_as_cpu.shape(),
            gpu_as_cpu
        );
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

// Helper for float comparison
fn assert_arrays_are_close_3d(a: &Array3<f32>, b: &Array3<f32>, epsilon: f32) {
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

#[tokio::test]
async fn test_gpu_add_broadcast_offset() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await);

    // 1. Setup CPU data
    // `a` is the hidden state: [batch=2, seq=3, hidden=2]
    let a_cpu = Array3::from_elem((2, 3, 2), 1.0);
    // `b` is the positional embedding table: [max_pos=10, hidden=2]
    let b_cpu = Array::from_shape_fn((10, 2), |(i, j)| (i as f32 * 1.0) + (j as f32 * 0.1));
    let b_row_offset = 2;

    // 2. Setup GPU tensors
    let a_gpu = GpuTensor::from_ndarray(&context, &a_cpu)?;
    let b_gpu = GpuTensor::from_ndarray(&context, &b_cpu)?;
    let vec: Vec<usize> = a_cpu.shape().iter().map(|&d| d as usize).collect();
    let output_gpu = GpuTensor::zeros(
        &context,
        vec,
        crate::gpu_ops::DType::F32,
        "f32",
    )?;

    // 3. Execute kernel
    let add_kernel = GpuAdd::new(&context);
    let mut encoder = context.device.create_command_encoder(&Default::default());
    add_kernel.encode_broadcast_offset(&mut encoder, &a_gpu, &b_gpu, b_row_offset, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    
    // Ensure GPU work completes
    let _ = context.device.poll(wgpu::PollType::wait_indefinitely());

    // 4. Verification
    let output_cpu = output_gpu.to_ndarray_3d().await?;

    // Manually compute expected output: output[i,j,k] = a[i,j,k] + b[j + offset, k]
    let expected_output = arr3(&[
        // Batch item 0
        [
            [1.0 + 2.0, 1.0 + 2.1], // seq 0 adds pos 2
            [1.0 + 3.0, 1.0 + 3.1], // seq 1 adds pos 3
            [1.0 + 4.0, 1.0 + 4.1],
        ], // seq 2 adds pos 4
        // Batch item 1 (same logic)
        [
            [1.0 + 2.0, 1.0 + 2.1],
            [1.0 + 3.0, 1.0 + 3.1],
            [1.0 + 4.0, 1.0 + 4.1],
        ],
    ]);

    assert_arrays_are_close_3d(&output_cpu, &expected_output, 1e-6);
    Ok(())
}

#[tokio::test]
async fn test_gpu_add_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await);
    let gpu_add = GpuAdd::new(&context);

    // 1. Setup: Create two tensors on the CPU and GPU
    let shape = (4, 256, 512);
    let a_cpu = Array3::from_shape_fn(shape, |(i, j, k)| (i + j + k) as f32 * 0.1);
    let b_cpu = Array3::from_shape_fn(shape, |(i, j, k)| (k + j + i) as f32 * -0.2);

    let a_gpu = GpuTensor::from_ndarray(&context, &a_cpu)?;
    let b_gpu = GpuTensor::from_ndarray(&context, &b_cpu)?;
    let output_gpu = GpuTensor::uninitialized(
        &context,
        vec![shape.0, shape.1, shape.2],
        a_gpu.dtype(),
        "Add Output",
    );

    // 2. CPU Ground Truth
    let expected_cpu = &a_cpu + &b_cpu;

    // 3. GPU Execution
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_add.encode(&mut encoder, &[&a_gpu, &b_gpu], &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    // 4. Compare
    assert_tensors_are_close(&expected_cpu, &output_gpu, "Add Output", 1e-6).await;

    println!("✅ GpuAdd passed parity test!");
    Ok(())
}
