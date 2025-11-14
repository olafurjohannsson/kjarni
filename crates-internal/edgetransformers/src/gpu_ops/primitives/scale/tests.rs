#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::gpu_ops::{DType, GpuTensor};
use anyhow::Result;
use common::read_gpu_tensor_to_vec;
use ndarray::{Array, arr2, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await.unwrap())
}

fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays not close. Max diff: {}",
        max_diff
    );
}
// Helper for float comparison
fn assert_arrays_are_close(a: &Array2<f32>, b: &Array2<f32>, epsilon: f32) {
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
async fn test_gpu_scale_out_of_place() -> Result<()> {
    println!("\n--- Testing GpuScale Out-of-Place Kernel ---");

    // Context + kernel
    let context = get_test_context().await;
    let scale_kernel = GpuScale::new(&context);

    // --- 1. Create CPU data ---
    let cpu_input = arr2(&[[1.0, 2.0, 3.0],
                           [4.0, 5.0, 6.0]]);
    let scale_factor = 2.5;

    // --- 2. Upload to GPU + allocate output ---
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let output_shape: Vec<usize> = cpu_input.shape().iter().map(|d| *d).collect();

    let gpu_output = GpuTensor::zeros(
        &context,
        output_shape,
        crate::gpu_ops::DType::F32,
        "f32",
    )?;

    // --- 3. Run GPU kernel ---
    let mut encoder = context.device.create_command_encoder(&Default::default());
    scale_kernel.encode_out_of_place(&mut encoder, &gpu_input, &gpu_output, scale_factor);
    context.queue.submit(std::iter::once(encoder.finish()));

    // Ensure GPU work completes
    let _ = context.device.poll(wgpu::PollType::wait_indefinitely());

    // --- 4. CPU baseline for comparison ---
    let cpu_expected = &cpu_input * scale_factor;

    // --- 5. Read back result from GPU ---
    let gpu_output_cpu = gpu_output.to_ndarray_2d().await?;

    // --- 6. Assert ---
    assert_arrays_are_close(&gpu_output_cpu, &cpu_expected, 1e-6);
    println!("✅ Passed!");

    Ok(())
}
#[tokio::test]
async fn test_scale_parity() -> Result<()> {
    println!("\n--- Testing GpuScale Kernel ---");
    let context = get_test_context().await;
    let scale_kernel = GpuScale::new(&context);

    let scale_factor = 1.0 / (64.0f32).sqrt(); // A realistic attention scaling factor

    // 1. Create Data
    let cpu_input = Array::random((12, 128, 128), Uniform::new(-10.0, 10.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;

    // 2. Execute GPU (in-place)
    let mut encoder = context.device.create_command_encoder(&Default::default());
    scale_kernel.encode_in_place(&mut encoder, &gpu_input, scale_factor);
    context.queue.submit(std::iter::once(encoder.finish()));
    
    // 3. Execute CPU
    let cpu_result = &cpu_input * scale_factor;

    // 4. Compare
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_input).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;

    assert_all_close(&gpu_result, &cpu_result, 1e-6);
    println!("✅ Passed!");

    Ok(())
}
