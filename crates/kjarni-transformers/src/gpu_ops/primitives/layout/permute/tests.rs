
#[path = "../../../../tests/common.rs"]
mod common;


use super::GpuPermute; // Assuming GpuPermute is in the parent `mod.rs`
use crate::gpu::{DType, GpuTensor};
use anyhow::Result;
use common::{read_gpu_tensor_to_vec};
use ndarray::{Array, Array2, Array4};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use common::get_test_context;


// Generic helper to compare arrays of any dimension
fn assert_all_close<D: ndarray::Dimension>(a: &Array<f32, D>, b: &Array<f32, D>, tolerance: f32) {
    assert_eq!(a.shape(), b.shape(), "Array shapes do not match");
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays not close. Max diff: {}",
        max_diff
    );
}

#[tokio::test]
async fn test_permute_2d_transpose() -> Result<()> {
    println!("\n--- Testing GpuPermute (2D Transpose) ---");
    let context = get_test_context().await;
    let permute_kernel = GpuPermute::new(&context);

    // 1. Create Data
    let cpu_input = Array::random((128, 768), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let perm = &[1, 0];
    let output_shape = vec![cpu_input.shape()[1], cpu_input.shape()[0]];

    // 2. Execute GPU
    let gpu_output = GpuTensor::uninitialized(&context, output_shape, DType::F32, "Permute Output");
    let mut encoder = context.device.create_command_encoder(&Default::default());
    permute_kernel.encode(&mut encoder, &gpu_input, &gpu_output, perm);
    context.queue.submit(std::iter::once(encoder.finish()));

    // 3. Execute CPU
    let cpu_result = cpu_input.permuted_axes(*perm);

    // 4. Compare
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array2::from_shape_vec((shape[0], shape[1]), gpu_vec)?;
    
    assert_all_close(&gpu_result, &cpu_result, 1e-6);
    Ok(())
}


#[tokio::test]
async fn test_permute_4d_attention_reshape() -> Result<()> {
    println!("\n--- Testing GpuPermute (Attention Q/V Reshape) ---");
    let context = get_test_context().await;
    let permute_kernel = GpuPermute::new(&context);

    let batch = 2;
    let seq = 128;
    let heads = 12;
    let dims = 64;

    // 1. Create Data in [B, S, H, D] layout
    let cpu_input = Array::random((batch, seq, heads, dims), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    
    // The permutation to get to [B, H, S, D]
    let perm = &[0, 2, 1, 3];
    let output_shape = vec![batch, heads, seq, dims];

    // 2. Execute GPU
    let gpu_output = GpuTensor::uninitialized(&context, output_shape, DType::F32, "Permute Output");
    let mut encoder = context.device.create_command_encoder(&Default::default());
    permute_kernel.encode(&mut encoder, &gpu_input, &gpu_output, perm);
    context.queue.submit(std::iter::once(encoder.finish()));

    // 3. Execute CPU
    let cpu_result = cpu_input.permuted_axes(*perm);

    // 4. Compare
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), gpu_vec)?;
    
    assert_all_close(&gpu_result, &cpu_result.as_standard_layout().to_owned(), 1e-6);
    Ok(())
}


#[tokio::test]
async fn test_permute_4d_attention_k_transpose() -> Result<()> {
    println!("\n--- Testing GpuPermute (Attention K Transpose) ---");
    let context = get_test_context().await;
    let permute_kernel = GpuPermute::new(&context);

    let batch = 2;
    let seq = 128;
    let heads = 12;
    let dims = 64;

    // 1. Create Data in [B, H, S, D] layout
    let cpu_input = Array::random((batch, heads, seq, dims), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    
    // The permutation to get to [B, H, D, S] for Q @ K^T
    let perm = &[0, 1, 3, 2];
    let output_shape = vec![batch, heads, dims, seq];

    // 2. Execute GPU
    let gpu_output = GpuTensor::uninitialized(&context, output_shape, DType::F32, "Permute Output");
    let mut encoder = context.device.create_command_encoder(&Default::default());
    permute_kernel.encode(&mut encoder, &gpu_input, &gpu_output, perm);
    context.queue.submit(std::iter::once(encoder.finish()));

    // 3. Execute CPU
    let cpu_result = cpu_input.permuted_axes(*perm);

    // 4. Compare
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), gpu_vec)?;
    
    assert_all_close(&gpu_result, &cpu_result.as_standard_layout().to_owned(), 1e-6);
    Ok(())
}