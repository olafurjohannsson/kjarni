use super::*;
use crate::utils::linear_algebra::{matmul_4d, matmul_3d_2d};
use crate::gpu_ops::{DType, GpuTensor};
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array, Array2, Array3, Array4};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::sync::Arc;

#[path = "../../../tests/common.rs"]
mod common;

use common::{read_gpu_tensor_to_vec};

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}


fn assert_all_close_4d(a: &Array4<f32>, b: &Array4<f32>, tolerance: f32) {
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

/// Helper function to run a single BMM test case.
async fn run_bmm_test(batch: usize, heads: usize, m: usize, k: usize, n: usize) -> Result<()> {
    let context = get_test_context().await;
    let bmm_kernel = GpuBatchedMatMul::new(&context);

    // Create CPU data in 4D (Batch, Heads, M, K)
    let cpu_a_4d = Array::random((batch, heads, m, k), Uniform::new(-1.0, 1.0));
    let cpu_b_4d = Array::random((batch, heads, k, n), Uniform::new(-1.0, 1.0));

    // Reshape to 3D for the GPU kernel, mimicking what the attention block does
    let cpu_a_3d = cpu_a_4d.to_shape((batch * heads, m, k)).unwrap().to_owned();
    let cpu_b_3d = cpu_b_4d.to_shape((batch * heads, k, n)).unwrap().to_owned();

    // Create GPU Tensors
    let gpu_a = GpuTensor::from_ndarray(&context, &cpu_a_3d)?;
    let gpu_b = GpuTensor::from_ndarray(&context, &cpu_b_3d)?;
    let gpu_c = GpuTensor::uninitialized(&context, vec![batch * heads, m, n], DType::F32, "BMM Output");

    // Execute GPU kernel
    let mut encoder = context.device.create_command_encoder(&Default::default());
    bmm_kernel.encode(&mut encoder, &[&gpu_a, &gpu_b], &gpu_c);
    context.queue.submit(std::iter::once(encoder.finish()));
    
    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_c).await?;

    // Read back GPU result and reshape to 4D for comparison
    // let gpu_result_3d: Array3<f32> = gpu_c.to_ndarray_3d().await?;
    let gpu_result_3d = Array3::from_shape_vec((result_shape[0], result_shape[1], result_shape[2]), gpu_result_vec)?;
    let gpu_result_4d = gpu_result_3d.into_shape((batch, heads, m, n)).unwrap();

    // Execute CPU ground truth (using your matmul_4d function)
    let cpu_result_4d = matmul_4d(&cpu_a_4d, &cpu_b_4d);

    // Assert
    println!("Verifying BMM [{}x{}, {}, {}, {}]...", batch, heads, m, k, n);
    assert_all_close_4d(&gpu_result_4d, &cpu_result_4d, 1e-4);
    println!("✅ Passed!");

    Ok(())
}

// --- Test Cases ---

#[tokio::test]
async fn test_bmm_q_k_scores() -> Result<()> {
    // Simulates a typical Q @ K^T operation in attention
    let batch_size = 1;
    let num_heads = 12;
    let seq_len = 128;
    let head_dim = 64;
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_scores_v() -> Result<()> {
    // Simulates a typical Scores @ V operation in attention
    let batch_size = 1;
    let num_heads = 12;
    let seq_len = 128;
    let head_dim = 64;
    run_bmm_test(batch_size, num_heads, seq_len, seq_len, head_dim).await
}

#[tokio::test]
async fn test_bmm_non_tile_aligned() -> Result<()> {
    // Critical test for off-by-one errors with dimensions not divisible by 32
    let batch_size = 1;
    let num_heads = 7; // Odd number of heads
    let seq_len = 50;
    let head_dim = 30;
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_large_batch() -> Result<()> {
    // Test performance and correctness with a larger batch size
    let batch_size = 8;
    let num_heads = 12;
    let seq_len = 256;
    let head_dim = 64;
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_single_token_generation() -> Result<()> {
    // This is a CRITICAL test for decoder models (like GPT).
    // Simulates Q @ K^T where Q is a single token (M=1) and K is the full cache.
    let batch_size = 1;
    let num_heads = 12;
    let query_len = 1;      // M=1
    let head_dim = 64;      // K=64
    let cache_len = 512;    // N=512
    run_bmm_test(batch_size, num_heads, query_len, head_dim, cache_len).await
}

#[tokio::test]
async fn test_bmm_multi_token_generation() -> Result<()> {
    // Simulates speculative decoding or re-computation with a short prompt.
    // Q has a short sequence length, K has the full cache.
    let batch_size = 1;
    let num_heads = 12;
    let query_len = 8;      // M=8
    let head_dim = 64;      // K=64
    let cache_len = 512;    // N=512
    run_bmm_test(batch_size, num_heads, query_len, head_dim, cache_len).await
}

#[tokio::test]
async fn test_bmm_odd_head_dim() -> Result<()> {
    // Some models don't have head dimensions that are a power of 2.
    // This stress-tests the inner loops of your shader.
    let batch_size = 1;
    let num_heads = 8;
    let seq_len = 128;
    let head_dim = 80; // e.g., T5 models
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_broadcast_3d_2d() -> Result<()> {
    // This test specifically verifies parity with your `matmul_3d_2d` CPU function
    let context = get_test_context().await;
    let bmm_kernel = GpuBatchedMatMul::new(&context);

    let batch = 8;
    let m = 64;
    let k = 128;
    let n = 32;

    // Create a 3D tensor A and a 2D tensor B
    let cpu_a = Array::random((batch, m, k), Uniform::new(-1.0, 1.0));
    let cpu_b = Array::random((k, n), Uniform::new(-1.0, 1.0));

    // Create GPU Tensors (note that gpu_b is created from a 2D array)
    let gpu_a = GpuTensor::from_ndarray(&context, &cpu_a)?; // 3D Tensor
    let gpu_b = GpuTensor::from_ndarray(&context, &cpu_b)?; // 2D Tensor
    let gpu_c = GpuTensor::uninitialized(&context, vec![batch, m, n], DType::F32, "BMM Broadcast Output");

    // Execute GPU kernel. The `encode` method will handle the broadcasting logic.
    let mut encoder = context.device.create_command_encoder(&Default::default());
    bmm_kernel.encode(&mut encoder, &[&gpu_a, &gpu_b], &gpu_c);
    context.queue.submit(std::iter::once(encoder.finish()));
    
    // Read back the result
    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_c).await?;
    let gpu_result = Array3::from_shape_vec((result_shape[0], result_shape[1], result_shape[2]), gpu_result_vec)?;

    // Execute CPU ground truth
    let cpu_result = matmul_3d_2d(&cpu_a, &cpu_b);
    
    // Assert
    println!("Verifying BMM Broadcast [{}, {}, {}] @ [{}, {}]...", batch, m, k, k, n);
    let diff = (&gpu_result - &cpu_result).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(max_diff < 1e-3, "Arrays are not close. Max diff: {}", max_diff);
    println!("✅ Passed!");

    Ok(())
}