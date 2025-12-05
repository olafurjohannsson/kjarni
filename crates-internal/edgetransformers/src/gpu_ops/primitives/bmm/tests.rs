use super::*;
use crate::WgpuContext;
use crate::gpu_ops::{DType, GpuTensor};
use crate::utils::linear_algebra::{matmul_3d_2d, matmul_4d};
use anyhow::Result;
use ndarray::{Array, Array2, Array3, Array4};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::sync::Arc;

#[path = "../../../tests/common.rs"]
mod common;

use common::{assert_all_close_4d, read_gpu_tensor_to_vec};

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

/// Helper function to run a single BMM test case.
async fn run_bmm_test(batch: usize, heads: usize, m: usize, k: usize, n: usize) -> Result<()> {
    let context = get_test_context().await;
    let bmm_kernel = GpuBatchedMatMul::new(&context);

    // Create CPU data in 4D (Batch, Heads, M, K)
    let cpu_a_4d = Array::random((batch, heads, m, k), Uniform::new(-1.0, 1.0));
    let cpu_b_4d = Array::random((batch, heads, k, n), Uniform::new(-1.0, 1.0));
    let cpu_a_3d = cpu_a_4d.to_shape((batch * heads, m, k)).unwrap().to_owned();
    let cpu_b_3d = cpu_b_4d.to_shape((batch * heads, k, n)).unwrap().to_owned();
    let gpu_a = GpuTensor::from_ndarray(&context, &cpu_a_3d)?;
    let gpu_b = GpuTensor::from_ndarray(&context, &cpu_b_3d)?;
    let gpu_c = GpuTensor::uninitialized(
        &context,
        vec![batch * heads, m, n],
        DType::F32,
        "BMM Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    bmm_kernel.encode(&mut encoder, &[&gpu_a, &gpu_b], &gpu_c);
    context.queue.submit(std::iter::once(encoder.finish()));
    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_c).await?;
    let gpu_result_3d = Array3::from_shape_vec(
        (result_shape[0], result_shape[1], result_shape[2]),
        gpu_result_vec,
    )?;
    let gpu_result_4d = gpu_result_3d.into_shape((batch, heads, m, n)).unwrap();
    let cpu_result_4d = matmul_4d(&cpu_a_4d, &cpu_b_4d);
    assert_all_close_4d(&gpu_result_4d, &cpu_result_4d, 1e-4);
    Ok(())
}
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
    let batch_size = 1;
    let num_heads = 12;
    let seq_len = 128;
    let head_dim = 64;
    run_bmm_test(batch_size, num_heads, seq_len, seq_len, head_dim).await
}

#[tokio::test]
async fn test_bmm_non_tile_aligned() -> Result<()> {
    let batch_size = 1;
    let num_heads = 7; // Odd number of heads
    let seq_len = 50;
    let head_dim = 30;
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_large_batch() -> Result<()> {
    let batch_size = 8;
    let num_heads = 12;
    let seq_len = 256;
    let head_dim = 64;
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_single_token_generation() -> Result<()> {
    let batch_size = 1;
    let num_heads = 12;
    let query_len = 1; // M=1
    let head_dim = 64; // K=64
    let cache_len = 512; // N=512
    run_bmm_test(batch_size, num_heads, query_len, head_dim, cache_len).await
}

#[tokio::test]
async fn test_bmm_multi_token_generation() -> Result<()> {
    let batch_size = 1;
    let num_heads = 12;
    let query_len = 8; // M=8
    let head_dim = 64; // K=64
    let cache_len = 512; // N=512
    run_bmm_test(batch_size, num_heads, query_len, head_dim, cache_len).await
}

#[tokio::test]
async fn test_bmm_odd_head_dim() -> Result<()> {
    let batch_size = 1;
    let num_heads = 8;
    let seq_len = 128;
    let head_dim = 80; // e.g., T5 models
    run_bmm_test(batch_size, num_heads, seq_len, head_dim, seq_len).await
}

#[tokio::test]
async fn test_bmm_broadcast_3d_2d() -> Result<()> {
    let context = get_test_context().await;
    let bmm_kernel = GpuBatchedMatMul::new(&context);

    let batch = 8;
    let m = 64;
    let k = 128;
    let n = 32;

    let cpu_a = Array::random((batch, m, k), Uniform::new(-1.0, 1.0));
    let cpu_b = Array::random((k, n), Uniform::new(-1.0, 1.0));
    let gpu_a = GpuTensor::from_ndarray(&context, &cpu_a)?; // 3D Tensor
    let gpu_b = GpuTensor::from_ndarray(&context, &cpu_b)?; // 2D Tensor
    let gpu_c = GpuTensor::uninitialized(
        &context,
        vec![batch, m, n],
        DType::F32,
        "BMM Broadcast Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    bmm_kernel.encode(&mut encoder, &[&gpu_a, &gpu_b], &gpu_c);
    context.queue.submit(std::iter::once(encoder.finish()));
    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_c).await?;
    let gpu_result = Array3::from_shape_vec(
        (result_shape[0], result_shape[1], result_shape[2]),
        gpu_result_vec,
    )?;
    let cpu_result = matmul_3d_2d(&cpu_a, &cpu_b);

    let diff = (&gpu_result - &cpu_result).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < 1e-3,
        "Arrays are not close. Max diff: {}",
        max_diff
    );

    Ok(())
}
