use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

use super::*;
use crate::gpu_ops::DType;
use crate::WgpuContext;

#[path = "../../../tests/common.rs"]
mod common;
use common::read_gpu_tensor_to_vec;

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

fn assert_all_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "arrays not close, max difference: {}",
        max_diff
    );
}

async fn run_matmul_test(m: usize, k: usize, n: usize) -> Result<()> {
    let context = get_test_context().await;
    let matmul_kernel = GpuMatMul::new(&context);

    let cpu_a = Array::random((m, k), Uniform::new(-1.0, 1.0));
    let cpu_b = Array::random((k, n), Uniform::new(-1.0, 1.0));

    let gpu_a = GpuTensor::from_ndarray(&context, &cpu_a)?;
    let gpu_b = GpuTensor::from_ndarray(&context, &cpu_b)?;
    let gpu_c = GpuTensor::uninitialized(&context, vec![m, n], DType::F32, "matmul_output");

    let mut encoder = context.device.create_command_encoder(&Default::default());
    matmul_kernel.encode(&mut encoder, &[&gpu_a, &gpu_b], &gpu_c);
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_c).await?;
    let gpu_result = Array2::from_shape_vec((result_shape[0], result_shape[1]), gpu_result_vec)?;

    let cpu_result = cpu_a.dot(&cpu_b);
    assert_all_close(&gpu_result, &cpu_result, 1e-3);

    Ok(())
}

#[tokio::test]
async fn test_matmul_small_square() -> Result<()> {
    run_matmul_test(64, 64, 64).await
}

#[tokio::test]
async fn test_matmul_large_square() -> Result<()> {
    run_matmul_test(512, 512, 512).await
}

#[tokio::test]
async fn test_matmul_rectangular_tall() -> Result<()> {
    run_matmul_test(1024, 256, 64).await
}

#[tokio::test]
async fn test_matmul_rectangular_wide() -> Result<()> {
    run_matmul_test(64, 256, 1024).await
}

#[tokio::test]
async fn test_matmul_non_tile_aligned() -> Result<()> {
    run_matmul_test(50, 100, 70).await
}

#[tokio::test]
async fn test_matmul_transformer_ffn_up() -> Result<()> {
    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;
    run_matmul_test(batch_size * seq_len, hidden_size, intermediate_size).await
}

#[tokio::test]
async fn test_matmul_transformer_ffn_down() -> Result<()> {
    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;
    run_matmul_test(batch_size * seq_len, intermediate_size, hidden_size).await
}