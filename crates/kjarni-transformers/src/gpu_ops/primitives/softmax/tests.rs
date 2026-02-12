use crate::activations::softmax_4d_inplace;



#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::gpu::GpuTensor;
use anyhow::Result;
use common::{read_gpu_tensor_to_vec, get_test_context, assert_all_close_4d};
use ndarray::{Array, Array2, Array4, Axis, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;


macro_rules! shape_4d {
    ($shape_vec:expr) => {
        [$shape_vec[0], $shape_vec[1], $shape_vec[2], $shape_vec[3]]
    };
}

/// CPU reference softmax implementation for a single row.
fn cpu_softmax(row: &mut [f32]) {
    if row.is_empty() {
        return;
    }
    let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0;
    for val in row.iter_mut() {
        *val = (*val - max_val).exp();
        sum += *val;
    }
    for val in row.iter_mut() {
        *val /= sum;
    }
}

fn assert_all_close_2d(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays not close. Max diff: {}",
        max_diff
    );
}

#[tokio::test]
async fn test_softmax_simple_case() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);

    let rows = 12 * 1; // batch * heads
    let cols = 128;
    let mut cpu_data = Array::random((rows, cols), Uniform::new(-1.0, 5.0));
    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_data)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode(&mut encoder, &gpu_tensor, 1.0); // scale = 1.0
    context.queue.submit(std::iter::once(encoder.finish()));
    for mut row in cpu_data.axis_iter_mut(Axis(0)) {
        cpu_softmax(row.as_slice_mut().unwrap());
    }
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array2::from_shape_vec((shape[0], shape[1]), gpu_vec)?;
    assert_all_close_2d(&gpu_result, &cpu_data, 1e-5);
    Ok(())
}

#[tokio::test]
async fn test_softmax_padded_case() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);

    let rows = 12;
    let physical_cols = 512;
    let logical_cols = 123;
    let mut cpu_data = Array::from_elem((rows, physical_cols), f32::NAN); // Fill padding with NaN
    let mut valid_part = Array::random((rows, logical_cols), Uniform::new(-1.0, 5.0));
    cpu_data
        .slice_mut(s![.., 0..logical_cols])
        .assign(&valid_part);

    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_data)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode_padded(&mut encoder, &gpu_tensor, logical_cols as u32, 1.0);
    context.queue.submit(std::iter::once(encoder.finish()));
    for mut row in valid_part.axis_iter_mut(Axis(0)) {
        cpu_softmax(row.as_slice_mut().unwrap());
    cpu_data
        .slice_mut(s![.., 0..logical_cols])
        .assign(&valid_part);
    cpu_data.slice_mut(s![.., logical_cols..]).fill(0.0);
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array2::from_shape_vec((shape[0], shape[1]), gpu_vec)?;
    assert_all_close_2d(&gpu_result, &cpu_data, 1e-5);
    Ok(())
}

#[tokio::test]
async fn test_softmax_simple_case2() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);
    let batch = 1;
    let heads = 12;
    let seq_len = 128;
    let mut cpu_scores = Array::random((batch, heads, seq_len, seq_len), Uniform::new(-1.0, 5.0));
    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode(&mut encoder, &gpu_tensor, 1.0); // scale = 1.0
    context.queue.submit(std::iter::once(encoder.finish()));
    softmax_4d_inplace(&mut cpu_scores);
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array4::from_shape_vec(shape_4d!(shape), gpu_vec)?;
    assert_all_close_4d(&gpu_result, &cpu_scores, 1e-5);
    Ok(())
}

#[tokio::test]
async fn test_softmax_padded_case2() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);

    let batch = 1;
    let heads = 12;
    let query_len = 1;
    let physical_cols = 512; // e.g., cache_capacity
    let logical_cols = 123; // e.g., total_seq_len

    let mut cpu_data_padded = Array::from_elem((batch, heads, query_len, physical_cols), f32::NAN);
    let mut valid_part = Array::random(
        (batch, heads, query_len, logical_cols),
        Uniform::new(-1.0, 5.0),
    );
    cpu_data_padded
        .slice_mut(s![.., .., .., 0..logical_cols])
        .assign(&valid_part);

    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_data_padded)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode_padded(&mut encoder, &gpu_tensor, logical_cols as u32, 1.0);
    context.queue.submit(std::iter::once(encoder.finish()));
    softmax_4d_inplace(&mut valid_part);
    let cpu_result_valid = valid_part.clone();
    let mut cpu_result_padded = Array4::<f32>::zeros((batch, heads, query_len, physical_cols));
    cpu_result_padded
        .slice_mut(s![.., .., .., 0..logical_cols])
        .assign(&cpu_result_valid);
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array4::from_shape_vec(shape_4d!(shape), gpu_vec)?;
    assert_all_close_4d(&gpu_result, &cpu_result_padded, 1e-5);
    Ok(())
}

#[tokio::test]
async fn test_softmax_with_scaling() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);

    let batch = 1;
    let heads = 12;
    let seq_len = 64;
    let scale = 0.125;

    let cpu_scores = Array::random((batch, heads, seq_len, seq_len), Uniform::new(-5.0, 5.0));
    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode(&mut encoder, &gpu_tensor, scale);
    context.queue.submit(std::iter::once(encoder.finish()));
    let mut cpu_scores_scaled = &cpu_scores * scale;
    softmax_4d_inplace(&mut cpu_scores_scaled);
    let cpu_result = cpu_scores_scaled.clone();
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array4::from_shape_vec(shape_4d!(shape), gpu_vec)?;
    assert_all_close_4d(&gpu_result, &cpu_result, 1e-5);
    Ok(())
}

#[tokio::test]
async fn test_softmax_numerical_stability() -> Result<()> {
    let context = get_test_context().await;
    let softmax_kernel = GpuSoftmax::new(&context);

    let batch = 1;
    let heads = 4;
    let seq_len = 32;

    // 1. Create data with large values to test stability
    let mut cpu_scores = Array::random((batch, heads, seq_len, seq_len), Uniform::new(100.0, 110.0));
    let gpu_tensor = GpuTensor::from_ndarray(&context, &cpu_scores)?;

    // 2. Execute GPU
    let mut encoder = context.device.create_command_encoder(&Default::default());
    softmax_kernel.encode(&mut encoder, &gpu_tensor, 1.0);
    context.queue.submit(std::iter::once(encoder.finish()));

    // 3. Execute CPU
    softmax_4d_inplace(&mut cpu_scores);
    let cpu_result = cpu_scores.clone();

    // 4. Read back and compare
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_tensor).await?;
    let gpu_result = Array4::from_shape_vec(shape_4d!(shape), gpu_vec)?;

    println!("Verifying Softmax (Numerical Stability)...");
    // We might need a slightly looser tolerance here due to large exponents
    assert_all_close_4d(&gpu_result, &cpu_result, 1e-4);
    println!("Passed!");

    Ok(())
}
