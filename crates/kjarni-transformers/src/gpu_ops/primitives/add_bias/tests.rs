#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::gpu_ops::{DType, GpuTensor};
use common::{read_gpu_tensor_to_vec};
use anyhow::Result;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

fn assert_all_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays are not close. Max difference: {}, Tolerance: {}",
        max_diff,
        tolerance
    );
}
async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

#[tokio::test]
async fn test_add_bias_parity() -> Result<()> {
    let context = get_test_context().await;
    let add_bias_kernel = GpuAddBias::new(&context);

    // Simulate the output of a typical linear layer
    let rows = 128;
    let cols = 768; // hidden_size

    // 1. Create CPU data
    let cpu_input = Array::random((rows, cols), Uniform::new(-10.0, 10.0));
    let cpu_bias = Array::random(cols, Uniform::new(-1.0, 1.0));

    // 2. Create GPU tensors
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_bias = GpuTensor::from_ndarray(&context, &cpu_bias)?; // from_ndarray works for 1D Array1 as well
    let gpu_output = GpuTensor::uninitialized(&context, vec![rows, cols], DType::F32, "AddBias Output");

    // 3. Execute GPU kernel
    let mut encoder = context.device.create_command_encoder(&Default::default());
    add_bias_kernel.encode(&mut encoder, &[&gpu_input, &gpu_bias], &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));

    // 4. Read back GPU result
    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array2::from_shape_vec((result_shape[0], result_shape[1]), gpu_result_vec)?;

    // 5. Execute CPU ground truth
    // ndarray handles broadcasting the 1D bias vector across the rows of the 2D input array.
    let cpu_result = &cpu_input + &cpu_bias;

    // 6. Assert
    println!("Verifying GpuAddBias parity...");
    assert_all_close(&gpu_result, &cpu_result, 1e-6); // Should be very precise
    println!("✅ GpuAddBias parity test passed!");

    Ok(())
}