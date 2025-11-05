#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::gpu_ops::{DType, GpuTensor};
use anyhow::Result;
use common::{read_gpu_tensor_to_vec};
use ndarray::{Array, Array3};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}


fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(max_diff < tolerance, "Arrays not close. Max diff: {}", max_diff);
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
    scale_kernel.encode(&mut encoder, &gpu_input, scale_factor);
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