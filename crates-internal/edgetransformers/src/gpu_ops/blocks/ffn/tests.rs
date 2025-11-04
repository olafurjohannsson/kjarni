#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::feedforward::FeedForward as CpuFeedForward;
use crate::gpu_ops::{DType, GpuTensor};
use common::{read_gpu_tensor_to_vec};
use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3};
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
async fn test_ffn_block_parity() -> Result<()> {
    let context = get_test_context().await;
    
    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    // 1. Create CPU weights and FFN block
    let cpu_fc1_w = Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1));
    let cpu_fc1_b = Array::random(intermediate_size, Uniform::new(-0.1, 0.1));
    let cpu_fc2_w = Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1));
    let cpu_fc2_b = Array::random(hidden_size, Uniform::new(-0.1, 0.1));
    
    let cpu_ffn = CpuFeedForward::new(
        cpu_fc1_w.t().as_standard_layout().to_owned(), // Your CPU FFN expects transposed weights
        cpu_fc1_b.clone(),
        cpu_fc2_w.t().as_standard_layout().to_owned(),
        cpu_fc2_b.clone(),
    );

    // 2. Create GPU weights (as Tensors) and GpuFeedForward block
    let gpu_fc1_w = GpuTensor::from_ndarray(&context, &cpu_fc1_w)?;
    let gpu_fc1_b = GpuTensor::from_ndarray(&context, &cpu_fc1_b)?;
    let gpu_fc2_w = GpuTensor::from_ndarray(&context, &cpu_fc2_w)?;
    let gpu_fc2_b = GpuTensor::from_ndarray(&context, &cpu_fc2_b)?;

    let gpu_ffn = GpuFeedForward::new(&context, gpu_fc1_w, gpu_fc1_b, gpu_fc2_w, gpu_fc2_b);

    // 3. Create input data
    let cpu_input = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    
    // Create output & temporary buffers for the GPU
    let gpu_intermediate = GpuTensor::uninitialized(&context, vec![batch_size, seq_len, intermediate_size], DType::F32, "FFN Intermediate");
    let gpu_output = GpuTensor::uninitialized(&context, vec![batch_size, seq_len, hidden_size], DType::F32, "FFN Output");

    // 4. Execute
    // a. GPU
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn.forward(&mut encoder, &gpu_input, &gpu_intermediate, &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;

    // b. CPU
    let cpu_result = cpu_ffn.forward(&cpu_input)?;

    // 5. Assert
    println!("Verifying GpuFeedForward block parity...");
    assert_all_close(&gpu_result, &cpu_result, 1e-3); // Looser tolerance for fused kernel
    println!("✅ GpuFeedForward passed!");

    Ok(())
}