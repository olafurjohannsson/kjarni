use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::GpuTensor;
use crate::rope::RoPE as CpuRoPE; // Import your CPU implementation
use anyhow::Result;
use ndarray::{Array, Array4, Ix4};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::sync::Arc;

// Helper to read a GPU tensor back to a generic ndarray for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
        .into_dimensionality::<D>()
        .unwrap())
}

/// A crucial helper function to compare CPU and GPU tensors with a tolerance.
async fn assert_tensors_are_close(
    cpu_tensor: &Array4<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix4>(gpu_tensor).await.unwrap();
    let close = cpu_tensor
        .iter()
        .zip(gpu_as_cpu.iter())
        .all(|(a, b)| (a - b).abs() < tolerance);

    if !close {
        // For smaller tensors, you can print the whole thing
        println!("CPU tensor: \n{:?}", cpu_tensor);
        println!("GPU tensor: \n{:?}", gpu_as_cpu);
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

#[tokio::test]
async fn test_gpu_rope_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
    let (b, h, s, d) = (2, 8, 32, 64); // Batch, Heads, SeqLen, HeadDim
    let max_seq = 128;
    let theta = 10000.0;
    let position_offset = 16;

    // 1. Setup: Create CPU RoPE and get the caches
    let cpu_rope = CpuRoPE::new(d, max_seq, theta);
    let cos_cache_cpu = cpu_rope.cos_cache.clone();
    let sin_cache_cpu = cpu_rope.sin_cache.clone();

    // Create GPU RoPE, which will upload the caches
    let gpu_rope = GpuRoPE::new(&context, &cos_cache_cpu, &sin_cache_cpu)?;

    // Create identical input tensors for CPU and GPU
    let q_cpu = Array::random((b, h, s, d), Uniform::new(-1.0, 1.0));
    let k_cpu = Array::random((b, h, s, d), Uniform::new(-1.0, 1.0));
    
    // The GPU tensors will be modified in-place, so we clone the CPU versions for the ground truth calculation
    let q_gpu = GpuTensor::from_ndarray(&context, &q_cpu)?;
    let k_gpu = GpuTensor::from_ndarray(&context, &k_cpu)?;

    // 2. CPU Ground Truth
    let (expected_q, expected_k) = cpu_rope.apply_4d(&q_cpu, &k_cpu, position_offset);
    
    // 3. GPU Execution
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_rope.encode(&mut encoder, &q_gpu, &k_gpu, position_offset);
    context.queue.submit(Some(encoder.finish()));

    // 4. Compare
    assert_tensors_are_close(&expected_q, &q_gpu, "Rotated Q", 1e-5).await;
    assert_tensors_are_close(&expected_k, &k_gpu, "Rotated K", 1e-5).await;

    println!("✅ GpuRoPE passed parity test!");
    Ok(())
}