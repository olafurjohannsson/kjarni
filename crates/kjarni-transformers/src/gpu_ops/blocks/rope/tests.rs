use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::GpuTensor;
use crate::rope::RoPE as CpuRoPE;
use crate::WgpuContext;
use anyhow::Result;
use common::assert_tensors_are_close_4d;
use ndarray::Array;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
#[path = "../../../tests/common.rs"]
mod common;

#[tokio::test]
async fn test_gpu_rope_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, h, s, d) = (2, 8, 32, 64); // Batch, Heads, SeqLen, HeadDim
    let max_seq = 128;
    let theta = 10000.0;
    let position_offset = 16;

    // --- 1. Setup CPU and GPU RoPE instances ---
    let cpu_rope = CpuRoPE::new(d, max_seq, theta);
    let gpu_rope = GpuRoPE::from_cpu_rope(&context, &cpu_rope)?;

    // --- 2. Create CPU and GPU input tensors ---
    let q_cpu = Array::random((b, h, s, d), Uniform::new(-1.0, 1.0));
    let k_cpu = Array::random((b, h, s, d), Uniform::new(-1.0, 1.0));

    // Use turbofish to specify the f32 type
    let q_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &q_cpu)?;
    let k_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &k_cpu)?;

    // --- 3. Calculate the expected result on the CPU ---
    let (expected_q, expected_k) = cpu_rope.apply_4d(&q_cpu, &k_cpu, position_offset);

    // --- 4. Run the GPU RoPE kernel (Out-of-Place) ---
    let mut encoder = context.device.create_command_encoder(&Default::default());

    // Create uninitialized output tensors for the results
    let q_rot_gpu = GpuTensor::uninitialized(
        &context,
        q_gpu.shape().to_vec(),
        crate::gpu_ops::DType::F32,
        "Rotated Q Output",
    );
    let k_rot_gpu = GpuTensor::uninitialized(
        &context,
        k_gpu.shape().to_vec(),
        crate::gpu_ops::DType::F32,
        "Rotated K Output",
    );

    // Call the new, out-of-place encode function
    gpu_rope.encode(&mut encoder, &q_gpu, &q_rot_gpu, position_offset);
    gpu_rope.encode(&mut encoder, &k_gpu, &k_rot_gpu, position_offset);

    context.queue.submit(Some(encoder.finish()));
    context.device.poll(wgpu::PollType::wait_indefinitely()); // Ensure GPU work is done

    // --- 5. Assert that the GPU output matches the CPU expectation ---
    assert_tensors_are_close_4d(&expected_q, &q_rot_gpu, "Rotated Q", 1e-5).await;
    assert_tensors_are_close_4d(&expected_k, &k_rot_gpu, "Rotated K", 1e-5).await;

    Ok(())
}
