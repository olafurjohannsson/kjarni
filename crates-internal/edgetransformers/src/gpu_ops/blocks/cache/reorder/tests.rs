use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::cache::reorder::GpuReorderCache;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::{s, Array, Array1, Array4, Axis, Ix4};
use std::sync::Arc;

// Helper to read a GpuTensor back to the CPU for comparison.
async fn read_gpu_tensor(tensor: &GpuTensor) -> Result<Array4<f32>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), data_slice.to_vec())?)
}

#[tokio::test]
async fn test_gpu_reorder_cache_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
    let reorder_kernel = GpuReorderCache::new(&context);

    // --- 1. SETUP ---
    // Let's simulate a cache with 4 beams (batch_size = 4).
    let (num_beams, num_heads, seq_len, head_dim) = (4, 2, 5, 8);

    // Create a source tensor where each beam has a unique, identifiable value.
    let source_cpu = Array4::from_shape_fn((num_beams, num_heads, seq_len, head_dim), |(b, _, _, _)| {
        (b as f32 + 1.0) * 100.0 // Beam 0 -> 100.0, Beam 1 -> 200.0, etc.
    });
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;
    
    // --- 2. DEFINE THE REORDERING ---
    // This is the `parent_indices` from a beam search step.
    // New beam 0 comes from old beam 2.
    // New beam 1 comes from old beam 0.
    // New beam 2 comes from old beam 2.
    // New beam 3 comes from old beam 1.
    let parent_indices_cpu = Array1::from(vec![2u32, 0, 2, 1]);
    let indices_gpu = GpuTensor::from_ndarray(&context, &parent_indices_cpu)?;

    // --- 3. CPU GROUND TRUTH ---
    // Manually construct the expected output tensor.
    let mut expected_cpu = Array4::zeros(source_cpu.dim());
    for i in 0..num_beams {
        let parent_idx = parent_indices_cpu[i] as usize;
        let mut dest_slice = expected_cpu.slice_mut(s![i, .., .., ..]);
        let src_slice = source_cpu.slice(s![parent_idx, .., .., ..]);
        dest_slice.assign(&src_slice);
    }

    // --- 4. GPU EXECUTION ---
    let output_gpu = GpuTensor::uninitialized(&context, source_cpu.shape().to_vec(), source_gpu.dtype(), "Reorder Dst");

    let mut encoder = context.device.create_command_encoder(&Default::default());
    reorder_kernel.encode(&mut encoder, &source_gpu, &output_gpu, &indices_gpu, seq_len);
    context.queue.submit(Some(encoder.finish()));

    // --- 5. COMPARE RESULTS ---
    let actual_gpu_result = read_gpu_tensor(&output_gpu).await?;
    
    // For a data copy operation, the results should be exact.
    assert_eq!(expected_cpu, actual_gpu_result, "GPU reorder result does not match CPU ground truth.");

    println!("✅ GpuReorderCache passed parity test!");
    Ok(())
}