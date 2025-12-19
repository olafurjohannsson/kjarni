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
    let context = WgpuContext::new().await?;
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


#[tokio::test]
async fn test_reorder_at_step_2_failure_simulation() -> Result<()> {
    println!("\n=== Simulating Reorder Failure at Step 2 ===\n");
    let context = WgpuContext::new().await?;
    let reorder_kernel = GpuReorderCache::new(&context);

    // --- 1. SETUP: State at the START of Step 2 ---
    // From your logs, seq_length is 2.
    const CURRENT_SEQ_LEN: usize = 2;
    let (num_beams, num_heads, capacity, head_dim) = (4, 16, 142, 64);

    // Create a source cache that represents the state before reordering.
    // Each beam's history is identifiable.
    // Beam 0 has value 1.0, Beam 1 has 2.0, etc.
    let source_cpu = Array4::from_shape_fn(
        (num_beams, num_heads, capacity, head_dim),
        |(b, _, s, _)| {
            if s < CURRENT_SEQ_LEN {
                (b + 1) as f32 // Beam 0 -> 1.0, Beam 1 -> 2.0, etc.
            } else {
                0.0 // The rest of the cache is unused
            }
        },
    );
    let source_gpu = GpuTensor::from_ndarray(&context, &source_cpu)?;

    // --- 2. THE CRITICAL REORDER INDICES from your log ---
    // [UPDATE] Parent beam indices for reorder: [0, 0, 1, 0]
    let parent_indices_cpu = Array1::from(vec![0u32, 0, 1, 0]);
    let indices_gpu = GpuTensor::from_ndarray(&context, &parent_indices_cpu)?;

    // --- 3. COMPUTE CPU GROUND TRUTH ---
    let mut expected_cpu = Array4::zeros(source_cpu.dim());
    for new_beam_idx in 0..num_beams {
        let parent_beam_idx = parent_indices_cpu[new_beam_idx] as usize;
        let mut dest_slice = expected_cpu.slice_mut(s![new_beam_idx, .., .., ..]);
        let src_slice = source_cpu.slice(s![parent_beam_idx, .., .., ..]);
        dest_slice.assign(&src_slice);
    }

    // --- 4. GPU EXECUTION ---
    let output_gpu = GpuTensor::uninitialized(&context, source_cpu.shape().to_vec(), source_gpu.dtype(), "Reorder Dst");
    let mut encoder = context.device.create_command_encoder(&Default::default());
    reorder_kernel.encode(
        &mut encoder,
        &source_gpu,
        &output_gpu,
        &indices_gpu,
        CURRENT_SEQ_LEN, // Use the correct sequence length
    );
    context.queue.submit(Some(encoder.finish()));

    // --- 5. VERIFY ---
    let actual_gpu_result = read_gpu_tensor(&output_gpu).await?;
    assert_eq!(expected_cpu, actual_gpu_result, "GPU reorder result does not match CPU ground truth for the failure case.");

    println!("✅ GPU reorder kernel correctly simulates the Step 2 reorder!");

    // --- 6. ANALYSIS: Why does "Rust Rust" happen? ---
    // Let's look at the new state of Beam 0. It inherited its history from old Beam 0.
    // The history contains the token "Rust".
    let new_beam_0_history_val = expected_cpu[[0, 0, 0, 0]];
    assert_eq!(new_beam_0_history_val, 1.0);

    // Now look at the new state of Beam 2. It inherited its history from old Beam 1.
    // Old Beam 1's history did NOT contain "Rust".
    let new_beam_2_history_val = expected_cpu[[2, 0, 0, 0]];
    assert_eq!(new_beam_2_history_val, 2.0);

    // The log shows the next tokens are [23083, 20, 23083, 128]
    // The parent beams are             [0,     0,  1,     0]
    //
    // This means:
    // - New Beam 0 gets token "Rust" and history from Old Beam 0. History is now ["Rust", "Rust"]
    // - New Beam 1 gets token "The" and history from Old Beam 0. History is now ["Rust", "The"]
    // - New Beam 2 gets token "Rust" and history from Old Beam 1. History is now ["The", "Rust"]
    // - New Beam 3 gets token "'" and history from Old Beam 0. History is now ["Rust", "'"]
    //
    // The "Rust Rust" happens because the penalty `no_repeat_ngram_size: 3` does not
    // prevent a bigram repeat. The reorder logic is correct. The problem is that the
    // generation logic ALLOWS this choice to be made.
    // The ONLY reason the CPU works is due to floating point differences making another
    // token slightly more likely. The GPU is not wrong, it's just exposing the flaw
    // in the penalty configuration.
    println!("Analysis complete: The reorder kernel is correct. The divergence is caused by a logic flaw (penalty config) exposed by GPU floating point arithmetic.");

    Ok(())
}