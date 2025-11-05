#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::feedforward::FeedForward as CpuFeedForward;
use crate::gpu_ops::{DType, GpuTensor};
use anyhow::Result;
use common::read_gpu_tensor_to_vec;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays not close. Max diff: {}, Tolerance: {} CPU: {:?}\n GPU: {:?}",
        max_diff,
        tolerance,
        a,
        b
    );
}
async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

#[tokio::test]
async fn test_fc1_kernel_parity() -> Result<()> {
    println!("\n--- Isolating and Testing FC1 Kernel ---");
    let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    // 1. Create data in the standard [in, out] layout
    let cpu_input = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let cpu_fc1_w = Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1));
    let cpu_fc1_b = Array::random(intermediate_size, Uniform::new(-0.1, 0.1));

    // --- CPU Ground Truth ---
    // The CpuFeedForward constructor is "dumb" and expects [in, out] weights.
    let cpu_ffn_partial = CpuFeedForward::new(
        cpu_fc1_w.clone(),
        cpu_fc1_b.clone(),
        // Dummy weights must have logically consistent shapes for the constructor's assertions.
        Array2::zeros((intermediate_size, hidden_size)),
        Array1::zeros(hidden_size),
        crate::activations::Activation::Gelu,
    );
    // Perform the exact operation of the fused kernel on the CPU.
    let mut cpu_result = cpu_ffn_partial.fc1(&cpu_input)?;
    cpu_ffn_partial.apply_activation(&mut cpu_result);

    // --- GPU Execution ---
    // The GpuFeedForward constructor ALSO expects [in, out] weights. NO TRANSPOSE NEEDED.
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_fc1_w = GpuTensor::from_ndarray(&context, &cpu_fc1_w)?; // Pass [in, out] directly
    let gpu_fc1_b = GpuTensor::from_ndarray(&context, &cpu_fc1_b)?;
    
    // Dummy weights for the constructor must also have the correct [in, out] shapes.
    let dummy_fc2_w = GpuTensor::uninitialized(&context, vec![intermediate_size, hidden_size], DType::F32, "dummy_w2");
    let dummy_fc2_b = GpuTensor::uninitialized(&context, vec![hidden_size], DType::F32, "dummy_b2");

    let gpu_ffn_partial = GpuFeedForward::new(
        &context,
        gpu_fc1_w,
        gpu_fc1_b,
        dummy_fc2_w,
        dummy_fc2_b,
        crate::activations::Activation::Gelu,
    )?;

    let gpu_intermediate = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, intermediate_size],
        DType::F32,
        "FC1 Output",
    );

    // a. Encode the GPU command
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn_partial.run_fc1(&mut encoder, &gpu_input, &gpu_intermediate);
    context.queue.submit(std::iter::once(encoder.finish()));

    // b. Read back the result
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_intermediate).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;

    // --- Assert ---
    println!("Verifying FC1 Fused Kernel parity...");
    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    println!("✅ FC1 Kernel Passed Parity Check!");

    Ok(())
}

#[tokio::test]
async fn test_fc2_kernel_parity() -> Result<()> {
   let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    // 1. Create data. Input to FC2 is intermediate size.
    let cpu_input = Array::random((batch_size, seq_len, intermediate_size), Uniform::new(-1.0, 1.0));
    // FC2 weight is [in, out] -> [intermediate_size, hidden_size]
    let raw_fc2_w = Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1));
    let cpu_fc2_b = Array::random(hidden_size, Uniform::new(-0.1, 0.1));

    // --- CPU Ground Truth ---
    // The CpuFeedForward constructor is "dumb" and expects [in, out] weights. We provide them.
    let cpu_ffn = CpuFeedForward::new(
        Array2::zeros((hidden_size, intermediate_size)), // Correct dummy shape [in, out]
        Array1::zeros(intermediate_size),                // Correct dummy shape
        raw_fc2_w.clone(),                               // The real [in, out] weight
        cpu_fc2_b.clone(),
        crate::activations::Activation::Gelu,
    );
    let cpu_result = cpu_ffn.fc2(&cpu_input)?;

    // --- GPU Execution ---
    // The GpuFeedForward constructor is ALSO "dumb" and expects [in, out] weights.
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_fc2_w = GpuTensor::from_ndarray(&context, &raw_fc2_w)?; // NO .t()
    let gpu_fc2_b = GpuTensor::from_ndarray(&context, &cpu_fc2_b)?;

    let gpu_ffn_partial = GpuFeedForward::new(
        &context,
        // Correct dummy shape [in, out] for FC1
        GpuTensor::uninitialized(&context, vec![hidden_size, intermediate_size], DType::F32, "dummy_w"),
        GpuTensor::uninitialized(&context, vec![intermediate_size], DType::F32, "dummy_b"),
        gpu_fc2_w, // The real [in, out] weight
        gpu_fc2_b,
        crate::activations::Activation::Gelu,
    )?;

    let gpu_output = GpuTensor::uninitialized(&context, vec![batch_size, seq_len, hidden_size], DType::F32, "FC2 Output");

    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn_partial.run_fc2(&mut encoder, &gpu_input, &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    
    // --- Assert ---
    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    println!("✅ FC2 Kernel Passed Parity Check!");

    Ok(())
}

async fn run_ffn_test(transpose_weights: bool) -> Result<()> {
    let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    // 1. Create RAW weights in the format they would be on disk.
    let (raw_fc1_w, raw_fc2_w) = if transpose_weights {
        // BERT-style: raw weights are [in, out]
        (
            Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1)),
            Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1)),
        )
    } else {
        // GPT-style: raw weights are [out, in]
        (
            Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1)),
            Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1)),
        )
    };
    let cpu_fc1_b = Array::random(intermediate_size, Uniform::new(-0.1, 0.1));
    let cpu_fc2_b = Array::random(hidden_size, Uniform::new(-0.1, 0.1));

    // --- LOADER LOGIC: Prepare weights into the standard [in, out] format ---
    // This logic is now identical for both CPU and GPU paths.
    let fc1_w_prepared = if transpose_weights { raw_fc1_w.clone() } else { raw_fc1_w.t().to_owned() };
    let fc2_w_prepared = if transpose_weights { raw_fc2_w.clone() } else { raw_fc2_w.t().to_owned() };
    
    // --- CPU PATH ---
    // The "dumb" CpuFeedForward constructor receives the prepared [in, out] weights.
    let cpu_ffn = CpuFeedForward::new(
        fc1_w_prepared.clone(),
        cpu_fc1_b.clone(),
        fc2_w_prepared.clone(),
        cpu_fc2_b.clone(),
        crate::activations::Activation::Gelu,
    );

    // --- GPU PATH ---
    // The "dumb" GpuFeedForward constructor ALSO receives the prepared [in, out] weights.
    let gpu_fc1_w = GpuTensor::from_ndarray(&context, &fc1_w_prepared)?;
    let gpu_fc2_w = GpuTensor::from_ndarray(&context, &fc2_w_prepared)?;
    let gpu_fc1_b = GpuTensor::from_ndarray(&context, &cpu_fc1_b)?;
    let gpu_fc2_b = GpuTensor::from_ndarray(&context, &cpu_fc2_b)?;
    
    let gpu_ffn = GpuFeedForward::new(
        &context,
        gpu_fc1_w,
        gpu_fc1_b,
        gpu_fc2_w,
        gpu_fc2_b,
        crate::activations::Activation::Gelu,
    )?;

    // 3. Create common input data and buffers
    let cpu_input = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_intermediate = GpuTensor::uninitialized(&context, vec![batch_size, seq_len, intermediate_size], DType::F32, "FFN Intermediate");
    let gpu_output = GpuTensor::uninitialized(&context, vec![batch_size, seq_len, hidden_size], DType::F32, "FFN Output");

    // 4. Execute and Assert
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn.forward(&mut encoder, &gpu_input, &gpu_intermediate, &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    let cpu_result = cpu_ffn.forward(&cpu_input)?;

    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    Ok(())
}

#[tokio::test]
async fn test_ffn_parity_with_transpose_true() -> Result<()> {
    println!("\n--- Testing FFN Parity (transpose_weights = true) ---");
    run_ffn_test(true).await?;
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_ffn_parity_with_transpose_false() -> Result<()> {
    println!("\n--- Testing FFN Parity (transpose_weights = false) ---");
    run_ffn_test(false).await?;
    println!("✅ Passed!");
    Ok(())
}
