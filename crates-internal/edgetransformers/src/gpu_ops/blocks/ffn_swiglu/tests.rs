use super::*; // Imports GpuSwiGLUFFN, GpuSwiGLUFFNWeights
use crate::feedforward::SwiGluFeedForward as CpuSwiGLUFFN; // Import your CPU implementation
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::TempStorage;
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::{Array, Array2, Ix2};
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
    cpu_tensor: &Array2<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix2>(gpu_tensor).await.unwrap();
    
    // Calculate the absolute differences for all elements
    let diffs = (cpu_tensor - &gpu_as_cpu).mapv(f32::abs);

    // Find the maximum difference and its index
    let mut max_diff = 0.0;
    let mut max_diff_index = 0;
    for (i, &d) in diffs.iter().enumerate() {
        if d > max_diff {
            max_diff = d;
            max_diff_index = i;
        }
    }

    // Check if any difference exceeds the tolerance
    if max_diff > tolerance {
        let mean_abs_diff = diffs.mean().unwrap_or(0.0);
        
        // Get the values at the point of maximum difference
        let cpu_val = cpu_tensor.iter().nth(max_diff_index).unwrap();
        let gpu_val = gpu_as_cpu.iter().nth(max_diff_index).unwrap();

        // Print a detailed report
        println!("\n--- TENSOR PARITY FAILURE: '{}' ---", label);
        println!("- Tolerance:           {}", tolerance);
        println!("- Mean Absolute Diff:  {}", mean_abs_diff);
        println!("- Max Absolute Diff:   {}", max_diff);
        println!("- Index of Max Diff:   {}", max_diff_index);
        println!("- CPU Value at Index:  {}", cpu_val);
        println!("- GPU Value at Index:  {}", gpu_val);
        println!("--------------------------------------------------\n");

        // Optional: Print a small slice of the tensors around the max difference
        // This is useful for seeing the local context of the error.
        let start = if max_diff_index > 8 { max_diff_index - 8 } else { 0 };
        let end = (max_diff_index + 8).min(cpu_tensor.len());
        println!("CPU Slice around diff: {:?}", &cpu_tensor.iter().skip(start).take(end - start).collect::<Vec<_>>());
        println!("GPU Slice around diff: {:?}", &gpu_as_cpu.iter().skip(start).take(end - start).collect::<Vec<_>>());
        println!("--------------------------------------------------\n");
        
        panic!("Tensor '{}' is not close enough to its GPU counterpart.", label);
    }
}

#[tokio::test]
async fn test_gpu_swiglu_ffn_parity() -> Result<()> {
    // --- 1. Arrange ---
    let context = Arc::new(WgpuContext::new().await);
    let (rows, hidden_size) = (128, 256); // (batch*seq_len), hidden_size
    let intermediate_size = 512;
    
    let gpu_swiglu = GpuSwiGLUFFN::new(&context)?;

    // Create identical random weights for CPU and GPU
    let gate_w_cpu = Array::random((hidden_size, intermediate_size), Uniform::new(-1.0, 1.0));
    let up_w_cpu = Array::random((hidden_size, intermediate_size), Uniform::new(-1.0, 1.0));
    let down_w_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-1.0, 1.0));

    let weights_gpu = GpuSwiGLUFFNWeights::new(
        GpuTensor::from_ndarray(&context, &gate_w_cpu)?,
        GpuTensor::from_ndarray(&context, &up_w_cpu)?,
        GpuTensor::from_ndarray(&context, &down_w_cpu)?,
    )?;
    
    // Create identical input (already in 2D format)
    let input_cpu = Array::random((rows, hidden_size), Uniform::new(-1.0, 1.0));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let output_gpu = GpuTensor::uninitialized(&context, input_cpu.shape().to_vec(), input_gpu.dtype(), "SwiGLU Output");

    // --- 2. CPU Ground Truth ---
    // We clone the weights because the CpuSwiGLUFFN constructor takes ownership.
    let cpu_swiglu = CpuSwiGLUFFN::new(gate_w_cpu.clone(), up_w_cpu.clone(), down_w_cpu.clone());
    
    // ✅ IMPROVEMENT: Call the `forward_2d` method directly. No more reshaping!
    let expected_cpu = cpu_swiglu.forward_2d(&input_cpu);
    
    // --- 3. GPU Execution ---
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut temp = TempStorage::new(context.clone());
    gpu_swiglu.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu, &mut temp);
    context.queue.submit(Some(encoder.finish()));

    // --- 4. Assert ---
    assert_tensors_are_close(&expected_cpu, &output_gpu, "SwiGLU FFN Output", 1e-2).await;

    println!("✅ GpuSwiGLUFFN passed parity test against the CPU implementation!");
    Ok(())
}