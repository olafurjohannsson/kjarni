use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, primitives::layout::reshape::GpuReshape};
use anyhow::Result;
use ndarray::{Array, Array3, Array4, Axis, Ix4};
use std::sync::Arc;

// Helper to get a test context.
async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await.unwrap())
}

// Helper to read a 4D GPU tensor back to the CPU for comparison.
async fn read_gpu_tensor_4d(tensor: &GpuTensor) -> Result<Array4<f32>> {
    let shape = tensor.shape();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(
        (shape[0], shape[1], shape[2], shape[3]),
        data_slice.to_vec(),
    )?)
}

#[tokio::test]
async fn test_reshape_q_v_path() -> Result<()> {
    println!("\n--- Testing GpuReshape (Q/V Path) ---");
    let context = get_test_context().await;
    let gpu_reshape = GpuReshape::new(&context);

    // Use non-square dimensions to catch indexing errors
    let (b, s, h, d) = (2, 7, 4, 5); // Batch, SeqLen, NumHeads, HeadDim
    let hidden_size = h * d;

    // 1. ARRANGE: Create CPU and GPU input tensors
    let cpu_input = Array3::from_shape_fn((b, s, hidden_size), |(i, j, k)| {
        (i * 100 + j * 10 + k) as f32
    });
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;

    // 2. ARRANGE: Compute the CPU ground truth
    let cpu_ground_truth = cpu_input
        .to_owned()
        .into_shape((b, s, h, d))? // [B, S, H, D]
        .permuted_axes([0, 2, 1, 3]) // [B, H, S, D]
        .as_standard_layout()
        .to_owned();

    // 3. ACT: Run the GPU kernel
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![b, h, s, d],
        crate::gpu_ops::DType::F32,
        "Reshape Output Q/V",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_reshape.encode(&mut encoder, &gpu_input, &gpu_output, false); // transpose_k = false
    context.queue.submit(Some(encoder.finish()));

    // 4. ASSERT: Compare the results
    let gpu_result = read_gpu_tensor_4d(&gpu_output).await?;

    assert_eq!(cpu_ground_truth.as_slice(), gpu_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_reshape_k_transpose_path() -> Result<()> {
    println!("\n--- Testing GpuReshape (K^T Path) ---");
    let context = get_test_context().await;
    let gpu_reshape = GpuReshape::new(&context);

    let (b, s, h, d) = (2, 7, 4, 5);
    let hidden_size = h * d;

    // 1. ARRANGE: Create CPU and GPU input tensors
    let cpu_input = Array3::from_shape_fn((b, s, hidden_size), |(i, j, k)| {
        (i * 100 + j * 10 + k) as f32
    });
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;

    // 2. ARRANGE: Compute the CPU ground truth for the transposed case
    let cpu_ground_truth = cpu_input
        .to_owned()
        .into_shape((b, s, h, d))? // [B, S, H, D]
        .permuted_axes([0, 2, 3, 1]) // [B, H, D, S]
        .as_standard_layout()
        .to_owned();

    // 3. ACT: Run the GPU kernel with transpose_k = true
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![b, h, d, s], // Note the output shape is different
        crate::gpu_ops::DType::F32,
        "Reshape Output K^T",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_reshape.encode(&mut encoder, &gpu_input, &gpu_output, true); // transpose_k = true
    context.queue.submit(Some(encoder.finish()));

    // 4. ASSERT: Compare the results
    let gpu_result = read_gpu_tensor_4d(&gpu_output).await?;

    assert_eq!(cpu_ground_truth.as_slice(), gpu_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}
