use crate::gpu::{GpuTensor, DType};
use crate::gpu_ops::primitives::layout::reshape::GpuReshape;
use anyhow::Result;
use ndarray::{ Array3};
use common::{read_gpu_tensor_4d, get_test_context};

#[path = "../../../../tests/common.rs"]
mod common;


#[tokio::test]
async fn test_reshape_q_v_path() -> Result<()> {
    println!("\n--- Testing GpuReshape (Q/V Path) ---");
    let context = get_test_context().await;
    let gpu_reshape = GpuReshape::new(&context);
    let (b, s, h, d) = (2, 7, 4, 5); // Batch, SeqLen, NumHeads, HeadDim
    let hidden_size = h * d;
    let cpu_input = Array3::from_shape_fn((b, s, hidden_size), |(i, j, k)| {
        (i * 100 + j * 10 + k) as f32
    });
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let cpu_ground_truth = cpu_input
        .to_owned()
        .into_shape_with_order((b, s, h, d))? // [B, S, H, D]
        .permuted_axes([0, 2, 1, 3]) // [B, H, S, D]
        .as_standard_layout()
        .to_owned();
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![b, h, s, d],
        DType::F32,
        "Reshape Output Q/V",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_reshape.encode(&mut encoder, &gpu_input, &gpu_output, false); // transpose_k = false
    context.queue.submit(Some(encoder.finish()));
    let gpu_result = read_gpu_tensor_4d(&gpu_output).await?;
    assert_eq!(cpu_ground_truth.as_slice(), gpu_result.as_slice());
    Ok(())
}

#[tokio::test]
async fn test_reshape_k_transpose_path() -> Result<()> {
    println!("\n--- Testing GpuReshape (K^T Path) ---");
    let context = get_test_context().await;
    let gpu_reshape = GpuReshape::new(&context);
    let (b, s, h, d) = (2, 7, 4, 5);
    let hidden_size = h * d;
    let cpu_input = Array3::from_shape_fn((b, s, hidden_size), |(i, j, k)| {
        (i * 100 + j * 10 + k) as f32
    });
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let cpu_ground_truth = cpu_input
        .to_owned()
        .into_shape_with_order((b, s, h, d))? // [B, S, H, D]
        .permuted_axes([0, 2, 3, 1]) // [B, H, D, S]
        .as_standard_layout()
        .to_owned();
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![b, h, d, s], 
        DType::F32,
        "Reshape Output K^T",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_reshape.encode(&mut encoder, &gpu_input, &gpu_output, true); // transpose_k = true
    context.queue.submit(Some(encoder.finish()));
    let gpu_result = read_gpu_tensor_4d(&gpu_output).await?;
    assert_eq!(cpu_ground_truth.as_slice(), gpu_result.as_slice());
    Ok(())
}
