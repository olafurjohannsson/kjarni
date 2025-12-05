use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, primitives::layout::unreshape::GpuUnreshape};
use anyhow::Result;
use ndarray::{Array, Array3, Array4};
use std::sync::Arc;

// Helper to get a test context.
async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

// Helper to read a 3D GPU tensor back to the CPU for comparison.
async fn read_gpu_tensor_3d(tensor: &GpuTensor) -> Result<Array3<f32>> {
    let shape = tensor.shape();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(
        (shape[0], shape[1], shape[2]),
        data_slice.to_vec(),
    )?)
}

#[tokio::test]
async fn test_unreshape() -> Result<()> {
    println!("\n--- Testing GpuUnreshape (Merge Heads) ---");
    let context = get_test_context().await;
    let gpu_unreshape = GpuUnreshape::new(&context);

    let (b, s, h, d) = (2, 7, 4, 5); // Batch, SeqLen, NumHeads, HeadDim
    let hidden_size = h * d;
    let cpu_input = Array4::from_shape_fn((b, h, s, d), |(i, j, k, l)| {
        (i * 1000 + j * 100 + k * 10 + l) as f32
    });
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let cpu_ground_truth = cpu_input
        .to_owned()
        .permuted_axes([0, 2, 1, 3]) // [B, S, H, D]
        .as_standard_layout()
        .to_owned()
        .into_shape_with_order((b, s, hidden_size))?; // [B, S, H*D]
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![b, s, hidden_size],
        crate::gpu_ops::DType::F32,
        "Unreshape Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_unreshape.encode(&mut encoder, &gpu_input, &gpu_output);
    context.queue.submit(Some(encoder.finish()));
    let gpu_result = read_gpu_tensor_3d(&gpu_output).await?;
    assert_eq!(cpu_ground_truth.as_slice(), gpu_result.as_slice());
    Ok(())
}
