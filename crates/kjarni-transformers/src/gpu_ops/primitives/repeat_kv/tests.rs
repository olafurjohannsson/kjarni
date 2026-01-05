use super::*;
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array};
#[path = "../../../tests/common.rs"]
mod common;

use common::assert_tensors_are_close_4d;

#[tokio::test]
async fn test_repeat_kv() -> Result<()> {
    let context = WgpuContext::new().await?;
    let repeat_kernel = GpuRepeatKV::new(&context);

    // --- Setup Data ---
    // Input shape: [batch=1, num_kv_heads=2, seq=2, dim=2]
    let input_vec: Vec<f32> = vec![
        1., 2., 3., 4., // KV Head 0
        5., 6., 7., 8., // KV Head 1
    ];
    let input_cpu = Array::from_shape_vec((1, 2, 2, 2), input_vec)?;
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    // Output shape: [batch=1, num_q_heads=4, seq=2, dim=2]
    let output_gpu = GpuTensor::zeros(
        &context,
        vec![1, 4, 2, 2],
        crate::gpu_ops::DType::F32,
        "output",
    )?;

    // --- Execute Kernel ---
    let mut encoder = context.device.create_command_encoder(&Default::default());
    repeat_kernel.encode(&mut encoder, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    let expected_vec: Vec<f32> = vec![
        1., 2., 3., 4., // Q Head 0 (from KV Head 0)
        1., 2., 3., 4., // Q Head 1 (from KV Head 0)
        5., 6., 7., 8., // Q Head 2 (from KV Head 1)
        5., 6., 7., 8., // Q Head 3 (from KV Head 1)
    ];
    let expected_cpu = Array::from_shape_vec((1, 4, 2, 2), expected_vec)?;

    assert_tensors_are_close_4d(&expected_cpu, &output_gpu, "Repeat KV", 1e-6).await;

    Ok(())
}
