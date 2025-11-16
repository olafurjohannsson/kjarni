use super::*;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::utils::{assert_vecs_are_close, read_buffer_2d};
use anyhow::Result;
use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::gpu_ops::DType;
use ndarray::{Array4};
#[path = "../../../tests/common.rs"]
mod common;

use common::{read_gpu_tensor_to_vec, assert_tensors_are_close_4d};


// Helper to compare two ndarray arrays for near-equality.
fn assert_all_close(a: &Array2<f32>, b: &Array2<f32>, tolerance: f32) {
    let diff = (a - b).mapv(f32::abs);
    let max_diff = diff.iter().fold(0.0f32, |max, &v| v.max(max));
    assert!(
        max_diff < tolerance,
        "Arrays are not close. Max difference: {}",
        max_diff
    );
}
#[tokio::test]
async fn test_repeat_kv() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
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
    let output_gpu = GpuTensor::zeros(&context, vec![1, 4, 2, 2], crate::gpu_ops::DType::F32, "output")?;

    // --- Execute Kernel ---
    let mut encoder = context.device.create_command_encoder(&Default::default());
    repeat_kernel.encode(&mut encoder, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    // let result_cpu: Array<f32, _> = output_gpu.to_ndarray_4d().await?;

    // --- Verification ---
    let expected_vec: Vec<f32> = vec![
        1., 2., 3., 4., // Q Head 0 (from KV Head 0)
        1., 2., 3., 4., // Q Head 1 (from KV Head 0)
        5., 6., 7., 8., // Q Head 2 (from KV Head 1)
        5., 6., 7., 8., // Q Head 3 (from KV Head 1)
    ];
    let expected_cpu = Array::from_shape_vec((1, 4, 2, 2), expected_vec)?;

    assert_tensors_are_close_4d(&expected_cpu, &output_gpu, "Repeat KV", 1e-6);

    Ok(())
}