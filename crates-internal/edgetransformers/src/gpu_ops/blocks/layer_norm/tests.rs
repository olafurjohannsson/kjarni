use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Axis, Ix3};
use std::sync::Arc;
use crate::tests::common::{assert_tensors_are_close};

#[path = "../../../tests/common.rs"]
mod common;


#[tokio::test]
async fn test_gpu_layernorm_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_layernorm = GpuLayerNorm::new(&context, eps);
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.01 - 5.0);
    let gamma_cpu = Array1::from_shape_fn(h, |i| 1.0 + i as f32 * 0.005);
    let beta_cpu = Array1::from_shape_fn(h, |i| 0.0 + i as f32 * -0.002);
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuLayerNormWeights::new(
        GpuTensor::from_ndarray(&context, &gamma_cpu)?,
        GpuTensor::from_ndarray(&context, &beta_cpu)?,
    )?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "LN Output");
    let mean_cpu = input_cpu.mean_axis(Axis(2)).unwrap().insert_axis(Axis(2));
    let var_cpu = input_cpu.var_axis(Axis(2), 0.0).insert_axis(Axis(2));
    let inv_std_cpu = 1.0 / (var_cpu + eps).mapv(f32::sqrt);
    let normalized_cpu = (&input_cpu - &mean_cpu) * &inv_std_cpu;
    let expected_cpu = &normalized_cpu * &gamma_cpu + &beta_cpu;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_layernorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "LayerNorm Output", 1e-4).await;
    Ok(())
}
