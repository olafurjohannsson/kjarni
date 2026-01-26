use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::blocks::rms_norm::{GpuRMSNorm, GpuRMSNormWeights};
use crate::cpu::normalization::rms_norm::RMSNorm;
use crate::tests::common::assert_tensors_are_close;
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Axis};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;

#[path = "../../../tests/common.rs"]
mod common;

#[tokio::test]
async fn test_gpu_rmsnorm_parity_with_cpu_impl() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_rmsnorm = GpuRMSNorm::new(&context, eps);
    let input_cpu = Array::random((b, s, h), Uniform::new(-5.0, 5.0));
    let gamma_cpu = Array::random(h, Uniform::new(0.5, 1.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuRMSNormWeights::new(GpuTensor::from_ndarray(&context, &gamma_cpu)?)?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "RMSNorm Output");
    let cpu_rmsnorm = RMSNorm::new(gamma_cpu.clone(), eps);
    let expected_cpu = cpu_rmsnorm.forward_3d(&input_cpu);
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_rmsnorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "RMSNorm Output", 1e-4).await;
    Ok(())
}
#[tokio::test]
async fn test_gpu_rmsnorm_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_rmsnorm = GpuRMSNorm::new(&context, eps);
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.01 - 5.0);
    let gamma_cpu = Array1::from_shape_fn(h, |i| 1.0 + i as f32 * 0.005);

    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuRMSNormWeights::new(GpuTensor::from_ndarray(&context, &gamma_cpu)?)?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "RMSNorm Output");
    let variance = input_cpu
        .mapv(|x| x.powi(2))
        .mean_axis(Axis(2))
        .unwrap()
        .insert_axis(Axis(2));
    let inv_rms = 1.0 / (variance + eps).mapv(f32::sqrt);
    let normalized_cpu = &input_cpu * &inv_rms;
    let expected_cpu = &normalized_cpu * &gamma_cpu;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_rmsnorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "RMSNorm Output", 1e-4).await;
    Ok(())
}
