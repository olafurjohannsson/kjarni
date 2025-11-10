use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::rms_norm::{GpuRMSNorm, GpuRMSNormWeights};
use crate::gpu_ops::GpuTensor;
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Axis, Ix3};
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
    cpu_tensor: &Array3<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix3>(gpu_tensor).await.unwrap();
    let close = cpu_tensor
        .iter()
        .zip(gpu_as_cpu.iter())
        .all(|(a, b)| (a - b).abs() < tolerance);

    if !close {
        println!("Mismatch in tensor '{}'", label);
        println!("CPU tensor: \n{:?}", cpu_tensor);
        println!("GPU tensor: \n{:?}", gpu_as_cpu);
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

#[tokio::test]
async fn test_gpu_rmsnorm_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await);
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_rmsnorm = GpuRMSNorm::new(&context, eps);

    // 1. Setup: Create input and weight tensors on CPU and GPU
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.01 - 5.0);
    let gamma_cpu = Array1::from_shape_fn(h, |i| 1.0 + i as f32 * 0.005);

    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu =
        GpuRMSNormWeights::new(GpuTensor::from_ndarray(&context, &gamma_cpu)?)?;
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "RMSNorm Output");

    // 2. CPU Ground Truth: Manually implement RMSNorm with ndarray
    let variance = input_cpu.mapv(|x| x.powi(2)).mean_axis(Axis(2)).unwrap().insert_axis(Axis(2));
    let inv_rms = 1.0 / (variance + eps).mapv(f32::sqrt);
    let normalized_cpu = &input_cpu * &inv_rms;
    let expected_cpu = &normalized_cpu * &gamma_cpu;

    // 3. GPU Execution
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_rmsnorm.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    // 4. Compare
    assert_tensors_are_close(&expected_cpu, &output_gpu, "RMSNorm Output", 1e-4).await;

    println!("✅ GpuRMSNorm passed parity test!");
    Ok(())
}