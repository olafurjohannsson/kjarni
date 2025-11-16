use crate::gpu_context::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::primitives::layer_norm::{GpuLayerNorm, GpuLayerNormWeights};
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Axis, Ix3};
use std::sync::Arc;

async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
        .into_dimensionality::<D>()
        .unwrap())
}

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
async fn test_gpu_layernorm_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
    let (b, s, h) = (4, 64, 256); // Batch, SeqLen, HiddenSize
    let eps = 1e-5;
    let gpu_layernorm = GpuLayerNorm::new(&context, eps);
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.01 - 5.0);
    let gamma_cpu = Array1::from_shape_fn(h, |i| 1.0 + i as f32 * 0.005);
    let beta_cpu = Array1::from_shape_fn(h, |i| 0.0 + i as f32 * -0.002);

    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let weights_gpu = GpuLayerNormWeights {
        gamma: GpuTensor::from_ndarray(&context, &gamma_cpu)?,
        beta: GpuTensor::from_ndarray(&context, &beta_cpu)?,
    };
    let output_gpu =
        GpuTensor::uninitialized(&context, vec![b, s, h], input_gpu.dtype(), "LN Output");
    let mean_cpu = input_cpu.mean_axis(Axis(2)).unwrap().insert_axis(Axis(2));
    let var_cpu = input_cpu.var_axis(Axis(2), 0.0).insert_axis(Axis(2));
    let inv_std_cpu = 1.0 / (var_cpu + eps).mapv(f32::sqrt);
    let normalized_cpu = (&input_cpu - &mean_cpu) * &inv_std_cpu;
    let expected_cpu = &normalized_cpu * &gamma_cpu + &beta_cpu;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_layernorm.encode(&mut encoder, &input_gpu, &weights_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    assert_tensors_are_close(&expected_cpu, &output_gpu, "LayerNorm Output", 1e-4).await;

    Ok(())
}
