use crate::WgpuContext;
use crate::gpu_ops::GpuTensor;
use crate::gpu_ops::primitives::layout::concatenate::GpuConcatenate;
use anyhow::Result;
use ndarray::{Array, Array4, Axis};

// You will need a `read_gpu_tensor` helper in this test module.
// Helper to read a GPU tensor back to a generic ndarray for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
        .into_dimensionality::<D>()
        .unwrap())
}

#[tokio::test]
async fn test_gpu_concatenate_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let concat_kernel = GpuConcatenate::new(&context);
    let a_shape = (1, 12, 10, 64);
    let b_shape = (1, 12, 1, 64);
    let concat_axis = 2; // Sequence dimension
    let a_cpu = Array4::from_shape_fn(a_shape, |(i, j, k, l)| (i + j + k + l) as f32);
    let b_cpu = Array4::from_shape_fn(b_shape, |(i, j, k, l)| (i + j + k + l) as f32 * -1.0);
    let a_gpu = GpuTensor::from_ndarray(&context, &a_cpu)?;
    let b_gpu = GpuTensor::from_ndarray(&context, &b_cpu)?;
    let expected_cpu = ndarray::concatenate(Axis(concat_axis), &[a_cpu.view(), b_cpu.view()])?;
    let output_shape = expected_cpu.shape().to_vec();
    let output_gpu =
        GpuTensor::uninitialized(&context, output_shape, a_gpu.dtype(), "Concat Output");
    let mut encoder = context.device.create_command_encoder(&Default::default());
    concat_kernel.encode(&mut encoder, &[&a_gpu, &b_gpu], &output_gpu, concat_axis);
    context.queue.submit(Some(encoder.finish()));
    let actual_gpu_result: Array4<f32> = read_gpu_tensor(&output_gpu).await?;
    assert_eq!(
        expected_cpu, actual_gpu_result,
        "GPU concatenate result does not match CPU ground truth."
    );
    Ok(())
}
