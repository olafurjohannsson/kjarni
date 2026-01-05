#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::gpu_ops::{GpuTensor};
use anyhow::Result;
use common::{assert_arrays_are_close_2d, get_test_context, assert_all_close, read_gpu_tensor_to_vec};
use ndarray::{Array, Array3, arr2};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;



#[tokio::test]
async fn test_gpu_scale_out_of_place() -> Result<()> {
    let context = get_test_context().await;
    let scale_kernel = GpuScale::new(&context);
    let cpu_input = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let scale_factor = 2.5;
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let output_shape: Vec<usize> = cpu_input.shape().iter().map(|d| *d).collect();
    let gpu_output = GpuTensor::zeros(&context, output_shape, crate::gpu_ops::DType::F32, "f32")?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    scale_kernel.encode_out_of_place(&mut encoder, &gpu_input, &gpu_output, scale_factor);
    context.queue.submit(std::iter::once(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    let cpu_expected = &cpu_input * scale_factor;
    let gpu_output_cpu = gpu_output.to_ndarray_2d().await?;
    assert_arrays_are_close_2d(&gpu_output_cpu, &cpu_expected, 1e-6);
    Ok(())
}
#[tokio::test]
async fn test_scale_parity() -> Result<()> {
    let context = get_test_context().await;
    let scale_kernel = GpuScale::new(&context);
    let scale_factor = 1.0 / (64.0f32).sqrt(); // A realistic attention scaling factor
    let cpu_input = Array::random((12, 128, 128), Uniform::new(-10.0, 10.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    scale_kernel.encode_in_place(&mut encoder, &gpu_input, scale_factor);
    context.queue.submit(std::iter::once(encoder.finish()));
    let cpu_result = &cpu_input * scale_factor;
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_input).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    assert_all_close(&gpu_result, &cpu_result, 1e-6);
    Ok(())
}
