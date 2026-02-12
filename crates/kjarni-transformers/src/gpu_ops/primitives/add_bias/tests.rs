
use anyhow::Result;
use ndarray::{Array, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

use super::*;
use crate::gpu::{DType, GpuTensor};

#[path = "../../../tests/common.rs"]
mod common;
use common::{read_gpu_tensor_to_vec, assert_arrays_are_close_2d, get_test_context};


#[tokio::test]
async fn test_add_bias_parity() -> Result<()> {
    let context = get_test_context().await;
    let add_bias_kernel = GpuAddBias::new(&context);

    let rows = 128;
    let cols = 768;

    let cpu_input = Array::random((rows, cols), Uniform::new(-10.0, 10.0));
    let cpu_bias = Array::random(cols, Uniform::new(-1.0, 1.0));

    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_bias = GpuTensor::from_ndarray(&context, &cpu_bias)?;
    let gpu_output =
        GpuTensor::uninitialized(&context, vec![rows, cols], DType::F32, "add_bias_output");

    let mut encoder = context.device.create_command_encoder(&Default::default());
    add_bias_kernel.encode(&mut encoder, &[&gpu_input, &gpu_bias], &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_result_vec, result_shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array2::from_shape_vec((result_shape[0], result_shape[1]), gpu_result_vec)?;

    let cpu_result = &cpu_input + &cpu_bias;

    assert_arrays_are_close_2d(&gpu_result, &cpu_result, 1e-6);

    Ok(())
}