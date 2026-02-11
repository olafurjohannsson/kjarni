#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::feedforward::LegacyFeedForward as CpuFeedForward;
use crate::gpu::{DType, GpuTensor, GpuTensorPool};
use anyhow::Result;
use common::read_gpu_tensor_to_vec;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use crate::tests::common::{assert_all_close, assert_tensors_are_close, get_test_context};

#[tokio::test]
async fn test_fc1_kernel_parity() -> Result<()> {
    let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;
    let cpu_input = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let cpu_fc1_w = Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1));
    let cpu_fc1_b = Array::random(intermediate_size, Uniform::new(-0.1, 0.1));
    let cpu_ffn_partial = CpuFeedForward::new(
        cpu_fc1_w.clone(),
        cpu_fc1_b.clone(),
        Array2::zeros((intermediate_size, hidden_size)),
        Array1::zeros(hidden_size),
        crate::activations::Activation::Gelu,
    );
    let mut cpu_result = cpu_ffn_partial.fc1(&cpu_input)?;
    cpu_ffn_partial.apply_activation(&mut cpu_result);

    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;

    let dummy_fc2_w_cpu = Array2::<f32>::zeros((intermediate_size, hidden_size));
    let dummy_fc2_b_cpu = Array1::<f32>::zeros(hidden_size);

    // 2. Use the single "smart" constructor to create the weights struct.
    //    This function correctly transposes the real `cpu_fc1_w` and the dummy `dummy_fc2_w_cpu`
    //    before creating the final GpuTensors.
    let weights = GpuFeedForwardWeights::from_ndarrays(
        &context,
        &cpu_fc1_w,
        &cpu_fc1_b,
        &dummy_fc2_w_cpu,
        &dummy_fc2_b_cpu,
    )?;

    let gpu_ffn_partial = GpuFeedForwardStd::new(&context, crate::activations::Activation::Gelu)?;
    let gpu_intermediate = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, intermediate_size],
        DType::F32,
        "FC1 Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn_partial.run_fc1(&mut encoder, &weights, &gpu_input, &gpu_intermediate);
    context.queue.submit(std::iter::once(encoder.finish()));
    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_intermediate).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    Ok(())
}

#[tokio::test]
async fn test_fc2_kernel_parity() -> Result<()> {
    let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    let cpu_input = Array::random(
        (batch_size, seq_len, intermediate_size),
        Uniform::new(-1.0, 1.0),
    );
    let raw_fc2_w = Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1));
    let cpu_fc2_b = Array::random(hidden_size, Uniform::new(-0.1, 0.1));

    let cpu_ffn = CpuFeedForward::new(
        Array2::zeros((hidden_size, intermediate_size)), // Correct dummy shape [in, out]
        Array1::zeros(intermediate_size),                // Correct dummy shape
        raw_fc2_w.clone(),                               // The real [in, out] weight
        cpu_fc2_b.clone(),
        crate::activations::Activation::Gelu,
    );
    let cpu_result = cpu_ffn.fc2(&cpu_input)?;

    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;

    let dummy_fc1_w_cpu = Array2::<f32>::zeros((hidden_size, intermediate_size));
    let dummy_fc1_b_cpu = Array1::<f32>::zeros(intermediate_size);

    // 2. Use the single "smart" constructor to create the weights struct.
    //    This function correctly transposes the real `raw_fc2_w` and the dummy `dummy_fc1_w_cpu`
    //    before creating the final GpuTensors.
    let weights = GpuFeedForwardWeights::from_ndarrays(
        &context,
        &dummy_fc1_w_cpu,
        &dummy_fc1_b_cpu,
        &raw_fc2_w,
        &cpu_fc2_b,
    )?;

    let gpu_ffn_partial = GpuFeedForwardStd::new(&context, crate::activations::Activation::Gelu)?;
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, hidden_size],
        DType::F32,
        "FC2 Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn_partial.run_fc2(&mut encoder, &weights, &gpu_input, &gpu_output);
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    Ok(())
}

async fn run_ffn_test(transpose_weights: bool) -> Result<()> {
    let context = get_test_context().await;

    let batch_size = 1;
    let seq_len = 128;
    let hidden_size = 768;
    let intermediate_size = hidden_size * 4;

    let (raw_fc1_w, raw_fc2_w) = if transpose_weights {
        (
            Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1)),
            Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1)),
        )
    } else {
        (
            Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1)),
            Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1)),
        )
    };
    let cpu_fc1_b = Array::random(intermediate_size, Uniform::new(-0.1, 0.1));
    let cpu_fc2_b = Array::random(hidden_size, Uniform::new(-0.1, 0.1));
    let fc1_w_prepared = if transpose_weights {
        raw_fc1_w.clone()
    } else {
        raw_fc1_w.t().to_owned()
    };
    let fc2_w_prepared = if transpose_weights {
        raw_fc2_w.clone()
    } else {
        raw_fc2_w.t().to_owned()
    };

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_prepared.clone(),
        cpu_fc1_b.clone(),
        fc2_w_prepared.clone(),
        cpu_fc2_b.clone(),
        crate::activations::Activation::GeluNew,
    );

    let weights = GpuFeedForwardWeights::from_ndarrays(
        &context,
        &fc1_w_prepared,
        &cpu_fc1_b,
        &fc2_w_prepared,
        &cpu_fc2_b,
    )?;

    let gpu_ffn = GpuFeedForwardStd::new(&context, crate::activations::Activation::GeluNew)?;
    let cpu_input = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let gpu_input = GpuTensor::from_ndarray(&context, &cpu_input)?;
    let gpu_intermediate = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, intermediate_size],
        DType::F32,
        "FFN Intermediate",
    );
    let gpu_output = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, hidden_size],
        DType::F32,
        "FFN Output",
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn.forward(
        &mut encoder,
        &weights,
        &gpu_input,
        &gpu_intermediate,
        &gpu_output,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let (gpu_vec, shape) = read_gpu_tensor_to_vec::<f32>(&gpu_output).await?;
    let gpu_result = Array3::from_shape_vec((shape[0], shape[1], shape[2]), gpu_vec)?;
    let cpu_result = cpu_ffn.forward(&cpu_input)?;

    assert_all_close(&gpu_result, &cpu_result, 1e-3);
    Ok(())
}

#[tokio::test]
async fn test_ffn_parity_with_transpose_true() -> Result<()> {
    run_ffn_test(true).await?;
    Ok(())
}

#[tokio::test]
async fn test_ffn_parity_with_transpose_false() -> Result<()> {
    run_ffn_test(false).await?;
    Ok(())
}

#[tokio::test]
async fn test_gpu_ffn_parity_encode() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::Gelu;

    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let input_cpu = Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-1.5, 1.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    let expected_cpu = cpu_ffn.forward(&input_cpu)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    let output_gpu = gpu_ffn.encode(&mut encoder, &input_gpu, &gpu_weights, &mut pool);
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    assert_tensors_are_close_relative(
        &expected_cpu,
        &output_gpu,
        "FFN FC2 Fused Output",
        1e-4, 
        1e-5, 
    )
    .await;

    Ok(())
}
#[tokio::test]
async fn test_gpu_ffn_fc1_isolated_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::GeluNew;

    // Create deterministic weights
    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);

    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );

    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let input_cpu = Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-0.5, 0.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    // Run CPU
    let mut expected_cpu_output = cpu_ffn.fc1(&input_cpu)?;
    cpu_ffn.apply_activation(&mut expected_cpu_output);

    // Run GPU
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let intermediate_shape = expected_cpu_output.shape().to_vec();
    let output_gpu = GpuTensor::uninitialized(
        &context,
        intermediate_shape,
        DType::F32,
        "FC1 Isolated Test Output",
    );
    gpu_ffn.run_fc1(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(_) => {} // Success
        Err(e) => panic!("GPU Poll failed: {:?}", e),
    }

    assert_tensors_are_close(
        &expected_cpu_output,
        &output_gpu,
        "FFN FC1 Fused Output",
        2e-4,
    )
    .await;

    Ok(())
}
#[tokio::test]
async fn test_gpu_ffn_fc1_isolated_relu() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::Relu;
    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    // Inputs
    let input_cpu = Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-0.5, 0.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    // CPU Run
    let mut expected_cpu_output = cpu_ffn.fc1(&input_cpu)?;
    cpu_ffn.apply_activation(&mut expected_cpu_output);

    // GPU Run
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let intermediate_shape = expected_cpu_output.shape().to_vec();
    let output_gpu =
        GpuTensor::uninitialized(&context, intermediate_shape, DType::F32, "Relu Output");
    gpu_ffn.run_fc1(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(_) => {}
        Err(e) => panic!("GPU Poll failed: {:?}", e),
    }

    // ReLU is numerically stable (max(0,x)), so we can use a tighter tolerance
    assert_tensors_are_close(
        &expected_cpu_output,
        &output_gpu,
        "FFN FC1 ReLU Output",
        1e-5,
    )
    .await;
    Ok(())
}

#[tokio::test]
async fn test_gpu_ffn_fc1_isolated_silu() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::SilU; // Ensure this exists

    // Deterministic weights
    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let input_cpu = Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-0.5, 0.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    let mut expected_cpu_output = cpu_ffn.fc1(&input_cpu)?;
    cpu_ffn.apply_activation(&mut expected_cpu_output);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let intermediate_shape = expected_cpu_output.shape().to_vec();
    let output_gpu =
        GpuTensor::uninitialized(&context, intermediate_shape, DType::F32, "SiLU Output");
    gpu_ffn.run_fc1(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(_) => {}
        Err(e) => panic!("GPU Poll failed: {:?}", e),
    }

    // SiLU uses exp/sigmoid, so we use 2e-4 tolerance
    assert_tensors_are_close(
        &expected_cpu_output,
        &output_gpu,
        "FFN FC1 SiLU Output",
        2e-4,
    )
    .await;
    Ok(())
}

#[tokio::test]
async fn test_gpu_ffn_fc1_isolated_tanh() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::Tanh; // Ensure this exists

    // Deterministic weights
    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let input_cpu = Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-0.5, 0.5));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    let mut expected_cpu_output = cpu_ffn.fc1(&input_cpu)?;
    cpu_ffn.apply_activation(&mut expected_cpu_output);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let intermediate_shape = expected_cpu_output.shape().to_vec();
    let output_gpu =
        GpuTensor::uninitialized(&context, intermediate_shape, DType::F32, "Tanh Output");
    gpu_ffn.run_fc1(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));

    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(_) => {}
        Err(e) => panic!("GPU Poll failed: {:?}", e),
    }

    // Tanh is transcendental, use 2e-4 tolerance
    assert_tensors_are_close(
        &expected_cpu_output,
        &output_gpu,
        "FFN FC1 Tanh Output",
        2e-4,
    )
    .await;
    Ok(())
}
#[tokio::test]
async fn test_gpu_ffn_fc2_isolated_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let activation = Activation::Gelu;

    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    })
    .as_standard_layout()
    .to_owned();
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);

    let cpu_ffn = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let fc2_w_gpu_transposed = GpuTensor::from_ndarray(&context, &fc2_w_cpu.t().to_owned())?;

    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let initial_input_cpu =
        Array3::random((batch_size, seq_len, hidden_size), Uniform::new(-1.5, 1.5));
    let mut intermediate_input_cpu = cpu_ffn.fc1(&initial_input_cpu)?;
    cpu_ffn.apply_activation(&mut intermediate_input_cpu);
    let intermediate_input_gpu = GpuTensor::from_ndarray(&context, &intermediate_input_cpu)?;

    let expected_cpu_output = cpu_ffn.fc2(&intermediate_input_cpu)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let final_output_shape = expected_cpu_output.shape().to_vec();
    let output_gpu = GpuTensor::uninitialized(
        &context,
        final_output_shape,
        DType::F32,
        "FC2 Isolated Test Output",
    );

    gpu_ffn.run_fc2(
        &mut encoder,
        &gpu_weights,
        &intermediate_input_gpu,
        &output_gpu,
    );

    context.queue.submit(Some(encoder.finish()));

    assert_tensors_are_close_relative(
        &expected_cpu_output,
        &output_gpu,
        "FFN FC2 Fused Output",
        1e-4, // Relative tolerance: 0.01%
        1e-5, // Absolute tolerance
    )
    .await;

    Ok(())
}
pub async fn assert_tensors_are_close_relative(
    cpu_tensor: &Array3<f32>,
    gpu_tensor: &GpuTensor,
    tensor_name: &str,
    relative_tolerance: f32,
    absolute_tolerance: f32,
) {
    let gpu_tensor_cpu = gpu_tensor
        .to_ndarray_3d::<f32>()
        .await
        .expect("Failed to read GPU tensor back to CPU for comparison");

    assert_eq!(
        cpu_tensor.shape(),
        gpu_tensor_cpu.shape(),
        "Tensor '{}' shape mismatch. CPU: {:?}, GPU: {:?}",
        tensor_name,
        cpu_tensor.shape(),
        gpu_tensor_cpu.shape()
    );
    let mut max_abs_diff = 0.0;
    let mut worst_cpu_val = 0.0;
    let mut worst_gpu_val = 0.0;
    let mut worst_index = (0, 0, 0);

    let shape = cpu_tensor.shape();
    let (dim0, dim1, dim2) = (shape[0], shape[1], shape[2]);

    for i in 0..dim0 {
        for j in 0..dim1 {
            for k in 0..dim2 {
                let cpu_val = cpu_tensor[[i, j, k]];
                let gpu_val = gpu_tensor_cpu[[i, j, k]];
                let abs_diff = (cpu_val - gpu_val).abs();

                if abs_diff > max_abs_diff {
                    max_abs_diff = abs_diff;
                    worst_cpu_val = cpu_val;
                    worst_gpu_val = gpu_val;
                    worst_index = (i, j, k);
                }
            }
        }
    }

    let allowed_diff = absolute_tolerance + (relative_tolerance * worst_cpu_val.abs());

    if max_abs_diff > allowed_diff {
        panic!(
            "\n\nTensor '{}' is not close enough to its GPU counterpart.\n\
             Worst failure at index {:?}:\n\
             - CPU Value:      {}\n\
             - GPU Value:      {}\n\
             - Absolute Diff:  {}  (> Allowed Diff)\n\
             - Allowed Diff:   {} (abs_tol: {} + rel_tol: {} * |CPU|)\n\n",
            tensor_name,
            worst_index,
            worst_cpu_val,
            worst_gpu_val,
            max_abs_diff,
            allowed_diff,
            absolute_tolerance,
            relative_tolerance
        );
    }
}
#[tokio::test]
async fn test_gpu_ffn_fc2_pass_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let activation = Activation::Gelu; // Activation doesn't matter for FC2
    let (batch_size, seq_len, intermediate_size, hidden_size) = (2, 16, 512, 128);
    let fc2_w_cpu = Array2::from_shape_fn((intermediate_size, hidden_size), |(i, j)| {
        (i + j) as f32 * -0.01
    });
    let fc2_b_cpu = Array1::from_shape_fn(hidden_size, |i| i as f32 * -0.01);
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let fc1_w_cpu = Array2::<f32>::zeros((hidden_size, intermediate_size));
    let fc1_b_cpu = Array1::<f32>::zeros(intermediate_size);

    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context, &fc1_w_cpu, &fc1_b_cpu, &fc2_w_cpu, &fc2_b_cpu,
    )?;

    let input_cpu = Array3::from_shape_fn((batch_size, seq_len, intermediate_size), |(i, j, k)| {
        (i + j + k) as f32 * 0.1
    });
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let output_gpu = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, hidden_size],
        input_gpu.dtype(),
        "FC2 Output",
    );
    let cpu_ffn_partial = CpuFeedForward::new(
        Array2::zeros((hidden_size, intermediate_size)),
        Array1::zeros(intermediate_size),
        fc2_w_cpu.clone(),
        fc2_b_cpu.clone(),
        activation,
    );
    let expected_cpu = cpu_ffn_partial.fc2(&input_cpu)?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn.run_fc2(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "FFN FC2 Output", 1e-1).await; // TODO FIND OUT WHY 1e-2 doesnt work
    Ok(())
}

#[tokio::test]
async fn test_gpu_ffn_fc1_pass_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let activation = Activation::Gelu;
    let (batch_size, seq_len, hidden_size, intermediate_size) = (2, 16, 128, 512);
    let fc1_w_cpu = Array2::from_shape_fn((hidden_size, intermediate_size), |(i, j)| {
        (i + j) as f32 * 0.01
    });
    let fc1_b_cpu = Array1::from_shape_fn(intermediate_size, |i| i as f32 * 0.01);
    let gpu_ffn = GpuFeedForwardStd::new(&context, activation)?;
    let dummy_fc2_w_cpu = Array2::<f32>::zeros((intermediate_size, hidden_size));
    let dummy_fc2_b_cpu = Array1::<f32>::zeros(hidden_size);

    let gpu_weights = GpuFeedForwardWeights::from_ndarrays(
        &context,
        &fc1_w_cpu,
        &fc1_b_cpu,
        &dummy_fc2_w_cpu,
        &dummy_fc2_b_cpu,
    )?;
    let input_cpu = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(i, j, k)| {
        (i + j + k) as f32 * 0.1
    });
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let output_gpu = GpuTensor::uninitialized(
        &context,
        vec![batch_size, seq_len, intermediate_size],
        input_gpu.dtype(),
        "FC1 Output",
    );
    let cpu_ffn_partial = CpuFeedForward::new(
        fc1_w_cpu.clone(),
        fc1_b_cpu.clone(),
        Array2::zeros((intermediate_size, hidden_size)),
        Array1::zeros(hidden_size),
        activation,
    );
    let mut matmul_plus_bias = cpu_ffn_partial.fc1(&input_cpu)?;
    cpu_ffn_partial.apply_activation(&mut matmul_plus_bias);
    let expected_cpu = matmul_plus_bias;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_ffn.run_fc1(&mut encoder, &gpu_weights, &input_gpu, &output_gpu);
    context.queue.submit(Some(encoder.finish()));
    assert_tensors_are_close(&expected_cpu, &output_gpu, "FFN FC1 Output", 1e-2).await;
    Ok(())
}
