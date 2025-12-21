use super::*; // Imports GpuSwiGLUFFN, GpuSwiGLUFFNWeights
use crate::feedforward::SwiGluFeedForward as CpuSwiGLUFFN; // Import your CPU implementation
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, GpuTensorPool, GpuFrameContext};
use anyhow::Result;
use ndarray::{Array};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::sync::Arc;
use crate::linear_layer::LinearLayer;
use crate::tests::common::{assert_tensors_are_close_2d as assert_tensors_are_close};

#[path = "../../../tests/common.rs"]
mod common;

#[tokio::test]
async fn test_gpu_swiglu_ffn_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (rows, hidden_size) = (128, 256); 
    let intermediate_size = 512;
    
    let gpu_swiglu = GpuSwiGLUFFN::new(&context)?;

    // CPU weights: [Out, In] (PyTorch/LinearLayer convention)
    let gate_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-1.0, 1.0));
    let up_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-1.0, 1.0));
    let down_cpu = Array::random((hidden_size, intermediate_size), Uniform::new(-1.0, 1.0));

    // GPU F32 matmul expects [In, Out] - transpose on upload
    let weights_gpu = GpuSwiGLUFFNWeights::new(
        GpuTensor::from_ndarray(&context, &gate_cpu.as_standard_layout().to_owned())?,
        GpuTensor::from_ndarray(&context, &up_cpu.as_standard_layout().to_owned())?,
        GpuTensor::from_ndarray(&context, &down_cpu.t().as_standard_layout().to_owned())?,
    )?;

    // CPU uses LinearLayer (handles transpose internally)
    let cpu_swiglu = CpuSwiGLUFFN::new(
        LinearLayer::from(gate_cpu),
        LinearLayer::from(up_cpu),
        LinearLayer::from(down_cpu),
    );

    let input_cpu = Array::random((rows, hidden_size), Uniform::new(-1.0, 1.0));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let output_gpu = GpuTensor::uninitialized(&context, vec![rows, hidden_size], crate::tensor::DType::F32, "SwiGLU Output");

    let expected_cpu = cpu_swiglu.forward_2d(&input_cpu)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());
    
    gpu_swiglu.encode(&mut encoder, &weights_gpu, &input_gpu, &output_gpu, &mut pool);
    context.queue.submit(Some(encoder.finish()));
    
    assert_tensors_are_close(&expected_cpu, &output_gpu, "SwiGLU FFN Output", 1e-2).await;
    Ok(())
}