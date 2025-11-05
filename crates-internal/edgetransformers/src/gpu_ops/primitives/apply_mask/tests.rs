#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::attention::{apply_causal_mask, apply_padding_mask};
use crate::gpu_ops::{DType, GpuTensor};
use crate::utils::linear_algebra::apply_attention_mask;
use crate::utils::masks::{
    create_batched_causal_mask, create_causal_mask, create_full_attention_mask,
    create_padding_mask_from_tokens, expand_mask_for_attention,
};
use anyhow::Result;
use common::read_gpu_tensor_to_vec;
use ndarray::{Array, Array2, Array4, Axis, s};
use std::sync::Arc;

const MASK_VALUE: f32 = -1e9;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

fn generate_expected_scores(
    scores: &Array4<f32>,
    padding_mask: &Array2<f32>,
    is_causal: bool,
    position_offset: u32,
) -> Result<Array4<f32>> {
    let mut current_scores = scores.clone();

    // Call the REAL production functions
    if is_causal {
        current_scores = apply_causal_mask(current_scores, position_offset as usize)?;
    }
    current_scores = apply_padding_mask(current_scores, padding_mask)?;

    Ok(current_scores)
}
#[tokio::test]
async fn test_mask_encoder_case() -> Result<()> {
    println!("\n--- Testing GpuApplyMask (Encoder) ---");
    let context = get_test_context().await;
    let kernel = GpuApplyMask::new(&context);

    let (b, h, s) = (1, 2, 4);
    let cpu_scores = Array::from_shape_fn((b, h, s, s), |(i, j, k, l)| (i + j + k + l) as f32);
    let cpu_tokens = Array2::from_shape_vec((b, s), vec![1.0, 1.0, 1.0, 0.0])?;
    let cpu_mask = create_padding_mask_from_tokens(&cpu_tokens, 0.0);

    let gpu_scores = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let gpu_mask = GpuTensor::from_ndarray(&context, &cpu_mask)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_scores, &gpu_mask, false, 0);
    context.queue.submit(std::iter::once(encoder.finish()));

    let expected_result = generate_expected_scores(&cpu_scores, &cpu_mask, false, 0)?;

    let gpu_result_vec = read_gpu_tensor_to_vec::<f32>(&gpu_scores).await?.0;
    let gpu_result = Array4::from_shape_vec((b, h, s, s), gpu_result_vec)?;

    assert_eq!(gpu_result.as_slice(), expected_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_mask_decoder_prompt_case() -> Result<()> {
    println!("\n--- Testing GpuApplyMask (Decoder Prompt) ---");
    let context = get_test_context().await;
    let kernel = GpuApplyMask::new(&context);

    let (b, h, s) = (1, 2, 4);
    let cpu_scores = Array::from_shape_fn((b, h, s, s), |(i, j, k, l)| (i + j + k + l) as f32);
    let cpu_mask = create_full_attention_mask(b, s);

    let gpu_scores = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let gpu_mask = GpuTensor::from_ndarray(&context, &cpu_mask)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_scores, &gpu_mask, true, 0);
    context.queue.submit(std::iter::once(encoder.finish()));

    let expected_result = generate_expected_scores(&cpu_scores, &cpu_mask, true, 0)?;

    let gpu_result_vec = read_gpu_tensor_to_vec::<f32>(&gpu_scores).await?.0;
    let gpu_result = Array4::from_shape_vec((b, h, s, s), gpu_result_vec)?;

    assert_eq!(gpu_result.as_slice(), expected_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_mask_decoder_generation_case() -> Result<()> {
    println!("\n--- Testing GpuApplyMask (Decoder Generation) ---");
    let context = get_test_context().await;
    let kernel = GpuApplyMask::new(&context);

    let (b, h) = (1, 2);
    let query_len = 1;
    let cache_capacity = 8;
    let position_offset = 4;

    let cpu_scores = Array::from_shape_fn((b, h, query_len, cache_capacity), |(i, j, k, l)| {
        (i + j + k + l) as f32
    });
    let cpu_mask =
        Array::from_shape_vec((b, cache_capacity), vec![1., 1., 1., 0., 1., 0., 0., 0.])?;

    let gpu_scores = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let gpu_mask = GpuTensor::from_ndarray(&context, &cpu_mask)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(
        &mut encoder,
        &gpu_scores,
        &gpu_mask,
        true,
        position_offset as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let expected_result =
        generate_expected_scores(&cpu_scores, &cpu_mask, true, position_offset as u32)?;

    let gpu_result_vec = read_gpu_tensor_to_vec::<f32>(&gpu_scores).await?.0;
    let gpu_result = Array4::from_shape_vec((b, h, query_len, cache_capacity), gpu_result_vec)?;

    assert_eq!(gpu_result.as_slice(), expected_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_mask_decoder_generation_offset_zero() -> Result<()> {
    println!("\n--- Testing GpuApplyMask (Decoder Edge Case: Offset 0) ---");
    let context = get_test_context().await;
    let kernel = GpuApplyMask::new(&context);

    let (b, h) = (1, 2);
    let query_len = 1;
    let cache_capacity = 4;
    let position_offset = 0;

    let cpu_scores = Array::from_shape_fn((b, h, query_len, cache_capacity), |(i, j, k, l)| {
        (i + j + k + l) as f32
    });
    let cpu_mask = Array::from_shape_vec((b, cache_capacity), vec![1., 1., 0., 1.])?;

    let gpu_scores = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let gpu_mask = GpuTensor::from_ndarray(&context, &cpu_mask)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(
        &mut encoder,
        &gpu_scores,
        &gpu_mask,
        true,
        position_offset as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let expected_result =
        generate_expected_scores(&cpu_scores, &cpu_mask, true, position_offset as u32)?;
    let gpu_result_vec = read_gpu_tensor_to_vec::<f32>(&gpu_scores).await?.0;
    let gpu_result = Array4::from_shape_vec((b, h, query_len, cache_capacity), gpu_result_vec)?;

    assert_eq!(gpu_result.as_slice(), expected_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}

#[tokio::test]
async fn test_mask_decoder_generation_batched() -> Result<()> {
    println!("\n--- Testing GpuApplyMask (Decoder Batched Generation) ---");
    let context = get_test_context().await;
    let kernel = GpuApplyMask::new(&context);

    let (b, h) = (2, 2);
    let query_len = 1;
    let cache_capacity = 4;
    let position_offset = 1;

    let cpu_scores = Array::from_shape_fn((b, h, query_len, cache_capacity), |(i, j, k, l)| {
        (i + j + k + l) as f32
    });
    let cpu_mask =
        Array::from_shape_vec((b, cache_capacity), vec![1., 0., 1., 1., 1., 1., 1., 0.])?;

    let gpu_scores = GpuTensor::from_ndarray(&context, &cpu_scores)?;
    let gpu_mask = GpuTensor::from_ndarray(&context, &cpu_mask)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(
        &mut encoder,
        &gpu_scores,
        &gpu_mask,
        true,
        position_offset as u32,
    );
    context.queue.submit(std::iter::once(encoder.finish()));

    let expected_result =
        generate_expected_scores(&cpu_scores, &cpu_mask, true, position_offset as u32)?;
    let gpu_result_vec = read_gpu_tensor_to_vec::<f32>(&gpu_scores).await?.0;
    let gpu_result = Array4::from_shape_vec((b, h, query_len, cache_capacity), gpu_result_vec)?;

    assert_eq!(gpu_result.as_slice(), expected_result.as_slice());
    println!("✅ Passed!");
    Ok(())
}
