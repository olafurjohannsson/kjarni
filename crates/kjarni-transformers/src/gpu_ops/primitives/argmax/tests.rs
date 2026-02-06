#[path = "../../../tests/common.rs"]
mod common;

use super::*;
use crate::WgpuContext;
use crate::gpu::GpuTensor;
use anyhow::Result;
use ndarray::Array2;
use std::sync::Arc;
use wgpu::{BufferDescriptor, BufferUsages};

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}

async fn read_output_index(context: &WgpuContext, buffer: &wgpu::Buffer) -> u32 {
    let staging = context.device.create_buffer(&BufferDescriptor {
        label: Some("Staging Buffer"),
        size: 4,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, 4);
    context.queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = futures::channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    context.device.poll(wgpu::PollType::wait_indefinitely());
    rx.await.unwrap().unwrap();

    let data = slice.get_mapped_range();
    let result = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    staging.unmap();

    result
}

fn create_output_buffer(context: &WgpuContext) -> wgpu::Buffer {
    context.device.create_buffer(&BufferDescriptor {
        label: Some("ArgMax Output"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

// ========================================================================
//  Basic Functionality Tests
// ========================================================================

#[tokio::test]
async fn test_argmax_simple() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Simple case: max at index 2
    let logits = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 5.0, 3.0, 4.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 2);

    Ok(())
}

#[tokio::test]
async fn test_argmax_first_element() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Max at index 0
    let logits = Array2::from_shape_vec((1, 5), vec![10.0, 2.0, 3.0, 4.0, 5.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 0);

    Ok(())
}

#[tokio::test]
async fn test_argmax_last_element() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Max at last index
    let logits = Array2::from_shape_vec((1, 5), vec![1.0, 2.0, 3.0, 4.0, 100.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 4);

    Ok(())
}

// ========================================================================
//  Negative Values Tests
// ========================================================================

#[tokio::test]
async fn test_argmax_with_negative_values() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // All negative, max at index 1
    let logits = Array2::from_shape_vec((1, 5), vec![-5.0, -1.0, -3.0, -4.0, -2.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 1);

    Ok(())
}

#[tokio::test]
async fn test_argmax_mixed_positive_negative() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Mixed values, max at index 3
    let logits = Array2::from_shape_vec((1, 5), vec![-5.0, 0.0, -3.0, 10.0, -2.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 3);

    Ok(())
}

// ========================================================================
//  Edge Cases
// ========================================================================

#[tokio::test]
async fn test_argmax_single_element() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Single element
    let logits = Array2::from_shape_vec((1, 1), vec![42.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 0);

    Ok(())
}

#[tokio::test]
async fn test_argmax_two_elements() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Two elements, max at index 1
    let logits = Array2::from_shape_vec((1, 2), vec![1.0, 2.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 1);

    Ok(())
}

#[tokio::test]
async fn test_argmax_equal_values_returns_first() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // All equal values - should return first occurrence (index 0)
    let logits = Array2::from_shape_vec((1, 5), vec![5.0, 5.0, 5.0, 5.0, 5.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 0);

    Ok(())
}

#[tokio::test]
async fn test_argmax_zeros() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // All zeros
    let logits = Array2::from_shape_vec((1, 5), vec![0.0, 0.0, 0.0, 0.0, 0.0])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 0);

    Ok(())
}

// ========================================================================
//  Realistic Vocab Size Tests
// ========================================================================

#[tokio::test]
async fn test_argmax_large_vocab() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Realistic vocab size (e.g., GPT-2: 50257)
    let vocab_size = 1024;
    let max_idx = 512;

    let mut logits_vec = vec![0.0f32; vocab_size];
    logits_vec[max_idx] = 100.0;

    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, max_idx as u32);

    Ok(())
}

#[tokio::test]
async fn test_argmax_gpt2_vocab_size() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // GPT-2 vocab size
    let vocab_size = 50257;
    let max_idx = 42000;

    let mut logits_vec = vec![-10.0f32; vocab_size];
    logits_vec[max_idx] = 15.0;

    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, max_idx as u32);

    Ok(())
}

#[tokio::test]
async fn test_argmax_llama_vocab_size() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Llama vocab size
    let vocab_size = 32000;
    let max_idx = 31999; // Last token

    let mut logits_vec = vec![0.0f32; vocab_size];
    logits_vec[max_idx] = 1.0;

    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, max_idx as u32);

    Ok(())
}

// ========================================================================
//  Numerical Precision Tests
// ========================================================================

#[tokio::test]
async fn test_argmax_very_close_values() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Very close values, max at index 2
    let logits = Array2::from_shape_vec(
        (1, 5),
        vec![1.0000001, 1.0000002, 1.0000003, 1.0000001, 1.0000000],
    )?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 2);

    Ok(())
}

#[tokio::test]
async fn test_argmax_large_magnitude() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Large magnitude values
    let logits = Array2::from_shape_vec((1, 5), vec![1e10, 1e11, 1e12, 1e9, 1e8])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 2);

    Ok(())
}

#[tokio::test]
async fn test_argmax_small_magnitude() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Small magnitude values
    let logits = Array2::from_shape_vec((1, 5), vec![1e-10, 1e-9, 1e-8, 1e-11, 1e-12])?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 2);

    Ok(())
}

// ========================================================================
//  CPU Reference Comparison Tests
// ========================================================================

fn cpu_argmax(logits: &[f32]) -> u32 {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx as u32)
        .unwrap_or(0)
}

#[tokio::test]
async fn test_argmax_matches_cpu_random_small() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Deterministic "random" values
    let logits_vec: Vec<f32> = (0..100)
        .map(|i| ((i * 7 + 13) % 100) as f32 - 50.0)
        .collect();

    let expected = cpu_argmax(&logits_vec);

    let logits = Array2::from_shape_vec((1, 100), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, expected);

    Ok(())
}

#[tokio::test]
async fn test_argmax_matches_cpu_random_large() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Larger deterministic "random" values
    let logits_vec: Vec<f32> = (0..10000)
        .map(|i| ((i * 17 + 31) % 1000) as f32 / 10.0 - 50.0)
        .collect();

    let expected = cpu_argmax(&logits_vec);

    let logits = Array2::from_shape_vec((1, 10000), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, expected);

    Ok(())
}

// ========================================================================
//  Typical LLM Logits Distribution Tests
// ========================================================================

#[tokio::test]
async fn test_argmax_softmax_like_distribution() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Simulate softmax-like distribution with one clear winner
    let vocab_size = 1000;
    let winner_idx = 777;

    let mut logits_vec: Vec<f32> = (0..vocab_size)
        .map(|i| -5.0 + (i as f32 / vocab_size as f32) * 2.0)
        .collect();
    logits_vec[winner_idx] = 10.0; // Clear winner

    let logits = Array2::from_shape_vec((1, vocab_size), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, winner_idx as u32);

    Ok(())
}

#[tokio::test]
async fn test_argmax_ascending_values() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Ascending values - max should be last
    let logits_vec: Vec<f32> = (0..256).map(|i| i as f32).collect();

    let logits = Array2::from_shape_vec((1, 256), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 255);

    Ok(())
}

#[tokio::test]
async fn test_argmax_descending_values() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Descending values - max should be first
    let logits_vec: Vec<f32> = (0..256).rev().map(|i| i as f32).collect();

    let logits = Array2::from_shape_vec((1, 256), logits_vec)?;
    let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
    let output_buffer = create_output_buffer(&context);

    let mut encoder = context.device.create_command_encoder(&Default::default());
    kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
    context.queue.submit(std::iter::once(encoder.finish()));

    let result = read_output_index(&context, &output_buffer).await;
    assert_eq!(result, 0);

    Ok(())
}

// ========================================================================
//  Kernel Reuse Test
// ========================================================================

#[tokio::test]
async fn test_argmax_kernel_reuse() -> Result<()> {
    let context = get_test_context().await;
    let kernel = GpuArgMax::new(&context);

    // Run multiple times with same kernel
    let test_cases = vec![
        (vec![1.0, 5.0, 3.0], 1u32),
        (vec![10.0, 2.0, 3.0], 0u32),
        (vec![1.0, 2.0, 30.0], 2u32),
    ];

    for (logits_vec, expected) in test_cases {
        let logits = Array2::from_shape_vec((1, logits_vec.len()), logits_vec)?;
        let gpu_logits = GpuTensor::from_ndarray(&context, &logits)?;
        let output_buffer = create_output_buffer(&context);

        let mut encoder = context.device.create_command_encoder(&Default::default());
        kernel.encode(&mut encoder, &gpu_logits, &output_buffer);
        context.queue.submit(std::iter::once(encoder.finish()));

        let result = read_output_index(&context, &output_buffer).await;
        assert_eq!(result, expected);
    }

    Ok(())
}
