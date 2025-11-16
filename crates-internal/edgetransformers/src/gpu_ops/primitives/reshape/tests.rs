#[path = "../../../tests/common.rs"]
mod common;
use common::{
    assert_all_close, assert_arrays_are_close_2d, get_test_context, read_gpu_tensor_to_vec,
};

use super::*;
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::utils::{assert_vecs_are_close, read_buffer_3d};
use ndarray::{Array, Array3};
use ndarray_rand::RandomExt;
use rand_distr::Uniform;
use std::sync::Arc;

use anyhow::Result;
use wgpu::util::DeviceExt;

#[tokio::test]
async fn test_reshape_unreshape_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;
    let (b, s, h, d) = (2, 8, 4, 16); // batch, seq_len, num_heads, head_dim
    let hidden_size = h * d;
    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));
    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Reshape Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage,
    });
    let reshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Reshaped Output"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let unreshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshaped Output"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let mut cpu_reshaped = Array3::<f32>::zeros((b * h, s, d));
    for batch_idx in 0..b {
        for seq_idx in 0..s {
            for head_idx in 0..h {
                for head_dim_idx in 0..d {
                    let val = input_cpu[[batch_idx, seq_idx, head_idx * d + head_dim_idx]];
                    cpu_reshaped[[batch_idx * h + head_idx, seq_idx, head_dim_idx]] = val;
                }
            }
        }
    }
    let mut encoder1 =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let pipeline = compile_reshape_pipeline(&context);
    run_gpu_reshape(
        &context,
        &mut encoder1,
        &pipeline,
        &input_gpu,
        &reshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    context.queue.submit(std::iter::once(encoder1.finish()));
    let gpu_reshaped_array = read_buffer_3d(&context, &reshaped_gpu, (b * h, s, d)).await?;
    assert_vecs_are_close(
        cpu_reshaped.as_slice().unwrap(),
        gpu_reshaped_array.as_slice().unwrap(),
        1e-6,
    );
    let mut encoder2 =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let pipeline = compile_unreshape_pipeline(&context);
    run_gpu_unreshape(
        &context,
        &mut encoder2,
        &pipeline,
        &reshaped_gpu,
        &unreshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
    );
    context.queue.submit(std::iter::once(encoder2.finish()));
    let gpu_unreshaped_array =
        read_buffer_3d(&context, &unreshaped_gpu, (b, s, hidden_size)).await?;
    assert_vecs_are_close(
        input_cpu.as_slice().unwrap(),
        gpu_unreshaped_array.as_slice().unwrap(),
        1e-6,
    );
    Ok(())
}

#[tokio::test]
async fn test_reshape_correctness() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;
    let (b, s, h, d) = (2, 8, 4, 16); // batch, seq_len, num_heads, head_dim
    let hidden_size = h * d;
    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));
    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;

    let input_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Reshape Input"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Reshape Output"),
        size: buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut cpu_reshaped = Array3::<f32>::zeros((b * h, s, d));
    for batch_idx in 0..b {
        for seq_idx in 0..s {
            for head_idx in 0..h {
                for head_dim_idx in 0..d {
                    let val = input_cpu[[batch_idx, seq_idx, head_idx * d + head_dim_idx]];
                    cpu_reshaped[[batch_idx * h + head_idx, seq_idx, head_dim_idx]] = val;
                }
            }
        }
    }

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Reshape Test Encoder"),
    });
    let pipeline = compile_reshape_pipeline(&context);
    run_gpu_reshape(
        &context,
        &mut encoder,
        &pipeline,
        &input_gpu,
        &output_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    context.queue.submit(std::iter::once(encoder.finish()));
    let gpu_reshaped_array = read_buffer_3d(&context, &output_gpu, (b * h, s, d)).await?;
    assert_vecs_are_close(
        cpu_reshaped.as_slice().unwrap(),
        gpu_reshaped_array.as_slice().unwrap(),
        1e-6,
    );
    Ok(())
}

/// Tests that the `run_gpu_unreshape` kernel is the mathematical inverse of `run_gpu_reshape`.
#[tokio::test]
async fn test_unreshape_is_inverse_of_reshape() -> Result<()> {
    let context = get_test_context().await;
    let device = &context.device;

    let (b, s, h, d) = (2, 8, 4, 16);
    let hidden_size = h * d;

    let input_cpu: Array3<f32> = Array::random((b, s, hidden_size), Uniform::new(-1.0, 1.0));
    let buffer_size = (b * s * hidden_size * std::mem::size_of::<f32>()) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;

    let original_gpu = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Test Unreshape Original"),
        contents: bytemuck::cast_slice(input_cpu.as_slice().unwrap()),
        usage,
    });
    let reshaped_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshape Intermediate"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });
    let final_gpu = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Test Unreshape Final"),
        size: buffer_size,
        usage,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Inverse Test Encoder"),
    });
    let pipeline = compile_reshape_pipeline(&context);
    run_gpu_reshape(
        &context,
        &mut encoder,
        &pipeline,
        &original_gpu,
        &reshaped_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
        false,
    );
    let pipeline1 = compile_unreshape_pipeline(&context);
    run_gpu_unreshape(
        &context,
        &mut encoder,
        &pipeline1,
        &reshaped_gpu,
        &final_gpu,
        b as u32,
        s as u32,
        h as u32,
        d as u32,
    );

    context.queue.submit(std::iter::once(encoder.finish()));

    let gpu_final_array = read_buffer_3d(&context, &final_gpu, (b, s, hidden_size)).await?;
    assert_vecs_are_close(
        input_cpu.as_slice().unwrap(),
        gpu_final_array.as_slice().unwrap(),
        1e-6,
    );
    Ok(())
}
