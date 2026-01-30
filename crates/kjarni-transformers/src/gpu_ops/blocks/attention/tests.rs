use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array1, Array2, Array3, Axis};

use common::assert_tensors_are_close;

use crate::attention::MultiHeadAttention;
use crate::cache::{Cache, GpuKVCache};
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::{GpuTensor, GpuTensorPool};
use crate::WgpuContext;

#[path = "../../../tests/common.rs"]
mod common;

fn create_cpu_attention(
    h: usize,
    n: usize,
) -> (
    MultiHeadAttention,
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
    Array2<f32>,
    Array1<f32>,
) {
    let q_w = Array2::from_elem((h, h), 0.1);
    let q_b = Array1::from_elem(h, 0.1);
    let k_w = Array2::from_elem((h, h), 0.2);
    let k_b = Array1::from_elem(h, 0.2);
    let v_w = Array2::from_elem((h, h), 0.3);
    let v_b = Array1::from_elem(h, 0.3);
    let o_w = Array2::from_elem((h, h), 0.4);
    let o_b = Array1::from_elem(h, 0.4);

    let attention = MultiHeadAttention::new(
        h,
        n,
        q_w.clone(),
        q_b.clone(),
        k_w.clone(),
        k_b.clone(),
        v_w.clone(),
        v_b.clone(),
        o_w.clone(),
        o_b.clone(),
        None,
    );
    (attention, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b)
}

fn create_gpu_attention(
    context: &Arc<WgpuContext>,
    h: u32,
    n: u32,
    q_w: &Array2<f32>,
    q_b: &Array1<f32>,
    k_w: &Array2<f32>,
    k_b: &Array1<f32>,
    v_w: &Array2<f32>,
    v_b: &Array1<f32>,
    o_w: &Array2<f32>,
    o_b: &Array1<f32>,
) -> (GpuAttention, GpuAttentionWeights) {
    let attention = GpuAttention::new(&context.clone(), h, n, n);
    let weights = GpuAttentionWeights {
        q_weight: GpuTensor::from_ndarray(context, q_w).unwrap(),
        q_bias: GpuTensor::from_ndarray(context, q_b).unwrap(),
        k_weight: GpuTensor::from_ndarray(context, k_w).unwrap(),
        k_bias: GpuTensor::from_ndarray(context, k_b).unwrap(),
        v_weight: GpuTensor::from_ndarray(context, v_w).unwrap(),
        v_bias: GpuTensor::from_ndarray(context, v_b).unwrap(),
        output_weight: GpuTensor::from_ndarray(context, o_w).unwrap(),
        output_bias: GpuTensor::from_ndarray(context, o_b).unwrap(),
    };
    (attention, weights)
}

#[tokio::test]
async fn test_attention_encoder_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, s, h, n) = (1, 4, 16, 4);

    let (cpu_attn, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b) = create_cpu_attention(h, n);
    let (gpu_attn, gpu_weights) = create_gpu_attention(
        &context, h as u32, n as u32, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b,
    );

    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.1);
    let mask_cpu = Array2::from_shape_vec((b, s), vec![1.0, 1.0, 1.0, 0.0])?;
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

    let (cpu_output, _cpu_new_k, _cpu_new_v) =
        cpu_attn.forward_with_cache(&input_cpu, None, Some(&mask_cpu), false, None, None)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());

    let gpu_output = gpu_attn.forward(
        &mut encoder,
        &input_gpu,
        None,
        &gpu_weights,
        Some(&mask_gpu),
        false,
        None,
        &mut pool,
    );
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    assert_tensors_are_close(&cpu_output, &gpu_output, "encoder output", 1e-4).await;

    Ok(())
}

#[tokio::test]
async fn test_attention_decoder_generation_parity() -> Result<()> {
    let context = WgpuContext::new().await?;
    let (b, h, n) = (1, 16, 4);
    let head_dim = h / n;
    let prompt_len = 3;
    let gen_len = 1;
    let total_len = prompt_len + gen_len;
    let max_len = 8;

    let (cpu_attn, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b) = create_cpu_attention(h, n);
    let (gpu_attn, gpu_weights) = create_gpu_attention(
        &context, h as u32, n as u32, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b,
    );

    let prompt_cpu =
        Array3::from_shape_fn((b, prompt_len, h), |(i, j, k)| (i + j + k) as f32 * 0.1);
    let query_cpu = Array3::from_shape_fn((b, gen_len, h), |(i, j, k)| {
        (i + j + k + prompt_len) as f32 * 0.1
    });
    let mask_cpu = Array2::ones((b, total_len));

    let prompt_gpu = GpuTensor::from_ndarray(&context, &prompt_cpu)?;
    let query_gpu = GpuTensor::from_ndarray(&context, &query_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

    // Project K and V
    let (prompt_k_cpu, prompt_v_cpu) = cpu_attn.project_kv(&prompt_cpu);
    let (cpu_new_k, cpu_new_v) = cpu_attn.project_kv(&query_cpu);

    let full_k_cpu = ndarray::concatenate(Axis(1), &[prompt_k_cpu.view(), cpu_new_k.view()])?;
    let full_v_cpu = ndarray::concatenate(Axis(1), &[prompt_v_cpu.view(), cpu_new_v.view()])?;

    let q_proj_cpu = crate::utils::linear_algebra::matmul_3d_2d(&query_cpu, &q_w) + &q_b;

    let cpu_output = cpu_attn.attend(
        &q_proj_cpu, 
        &full_k_cpu,
        &full_v_cpu,
        Some(&mask_cpu),
        true,
        prompt_len,
    )?;
    let cpu_output_projected = crate::utils::linear_algebra::matmul_3d_2d(&cpu_output, &o_w) + &o_b;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());
    let mut gpu_cache = GpuKVCache::new(&context, 1, b, n, head_dim, max_len)?;
    let gpu_slicer = GpuSlice::new(&context);

    let (prompt_k_gpu, prompt_v_gpu) =
        gpu_attn.project_kv(&mut encoder, &prompt_gpu, &gpu_weights, 0, &mut pool, None);
    gpu_cache.update(&mut encoder, 0, &prompt_k_gpu, &prompt_v_gpu, 0)?;
    gpu_cache.increment_len(prompt_len);

    let (gpu_new_k, gpu_new_v) =
        gpu_attn.project_kv(&mut encoder, &query_gpu, &gpu_weights, prompt_len, &mut pool, None);
    gpu_cache.update(&mut encoder, 0, &gpu_new_k, &gpu_new_v, prompt_len)?;
    gpu_cache.increment_len(gen_len);

    let (full_cache_k_gpu, full_cache_v_gpu) = gpu_cache.get(0).unwrap();

    let cache_k_view = full_cache_k_gpu.slice(
        &mut encoder,
        &gpu_slicer,
        &[0, 0, 0, 0],
        &[b, n, total_len, head_dim],
    )?;
    let cache_v_view = full_cache_v_gpu.slice(
        &mut encoder,
        &gpu_slicer,
        &[0, 0, 0, 0],
        &[b, n, total_len, head_dim],
    )?;

    let gpu_output = gpu_attn.attend(
        &mut encoder,
        &query_gpu,
        &gpu_weights,
        &mask_gpu,
        true,
        (&cache_k_view, &cache_v_view),
        prompt_len,
        &mut pool,
    );
    context.queue.submit(Some(encoder.finish()));
    pool.next_frame();

    assert_tensors_are_close(&cpu_output_projected, &gpu_output, "decoder output", 1e-4).await;

    Ok(())
}