use crate::attention::MultiHeadAttention;
use crate::cache::{Cache, CpuKVCache, GpuKVCache};
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::cache::GpuUpdateCache;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::{
    DType, GpuTensor, GpuTensorPool, GpuFrameContext,
    blocks::attention::attention::{GpuAttention, GpuAttentionWeights, TempStorage},
};
use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, Ix3, Ix4};
use std::sync::Arc;
use common::{assert_tensors_are_close, read_gpu_tensor_to_vec};
#[path = "../../../tests/common.rs"]
mod common;



/// Helper to create a CPU attention block with dummy weights.
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
    );
    (attention, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b)
}

/// Helper to create the corresponding GPU attention block and weights.
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
    let attention = GpuAttention::new(&context.clone(), h, n);
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
    let context = Arc::new(WgpuContext::new().await?);
    let (b, s, h, n) = (1, 4, 16, 4); // Batch, SeqLen, HiddenSize, NumHeads

    let (cpu_attn, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b) = create_cpu_attention(h, n);
    let (gpu_attn, gpu_weights) = create_gpu_attention(
        &context, h as u32, n as u32, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b,
    );

    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.1);
    let mask_cpu = Array2::from_shape_vec((b, s), vec![1.0, 1.0, 1.0, 0.0])?;
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

    let (cpu_output, cpu_new_k, cpu_new_v) =
        cpu_attn.forward_with_cache(&input_cpu, None, Some(&mask_cpu), false, None, None)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut pool = GpuTensorPool::new(context.clone());
    let mut frame = GpuFrameContext::new(&context, &mut pool);

    let (gpu_output, gpu_new_k, gpu_new_v) = gpu_attn.forward(
        &mut encoder,
        &input_gpu,
        &input_gpu,
        &gpu_weights,
        &mask_gpu,
        false,
        None,
        0,
        frame.pool(),
    );
    context.queue.submit(Some(encoder.finish()));
    frame.finish();

    // 5. Compare results
    assert_tensors_are_close(&cpu_output, &gpu_output, "Encoder Output", 1e-4).await;
    assert_tensors_are_close(&cpu_new_k, &gpu_new_k, "Encoder New K", 1e-4).await;
    assert_tensors_are_close(&cpu_new_v, &gpu_new_v, "Encoder New V", 1e-4).await;

    println!("✅ GpuAttention passed encoder parity test!");
    Ok(())
}

#[tokio::test]
async fn test_attention_decoder_generation_parity() -> Result<()> {
    let context = Arc::new(WgpuContext::new().await?);
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
    let (prompt_k_cpu, prompt_v_cpu) = cpu_attn.project_kv(&prompt_cpu);
    let (cpu_new_k, cpu_new_v) = cpu_attn.project_kv(&query_cpu);

    let full_k_cpu = ndarray::concatenate(Axis(1), &[prompt_k_cpu.view(), cpu_new_k.view()])?;
    let full_v_cpu = ndarray::concatenate(Axis(1), &[prompt_v_cpu.view(), cpu_new_v.view()])?;

    let cpu_output = cpu_attn.attend(
        &query_cpu,
        &full_k_cpu,
        &full_v_cpu,
        Some(&mask_cpu),
        true,
        prompt_len,
        None,
    )?;

    let mut gpu_cache = GpuKVCache::new(&context, 1, b, n, head_dim, max_len)?;
    let gpu_slicer = GpuSlice::new(&context);
    let pool_guard = self.pool.lock().await;
    let mut frame = GpuFrameContext::new(&self.context, pool_guard);
    let (prompt_k_gpu, prompt_v_gpu) =
        gpu_attn.project_kv(&mut encoder, &prompt_gpu, &gpu_weights, &mut temp);
    gpu_cache.update(&mut encoder, 0, &prompt_k_gpu, &prompt_v_gpu)?;
    gpu_cache.increment_len(prompt_len);

    let (gpu_new_k, gpu_new_v) =
        gpu_attn.project_kv(&mut encoder, &query_gpu, &gpu_weights, &mut temp);
    gpu_cache.update(&mut encoder, 0, &gpu_new_k, &gpu_new_v)?;

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
        true, // Causal
        (&cache_k_view, &cache_v_view),
        prompt_len,
        &mut temp,
    );
    frame.finish();

    assert_tensors_are_close(&cpu_output, &gpu_output, "Decoder Output", 1e-4).await;
    assert_tensors_are_close(&cpu_new_k, &gpu_new_k, "Decoder New K", 1e-4).await;
    assert_tensors_are_close(&cpu_new_v, &gpu_new_v, "Decoder New V", 1e-4).await;

    println!("✅ GpuAttention passed decoder generation parity test!");
    Ok(())
}
