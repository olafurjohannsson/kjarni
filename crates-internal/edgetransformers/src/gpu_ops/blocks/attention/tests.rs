use crate::attention::MultiHeadAttention;
use crate::cache::{Cache, CpuKVCache, GpuKVCache};
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::cache::GpuUpdateCache;
use crate::gpu_ops::primitives::layout::slice::GpuSlice;
use crate::gpu_ops::{
    DType, GpuTensor,
    blocks::attention::attention::{GpuAttention, GpuAttentionWeights, TempStorage},
};
use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3, Array4, Axis, Ix3, Ix4};
use std::sync::Arc;

// Helper to read a GpuTensor back to the CPU for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let rank = tensor.rank();
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?.into_dimensionality::<D>()?)
}

/// A crucial helper function to compare CPU and GPU tensors with a tolerance.
/// GPU floating point math is not always bit-for-bit identical to CPU math.
async fn assert_tensors_are_close(
    cpu_tensor: &Array3<f32>,
    gpu_tensor: &GpuTensor,
    label: &str,
    tolerance: f32,
) {
    let gpu_as_cpu = read_gpu_tensor::<Ix3>(gpu_tensor).await.unwrap();
    let close = cpu_tensor
        .iter()
        .zip(gpu_as_cpu.iter())
        .all(|(a, b)| (a - b).abs() < tolerance);

    if !close {
        println!("Mismatch in tensor '{}'", label);
        println!("CPU tensor: \n{:?}", cpu_tensor);
        println!("GPU tensor: \n{:?}", gpu_as_cpu);
        panic!(
            "Tensor '{}' is not close enough to its GPU counterpart.",
            label
        );
    }
}

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

    // 1. Setup CPU and GPU versions with identical weights
    let (cpu_attn, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b) = create_cpu_attention(h, n);
    let (gpu_attn, gpu_weights) = create_gpu_attention(
        &context, h as u32, n as u32, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b,
    );

    // 2. Create identical inputs
    let input_cpu = Array3::from_shape_fn((b, s, h), |(i, j, k)| (i + j + k) as f32 * 0.1);
    let mask_cpu = Array2::from_shape_vec((b, s), vec![1.0, 1.0, 1.0, 0.0])?;
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

    // 3. Run CPU forward pass
    let (cpu_output, cpu_new_k, cpu_new_v) =
        cpu_attn.forward_with_cache(&input_cpu, None, Some(&mask_cpu), false, None, None)?;

    // 4. Run GPU forward pass
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut temp = TempStorage::new(context.clone());
    let (gpu_output, gpu_new_k, gpu_new_v) = gpu_attn.forward(
        &mut encoder,
        &input_gpu,
        &input_gpu,
        &gpu_weights,
        &mask_gpu,
        false,
        None,
        0,
        &mut temp,
    );
    context.queue.submit(Some(encoder.finish()));
    temp.reclaim();

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
    let (b, h, n) = (1, 16, 4); // Batch, HiddenSize, NumHeads
    let head_dim = h / n;
    let prompt_len = 3;
    let gen_len = 1; // In decoding, we always process one token at a time
    let total_len = prompt_len + gen_len;
    let max_len = 8; // The total capacity of our cache

    // 1. --- SETUP ---
    // Create identical CPU and GPU components.
    let (cpu_attn, q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b) = create_cpu_attention(h, n);
    let (gpu_attn, gpu_weights) = create_gpu_attention(
        &context, h as u32, n as u32, &q_w, &q_b, &k_w, &k_b, &v_w, &v_b, &o_w, &o_b,
    );

    // 2. --- INPUTS ---
    // The prompt that "primes" the cache.
    let prompt_cpu =
        Array3::from_shape_fn((b, prompt_len, h), |(i, j, k)| (i + j + k) as f32 * 0.1);
    // The new token's hidden state for the generation step.
    let query_cpu = Array3::from_shape_fn((b, gen_len, h), |(i, j, k)| {
        (i + j + k + prompt_len) as f32 * 0.1
    });
    // The attention mask must cover the total length (prompt + new token).
    let mask_cpu = Array2::ones((b, total_len));

    let prompt_gpu = GpuTensor::from_ndarray(&context, &prompt_cpu)?;
    let query_gpu = GpuTensor::from_ndarray(&context, &query_cpu)?;
    let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

    // 3. --- CPU GROUND TRUTH ---
    // This follows the new, clean, decoupled architecture.

    // Step A: Project the prompt to get its K/V state.
    let (prompt_k_cpu, prompt_v_cpu) = cpu_attn.project_kv(&prompt_cpu);

    // Step B: Project the new query token to get its K/V state.
    // These are the "new_k" and "new_v" we expect as output.
    let (cpu_new_k, cpu_new_v) = cpu_attn.project_kv(&query_cpu);

    // Step C: Manually simulate the cache manager by concatenating the states.
    let full_k_cpu = ndarray::concatenate(Axis(1), &[prompt_k_cpu.view(), cpu_new_k.view()])?;
    let full_v_cpu = ndarray::concatenate(Axis(1), &[prompt_v_cpu.view(), cpu_new_v.view()])?;

    // Step D: Call `attend` with the query and the complete, final cache state.
    let cpu_output = cpu_attn.attend(
        &query_cpu,
        &full_k_cpu,
        &full_v_cpu,
        Some(&mask_cpu),
        true,       // Causal attention is a must for decoders
        prompt_len, // The offset is the length of what was already in the cache
        None,
    )?;

    // 4. --- GPU EXECUTION ---
    // This simulates the "mini-orchestrator" logic of the future GpuTransformerDecoder.

    // We need a GpuKVCache and the GpuUpdateCache kernel.
    // Make sure to `use crate::gpu_ops::blocks::cache::GpuUpdateCache;`
    let mut gpu_cache = GpuKVCache::new(&context, 1, b, n, head_dim, max_len)?;
    let gpu_slicer = GpuSlice::new(&context);
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut temp = TempStorage::new(context.clone());

    // Step A: "Priming" pass - project the prompt and update the cache.
    let (prompt_k_gpu, prompt_v_gpu) =
        gpu_attn.project_kv(&mut encoder, &prompt_gpu, &gpu_weights, &mut temp);
    gpu_cache.update(&mut encoder, 0, &prompt_k_gpu, &prompt_v_gpu)?;
    gpu_cache.increment_len(prompt_len);

    // Step B: "Generation" pass - project the new query token.
    let (gpu_new_k, gpu_new_v) =
        gpu_attn.project_kv(&mut encoder, &query_gpu, &gpu_weights, &mut temp);
    // Update the cache *again* at the new offset.
    gpu_cache.update(&mut encoder, 0, &gpu_new_k, &gpu_new_v)?;

    // Step C: Get the full cache tensor. This will have shape [B, H, max_len, D].
    let (full_cache_k_gpu, full_cache_v_gpu) = gpu_cache.get(0).unwrap();

    // CRITICAL: We need a view/slice of the cache that only contains the valid data.
    // This assumes your GpuTensor has a `slice` method that performs a GPU-side copy or view.
    // Let's assume for this test that we can create a view.
    let cache_k_view = full_cache_k_gpu.slice(
        &mut encoder,
        &gpu_slicer,
        &[0, 0, 0, 0],                // offset
        &[b, n, total_len, head_dim], // shape
    )?;
    let cache_v_view = full_cache_v_gpu.slice(
        &mut encoder,
        &gpu_slicer,
        &[0, 0, 0, 0],
        &[b, n, total_len, head_dim],
    )?;

    // Note: A real `slice` operation might be needed here if `view` cannot handle this.
    // For now, `view` might panic if `max_len` != `total_len`. This exposes the need for `slice`.

    // Step D: Call `attend` with the query and the correctly-sized view of the cache.
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
    context.queue.submit(Some(encoder.finish()));
    temp.reclaim();

    // 5. --- COMPARE RESULTS ---
    assert_tensors_are_close(&cpu_output, &gpu_output, "Decoder Output", 1e-4).await;
    assert_tensors_are_close(&cpu_new_k, &gpu_new_k, "Decoder New K", 1e-4).await;
    assert_tensors_are_close(&cpu_new_v, &gpu_new_v, "Decoder New V", 1e-4).await;

    println!("✅ GpuAttention passed decoder generation parity test!");
    Ok(())
}
