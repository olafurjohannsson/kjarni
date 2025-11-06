use super::{CpuKVCache, GpuKVCache, Cache};
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{blocks::cache::GpuUpdateCache, GpuTensor}; // Ensure GpuUpdateCache is public
use anyhow::Result;
use ndarray::{s, Array, Array3, Array4, Ix3, Ix4};
use std::sync::Arc;

// Helper to read a GPU tensor back to a generic ndarray for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
        .into_dimensionality::<D>()
        .unwrap())
}

#[tokio::test]
async fn test_cache_symmetry() -> Result<()> {
    println!("\n--- Testing CPU/GPU KV Cache Symmetry ---");
    let context = Arc::new(WgpuContext::new().await);

    // 1. SETUP: Define shared parameters and instantiate both caches.
    let (num_layers, batch_size, max_len, hidden_size) = (1, 1, 8, 16);
    let layer_idx = 0;

    let mut cpu_cache = CpuKVCache::new(num_layers, batch_size, max_len, hidden_size);

    // GPU cache operates on head-split data.
    let num_heads = 4;
    let head_dim = hidden_size / num_heads;
    let mut gpu_cache = GpuKVCache::new(
        &context,
        num_layers,
        batch_size,
        num_heads,
        head_dim,
        max_len,
    )?;

    // 2. SIMULATE STEP 1 (Prompt processing, seq_len = 3)
    let prompt_len = 3;
    let new_k_cpu_1 = Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
        (b * 100 + s * 10 + h) as f32
    });
    let new_v_cpu_1 = Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
        (b * 100 + s * 10 + h) as f32 * 10.0
    });

    // Update CPU Cache
    cpu_cache.update(layer_idx, &new_k_cpu_1, &new_v_cpu_1)?;
    cpu_cache.increment_len(prompt_len);

    // Update GPU Cache
    let new_k_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;
    let new_v_gpu_1 = GpuTensor::from_ndarray(&context, &new_v_cpu_1)?;
    let mut encoder1 = context.device.create_command_encoder(&Default::default());
    gpu_cache.update(&mut encoder1, layer_idx, &new_k_gpu_1, &new_v_gpu_1)?;
    context.queue.submit(Some(encoder1.finish()));
    gpu_cache.increment_len(prompt_len);

    // 3. SIMULATE STEP 2 (Token generation, seq_len = 1)
    let gen_len = 1;
    let new_k_cpu_2 =
        Array3::<f32>::from_shape_fn((batch_size, gen_len, hidden_size), |(b, s, h)| {
            (b * 100 + (s + prompt_len) * 10 + h) as f32
        });
    let new_v_cpu_2 =
        Array3::<f32>::from_shape_fn((batch_size, gen_len, hidden_size), |(b, s, h)| {
            (b * 100 + (s + prompt_len) * 10 + h) as f32 * 10.0
        });

    // Update CPU Cache
    cpu_cache.update(layer_idx, &new_k_cpu_2, &new_v_cpu_2)?;
    cpu_cache.increment_len(gen_len);

    // Update GPU Cache
    let new_k_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;
    let new_v_gpu_2 = GpuTensor::from_ndarray(&context, &new_v_cpu_2)?;
    let mut encoder2 = context.device.create_command_encoder(&Default::default());
    gpu_cache.update(&mut encoder2, layer_idx, &new_k_gpu_2, &new_v_gpu_2)?;
    context.queue.submit(Some(encoder2.finish()));
    gpu_cache.increment_len(gen_len);

    // 4. ASSERT: The final state of both caches must be identical.
    let final_len = prompt_len + gen_len;
    assert_eq!(cpu_cache.get_seq_length(), final_len);
    assert_eq!(gpu_cache.get_seq_length(), final_len);

    // Get the final content of both caches
    let (cpu_k_final_view, cpu_v_final_view) = cpu_cache.get(layer_idx).unwrap();
    // THE FIX: `get` now returns a GpuTensor representing the *full* buffer.
    let (gpu_k_full_buffer, gpu_v_full_buffer) = gpu_cache.get(layer_idx).unwrap();

    // Download the full GPU buffers
    let gpu_k_full_cpu: Array4<f32> = read_gpu_tensor(&gpu_k_full_buffer).await?;
    let gpu_v_full_cpu: Array4<f32> = read_gpu_tensor(&gpu_v_full_buffer).await?;

    // THE FIX: Slice the downloaded GPU data to the active length for comparison.
    let gpu_k_active_view = gpu_k_full_cpu.slice(s![.., .., 0..final_len, ..]);
    let gpu_v_active_view = gpu_v_full_cpu.slice(s![.., .., 0..final_len, ..]);
    
    // CPU data needs to be reshaped to match the GPU's head-split layout for comparison.
    let cpu_k_reshaped = cpu_k_final_view
        .to_owned()
        .into_shape((batch_size, final_len, num_heads, head_dim))?
        .permuted_axes([0, 2, 1, 3]);
    
    let cpu_v_reshaped = cpu_v_final_view
        .to_owned()
        .into_shape((batch_size, final_len, num_heads, head_dim))?
        .permuted_axes([0, 2, 1, 3]);

    // Compare K caches
    let cpu_k_standard = cpu_k_reshaped.as_standard_layout();
    let gpu_k_standard = gpu_k_active_view.as_standard_layout();

    assert_eq!(
        cpu_k_standard.as_slice(),
        gpu_k_standard.as_slice(),
        "Final K-cache states do not match!"
    );

    // Compare V caches
    let cpu_v_standard = cpu_v_reshaped.as_standard_layout();
    let gpu_v_standard = gpu_v_active_view.as_standard_layout();

    assert_eq!(
        cpu_v_standard.as_slice(),
        gpu_v_standard.as_slice(),
        "Final V-cache states do not match!"
    );

    println!("✅ CPU and GPU KV Caches passed symmetry test!");
    Ok(())
}