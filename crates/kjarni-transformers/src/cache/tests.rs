use super::{Cache, CpuKVCache, GpuKVCache};
use crate::cache::{CpuBeamKVCache, GpuBeamKVCache};
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::{GpuTensor, blocks::cache::GpuUpdateCache}; // Ensure GpuUpdateCache is public
use anyhow::Result;
use ndarray::{Array, Array1, Array3, Array4, Ix3, Ix4, s};
use ndarray_rand::rand_distr::Uniform;
use std::sync::Arc;
use ndarray_rand::RandomExt;

// Helper to read a GPU tensor back to a generic ndarray for comparison.
async fn read_gpu_tensor<D: ndarray::Dimension>(tensor: &GpuTensor) -> Result<Array<f32, D>> {
    let shape = tensor.shape().to_vec();
    let raw_data = tensor.read_raw_data().await?;
    let data_slice: &[f32] = bytemuck::cast_slice(&raw_data);
    Ok(Array::from_shape_vec(shape, data_slice.to_vec())?
        .into_dimensionality::<D>()
        .unwrap())
}
fn assert_all_close_3d(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
    if a.shape() != b.shape() {
        panic!("[{}] Shape mismatch: {:?} vs {:?}", context, a.shape(), b.shape());
    }
    let mut max_diff = 0.0;
    for (a_val, b_val) in a.iter().zip(b.iter()) {
        let diff = (a_val - b_val).abs();
        if diff > max_diff {
            max_diff = diff;
        }
        let tolerance = atol + rtol * b_val.abs();
        if diff > tolerance {
            panic!(
                "[{}] Arrays not close. Max diff {} > tolerance {} at values a={}, b={}",
                context, diff, tolerance, a_val, b_val
            );
        }
    }
    println!("[{}] Check passed. Max absolute difference: {:.6e}", context, max_diff);
}
#[tokio::test]
async fn test_cache_symmetry() -> Result<()> {
    println!("\n--- Testing CPU/GPU KV Cache Symmetry ---");
    let context = WgpuContext::new().await?;

    // 1. SETUP: Define shared parameters and instantiate both caches.
    let (num_layers, batch_size, max_len, hidden_size) = (1, 1, 8, 16);
    let layer_idx = 0;

    let mut cpu_cache = CpuKVCache::new(num_layers, batch_size, max_len, hidden_size);

    // GPU cache operates on head-split data.
    let num_heads = 4;
    let head_dim = hidden_size / num_heads;
    let mut gpu_cache = GpuKVCache::new(
        &context, num_layers, batch_size, num_heads, head_dim, max_len,
    )?;

    // 2. SIMULATE STEP 1 (Prompt processing, seq_len = 3)
    let prompt_len = 3;
    let new_k_cpu_1 =
        Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
            (b * 100 + s * 10 + h) as f32
        });
    let new_v_cpu_1 =
        Array3::<f32>::from_shape_fn((batch_size, prompt_len, hidden_size), |(b, s, h)| {
            (b * 100 + s * 10 + h) as f32 * 10.0
        });

    // Update CPU Cache
    cpu_cache.update(layer_idx, &new_k_cpu_1, &new_v_cpu_1)?;
    cpu_cache.increment_len(prompt_len);

    // Update GPU Cache
    let new_k_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;
    let new_v_gpu_1 = GpuTensor::from_ndarray(&context, &new_v_cpu_1)?;
    let mut encoder1 = context.device.create_command_encoder(&Default::default());
    gpu_cache.update(&mut encoder1, layer_idx, &new_k_gpu_1, &new_v_gpu_1, gpu_cache.get_seq_length())?;
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
    
    gpu_cache.update(&mut encoder2, layer_idx, &new_k_gpu_2, &new_v_gpu_2, gpu_cache.get_seq_length())?;
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

    Ok(())
}

#[test]
fn test_cache_initialization() {
    let cache = CpuKVCache::new(16, 1, 100, 512);

    assert_eq!(cache.get_seq_length(), 0);
    assert_eq!(cache.layers().len(), 16);

    for i in 0..16 {
        let (k, v) = cache.get(i).unwrap();
        assert_eq!(k.shape(), &[1, 0, 512]);
        assert_eq!(v.shape(), &[1, 0, 512]);
    }

    println!("✓ Cache initialization test passed");
}

#[test]
fn test_cache_update_and_grow() {
    use ndarray::Array3;

    let mut cache = CpuKVCache::new(2, 1, 100, 512);

    // First update - add 11 tokens
    let k1 = Array3::ones((1, 11, 512));
    let v1 = Array3::ones((1, 11, 512));

    cache.update(0, &k1, &v1).unwrap();
    cache.increment_len(11);

    assert_eq!(cache.get_seq_length(), 11);
    let (cached_k, cached_v) = cache.get(0).unwrap();
    assert_eq!(cached_k.shape(), &[1, 11, 512]);
    assert_eq!(cached_v.shape(), &[1, 11, 512]);

    // Second update - add 1 more token
    let k2 = Array3::ones((1, 1, 512)) * 2.0;
    let v2 = Array3::ones((1, 1, 512)) * 2.0;

    cache.update(0, &k2, &v2).unwrap();
    cache.increment_len(1);

    assert_eq!(cache.get_seq_length(), 12);
    let (cached_k, cached_v) = cache.get(0).unwrap();
    assert_eq!(cached_k.shape(), &[1, 12, 512]);
    assert_eq!(cached_v.shape(), &[1, 12, 512]);

    // Check values are correct
    assert_eq!(cached_k[[0, 0, 0]], 1.0); // First token
    assert_eq!(cached_k[[0, 11, 0]], 2.0); // Last token

    println!("✓ Cache update and grow test passed");
}

#[test]
fn test_cache_multiple_layers() {
    use ndarray::Array3;

    let mut cache = CpuKVCache::new(3, 1, 100, 512);

    // Update each layer with different values
    for layer in 0..3 {
        let k = Array3::ones((1, 5, 512)) * (layer as f32 + 1.0);
        let v = Array3::ones((1, 5, 512)) * (layer as f32 + 1.0);
        cache.update(layer, &k, &v).unwrap();
    }
    cache.increment_len(5);

    // Verify each layer has correct values
    for layer in 0..3 {
        let (k, v) = cache.get(layer).unwrap();
        assert_eq!(k.shape(), &[1, 5, 512]);
        assert_eq!(k[[0, 0, 0]], (layer as f32 + 1.0));
        assert_eq!(v[[0, 0, 0]], (layer as f32 + 1.0));
    }

    println!("✓ Cache multiple layers test passed");
}
#[tokio::test]
async fn test_gpu_kv_cache_update_and_readback() -> anyhow::Result<()> {
    // --- 1. Arrange ---
    let context = WgpuContext::new().await?;
    let num_layers = 2;
    let batch_size = 1;
    let num_heads = 4;
    let head_dim = 32;
    let capacity = 16; // Max sequence length

    let mut cache = GpuKVCache::new(
        &context, num_layers, batch_size, num_heads, head_dim, capacity,
    )?;

    let new_seq_len = 3;
    let position_offset = 5; // Write into the middle of the cache
    let layer_idx_to_test = 1;

    // Create dummy data on the CPU
    let new_k_cpu = Array::random(
        (batch_size, new_seq_len, num_heads * head_dim),
        Uniform::new(-1.0, 1.0),
    );
    let new_v_cpu = Array::random(
        (batch_size, new_seq_len, num_heads * head_dim),
        Uniform::new(-1.0, 1.0),
    );

    // Upload to GPU
    let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;
    let new_v_gpu = GpuTensor::from_ndarray(&context, &new_v_cpu)?;
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

    // --- 2. Act ---
    // Update the cache at the specified offset
    cache.update(&mut encoder, layer_idx_to_test, &new_k_gpu, &new_v_gpu, position_offset)?;
    context.queue.submit(Some(encoder.finish()));

    // Get the full physical buffer and download it
    let (k_cache_gpu, _) = cache.get(layer_idx_to_test).unwrap();
    let k_cache_cpu_result: Array4<f32> = k_cache_gpu.to_ndarray_4d().await?;

    // --- 3. Assert ---
    // Reshape the original CPU data to match the cache layout for comparison
    let new_k_cpu_reshaped = new_k_cpu
        .into_shape((batch_size, new_seq_len, num_heads, head_dim))?
        .permuted_axes([0, 2, 1, 3]);

    // Check that the data was written to the correct slice
    let updated_slice = k_cache_cpu_result.slice(s![
        ..,
        ..,
        position_offset..position_offset + new_seq_len,
        ..
    ]);
    assert_eq!(updated_slice, new_k_cpu_reshaped.as_standard_layout());

    // Check that the data BEFORE the slice is still zero
    let prefix_slice = k_cache_cpu_result.slice(s![.., .., 0..position_offset, ..]);
    assert!(prefix_slice.iter().all(|&x| x == 0.0));

    println!("✓ GPU cache update and readback test passed.");
    Ok(())
}

#[tokio::test]
async fn test_gpu_cache_stateful_update_simulation() -> Result<()> {
    println!("\n--- Testing GPU Cache State Management Across Steps ---");
    // --- 1. Arrange ---
    let context = WgpuContext::new().await?;
    let (num_layers, batch_size, num_heads, head_dim, capacity) = (1, 1, 2, 4, 10);
    let layer_idx = 0;

    // This is the cache object that persists across the generation loop.
    let mut gpu_cache = GpuKVCache::new(
        &context, num_layers, batch_size, num_heads, head_dim, capacity,
    )?;

    // --- 2. Act: Step 1 (e.g., Processing a 3-token prompt) ---
    {
        println!("--- Simulating Step 1 (Prompt Processing) ---");
        let prompt_len = 3;
        let new_k_cpu_1 = Array3::from_elem((batch_size, prompt_len, num_heads * head_dim), 1.0);
        let new_k_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?;
        let new_v_gpu_1 = GpuTensor::from_ndarray(&context, &new_k_cpu_1)?; // Using same data for simplicity

        // The logic from the start of a `forward` call
        let position_offset = gpu_cache.get_seq_length();
        assert_eq!(position_offset, 0, "Initial position_offset should be 0");

        // The logic from inside the `forward` loop
        let mut encoder = context.device.create_command_encoder(&Default::default());
        gpu_cache.update(&mut encoder, layer_idx, &new_k_gpu_1, &new_v_gpu_1, position_offset)?;
        context.queue.submit(Some(encoder.finish()));

        // The logic from the end of a `forward` call
        gpu_cache.set_seq_length(position_offset + prompt_len);
        assert_eq!(gpu_cache.get_seq_length(), 3, "Length after step 1 should be 3");
    }

    // --- 3. Act: Step 2 (e.g., Generating 1 new token) ---
    {
        println!("--- Simulating Step 2 (Token Generation) ---");
        let gen_len = 1;
        let new_k_cpu_2 = Array3::from_elem((batch_size, gen_len, num_heads * head_dim), 99.0);
        let new_k_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;
        let new_v_gpu_2 = GpuTensor::from_ndarray(&context, &new_k_cpu_2)?;

        // Logic from the start of the *next* `forward` call
        let position_offset = gpu_cache.get_seq_length();
        assert_eq!(position_offset, 3, "Position_offset for step 2 should be 3");

        // Logic from inside the `forward` loop
        let mut encoder = context.device.create_command_encoder(&Default::default());
        gpu_cache.update(&mut encoder, layer_idx, &new_k_gpu_2, &new_v_gpu_2, position_offset)?;
        context.queue.submit(Some(encoder.finish()));

        // Logic from the end of the `forward` call
        gpu_cache.set_seq_length(position_offset + gen_len);
        assert_eq!(gpu_cache.get_seq_length(), 4, "Final length should be 4");
    }

    // --- 4. Assert ---
    println!("--- Verifying Final Cache State ---");
    let (k_cache_gpu, _) = gpu_cache.get(layer_idx).unwrap();
    let k_cache_cpu: Array4<f32> = read_gpu_tensor(&k_cache_gpu).await?;

    // Check the data from the first update
    let slice1 = k_cache_cpu.slice(s![0, 0, 0..3, ..]);
    assert!(slice1.iter().all(|&x| x == 1.0), "Data from step 1 is incorrect or was overwritten");

    // CRITICAL: Check the data from the second update
    let slice2 = k_cache_cpu.slice(s![0, 0, 3..4, ..]);
    assert!(slice2.iter().all(|&x| x == 99.0), "Data from step 2 was not written to the correct offset");
    
    println!("✓ GPU cache stateful update test passed.");
    Ok(())
}

#[tokio::test]
async fn test_cache_reorder_parity() -> Result<()> {
    println!("\n=== Testing CPU vs. GPU Cache Reorder Parity ===\n");

    // 1. SETUP: Define common parameters
    const NUM_LAYERS: usize = 2;
    const NUM_BEAMS: usize = 4;
    const NUM_HEADS: usize = 8;
    const HEAD_DIM: usize = 64;
    const HIDDEN_SIZE: usize = NUM_HEADS * HEAD_DIM; // 512
    const CAPACITY: usize = 10;
    const NUM_STEPS_TO_POPULATE: usize = 3;

    let context = WgpuContext::new().await?;

    // 2. CREATE CACHES: Instantiate both CPU and GPU caches with identical configs.
    let mut cpu_cache =
        CpuBeamKVCache::new(NUM_LAYERS, NUM_BEAMS, CAPACITY, HIDDEN_SIZE);
    let mut gpu_cache = GpuBeamKVCache::new(
        &context, NUM_LAYERS, NUM_BEAMS, NUM_HEADS, HEAD_DIM, CAPACITY,
    )?;

    // 3. POPULATE CACHES: Fill both caches with the exact same data for a few steps.
    println!("Populating caches for {} steps...", NUM_STEPS_TO_POPULATE);
    for step in 0..NUM_STEPS_TO_POPULATE {
        // Create unique, deterministic data for this step
        let new_k_cpu = Array3::from_shape_fn((NUM_BEAMS, 1, HIDDEN_SIZE), |(b, _, h)| {
            (step as f32 * 1000.0) + (b as f32 * 100.0) + h as f32
        });
        let new_v_cpu = &new_k_cpu + 5000.0;

        // Update CPU cache
        for layer in 0..NUM_LAYERS {
            cpu_cache.update(layer, &new_k_cpu, &new_v_cpu)?;
        }
        cpu_cache.increment_len(1);

        // Update GPU cache
        let new_k_gpu = GpuTensor::from_ndarray(&context, &new_k_cpu)?;
        let new_v_gpu = GpuTensor::from_ndarray(&context, &new_v_cpu)?;
        let mut encoder = context.device.create_command_encoder(&Default::default());
        for layer in 0..NUM_LAYERS {
            gpu_cache.update(&mut encoder, layer, &new_k_gpu, &new_v_gpu)?;
        }
        context.queue.submit(Some(encoder.finish()));
        gpu_cache.increment_len(1);
    }
    assert_eq!(cpu_cache.get_seq_length(), NUM_STEPS_TO_POPULATE);
    assert_eq!(gpu_cache.get_seq_length(), NUM_STEPS_TO_POPULATE);
    println!("Caches populated successfully.");

    // 4. REORDER: Apply the same reordering indices to both caches.
    let reorder_indices = vec![2, 0, 2, 1]; // A non-trivial reorder
    println!("\nReordering with indices: {:?}", reorder_indices);

    // Reorder CPU
    cpu_cache.reorder(&reorder_indices);

    // Reorder GPU
    let indices_gpu = GpuTensor::from_ndarray(
        &context,
        &Array1::from_vec(reorder_indices.iter().map(|&i| i as u32).collect()),
    )?;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    gpu_cache.reorder(&mut encoder, &indices_gpu);
    context.queue.submit(Some(encoder.finish()));
    println!("Reordering complete.");

    // 5. VERIFY: Download the GPU cache state and compare it to the CPU state.
    println!("\nVerifying parity post-reorder...");
    for layer_idx in 0..NUM_LAYERS {
        // Get CPU data (it's already in [beam, seq, hidden] format)
        let (cpu_k, cpu_v) = cpu_cache.get(layer_idx).unwrap();

        // Get GPU data and bring it to the CPU
        let (gpu_k_tensor, gpu_v_tensor) = gpu_cache.get_layer_tensors(layer_idx).unwrap();
        let gpu_k_4d: Array4<f32> = gpu_k_tensor.to_ndarray_4d().await?;
        let gpu_v_4d: Array4<f32> = gpu_v_tensor.to_ndarray_4d().await?;

        let p1 = gpu_k_4d.permuted_axes([0, 2, 1, 3]);
        let p2 = gpu_v_4d.permuted_axes([0, 2, 1, 3]);
        // CRITICAL: Reshape GPU data from [beam, head, seq, dim] to [beam, seq, hidden] to match CPU layout
        let gpu_k_reshaped = p1
            .as_standard_layout()
            .into_shape((NUM_BEAMS, CAPACITY, HIDDEN_SIZE))?;
        let gpu_v_reshaped = p2
            .as_standard_layout()
            .into_shape((NUM_BEAMS, CAPACITY, HIDDEN_SIZE))?;

        // Compare only the active parts of the cache
        let active_slice = s![.., 0..NUM_STEPS_TO_POPULATE, ..];
        let cpu_k_active = cpu_k.slice(active_slice);
        let gpu_k_active = gpu_k_reshaped.slice(active_slice);
        let cpu_v_active = cpu_v.slice(active_slice);
        let gpu_v_active = gpu_v_reshaped.slice(active_slice);

        let context_k = format!("Layer {} Key Cache", layer_idx);
        assert_all_close_3d(&cpu_k_active.to_owned(), &gpu_k_active.to_owned(), 1e-5, 1e-5, &context_k);

        let context_v = format!("Layer {} Value Cache", layer_idx);
        assert_all_close_3d(&cpu_v_active.to_owned(), &gpu_v_active.to_owned(), 1e-5, 1e-5, &context_v);
    }

    println!("\n✅✅✅ CPU and GPU cache reordering implementations have perfect parity!");
    Ok(())
}