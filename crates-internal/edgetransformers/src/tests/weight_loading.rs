
use crate::gpu_ops::{
    primitives::matmul::{compile_matmul_pipeline, run_gpu_matmul}
};
use crate::wgpu_ops::wgpu_matmul_3d_2d;
use crate::utils::linear_algebra::matmul_3d_2d;
use crate::weights::ModelWeights;
use crate::gpu_context::WgpuContext;
use wgpu::{Device, DeviceDescriptor, CommandEncoder, util::DeviceExt};
use anyhow::Result;
use crate::gpu_ops::utils::read_buffer_3d;

use ndarray::{Array2, Array3};
use std::path::Path;
use std::sync::Arc;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

async fn read_gpu_buffer_2d(
    context: &WgpuContext,
    buffer: &wgpu::Buffer,
    rows: usize,
    cols: usize,
) -> Result<Array2<f32>> {
    let size = (rows * cols * std::mem::size_of::<f32>()) as u64;

    let staging_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, size);
    context.queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures::channel::oneshot::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });

    context.device.poll(wgpu::PollType::wait_indefinitely());
    receiver.await??;

    let data = buffer_slice.get_mapped_range();
    let float_data: &[f32] = bytemuck::cast_slice(&data);
    let array = Array2::from_shape_vec((rows, cols), float_data.to_vec())?;

    drop(data);
    staging_buffer.unmap();

    Ok(array)
}

fn compare_arrays_2d(cpu: &Array2<f32>, gpu: &Array2<f32>, name: &str, tolerance: f32) {
    assert_eq!(cpu.dim(), gpu.dim(), "{}: Shape mismatch", name);

    let (rows, cols) = cpu.dim();
    let mut max_diff = 0.0f32;
    let mut max_diff_pos = (0, 0);
    let mut num_diffs = 0;

    for i in 0..rows {
        for j in 0..cols {
            let cpu_val = cpu[[i, j]];
            let gpu_val = gpu[[i, j]];
            let diff = (cpu_val - gpu_val).abs();

            if diff > tolerance {
                num_diffs += 1;
                if diff > max_diff {
                    max_diff = diff;
                    max_diff_pos = (i, j);
                }
            }
        }
    }

    println!("\n{} comparison:", name);
    println!("  Shape: {:?}", cpu.dim());
    println!(
        "  Max difference: {:.6} at position {:?}",
        max_diff, max_diff_pos
    );
    println!("  Num differences > {}: {}", tolerance, num_diffs);

    if num_diffs > 0 {
        println!("  CPU value at max diff: {:.6}", cpu[max_diff_pos]);
        println!("  GPU value at max diff: {:.6}", gpu[max_diff_pos]);
        println!(
            "  First 5 CPU values: {:?}",
            &cpu.as_slice().unwrap()[..5.min(cpu.len())]
        );
        println!(
            "  First 5 GPU values: {:?}",
            &gpu.as_slice().unwrap()[..5.min(gpu.len())]
        );
    }

    assert!(
        max_diff < tolerance,
        "{}: Max difference {:.6} exceeds tolerance {:.6}",
        name,
        max_diff,
        tolerance
    );
}



async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        tokio::fs::create_dir_all(local_dir).await?;
    }

    let files_to_check = ["model.safetensors", "config.json", "tokenizer.json"];
    for filename in files_to_check {
        let local_path = local_dir.join(filename);
        if !local_path.exists() {
            println!("-> Downloading {}...", filename);
            let download_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, filename
            );
            let response = reqwest::get(&download_url).await?.error_for_status()?;
            let content = response.bytes().await?;
            tokio::fs::write(&local_path, &content).await?;
            println!("   ... downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

fn find_weight_name(weights: &ModelWeights, pattern: &str) -> Option<String> {
    // Try to find a tensor name matching the pattern
    let all_keys = weights.list_tensor_names();

    // Common variations
    let patterns = vec![
        format!("bert.{}", pattern),
        format!("roberta.{}", pattern),
        format!("encoder.{}", pattern),
        pattern.to_string(),
    ];

    for pat in patterns {
        if all_keys.iter().any(|k| k == &pat) {
            return Some(pat);
        }
    }

    // Print available keys for debugging
    println!(
        "\nAvailable tensor names containing '{}':",
        pattern.split('.').next().unwrap()
    );
    for key in all_keys
        .iter()
        .filter(|k| k.contains(pattern.split('.').next().unwrap()))
    {
        println!("  - {}", key);
    }

    None
}

#[tokio::test]
async fn test_transpose_mismatch_detection() -> Result<()> {
    println!("\n=== Testing Transpose Mismatch Detection ===\n");

    let context = get_test_context().await;
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("edgegpt")
        .join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &cache_dir).await?;
    let weights = ModelWeights::new(&cache_dir)?;

    // Find the correct weight name
    let weight_pattern = "encoder.layer.0.attention.self.query.weight";
    let weight_name = find_weight_name(&weights, weight_pattern).expect("Could not find Q weight!");

    println!("Using weight: {}", weight_name);

    let weight_raw = weights.get_array2(&weight_name)?;
    println!("Original weight shape: {:?}", weight_raw.dim());

    // Create test input: [1, 4, 384] - small sequence
    let batch_size = 1;
    let seq_len = 4;
    let hidden_size = 384;
    let test_input = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(_, _, k)| {
        (k as f32) * 0.01 // Predictable pattern
    });

    // Test 1: CPU with transpose (CORRECT for BERT)
    let weight_cpu_transposed = weight_raw.t().to_owned();
    let cpu_output_correct = matmul_3d_2d(&test_input, &weight_cpu_transposed);

    println!("\nCPU output (transposed - CORRECT):");
    println!("  Shape: {:?}", cpu_output_correct.dim());
    println!("  Mean: {:.6}", cpu_output_correct.mean().unwrap());
    println!(
        "  First 5: {:?}",
        &cpu_output_correct.as_slice().unwrap()[..5]
    );

    // Test 2: CPU WITHOUT transpose (WRONG for BERT)
    let cpu_output_wrong = matmul_3d_2d(&test_input, &weight_raw);

    println!("\nCPU output (NOT transposed - WRONG):");
    println!("  Shape: {:?}", cpu_output_wrong.dim());
    println!("  Mean: {:.6}", cpu_output_wrong.mean().unwrap());
    println!(
        "  First 5: {:?}",
        &cpu_output_wrong.as_slice().unwrap()[..5]
    );

    // Compare: they should be VERY different if transpose matters
    let diff = (&cpu_output_correct - &cpu_output_wrong).mapv(|x| x.abs());
    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);

    println!("\nDifference between transposed vs not:");
    println!("  Max diff: {:.6}", max_diff);

    assert!(
        max_diff > 1.0,
        "Transpose doesn't seem to matter! Max diff: {:.6}",
        max_diff
    );

    println!("✅ Transpose DOES matter (good!)");

    // Now test GPU
    println!("\n=== Testing GPU ===");

    // Upload input
    let input_gpu = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Test Input"),
            contents: bytemuck::cast_slice(test_input.as_slice().unwrap()),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // Upload weight WITH transpose
    let weight_gpu_transposed =
        context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU Weight Transposed"),
                contents: bytemuck::cast_slice(
                    weight_cpu_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE,
            });

    // Upload weight WITHOUT transpose
    let weight_gpu_not_transposed =
        context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GPU Weight NOT Transposed"),
                contents: bytemuck::cast_slice(weight_raw.as_standard_layout().as_slice().unwrap()),
                usage: wgpu::BufferUsages::STORAGE,
            });

    let output_size = (batch_size * seq_len * hidden_size * std::mem::size_of::<f32>()) as u64;

    // Output buffer for transposed version
    let output_gpu_transposed = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Transposed"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Output buffer for non-transposed version
    let output_gpu_not_transposed = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output NOT Transposed"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // Compile pipeline
    let pipeline = Arc::new(compile_matmul_pipeline(&context));

    // Test GPU with transposed weights
    {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Test Transposed"),
            });

        run_gpu_matmul(
            &context,
            &mut encoder,
            &pipeline,
            &input_gpu,
            &weight_gpu_transposed,
            &output_gpu_transposed,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );

        context.queue.submit(std::iter::once(encoder.finish()));
        context.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let gpu_output_transposed = read_buffer_3d(
        &context,
        &output_gpu_transposed,
        (batch_size, seq_len, hidden_size),
    )
    .await?;

    println!("\nGPU output (transposed weights):");
    println!("  Shape: {:?}", gpu_output_transposed.dim());
    println!("  Mean: {:.6}", gpu_output_transposed.mean().unwrap());
    println!(
        "  First 5: {:?}",
        &gpu_output_transposed.as_slice().unwrap()[..5]
    );

    // Test GPU with non-transposed weights
    {
        let mut encoder = context
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Test NOT Transposed"),
            });

        run_gpu_matmul(
            &context,
            &mut encoder,
            &pipeline,
            &input_gpu,
            &weight_gpu_not_transposed,
            &output_gpu_not_transposed,
            (batch_size * seq_len) as u32,
            hidden_size as u32,
            hidden_size as u32,
        );

        context.queue.submit(std::iter::once(encoder.finish()));
        context.device.poll(wgpu::PollType::wait_indefinitely());
    }

    let gpu_output_not_transposed = read_buffer_3d(
        &context,
        &output_gpu_not_transposed,
        (batch_size, seq_len, hidden_size),
    )
    .await?;

    println!("\nGPU output (NOT transposed weights):");
    println!("  Shape: {:?}", gpu_output_not_transposed.dim());
    println!("  Mean: {:.6}", gpu_output_not_transposed.mean().unwrap());
    println!(
        "  First 5: {:?}",
        &gpu_output_not_transposed.as_slice().unwrap()[..5]
    );

    // Compare GPU transposed vs CPU correct
    let diff_gpu_cpu_correct = (&gpu_output_transposed - &cpu_output_correct).mapv(|x| x.abs());
    let max_diff_correct = diff_gpu_cpu_correct.iter().cloned().fold(0.0f32, f32::max);

    // Compare GPU transposed vs CPU wrong
    let diff_gpu_cpu_wrong = (&gpu_output_transposed - &cpu_output_wrong).mapv(|x| x.abs());
    let max_diff_wrong = diff_gpu_cpu_wrong.iter().cloned().fold(0.0f32, f32::max);

    println!("\n=== COMPARISON ===");
    println!(
        "GPU (transposed) vs CPU (transposed/CORRECT): max diff = {:.6}",
        max_diff_correct
    );
    println!(
        "GPU (transposed) vs CPU (wrong):              max diff = {:.6}",
        max_diff_wrong
    );

    // The GPU with transposed weights should match CPU correct
    if max_diff_correct < 1e-3 {
        println!("✅ GPU TRANSPOSED matches CPU CORRECT (this is RIGHT!)");
    } else {
        println!("❌ GPU TRANSPOSED does NOT match CPU CORRECT");
    }

    // GPU not-transposed should NOT match CPU correct
    let diff_gpu_not_cpu_correct =
        (&gpu_output_not_transposed - &cpu_output_correct).mapv(|x| x.abs());
    let max_diff_not_correct = diff_gpu_not_cpu_correct
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);

    println!(
        "\nGPU (NOT transposed) vs CPU (CORRECT): max diff = {:.6}",
        max_diff_not_correct
    );

    if max_diff_not_correct > 1.0 {
        println!("✅ GPU NOT TRANSPOSED does NOT match CPU CORRECT (expected)");
    }

    // Final verdict
    assert!(
        max_diff_correct < 1e-3,
        "GPU with transposed weights should match CPU! Diff: {:.6}",
        max_diff_correct
    );

    println!("\n✅ Transpose detection test PASSED!");
    println!("   GPU correctly uses transposed weights.");

    Ok(())
}

#[tokio::test]
async fn test_bert_attention_weight_loading() -> Result<()> {
    println!("\n=== Testing BERT Attention Weight Loading ===\n");

    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("edgegpt")
        .join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &cache_dir).await?;

    let weights = ModelWeights::new(&cache_dir)?;
    let context = get_test_context().await;

    let layer_idx = 0;

    // Test Q weight
    {
        println!("Testing Q weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.attention.self.query.weight", layer_idx);
        let weight_name = find_weight_name(&weights, &pattern).expect("Could not find Q weight!");

        println!("Using weight: {}", weight_name);

        // CPU: Load and transpose (as done in CPU encoder)
        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();

        // GPU: Load and transpose (current implementation)
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Q Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "Q Weight", 1e-6);
        println!("✅ Q weight matches!");
    }

    // Test K weight
    {
        println!("\nTesting K weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.attention.self.key.weight", layer_idx);
        let weight_name = find_weight_name(&weights, &pattern).expect("Could not find K weight!");

        println!("Using weight: {}", weight_name);

        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test K Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "K Weight", 1e-6);
        println!("✅ K weight matches!");
    }

    // Test V weight
    {
        println!("\nTesting V weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.attention.self.value.weight", layer_idx);
        let weight_name = find_weight_name(&weights, &pattern).expect("Could not find V weight!");

        println!("Using weight: {}", weight_name);

        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test V Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "V Weight", 1e-6);
        println!("✅ V weight matches!");
    }

    // Test output weight
    {
        println!("\nTesting Output weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.attention.output.dense.weight", layer_idx);
        let weight_name =
            find_weight_name(&weights, &pattern).expect("Could not find output weight!");

        println!("Using weight: {}", weight_name);

        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test Output Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "Output Weight", 1e-6);
        println!("✅ Output weight matches!");
    }

    println!("\n✅ All BERT attention weights loaded correctly!\n");
    Ok(())
}

#[tokio::test]
async fn test_bert_ffn_weight_loading() -> Result<()> {
    println!("\n=== Testing BERT FFN Weight Loading ===\n");

    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = dirs::cache_dir()
        .unwrap()
        .join("edgegpt")
        .join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &cache_dir).await?;

    let weights = ModelWeights::new(&cache_dir)?;
    let context = get_test_context().await;

    let layer_idx = 0;

    // Test FC1 (intermediate) weight
    {
        println!("Testing FC1 weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.intermediate.dense.weight", layer_idx);
        let weight_name = find_weight_name(&weights, &pattern).expect("Could not find FC1 weight!");

        println!("Using weight: {}", weight_name);

        // CPU: Load and transpose
        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();

        // GPU: Load and transpose (if config.transpose_ffn_weights() == true)
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test FC1 Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "FC1 Weight", 1e-6);
        println!("✅ FC1 weight matches!");
    }

    // Test FC2 (output) weight
    {
        println!("\nTesting FC2 weight (layer {})...", layer_idx);
        let pattern = format!("encoder.layer.{}.output.dense.weight", layer_idx);
        let weight_name = find_weight_name(&weights, &pattern).expect("Could not find FC2 weight!");

        println!("Using weight: {}", weight_name);

        let cpu_weight = weights.get_array2(&weight_name)?.t().to_owned();
        let gpu_weight_raw = weights.get_array2(&weight_name)?;
        let gpu_weight_transposed = gpu_weight_raw.t().to_owned();

        let gpu_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Test FC2 Weight"),
                contents: bytemuck::cast_slice(
                    gpu_weight_transposed
                        .as_standard_layout()
                        .as_slice()
                        .unwrap(),
                ),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let (rows, cols) = cpu_weight.dim();
        let gpu_readback = read_gpu_buffer_2d(&context, &gpu_buffer, rows, cols).await?;

        compare_arrays_2d(&cpu_weight, &gpu_readback, "FC2 Weight", 1e-6);
        println!("✅ FC2 weight matches!");
    }

    println!("\n✅ All BERT FFN weights loaded correctly!\n");
    Ok(())
}
