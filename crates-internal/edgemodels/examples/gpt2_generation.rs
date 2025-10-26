use anyhow::{Result, anyhow};
use edgemodels::gpt2::GPT2Model;
use edgetransformers::prelude::Device;
use edgetransformers::wgpu_context::WgpuContext;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

const MAX_NEW_TOKENS: usize = 50;

/// Helper function to ensure model files are available
async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        println!("-> Creating cache directory: {:?}", local_dir);
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

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== GPU vs CPU Comparison Test ===\n");

    // --- 1. Setup Model Files ---
    let model_repo = "distilgpt2";
    
    let cache_dir = dirs::cache_dir()
        .ok_or_else(|| anyhow!("Failed to get cache directory"))?
        .join("edgegpt");

    let model_dir = cache_dir.join(model_repo.replace('/', "_"));

    ensure_model_files(model_repo, &model_dir).await?;
    println!("Model files are available in: {:?}\n", model_dir);

    // --- 2. Initialize GPU Model ---
    println!("Initializing GPU Model...");
    let wgpu_context = Arc::new(WgpuContext::new().await);
    let gpu_model = GPT2Model::from_pretrained(
        &model_dir, 
        Device::Wgpu, 
        Some(wgpu_context.clone())
    )?;
    println!("✓ GPU model ready\n");

    // --- 3. Initialize CPU Model ---
    println!("Initializing CPU Model...");
    let cpu_model = GPT2Model::from_pretrained(
        &model_dir, 
        Device::Cpu, 
        None
    )?;
    println!("✓ CPU model ready\n");

    // --- 4. Test Single Generation ---
    let test_prompt = "Once upon a time";
    
    println!("=== Single Generation Test ===");
    println!("Prompt: \"{}\"\n", test_prompt);

    // GPU generation
    println!("🖥️  GPU generating...");
    let gpu_start = Instant::now();
    let gpu_output = gpu_model.generate(
        test_prompt,
        MAX_NEW_TOKENS,
        Some(1.0),  // temperature
        Some(50),   // top_k
    ).await?;
    let gpu_time = gpu_start.elapsed();
    
    println!("Generated: {}", gpu_output);
    println!("Time: {:.2}s ({:.1} tokens/sec)\n", 
        gpu_time.as_secs_f32(),
        MAX_NEW_TOKENS as f32 / gpu_time.as_secs_f32()
    );

    // CPU generation
    println!("💻 CPU generating...");
    let cpu_start = Instant::now();
    let cpu_output = cpu_model.generate(
        test_prompt,
        MAX_NEW_TOKENS,
        Some(1.0),  // temperature
        Some(50),   // top_k
    ).await?;
    let cpu_time = cpu_start.elapsed();
    
    println!("Generated: {}", cpu_output);
    println!("Time: {:.2}s ({:.1} tokens/sec)\n", 
        cpu_time.as_secs_f32(),
        MAX_NEW_TOKENS as f32 / cpu_time.as_secs_f32()
    );

    // Compare outputs
    println!("=== Output Comparison ===");
    if gpu_output == cpu_output {
        println!("✅ PERFECT MATCH! GPU and CPU outputs are identical.\n");
    } else {
        println!("⚠️  OUTPUTS DIFFER!");
        println!("GPU: {}", gpu_output);
        println!("CPU: {}", cpu_output);
        println!();
    }

    // --- 5. Speed Benchmark ---
    println!("=== Speed Benchmark ===");
    println!("Generating {} tokens per run...\n", MAX_NEW_TOKENS);

    let bench_prompt = "The future of artificial intelligence";
    let num_runs = 3;

    // GPU Benchmark
    println!("🖥️  GPU Benchmark:");
    let mut gpu_total_time = 0.0;
    for run in 1..=num_runs {
        let start = Instant::now();
        let output = gpu_model.generate(
            bench_prompt, 
            MAX_NEW_TOKENS, 
            Some(1.0), 
            Some(50)
        ).await?;
        let elapsed = start.elapsed();
        gpu_total_time += elapsed.as_secs_f32();
        
        println!("Run {}/{}: {:.2}s ({:.1} tokens/sec)", 
            run, num_runs,
            elapsed.as_secs_f32(),
            MAX_NEW_TOKENS as f32 / elapsed.as_secs_f32()
        );
        
        if run == 1 {
            println!("  Output: {}", output);
        }
    }

    let gpu_avg_time = gpu_total_time / num_runs as f32;
    let gpu_avg_speed = MAX_NEW_TOKENS as f32 / gpu_avg_time;

    println!("\n--- GPU Results ---");
    println!("Average time: {:.2}s", gpu_avg_time);
    println!("Average speed: {:.1} tokens/sec", gpu_avg_speed);
    println!("-------------------\n");

    // CPU Benchmark
    println!("💻 CPU Benchmark:");
    let mut cpu_total_time = 0.0;
    for run in 1..=num_runs {
        let start = Instant::now();
        let output = cpu_model.generate(
            bench_prompt, 
            MAX_NEW_TOKENS, 
            Some(1.0), 
            Some(50)
        ).await?;
        let elapsed = start.elapsed();
        cpu_total_time += elapsed.as_secs_f32();
        
        println!("Run {}/{}: {:.2}s ({:.1} tokens/sec)", 
            run, num_runs,
            elapsed.as_secs_f32(),
            MAX_NEW_TOKENS as f32 / elapsed.as_secs_f32()
        );
        
        if run == 1 {
            println!("  Output: {}", output);
        }
    }

    let cpu_avg_time = cpu_total_time / num_runs as f32;
    let cpu_avg_speed = MAX_NEW_TOKENS as f32 / cpu_avg_time;

    println!("\n--- CPU Results ---");
    println!("Average time: {:.2}s", cpu_avg_time);
    println!("Average speed: {:.1} tokens/sec", cpu_avg_speed);
    println!("-------------------\n");

    // --- 6. Final Comparison ---
    println!("=== Final Comparison ===");
    let speedup = cpu_avg_time / gpu_avg_time;
    
    println!("GPU: {:.1} tokens/sec", gpu_avg_speed);
    println!("CPU: {:.1} tokens/sec", cpu_avg_speed);
    println!("Speedup: {:.2}x", speedup);
    
    if speedup > 1.0 {
        println!("✓ GPU is {:.1}% faster", (speedup - 1.0) * 100.0);
    } else {
        println!("⚠️  GPU is {:.1}% slower (needs optimization!)", (1.0 - speedup) * 100.0);
    }
    
    println!("\n=== Note ===");
    println!("Without KV cache, speedup is limited due to recomputation.");
    println!("Expected speedup with KV cache: 10-25x");
    println!("Expected speedup with KV cache + batch=32: 50-100x");

    Ok(())
}