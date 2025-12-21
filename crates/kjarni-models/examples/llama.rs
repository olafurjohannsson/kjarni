use kjarni_models::models::llama::model::LlamaModel;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::{Device, ModelType};
use std::io;
use std::io::Write;

fn bench_matmul_bf16() {
    let m = 1;
    let k = 2048;
    let n = 8192;  // FFN up/gate projection

    let a = ndarray::Array2::<f32>::ones((m, k));
    let b = ndarray::Array2::<u16>::zeros((n, k));  // BF16 weights

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = kjarni_transformers::utils::linear_algebra::matmul_2d_mixed_bf16(&a.view(), &b.view());
    }
    let elapsed = start.elapsed();

    println!("100 iters: {:?}", elapsed);
    println!("Per iter: {:?}", elapsed / 100);
    println!("GFLOPS: {:.2}", (m * k * n * 2 * 100) as f64 / elapsed.as_secs_f64() / 1e9);
}
fn bench_down_projection() {
    kjarni_transformers::utils::configure_threading();

    // Down projection: [1, 8192] @ [2048, 8192]
    let m = 1;
    let k = 8192;
    let n = 2048;  // This was hitting serial path!

    let a = ndarray::Array2::<f32>::ones((m, k));
    let b = ndarray::Array2::<u16>::zeros((n, k));

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = kjarni_transformers::utils::linear_algebra::matmul_2d_mixed_bf16(&a.view(), &b.view());
    }
    let elapsed = start.elapsed();

    println!("Down projection: 100 iters: {:?}", elapsed);
    println!("Per iter: {:?}", elapsed / 100);
    println!("GFLOPS: {:.2}", (m * k * n * 2 * 100) as f64 / elapsed.as_secs_f64() / 1e9);
}
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // It's good practice to use a logger for backend info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();


    // --- Shared Setup ---
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    let config = GenerationConfig {
        max_new_tokens: Some(15),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.2,
        ..Default::default()
    };

    // =========================================================================
    //                            GPU Generation
    // =========================================================================
    println!("\n--- Running Llama 3 on GPU ---");

    // Step 0: Create GPU context and tensor pool (needed for the backend)
    // let context = WgpuContext::new().await?;
    //
    let d = DecoderLoadConfig {
        gpu_layers: None,
        offload_embeddings: false,
        offload_lm_head: false,
        target_dtype: None,
    };
    // let model_gpu = LlamaModel::from_registry(
    //     ModelType::Llama3_2_1B,
    //     None,
    //     Device::Wgpu,
    //     Some(context.clone()),
    //     Some(d),
    // ).await?;
    // let generator_gpu = DecoderGenerator::new(Box::new(model_gpu))?;
    //
    // println!("prompt: {}", prompt);
    // io::stdout().flush().unwrap();
    //
    // let mut stream_gpu = generator_gpu.generate_stream(prompt, &config).await?;
    // futures_util::pin_mut!(stream_gpu);
    // while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_gpu).await? {
    //     print!("{}", token.text);
    //     io::stdout().flush().unwrap();
    // }
    // println!();
    //
    // io::stdout().flush().unwrap();
    let model_cpu = LlamaModel::from_registry(
        ModelType::Llama3_2_1B,
        None,
        Device::Cpu,
        None,
        Some(d),
    ).await?;
    // let model_path = std::path::Path::new("/home/olafurj/.cache/kjarni/llama-3.2-1b-instruct-q4_k_m/llama-3.2-1b-instruct-q4_k_m.gguf");

    // let model_cpu = LlamaModel::from_pretrained(
    //     model_path,
    //     Device::Cpu,
    //     None, // No WgpuContext for CPU
    //     Some(d), // Your LoadConfig
    // )?;
    let generator_cpu = DecoderGenerator::new(Box::new(model_cpu))?;
    let mut stream_cpu = generator_cpu.generate_stream(prompt, &config).await?;
    futures_util::pin_mut!(stream_cpu);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_cpu).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }
    println!();
    Ok(())
}
