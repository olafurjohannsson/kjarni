use kjarni_models::models::llama::model::LlamaModel;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::stats::GenerationStats;
use kjarni_transformers::{Device, ModelType, WgpuContext};
use std::io;
use std::io::Write;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // It's good practice to use a logger for backend info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    kjarni_transformers::utils::configure_threading();
    // --- Shared Setup ---
    let prompt = "Describe the theory of relativity in simple terms(max 50 words):\n";
    let config = GenerationConfig {
        max_new_tokens: Some(150),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    // =========================================================================
    //                            GPU Generation
    // =========================================================================
    println!("\n--- Running Llama 3 on GPU ---");

    // Step 0: Create GPU context and tensor pool (needed for the backend)
    let context = WgpuContext::new().await?;
    //
    let d = ModelLoadConfig {
        offload_embeddings: false,
        offload_lm_head: false,
        target_dtype: None,
        quantize_lm_head: None, // Some(DType::Q8_0),
        use_gguf: false,
        ..Default::default()
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
    GenerationStats::enable();
    let model_cpu = LlamaModel::from_registry(
        ModelType::Llama3_2_3B_Instruct,
        None,
        Device::Cpu,
        None,
        Some(d),
    ).await?;
        let model_gpu = LlamaModel::from_registry(
        ModelType::Llama3_2_3B_Instruct,
        None,
        Device::Wgpu,
        None,
        Some(d),
    ).await?;
    //  let qwen_cpu = QwenModel::from_registry(
    //     ModelType::Qwen2,
    //     None,
    //     Device::Cpu,
    //     None,
    //     Some(d),
    // ).await?;
    let model_path = std::path::Path::new("/home/olafurj/.cache/kjarni/llama-3.2-3b-instruct-q4_k_m/Llama-3.2-3B-Instruct-Q4_K_M.gguf");

    // let model_cpu = LlamaModel::from_pretrained(
    //     model_path,
    //     Device::Cpu,
    //     None, // No WgpuContext for CPU
    //     Some(d),
    //     Some(ModelType::Llama3_2_3B_Instruct),
    // )?;
    let generator_cpu = DecoderGenerator::new(Arc::new(model_cpu))?;
    let mut stream_cpu = generator_cpu.generate_stream(prompt, &config, None).await?;
    futures_util::pin_mut!(stream_cpu);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_cpu).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }
    println!();
    let generator_gpu = DecoderGenerator::new(Arc::new(model_gpu))?;
    let mut stream_gpu = generator_gpu.generate_stream(prompt, &config, None).await?;
    futures_util::pin_mut!(stream_gpu);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_gpu).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }
    println!();
    Ok(())
}
