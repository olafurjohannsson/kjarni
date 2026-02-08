use futures::{pin_mut, TryStreamExt};
use kjarni_models::models::llama::model::LlamaModel;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::stats::GenerationStats;
use kjarni_transformers::{Device, ModelType};
use std::io::{self, Write};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    kjarni_transformers::utils::configure_threading();

    let prompt = "Describe the theory of relativity in simple terms (max 50 words):\n";
    let model_type = ModelType::Llama3_2_3B_Instruct;

    let generation_config = GenerationConfig {
        max_new_tokens: Some(150),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.0,
        ..Default::default()
    };

    let load_config = ModelLoadConfig::default();

    GenerationStats::enable();

    println!("\n--- Running on CPU ---");
    println!("Prompt: {}", prompt);

    let model_cpu = LlamaModel::from_registry(
        model_type,
        None,
        Device::Cpu,
        None,
        Some(load_config.clone()),
    )
    .await?;

    let generator_cpu = DecoderGenerator::new(Arc::new(model_cpu))?;
    let stream_cpu = generator_cpu.generate_stream(prompt, &generation_config, None).await?;
    pin_mut!(stream_cpu);

    while let Some(token) = stream_cpu.try_next().await? {
        print!("{}", token.text);
        io::stdout().flush()?;
    }
    println!("\n");

    println!("--- Running on GPU ---");
    println!("Prompt: {}", prompt);

    let model_gpu = LlamaModel::from_registry(
        model_type,
        None,
        Device::Wgpu,
        None,
        Some(load_config),
    )
    .await?;

    let generator_gpu = DecoderGenerator::new(Arc::new(model_gpu))?;
    let stream_gpu = generator_gpu.generate_stream(prompt, &generation_config, None).await?;
    pin_mut!(stream_gpu);

    while let Some(token) = stream_gpu.try_next().await? {
        print!("{}", token.text);
        io::stdout().flush()?;
    }
    println!("\n");

    Ok(())
}