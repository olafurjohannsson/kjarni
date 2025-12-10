use anyhow::Result;
use edgemodels::generation::decoder::Generator;
use edgemodels::generation::encoder_decoder::CpuTensor;
use edgemodels::generation::encoder_decoder::GpuSeq2SeqTensor;
use edgemodels::generation::encoder_decoder::Seq2SeqGenerator;
use edgemodels::generation::encoder_decoder::{CpuBackend, GpuBackend};
use edgemodels::models::bart::model::BartModel;
use edgemodels::models::llama::model::LlamaModel;
use edgetransformers::encoder_decoder::traits::EncoderDecoderGenerationBackend;
use edgetransformers::encoder_decoder::traits::EncoderDecoderLanguageModel;
use edgetransformers::gpu_ops::tensor::GpuTensor;
use edgetransformers::gpu_ops::GpuTensorPool;
use edgetransformers::models::base::DecoderLoadConfig;
use edgetransformers::models::base::LanguageModel;
use edgetransformers::models::base::{DecodingStrategy, GenerationConfig};
use edgetransformers::WgpuContext;
use edgetransformers::{Device, ModelType};
use ndarray::{ArrayViewD, IxDyn};
use std::sync::Arc;
use tokio::sync::Mutex;

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}
use anyhow::anyhow;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let article = "Rust is a multi-paradigm, general-purpose programming language.";
    // --- 1. SETUP ---
    log::info!("Loading models for CPU and GPU...");
    let ctx = get_test_context().await;
    let model_type = ModelType::DistilBartCnn;

    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None).await?;
    let gpu_model =
        BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone())).await?;
    use std::io::Write;
    use std::io;
    let cpu_generator = Seq2SeqGenerator::new(Box::new(cpu_model))?;
    let mut cpu_generation_config = cpu_generator.model.get_default_generation_config();
    cpu_generation_config.max_new_tokens = Some(100);

    io::stdout().flush().unwrap();
    let cpu_summary = cpu_generator
        .generate_stream(article, Some(&cpu_generation_config));

    // io::stdout().flush().unwrap();
    //
    // let mut stream_gpu = generator_gpu.generate_stream(formatted.as_str(), &config).await?;
    // futures_util::pin_mut!(stream_gpu);
    // while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_gpu).await? {
    //     print!("{}", token.text);
    //     io::stdout().flush().unwrap();
    // }
    // let mut stream_gpu = generator_gpu.generate_stream(formatted.as_str(), &config).await?;


    futures_util::pin_mut!(cpu_summary);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut cpu_summary).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }


    let gpu_generator = Seq2SeqGenerator::new(Box::new(gpu_model))?;
    let gpu_generation_config = gpu_generator.model.get_default_generation_config();
    let gpu_summary = gpu_generator
        .generate_stream(article, Some(&gpu_generation_config));
    futures_util::pin_mut!(gpu_summary);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut gpu_summary).await? {
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
    }

    println!("\n--- FINAL GPU GENERATED SUMMARY ---");
    // println!("Summary: {}", gpu_summary);
    println!("");
    println!("\n--- FINAL GPU GENERATED SUMMARY ---");
    // println!("Summary: {}", gpu_summary);

    Ok(())
}
