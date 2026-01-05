use anyhow::Result;
use kjarni_models::models::bart::model::BartModel;
use kjarni_transformers::common::{BeamSearchParams, DecodingStrategy, GenerationConfig};
use kjarni_transformers::encoder_decoder::EncoderDecoderGenerator;
use kjarni_transformers::WgpuContext;
use kjarni_transformers::{Device, ModelType};
use std::sync::Arc;

async fn get_test_context() -> Arc<WgpuContext> {
    WgpuContext::new().await.unwrap()
}
use anyhow::anyhow;

#[tokio::main]
async fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
like C++, Haskell, and Erlang.";
    // --- 1. SETUP ---
    log::info!("Loading models for CPU and GPU...");
    let ctx = get_test_context().await;
    let model_type = ModelType::DistilBartCnn;

    let cpu_model = BartModel::from_registry(model_type, None, Device::Cpu, None, None).await?;
    let gpu_model =
        BartModel::from_registry(model_type, None, Device::Wgpu, Some(ctx.clone()), None).await?;
    use std::io;
    use std::io::Write;
    let cpu_generator = EncoderDecoderGenerator::new(Box::new(cpu_model))?;
    let config = GenerationConfig {
        max_length: 142,
        min_length: 56,
        no_repeat_ngram_size: 3,
        repetition_penalty: 1.0,
        max_new_tokens: None,
        add_bos_token: false,
        strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
            num_beams: 4,
            length_penalty: 2.0,
            early_stopping: true,
        }),
    };
    let mut cpu_generation_config = cpu_generator.model.get_default_generation_config();
    // cpu_generation_config.max_new_tokens = Some(100);

    let t = cpu_generator.generate(&article, Some(&config)).await?;
    println!("CPU Generation test: {}", t);
    io::stdout().flush().unwrap();

    println!("GenConfig {:?}", cpu_generation_config);
    io::stdout().flush().unwrap();
    let cpu_summary = cpu_generator.generate_stream(article, Some(&cpu_generation_config));

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
    println!("\n--- FINAL CPU GENERATED SUMMARY ---");
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut cpu_summary).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }

    let gpu_generator = EncoderDecoderGenerator::new(Box::new(gpu_model))?;
    let gpu_generation_config = gpu_generator.model.get_default_generation_config();
    let gpu_summary = gpu_generator.generate_stream(article, Some(&gpu_generation_config));
    futures_util::pin_mut!(gpu_summary);
    println!("\n--- FINAL GPU GENERATED SUMMARY ---");
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut gpu_summary).await? {
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
    }

    // println!("Summary: {}", gpu_summary);
    println!("");

    // println!("Summary: {}", gpu_summary);

    Ok(())
}
