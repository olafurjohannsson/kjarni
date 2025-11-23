// In your main application or examples folder


use edgemodels::models::gpt2::Gpt2Model; // The new, refactored struct
// use edgemodels::text_generation::LLamaModel2;
// use edgemodels::text_generation::TextGenerator; 
use edgetransformers::WgpuContext;
use edgetransformers::models::base::{GenerationConfig, DecodingStrategy};
use edgetransformers::{Device, ModelType};
use std::sync::Arc;
use tokio::sync::Mutex;
use edgetransformers::gpu_ops::GpuTensorPool;
use edgemodels::generation::decoder::{CpuDecoderBackend, Generator};
use edgemodels::generation::generator::DecoderGenerationBackend;
use edgemodels::generation::decoder::GpuDecoderBackend;

use std::io::Write;


#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let context = Arc::new(WgpuContext::new().await?);
    let pool = Arc::new(Mutex::new(GpuTensorPool::new(context.clone())));

    let gpt2_model = Gpt2Model::from_registry(
        ModelType::DistilGpt2,
        None, // Use default cache dir
        Device::Wgpu,
        Some(context.clone()), // No WGPU context needed for CPU
    )
    .await?;

    // // 2. Create the generic Generator, handing it the model.
    let backend = GpuDecoderBackend::new(context.clone(), pool.clone())?;
    
    let backend_cpu = CpuDecoderBackend;

    let generator_gpu = Generator::new(Box::new(gpt2_model), backend);


    // 3. Configure the generation parameters.
    let config = GenerationConfig {
        max_new_tokens: Some(100),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        add_bos_token: false,
        ..Default::default()
    };
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";;
    println!("\n--- Streaming text ---");
    
    let stream = generator_gpu.generate_stream(prompt, &config).await?;

    futures_util::pin_mut!(stream);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream).await? {
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
    }
    println!();

    // println!()

    // let llama_model = LLamaModel2::from_registry(
    //     ModelType::Llama3_2_1B,
    //     None,
    //     Device::Cpu,
    //     None, ).await?;

    // let llama_generator = Generator::new(Box::new(llama_model));
    // println!("LLama gen: ");
    // let stream = llama_generator.generate_stream("Rust is a language that is", &config).await?;
    // futures_util::pin_mut!(stream);
    // while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream).await? {
    //     print!("{}", token);
    //     std::io::stdout().flush().unwrap();
    // }
    // println!();

    Ok(())
}
