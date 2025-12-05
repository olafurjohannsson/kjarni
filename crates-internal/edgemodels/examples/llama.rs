use edgemodels::generation::decoder::{CpuDecoderBackend, Generator};
use edgemodels::models::llama::model::LlamaModel;
// use edgemodels::text_generation::TextGenerator;
use edgemodels::generation::decoder::GpuDecoderBackend;
// use edgemodels::generation::generator::DecoderGenerationBackend;
use edgetransformers::decoder::DecoderGenerationBackend;
use edgetransformers::WgpuContext;
use edgetransformers::gpu_ops::GpuTensorPool;
use edgetransformers::models::base::DecoderLoadConfig;
use edgetransformers::models::base::{DecodingStrategy, GenerationConfig};
use edgetransformers::{Device, ModelType};
use std::io;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // It's good practice to use a logger for backend info
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // --- Shared Setup ---
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    let config = GenerationConfig {
        max_new_tokens: Some(10),
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
    let pool = Arc::new(Mutex::new(GpuTensorPool::new(context.clone())));
    let d = DecoderLoadConfig {
        gpu_layers: None,
        offload_embeddings: false,
        offload_lm_head: false,
        target_dtype: Some(edgetransformers::weights::DType::F32),
    };
    // Step 1: Load the model onto the desired device
    let model_gpu = LlamaModel::from_registry(
        ModelType::Llama3_2_1B,
        None,
        Device::Cpu,
        None, //Some(context.clone()),
        Some(d),
    )
    .await?;

    // Step 2: Create the appropriate backend for the device
    // let cpu_backend = CpuDecoderBackend;
    // let gpu_backend = GpuDecoderBackend::new(context.clone(), pool.clone())?;

    // Step 3: Create the generic Generator with the model and backend
    let generator_gpu = Generator::new(Box::new(model_gpu))?;

    // Step 4: Generate text. The user-facing API is unchanged!
    println!("prompt: {}", prompt);
    io::stdout().flush().unwrap();

    let mut stream_gpu = generator_gpu.generate_stream(prompt, &config).await?;
    futures_util::pin_mut!(stream_gpu);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_gpu).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }
    println!();
    Ok(())
}
