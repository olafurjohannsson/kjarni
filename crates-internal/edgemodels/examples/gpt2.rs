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
// use edgemodels::generation::generator::DecoderGenerationBackend;
use edgetransformers::decoder::DecoderGenerationBackend;
use edgemodels::generation::decoder::GpuDecoderBackend;
use edgetransformers::models::base::DecoderLoadConfig;
use std::io::Write;

pub fn benchmark_faer() {
    use faer::prelude::*;
    use std::time::Instant;
    
    let a = faer::Mat::<f32>::zeros(2048, 2048);
    let b = faer::Mat::<f32>::zeros(2048, 2048);
    
    let start = Instant::now();
    let _c = &a * &b;
    println!("faer 2048x2048: {:?}", start.elapsed());
}
pub fn benchmark_blas() {
    use ndarray::Array2;
    use std::time::Instant;
    
    let a = Array2::<f32>::ones((2048, 2048));
    let b = Array2::<f32>::ones((2048, 2048));
    
    let start = Instant::now();
    let _c = a.dot(&b);
    let elapsed = start.elapsed();
    
    println!("2048x2048 matmul: {:?}", elapsed);
    if elapsed.as_millis() > 100 {
        println!("WARNING: BLAS is NOT working! Expected <50ms, got {:?}", elapsed);
    } else {
        println!("BLAS is working correctly");
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let context = WgpuContext::new().await?;
    // let pool = Arc::new(Mutex::new(GpuTensorPool::new(context.clone())));
    let d = DecoderLoadConfig {
        gpu_layers: None,
        offload_embeddings: false,
        offload_lm_head: false,
        target_dtype: None,
    };
    let gpt2_model = Gpt2Model::from_registry(
        ModelType::DistilGpt2,
        None, // Use default cache dir
        Device::Wgpu,
        Some(context.clone()), // No WGPU context needed for CPU
        Some(d),
    )
    .await?;

    // // 2. Create the generic Generator, handing it the model.
    // let backend = GpuDecoderBackend::new(context.clone(), pool.clone())?;
    
    // let backend_cpu = CpuDecoderBackend;

    let generator_gpu = Generator::new(Box::new(gpt2_model))?;


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
    benchmark_blas();
    benchmark_faer();
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
