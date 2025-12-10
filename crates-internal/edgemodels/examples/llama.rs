use edgemodels::generation::decoder::Generator;
use edgemodels::models::llama::model::LlamaModel;
use edgetransformers::gpu_ops::GpuTensorPool;
use edgetransformers::models::base::DecoderLoadConfig;
use edgetransformers::models::base::{DecodingStrategy, GenerationConfig};
use edgetransformers::WgpuContext;
use edgetransformers::{Device, ModelType};
use std::io;
use std::io::Write;
use std::sync::Arc;
use tokio::sync::Mutex;


fn bench_matmul_bf16() {
    let m = 1;
    let k = 2048;
    let n = 8192;  // FFN up/gate projection

    let a = ndarray::Array2::<f32>::ones((m, k));
    let b = ndarray::Array2::<u16>::zeros((n, k));  // BF16 weights

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = edgetransformers::utils::linear_algebra::matmul_2d_mixed_bf16(&a.view(), &b.view());
    }
    let elapsed = start.elapsed();

    println!("100 iters: {:?}", elapsed);
    println!("Per iter: {:?}", elapsed / 100);
    println!("GFLOPS: {:.2}", (m * k * n * 2 * 100) as f64 / elapsed.as_secs_f64() / 1e9);
}
fn bench_down_projection() {
    edgetransformers::utils::configure_threading();

    // Down projection: [1, 8192] @ [2048, 8192]
    let m = 1;
    let k = 8192;
    let n = 2048;  // This was hitting serial path!

    let a = ndarray::Array2::<f32>::ones((m, k));
    let b = ndarray::Array2::<u16>::zeros((n, k));

    let start = std::time::Instant::now();
    for _ in 0..100 {
        let _ = edgetransformers::utils::linear_algebra::matmul_2d_mixed_bf16(&a.view(), &b.view());
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
        max_new_tokens: Some(150),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.2,
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
        target_dtype: None,
    };
    let formatted = format!(
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        prompt
    );
    // Step 1: Load the model onto the desired device
    let model_gpu = LlamaModel::from_registry(
        ModelType::Llama3_2_1B,
        None,
        Device::Wgpu,
        Some(context.clone()),
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

    let mut stream_gpu = generator_gpu.generate_stream(formatted.as_str(), &config).await?;
    futures_util::pin_mut!(stream_gpu);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream_gpu).await? {
        print!("{}", token.text);
        io::stdout().flush().unwrap();
    }
    println!();
    bench_matmul_bf16();
    bench_down_projection();
    Ok(())
}
