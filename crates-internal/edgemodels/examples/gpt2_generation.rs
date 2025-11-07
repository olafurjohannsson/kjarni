use edgetransformers::models::base::{GenerationConfig, SamplingStrategy};
use edgemodels::text_generation::TextGenerator; 
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;
use env_logger;



#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _ = env_logger::try_init();
    println!("--- DistilGPT-2 Text Generation Example (CPU) ---");

    println!("Loading DistilGPT-2 model...");
    let context = std::sync::Arc::new(edgetransformers::WgpuContext::new().await);
    let generator = TextGenerator::from_registry(
        ModelType::DistilGpt2,
        None, 
        Device::Wgpu,
        Some(context), 
    ).await?;
    println!("✓ Model loaded.");

    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    println!("\n--- PROMPT ---");
    println!("{}", prompt);

    println!("\nGenerating text using Greedy Search...");
    
    let config = GenerationConfig {
        max_new_tokens: Some(100),
        sampling_strategy: SamplingStrategy::Greedy,
        repetition_penalty: 1.1,
        temperature: 0.7,
        ..Default::default()
    };
    
    let generated_text = generator.generate(prompt, &config).await?;

    println!("\n--- GENERATED TEXT ---");
    println!("{}", generated_text);
    
    Ok(())
}