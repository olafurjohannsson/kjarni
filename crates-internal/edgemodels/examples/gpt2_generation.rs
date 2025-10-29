use edgemodels::text_generation::TextGenerator;
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("--- DistilGPT-2 Text Generation Example (CPU) ---");

    // Load the DistilGPT2 model from the registry.
    // This will automatically download the model files on the first run.
    println!("Loading DistilGPT-2 model...");
    let generator = TextGenerator::from_registry(
        ModelType::DistilGpt2,
        None, // Use default cache directory
        Device::Cpu,
        None, // No WGPU context needed for CPU
    ).await?;
    println!("✓ Model loaded.");

    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    println!("\n--- PROMPT ---");
    println!("{}", prompt);

    // Generate text
    println!("\nGenerating text...");
    let generated_text = generator.generate(prompt, 50).await?; // Generate 50 new tokens

    println!("\n--- GENERATED TEXT ---");
    println!("{}", generated_text);
    
    Ok(())
}