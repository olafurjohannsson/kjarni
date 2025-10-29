// --- FIX 1: Correct the import paths ---
// GenerationConfig and SamplingStrategy are generic and live in the `edgetransformers` library.
use edgetransformers::models::base::{GenerationConfig, SamplingStrategy};
// This path assumes your TextGenerator is in a `text_generator` module within `edgemodels`. Adjust if needed.
use edgemodels::text_generation::TextGenerator; 
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("--- DistilGPT-2 Text Generation Example (CPU) ---");

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

    println!("\nGenerating text using Greedy Search...");
    
    // --- FIX 2 & 3: Correctly create the GenerationConfig ---
    let config = GenerationConfig {
        // `max_new_tokens` is now an Option, so wrap it in `Some()`
        max_new_tokens: Some(100),
        
        // Use the corrected path for the enum variant
        sampling_strategy: SamplingStrategy::Greedy,
        
        // Repetition penalty is useful for all generation types
        repetition_penalty: 1.1,
        temperature: 0.7,

        // Use the "struct update syntax" to fill in all other fields
        // with their default values (e.g., max_length, num_beams, etc.).
        // This makes our config robust to future changes.
        ..Default::default()
    };
    
    let generated_text = generator.generate(prompt, &config).await?;

    println!("\n--- GENERATED TEXT ---");
    println!("{}", generated_text);
    
    Ok(())
}