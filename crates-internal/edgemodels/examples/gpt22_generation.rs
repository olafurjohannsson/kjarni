// In your main application or examples folder

use edgemodels::generation::Generator;
use edgemodels::text_generation::Gpt2Model; // The new, refactored struct
use edgetransformers::models::base::GenerationConfig;
use edgetransformers::{Device, ModelType};
use std::io::Write;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // 1. Load the model container for GPT-2.
    //    This can be for CPU or GPU.
    let gpt2_model = Gpt2Model::from_registry(
        ModelType::Gpt2,
        None, // Use default cache dir
        Device::Cpu,
        None, // No WGPU context needed for CPU
    ).await?;

    // 2. Create the generic Generator, handing it the model.
    let generator = Generator::new(Box::new(gpt2_model));

    // 3. Configure the generation parameters.
    let config = GenerationConfig {
        max_new_tokens: Some(2550),
        ..Default::default()
    };
    
    
    println!("\n--- Streaming text ---");
    let stream = generator.generate_stream("Rust is a language that is", &config).await?;
    futures_util::pin_mut!(stream);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream).await? {
        print!("{}", token);
        std::io::stdout().flush().unwrap();
    }
    println!();

    Ok(())
}