use anyhow::Result;
use kjarni_models::models::t5::model::T5Model;
use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::encoder_decoder::EncoderDecoderGenerator;
use kjarni_transformers::{Device, ModelType};
use std::io;
use std::io::Write;
use futures::{TryStreamExt, pin_mut};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to see model loading progress
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // 1. Define the input following the T5 task format
    let input_text = "translate English to German: How old are you?";

    // 2. Setup - Load the model from registry for CPU
    // We use Flan-T5-Base as in your Python example
    log::info!("Loading Flan-T5-Base on CPU...");
    let model_type = ModelType::FlanT5Large;

    let model = T5Model::from_registry(
        model_type,
        None, // Default cache directory
        Device::Cpu,
        None, // No WgpuContext needed for CPU
        None, // Use default ModelLoadConfig
    )
        .await?;
    let c = GenerationConfig {
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };

    // 3. Initialize the Generator
    // This orchestrates the Encoder-Decoder loop
    let generator = EncoderDecoderGenerator::new(Box::new(model))?;

    // 4. Get Default Generation Config
    // This pulls the beam search settings (num_beams: 4) defined in your T5Model trait
    // let mut config = generator.model.get_default_generation_config();
    // config.strategy = DecodingStrategy::Greedy;
    log::info!("Input: {}", input_text);
    // log::info!("Generating with config: {:?}", config);

    // 5. Run Complete Generation (Blocking)
    println!("\n--- Full Generation ---");
    let output = generator.generate(input_text, None).await?;
    println!("Result: {}", output);

    // 6. Run Streaming Generation (Optional, for real-time output)
    println!("\n--- Streaming Generation ---");
    print!("Result: ");
    io::stdout().flush()?;

    let mut stream = generator.generate_stream(input_text, None);

    pin_mut!(stream);
    while let Some(token) = TryStreamExt::try_next(&mut stream).await? {
        print!("{} ", token.text);
        std::io::stdout().flush().unwrap();
    }

    println!("\n");

    Ok(())
}
