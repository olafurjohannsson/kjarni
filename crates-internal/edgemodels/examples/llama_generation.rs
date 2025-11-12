use edgemodels::text_generation::LlamaModel;
use edgetransformers::models::{ModelType, LanguageModel};
use edgetransformers::traits::Device;
use env_logger;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();

    let generator = LlamaModel::from_registry(
        ModelType::Llama3_2_1B,
        None, 
        Device::Cpu,
        None,
    ).await?;
    
    println!("LLaMA 3.2 1B Model loaded.\n");

    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    println!("=== INPUT ===");
    println!("Text: {}", prompt);
    
    // ✅ Tokenize and print input IDs
    let encoding = generator.tokenizer().encode(prompt, false).unwrap();
    let input_ids: Vec<u32> = encoding.get_ids().to_vec();
    println!("Token IDs: {:?}", input_ids);
    println!("Number of tokens: {}\n", input_ids.len());
    
    // ✅ Generate with logging
// Rust: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field has been able to make a lot
// Pyth: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field has been able to make a lot
// Rust NEW: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field is now being used in many different
    let (generated_text, generated_ids) = generator.generate_with_ids(prompt, 20, 0.0, None).await?;
    

    println!("=== RUST OUTPUT ===");
    println!("{}", generated_text);
    println!("\n=== GENERATED TOKEN IDs ===");
    println!("{:?}", generated_ids);
    
    Ok(())
}