// In your main application or examples folder

use edgemodels::generation::Generator;
use edgemodels::text_generation::Gpt2Model; // The new, refactored struct
use edgemodels::text_generation::LLamaModel2;
use edgemodels::text_generation::TextGenerator; 
use edgetransformers::WgpuContext;
use edgetransformers::models::base::{GenerationConfig, SamplingStrategy};
use edgetransformers::{Device, ModelType};
use std::io::Write;
use std::sync::Arc;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = get_test_context().await;
    let gpt2_model = Gpt2Model::from_registry(
        ModelType::DistilGpt2,
        None, // Use default cache dir
        Device::Cpu,
        None, //Some(ctx), // No WGPU context needed for CPU
    )
    .await?;

    // // 2. Create the generic Generator, handing it the model.
    // let generator = Generator::new(Box::new(gpt2_model));

    // 3. Configure the generation parameters.
    let config: GenerationConfig = GenerationConfig {
        max_new_tokens: Some(20),
        sampling_strategy: SamplingStrategy::Greedy,
        repetition_penalty: 1.0,
        temperature: 0.0,
        ..Default::default()
    };
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";;
    // println!("\n--- Streaming text ---");
    // let stream = generator.generate_stream(prompt, &config).await?;
    // futures_util::pin_mut!(stream);
    // while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream).await? {
    //     print!("{}", token.text);
    //     std::io::stdout().flush().unwrap();
    // }
    // println!();

    // println!()

    // RUST NEW: <|begin_of_text|>The field of Artificial Intelligence has seen a lot of progress in the last few years. The technology is now able to perform tasks thats
// Rust: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field has been able to make a lot of progress in the last
// Pyth: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field has been able to make a lot of progress in the last
// Rust NEW: The field of Artificial Intelligence has seen a lot of progress in the last few years. The field is now being used in many different
    let llama_model = LLamaModel2::from_registry(
        ModelType::Llama3_2_1B,
        None,
        Device::Cpu,
        None, ).await?;

    let llama_generator = Generator::new(Box::new(llama_model));
    println!("LLama gen: "); // The field of Artificial Intelligence has seen a lot of progress in the last few years. The field of AI is now being used in
    let stream = llama_generator.generate_stream(prompt, &config).await?;
    futures_util::pin_mut!(stream);
    while let Some(token) = futures_util::TryStreamExt::try_next(&mut stream).await? {
        print!("{}", token.text);
        std::io::stdout().flush().unwrap();
    }
    println!();

    Ok(())
}
