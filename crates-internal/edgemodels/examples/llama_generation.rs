// In your main application or examples folder

use edgemodels::generation::Generator;
use edgemodels::text_generation::Gpt2Model; // The new, refactored struct
use edgemodels::text_generation::LlamaModel;
// use edgemodels::text_generation::TextGenerator;
use edgetransformers::WgpuContext;
use edgetransformers::models::base::{DecodingStrategy, GenerationConfig};
use edgetransformers::{Device, ModelType};
use std::io::Write;
use std::sync::Arc;

async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await.unwrap())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = get_test_context().await;
    let config: GenerationConfig = GenerationConfig {
        max_new_tokens: Some(100),
        strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.1,
        ..Default::default()
    };
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";
    // let llama_model =
    //     LlamaModel::from_registry(ModelType::Llama3_2_1B, None, Device::Cpu, None).await?;
    let llama_model =
        LlamaModel::from_registry(ModelType::Llama3_2_1B, None,
                                  Device::Wgpu, Some(ctx)).await?;

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
