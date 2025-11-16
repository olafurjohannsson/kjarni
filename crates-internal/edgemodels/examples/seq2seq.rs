use std::sync::Arc;
use anyhow::{Result, anyhow};
use edgemodels::generation::seq2seq::Seq2SeqGenerator; // Your new generator
use edgemodels::seq2seq::{AnySeq2SeqModel, BartConfig, Seq2SeqModel, TaskSpecificParams};
use edgetransformers::models::base::GenerationConfig;
use edgetransformers::{Device, ModelType, WgpuContext};
use edgemodels::generation::{DecodingStrategy};
use edgetransformers::models::base::{BeamSearchParams};
async fn get_test_context() -> Arc<WgpuContext> {
    Arc::new(WgpuContext::new().await.unwrap())
}
#[tokio::main]
async fn main() -> Result<()> {
    let ctx = get_test_context().await;

    let any_model = AnySeq2SeqModel::from_registry(
        ModelType::DistilBartCnn, 
        None,
        // Device::Cpu,
        // None,
        Device::Wgpu,
        Some(ctx),
    )
    .await?;
    let model = match any_model {
        AnySeq2SeqModel::Bart(m) => m,
    };
    //
    //
    // TODO: ADD GELU CONFIG
    //
    //
    let generator = Seq2SeqGenerator::new(Box::new(model));

    let mut generation_config = generator.model.get_default_generation_config();
    generation_config.max_new_tokens = Some(40);

    let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
    type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
    using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
    tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
    like C++, Haskell, and Erlang.";

    let summary = generator.generate(article, &generation_config).await?;

    println!("--- SUMMARY ---");
    println!("{}", summary);

    Ok(())
}
