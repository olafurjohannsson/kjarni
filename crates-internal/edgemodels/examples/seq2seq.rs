use std::sync::Arc;
use anyhow::{Result, anyhow};

// use edgemodels::generation::seq2seq::Seq2SeqGenerator; // Your new generator
// use edgemodels::generation::seq2seq2::Seq2SeqGenerator as Seq2SeqGeneratorNew; // Your new generator
use edgemodels::generation::encoder_decoder::Seq2SeqGenerator;
// use edgemodels::generation::seq2seq::Seq2SeqGenerator as Seq2SeqGeneratorOld;

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

    println!("Loading model...");
    let any_model = AnySeq2SeqModel::from_registry(
        ModelType::DistilBartCnn, 
        None,
        Device::Cpu,
        None, //Some(ctx),
    )
    .await?;
    
    let model = match any_model {
        AnySeq2SeqModel::Bart(m) => m,
    };

    let generator = Seq2SeqGenerator::new(Box::new(model));

    // Get default config
    let mut generation_config = generator.model.get_default_generation_config();
    
    // Print config to match Python output
    println!("\n--- DEFAULT GENERATION CONFIG ---");
    println!("GenerationConfig {{");
    println!("  max_length: {},", generation_config.max_length);
    println!("  min_length: {},", generation_config.min_length);
    println!("  no_repeat_ngram_size: {},", generation_config.no_repeat_ngram_size);
    println!("  repetition_penalty: {:.1},", generation_config.repetition_penalty);
    
    if let DecodingStrategy::BeamSearch(params) = &generation_config.strategy {
        println!("  num_beams: {},", params.num_beams);
        println!("  length_penalty: {:.1},", params.length_penalty);
        println!("  early_stopping: {},", params.early_stopping);
    }
    println!("}}");

    let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
    type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
    using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
    tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
    like C++, Haskell, and Erlang.";

    println!("\n--- GENERATING SUMMARY ---");
    let summary = generator.generate(article, &generation_config).await?;

    println!("\n--- FINAL GENERATED TEXT ---");
    println!("{}", summary);

    Ok(())
}