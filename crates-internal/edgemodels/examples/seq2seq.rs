
use edgemodels::seq2seq::{Seq2SeqModel, configs::TaskSpecificParams};
use edgemodels::seq2seq::configs::BartConfig;
use edgemodels::generation::seq2seq::Seq2SeqGenerator; // Your new generator
use edgetransformers::models::base::GenerationConfig;
use anyhow::{anyhow, Result};
use edgetransformers::{Device, ModelType};

#[tokio::main]
async fn main() -> Result<()> {
   let model = Seq2SeqModel::from_registry(
        ModelType::DistilBartCnn, // Or BartLargeCnn
        None,
        Device::Cpu,
        None,
    ).await?;

    let generator = Seq2SeqGenerator::new(Box::new(model));
    
    // ✅ This is now the clean, correct way to get the config
    let generation_config = generator.model.generation_config_from_preset();

    let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
    type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
    using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
    tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
    like C++, Haskell, and Erlang.";
    
    let cfg = GenerationConfig {
    num_beams: 4,
    max_new_tokens: Some(40),
    ..Default::default()
    };
    let summary = generator.generate(article, &cfg).await?;

    println!("--- SUMMARY ---");
    println!("{}", summary);

    Ok(())
}