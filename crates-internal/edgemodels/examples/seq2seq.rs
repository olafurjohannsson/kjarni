use anyhow::{Result, anyhow};
use edgemodels::generation::seq2seq::Seq2SeqGenerator; // Your new generator
use edgemodels::seq2seq::{AnySeq2SeqModel, BartConfig, Seq2SeqModel, TaskSpecificParams};
use edgetransformers::models::base::GenerationConfig;
use edgetransformers::{Device, ModelType};
use edgemodels::generation::{DecodingStrategy};
use edgetransformers::models::base::{BeamSearchParams};

#[tokio::main]
async fn main() -> Result<()> {
    let any_model = AnySeq2SeqModel::from_registry(
        ModelType::DistilBartCnn, 
        None,
        Device::Cpu,
        None,
    )
    .await?;
    let model = match any_model {
        AnySeq2SeqModel::Bart(m) => m,
    };

    let generator = Seq2SeqGenerator::new(Box::new(model));

    // ✅ This is now the clean, correct way to get the config
    let mut generation_config = generator.model.get_default_generation_config();
    generation_config.max_new_tokens = Some(40);
    generation_config.repetition_penalty = 1.1;
    // generation_config.strategy = DecodingStrategy::BeamSearch(BeamSearchParams{
    //     num_beams: 4,
    //     length_penalty: 2.0,
    //     early_stopping: true,
    // });

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
