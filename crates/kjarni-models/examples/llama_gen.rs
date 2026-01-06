use kjarni_models::models::llama::model::LlamaModel;
use kjarni_transformers::common::GenerationConfig;
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::models::base::ModelLoadConfig;
use kjarni_transformers::stats::GenerationStats;
use kjarni_transformers::{Device, ModelType, WgpuContext};
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // This test verifies that our Llama implementation produces a known, correct output
    // for a deterministic (greedy) generation task.
    kjarni_transformers::utils::configure_threading();
    // 1. Setup: Define the model, prompt, and expected output.
    // let model_type = ModelType::Llama3_2_1B;
    let prompt = "The field of Artificial Intelligence has seen a lot of progress";

    // The "golden" output string for generating 5 new tokens, based on previous correct runs.
    let expected_output =
        "The field of Artificial Intelligence has seen a lot of progress in the last few years.";

    // Create a config for deterministic, greedy decoding.
    let config = GenerationConfig {
        max_new_tokens: Some(6),
        // strategy: DecodingStrategy::Greedy,
        repetition_penalty: 1.0, // No penalty.
        add_bos_token: true,     // CRITICAL for Llama models.
        ..Default::default()
    };
    // decoder_layer::tests::test_decoder_layer_with_rope_and_gqa

    // 2. Load model and create the generator.
    println!("Running Meta Llama 3.2 1B");
    let llama_model = LlamaModel::from_pretrained(
        std::path::Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B-Instruct"),
        Device::Cpu,
        None,
        None,
        None,
    )?;
    GenerationStats::enable();
    println!("Creating DecoderGenerator");
    let generator = DecoderGenerator::new(std::sync::Arc::new(llama_model))?;
    println!("Generating text");
    // 3. Execute the generation.
    let generated_text = generator.generate(prompt, &config, None).await?;

    let concat_prompt = prompt.to_string() + "" + &generated_text;
    assert_eq!(concat_prompt.trim(), expected_output.trim());

    Ok(())
}
