use kjarni_models::sequence_classifier::{SequenceClassifier, zero_shot::ZeroShotClassifier};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use anyhow::Result;
#[tokio::main]
async fn main() -> Result<()> {
    let nli_model_type = ModelType::BartLargeMNLI;
    
    // --- BATCH INPUTS OF DIFFERENT LENGTHS ---
    let texts_to_classify = &[
        "The company announced record profits and a stock buyback plan.", // Long sentence
        "A player scored a goal.", // Short sentence
    ];
    let candidate_labels = ["politics", "sports", "technology", "business"];

    println!("--- Step 1: Loading the underlying NLI model ('{}') ---", nli_model_type.cli_name());
    let nli_model = SequenceClassifier::from_registry(
        nli_model_type,
        None,
        Device::Cpu,
        None,
        None,
    ).await?;
    println!("NLI model loaded successfully.");

    println!("\n--- Step 2: Creating the ZeroShotClassifier engine ---");
    let zero_shot_classifier = ZeroShotClassifier::new(nli_model)?;
    println!("ZeroShotClassifier engine created.");

    println!("\n--- Step 3: Running classification on a mixed-length batch ---");
    println!("Classifying {} texts against {} labels...", texts_to_classify.len(), candidate_labels.len());

    // This single call will create a batch of 8 pairs (4 for the long text, 4 for the short one)
    let results_batch = zero_shot_classifier
        .classify(texts_to_classify, &candidate_labels)
        .await?;

    println!("\n--- ALL DONE ---");
    
    // --- Print results for the first text ---
    println!("\n--- Results for: \"{}\" ---", texts_to_classify[0]);
    let first_results = &results_batch[0];
    for (label, score) in first_results {
        let bar_len = (score * 50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  - {:<12}: {:.2}% {}", label, score * 100.0, bar);
    }
    
    // --- Print results for the second text ---
    println!("\n--- Results for: \"{}\" ---", texts_to_classify[1]);
    let second_results = &results_batch[1];
    for (label, score) in second_results {
        let bar_len = (score * 50.0) as usize;
        let bar = "█".repeat(bar_len);
        println!("  - {:<12}: {:.2}% {}", label, score * 100.0, bar);
    }

    Ok(())
}