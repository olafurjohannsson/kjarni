use edgemodels::seq2seq::Seq2SeqModel;
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;
use edgetransformers::Seq2SeqLanguageModel;
use edgetransformers::models::base::GenerationConfig;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("--- BART Summarization Example (CPU) ---");

    // Load the BART model from the registry.
    // This will automatically download the model files on the first run.
    println!("Loading BART model...");
    let summarizer = Seq2SeqModel::from_registry(
        ModelType::DistilBartCnn,
        None, // Use default cache directory
        Device::Cpu,
        None, // No WGPU context needed for CPU
    ).await?;
    println!("✓ Model loaded.");

    let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
    type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
    using a garbage collector. To simultaneously enforce memory safety and prevent data races, its 'borrow checker' \
    tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages \
    like C++, Haskell, and Erlang.";

    println!("\n--- ARTICLE ---");
    println!("{}", article);

    // Generate the summary
    println!("\nGenerating summary...");
    let cfg = summarizer.generation_config_from_preset();
    
    let summary = summarizer.generate(article, &cfg).await?; // Generate a summary with a max length of 60 tokens

    println!("\n--- GENERATED SUMMARY ---");
    println!("{}", summary);
    
    Ok(())
}