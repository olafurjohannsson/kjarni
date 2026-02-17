
use kjarni::chat::Chat;
use kjarni::GenerationStats;
use futures::StreamExt;
use std::path::{Path, PathBuf};
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    env_logger::init();
    GenerationStats::enable();
        let gguf_dir = dirs::cache_dir()
        .expect("no cache dir")
        .join("kjarni")
        .join("llama-3.2-3b-instruct-q4_k_m");
    let chat = Chat::builder("llama3.2-3b-instruct").
        model_path(gguf_dir)
        .temperature(0.7)
        // .greedy()
        .top_p(0.9)
        .max_tokens(256)
        // .gpu()
        .repetition_penalty(1.1)
        .build()
        .await?;

    println!("=== RUST SAFETENSORS ===");
    println!("Model: {}", chat.model_name());
    println!();

    // Non-streaming first
    let response = chat.send("Explain three benefits of quantization in neural networks.").await?;
    println!("{}", response);
    
    Ok(())
}