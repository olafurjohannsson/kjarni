
use kjarni::chat::Chat;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let gguf_dir = dirs::cache_dir()
        .expect("no cache dir")
        .join("kjarni")
        .join("llama-3.2-3b-instruct-q4_k_m");

    let chat = Chat::builder("llama3.2-3b-instruct")
        .model_path(gguf_dir)
        .temperature(0.7)
        .top_p(0.9)
        .max_tokens(256)
        .repetition_penalty(1.1)
        .build()
        .await?;

    println!("=== RUST GGUF Q4_K_M ===");
    println!("Model: {}", chat.model_name());
    println!();

    // Non-streaming first
    let response = chat.send("Explain three benefits of quantization in neural networks.").await?;
    println!("{}", response);

    Ok(())
}