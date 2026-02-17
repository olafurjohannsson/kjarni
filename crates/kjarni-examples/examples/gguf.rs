
use kjarni::chat::Chat;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let gguf_dir = dirs::cache_dir()
        .expect("no cache dir")
        .join("kjarni")
        .join("llama-3.2-3b-instruct-q4_k_m");

    println!("Loading GGUF from {:?}...", gguf_dir);

    let chat = Chat::builder("llama3.2-3b-instruct")
        .model_path(gguf_dir)
        .build()
        .await?;

    println!("Model: {}", chat.model_name());
    println!("Context: {} tokens\n", chat.context_size());

    // The quantized model works exactly like the full-precision version.
    // Same API, same streaming, same conversation management.

    let response = chat.send("What is Rust? One sentence.").await?;
    println!("Response: {}\n", response);

    // Streaming works the same way
    print!("Streaming: ");
    let mut stream = chat.stream("Name three benefits of quantization.").await?;
    while let Some(token) = stream.next().await {
        print!("{}", token?);
    }
    println!("\n");

    // Multi-turn conversation
    let mut convo = chat.conversation();

    let r1 = convo.send("I'm building a search engine in Rust.").await?;
    println!("Turn 1: {}", r1);

    let r2 = convo.send("What crates should I look at?").await?;
    println!("Turn 2: {}", r2);

    let r3 = convo.send("Which one would you start with and why?").await?;
    println!("Turn 3: {}", r3);

    Ok(())
}