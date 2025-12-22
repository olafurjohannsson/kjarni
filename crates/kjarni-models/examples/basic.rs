use kjarni_models::sentence_encoder::SentenceEncoder;
use kjarni_transformers::models::{LanguageModel, ModelType};
use kjarni_transformers::traits::Device;
use kjarni_transformers::WgpuContext;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let context = WgpuContext::new().await?;

    let model = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None, // Use default cache
        Device::Wgpu,
        Some(context),
    )
        .await?;

    let texts = ["Hello world", "How are you"];
    let embeddings = model.encode_batch(&texts).await?;

    for (i, embedding) in embeddings.iter().enumerate() {
        let n = embedding.len().min(10);
        println!("Text: {} == {:?}...", texts[i], &embedding[0..n]);
    }

    let dot: f32 = embeddings[0]
        .iter()
        .zip(&embeddings[1])
        .map(|(a, b)| a * b)
        .sum();
    let norm_a: f32 = embeddings[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = embeddings[1].iter().map(|v| v * v).sum::<f32>().sqrt();
    let cos_sim = dot / (norm_a * norm_b);

    println!(
        "\nCosine similarity ('{}' vs '{}'): {:.4}",
        texts[0], texts[1], cos_sim
    );

    let encoding = model.tokenizer().encode("Hello world", true).unwrap();
    println!("New implementation token IDs: {:?}", encoding.get_ids());

    Ok(())
}
