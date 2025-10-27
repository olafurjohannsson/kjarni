use edgemodels::sentence_encoder::SentenceEncoder;
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Automatically download and load MiniLM
    let encoder = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,  // Use default cache
        Device::Cpu,
        None,
    ).await?;
    
    // Encode sentences
    let sentences = [
        "The cat sits on the mat",
        "A feline rests on a rug",
        "Dogs are playing in the park",
    ];
    
    let embeddings = encoder.encode_batch(&sentences).await?;
    
    // Compute cosine similarity
    let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("Similarity: {:.4}", sim);
    
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}