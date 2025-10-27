use edgemodels::sentence_encoder::SentenceEncoder;
use edgetransformers::models::ModelType;
use edgetransformers::traits::Device;
use edgetransformers::gpu_context::WgpuContext;
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = Arc::new(WgpuContext::new().await);

    // Automatically download and load MiniLM
    let encoder = SentenceEncoder::from_registry(
        ModelType::MiniLML6V2,
        None,  // Use default cache
        Device::Wgpu,
        Some(ctx),
    ).await?;
    let encpuder_cpu = SentenceEncoder::from_registry(
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
    let embeddings2 = encpuder_cpu.encode_batch(&sentences).await?;
    
    // Compute cosine similarity
    let sim = cosine_similarity(&embeddings[0], &embeddings[1]);
    println!("Similarity GPU: {:.4}", sim);
    let sim2 = cosine_similarity(&embeddings2[0], &embeddings2[1]);
    println!("Similarity CPU: {:.4}", sim2);
    
    Ok(())
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}