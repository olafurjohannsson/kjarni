use kjarni_models::cross_encoder::CrossEncoder;
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use kjarni_transformers::WgpuContext;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let ctx = WgpuContext::new().await?;
    // Load cross-encoder
    let cross_encoder = CrossEncoder::from_registry(
        ModelType::MiniLML6V2CrossEncoder,
        None,
        Device::Wgpu,
        Some(ctx),
        None,
    )
        .await?;

    // === Example 1: Score a single pair ===
    let score = cross_encoder
        .predict_pair(
            "How do I train a neural network?",
            "Neural networks are trained using backpropagation and gradient descent.",
        )
        .await?;
    println!("Relevance score: {:.4}", score);

    // === Example 2: Rerank search results ===
    let query = "machine learning algorithms";
    let documents = vec![
        "Machine learning algorithms include decision trees, neural networks, and SVMs.",
        "The weather forecast predicts rain tomorrow.",
        "Deep learning is a subset of machine learning using neural networks.",
        "Cooking recipes for Italian pasta dishes.",
    ];

    let ranked_indices = cross_encoder.rerank(query, &documents).await?;

    println!("\nReranked results:");
    for (rank, &(idx, _)) in ranked_indices.iter().enumerate() {
        let score = cross_encoder.predict_pair(query, documents[idx]).await?;
        println!("{}. [Score: {:.4}] {}", rank + 1, score, documents[idx]);
    }

    Ok(())
}
