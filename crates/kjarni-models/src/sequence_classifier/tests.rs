use crate::cross_encoder::CrossEncoder;
use anyhow::Result;
use kjarni_transformers::models::ModelType;
use kjarni_transformers::traits::Device;
use kjarni_transformers::WgpuContext;
use tokio;

#[tokio::test]
async fn test_torch_cross_encoder_predict() -> Result<()> {
    let cpu_encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let context = WgpuContext::new().await?;
    let gpu_encoder = CrossEncoder::from_registry(
        ModelType::MiniLML6V2CrossEncoder,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
        .await?;
    let cpu_score = cpu_encoder
        .predict_pair("i love edgeGPT", "edgeGPT is a new model inference library")
        .await?;

    let torch_value = 3.1776933670043945;
    assert!((cpu_score - torch_value).abs() < 1e-3);
    let gpu_score = gpu_encoder
        .predict_pair("i love edgeGPT", "edgeGPT is a new model inference library")
        .await?;

    assert!((gpu_score - torch_value).abs() < 1e-3);

    assert!((cpu_score - gpu_score).abs() < 1e-3);
    Ok(())
}
#[tokio::test]
async fn test_cross_encoder_rerank_simple_validation() -> Result<()> {
    let cpu_encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let ranked = cpu_encoder.rerank("query", &[]).await?;
    assert!(
        ranked.is_empty(),
        "Reranking with no documents should return an empty vec"
    );
    let docs = ["doc1", "doc2", "doc3"];
    let ranked = cpu_encoder.rerank("query", &docs).await?;
    assert_eq!(
        ranked.len(),
        docs.len(),
        "Reranked output should have the same length as input documents"
    );
    Ok(())
}
#[tokio::test]
async fn test_cross_encoder_rerank_torch_parity() -> Result<()> {
    let query = "machine learning algorithms";
    let documents = vec![
        "Machine learning algorithms include decision trees, neural networks, and SVMs.", // Index 0
        "The weather forecast predicts rain tomorrow.",                                   // Index 1
        "Deep learning is a subset of machine learning using neural networks.",           // Index 2
        "Cooking recipes for Italian pasta dishes.",                                      // Index 3
    ];
    let expected_indices: Vec<usize> = vec![0, 2, 3, 1];
    let cpu_encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let cpu_indices = cpu_encoder.rerank(query, &documents).await?;
    let u: Vec<usize> = cpu_indices.iter().map(|v| v.0).collect();
    assert_eq!(
        u, expected_indices,
        "CPU rerank order does not match Torch!"
    );
    let context = WgpuContext::new().await?;
    let gpu_encoder = CrossEncoder::from_registry(
        ModelType::MiniLML6V2CrossEncoder,
        None,
        Device::Wgpu,
        Some(context),
        None,
    )
        .await?;
    let gpu_indices = gpu_encoder.rerank(query, &documents).await?;
    let gpu_rank: Vec<usize> = gpu_indices.iter().map(|v| v.0).collect();
    let cpu_rank: Vec<usize> = cpu_indices.iter().map(|v| v.0).collect();
    assert_eq!(gpu_rank, cpu_rank, "GPU rerank order does not match CPU!");

    Ok(())
}
#[tokio::test]
async fn test_rerank_returns_scores() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let query = "machine learning";
    let documents = vec![
        "Machine learning is awesome",
        "The weather is nice today",
        "Deep learning uses neural networks",
    ];
    let ranked = encoder.rerank(query, &documents).await?;
    assert_eq!(ranked.len(), 3);
    for (idx, score) in &ranked {
        assert!(*idx < documents.len(), "Index {} out of bounds", idx);
        assert!(score.is_finite(), "Score should be finite");
    }
    for i in 1..ranked.len() {
        assert!(
            ranked[i - 1].1 >= ranked[i].1,
            "Scores should be sorted in descending order"
        );
    }
    Ok(())
}

#[tokio::test]
async fn test_rerank_indices_backwards_compat() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let query = "machine learning";
    let documents = vec![
        "Machine learning is awesome",
        "The weather is nice today",
        "Deep learning uses neural networks",
    ];
    let ranked_with_scores = encoder.rerank(query, &documents).await?;
    let ranked_indices = encoder.rerank_indices(query, &documents).await?;
    let indices_from_scored: Vec<usize> = ranked_with_scores.iter().map(|(idx, _)| *idx).collect();
    assert_eq!(ranked_indices, indices_from_scored);
    Ok(())
}

#[tokio::test]
async fn test_rerank_top_k() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let query = "machine learning";
    let documents = vec![
        "Machine learning is awesome",          // Should rank high
        "The weather is nice today",            // Should rank low
        "Deep learning uses neural networks",   // Should rank high
        "I like pizza",                         // Should rank low
        "AI and ML are transforming the world", // Should rank high
    ];
    let top_2 = encoder.rerank_top_k(query, &documents, 2).await?;
    assert_eq!(top_2.len(), 2, "Should return exactly 2 results");
    assert!(
        top_2[0].1 >= top_2[1].1,
        "Top K results should be sorted by score"
    );
    let all_ranked = encoder.rerank(query, &documents).await?;
    assert_eq!(top_2[0], all_ranked[0]);
    assert_eq!(top_2[1], all_ranked[1]);
    Ok(())
}

#[tokio::test]
async fn test_rerank_top_k_indices() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let query = "machine learning";
    let documents = vec![
        "Machine learning is awesome",
        "The weather is nice today",
        "Deep learning uses neural networks",
        "I like pizza",
    ];
    let top_2_with_scores = encoder.rerank_top_k(query, &documents, 2).await?;
    let top_2_indices = encoder.rerank_top_k_indices(query, &documents, 2).await?;
    assert_eq!(top_2_indices.len(), 2);
    let expected_indices: Vec<usize> = top_2_with_scores.iter().map(|(idx, _)| *idx).collect();
    assert_eq!(top_2_indices, expected_indices);
    Ok(())
}

#[tokio::test]
async fn test_rerank_empty_documents() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let ranked = encoder.rerank("query", &[]).await?;
    assert!(ranked.is_empty());
    let ranked_indices = encoder.rerank_indices("query", &[]).await?;
    assert!(ranked_indices.is_empty());
    let top_k = encoder.rerank_top_k("query", &[], 5).await?;
    assert!(top_k.is_empty());
    Ok(())
}

#[tokio::test]
async fn test_rerank_top_k_exceeds_document_count() -> Result<()> {
    let encoder =
        CrossEncoder::from_registry(ModelType::MiniLML6V2CrossEncoder, None, Device::Cpu, None, None)
            .await?;
    let documents = vec!["Doc 1", "Doc 2", "Doc 3"];
    let top_10 = encoder.rerank_top_k("query", &documents, 10).await?;
    assert_eq!(top_10.len(), 3);
    Ok(())
}
