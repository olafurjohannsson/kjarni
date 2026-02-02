use anyhow::{Result, anyhow};
use kjarni_transformers::gpu::{GpuFrameContext, GpuTensor};
use kjarni_transformers::models::base::{ModelInput, ModelLoadConfig};

use kjarni_transformers::{
    models::{ModelArchitecture, ModelTask, ModelType},
    traits::{Device, ModelConfig, ModelLayout, ModelMetadata},
    weights::ModelWeights,
};
use ndarray::{Array2, Array3, s};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::{EncodeInput, Tokenizer};

#[cfg(test)]
mod cross_encoder_tests {
    use kjarni_transformers::WgpuContext;

    use crate::CrossEncoder;

    use super::*;

    #[tokio::test]
    async fn test_cross_encoder_predict() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let score = encoder
            .predict_pair("i love kjarni", "kjarni is a new model inference library")
            .await?;

        // Cross-encoder should return a finite score
        assert!(score.is_finite());
        println!("Score: {}", score);

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_torch_parity() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let score = encoder
            .predict_pair("i love edgeGPT", "edgeGPT is a new model inference library")
            .await?;

        // Golden value from PyTorch
        let torch_value = 3.1776933670043945;
        assert!(
            (score - torch_value).abs() < 1e-3,
            "Score {} doesn't match torch value {}",
            score,
            torch_value
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let query = "machine learning algorithms";
        let documents = vec![
            "Machine learning algorithms include decision trees, neural networks, and SVMs.",
            "The weather forecast predicts rain tomorrow.",
            "Deep learning is a subset of machine learning using neural networks.",
            "Cooking recipes for Italian pasta dishes.",
        ];

        let ranked = encoder.rerank(query, &documents).await?;

        // Should return all 4 documents
        assert_eq!(ranked.len(), 4);

        // ML-related docs should rank higher than unrelated ones
        // Expected order: [0, 2, 3, 1] based on torch golden values
        let expected_indices: Vec<usize> = vec![0, 2, 3, 1];
        let actual_indices: Vec<usize> = ranked.iter().map(|v| v.0).collect();
        assert_eq!(actual_indices, expected_indices);

        // Scores should be sorted descending
        for i in 1..ranked.len() {
            assert!(ranked[i - 1].1 >= ranked[i].1);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank_empty() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let ranked = encoder.rerank("query", &[]).await?;
        assert!(ranked.is_empty());

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_rerank_top_k() -> Result<()> {
        let encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let documents = vec!["Doc 1", "Doc 2", "Doc 3", "Doc 4", "Doc 5"];

        let top_2 = encoder.rerank_top_k("query", &documents, 2).await?;
        assert_eq!(top_2.len(), 2);

        // Requesting more than available should return all
        let top_10 = encoder.rerank_top_k("query", &documents, 10).await?;
        assert_eq!(top_10.len(), 5);

        Ok(())
    }

    #[tokio::test]
    async fn test_cross_encoder_gpu() -> Result<()> {
        let context = WgpuContext::new().await?;
        let gpu_encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Wgpu,
            Some(context),
            None,
        )
        .await?;

        let cpu_encoder = CrossEncoder::from_registry(
            ModelType::MiniLML6V2CrossEncoder,
            None,
            Device::Cpu,
            None,
            None,
        )
        .await?;

        let cpu_score = cpu_encoder
            .predict_pair("test query", "test document")
            .await?;
        let gpu_score = gpu_encoder
            .predict_pair("test query", "test document")
            .await?;

        // CPU and GPU should produce same results
        assert!(
            (cpu_score - gpu_score).abs() < 1e-3,
            "CPU {} vs GPU {}",
            cpu_score,
            gpu_score
        );

        Ok(())
    }
}
