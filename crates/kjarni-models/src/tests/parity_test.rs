use std::path::Path;

use anyhow::Result;
use tokio::fs;

use kjarni_transformers::models::ModelType;
use kjarni_transformers::prelude::Device;
use kjarni_transformers::WgpuContext;

use crate::models::sentence_encoder::SentenceEncoder;

async fn ensure_model_files(repo_id: &str, local_dir: &Path) -> Result<()> {
    if !local_dir.exists() {
        tokio::fs::create_dir_all(local_dir).await?;
    }

    let files_to_check = ["model.safetensors", "config.json", "tokenizer.json"];
    for filename in files_to_check {
        let local_path = local_dir.join(filename);
        if !local_path.exists() {
            println!("downloading {}...", filename);
            let download_url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                repo_id, filename
            );

            let response = reqwest::get(&download_url).await?.error_for_status()?;
            let content = response.bytes().await?;
            fs::write(&local_path, &content).await?;
            println!("downloaded to {:?}", local_path);
        }
    }
    Ok(())
}

fn assert_embeddings_close(cpu: &[Vec<f32>], gpu: &[Vec<f32>], tolerance: f32, name: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{}: batch size mismatch", name);

    for (batch_idx, (cpu_emb, gpu_emb)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert_eq!(
            cpu_emb.len(),
            gpu_emb.len(),
            "{}: embedding dimension mismatch at batch {}",
            name,
            batch_idx
        );

        let mut max_diff = 0.0f32;
        let mut max_rel_diff = 0.0f32;
        let mut num_large_diffs = 0;

        for (i, (&cpu_val, &gpu_val)) in cpu_emb.iter().zip(gpu_emb.iter()).enumerate() {
            if cpu_val.is_nan() || gpu_val.is_nan() {
                panic!(
                    "{}: nan at batch {} index {}: cpu={}, gpu={}",
                    name, batch_idx, i, cpu_val, gpu_val
                );
            }

            let diff = (cpu_val - gpu_val).abs();
            let rel_diff = if cpu_val.abs() > 1e-6 {
                diff / cpu_val.abs()
            } else {
                diff
            };

            max_diff = max_diff.max(diff);
            max_rel_diff = max_rel_diff.max(rel_diff);

            if diff > tolerance {
                num_large_diffs += 1;
                if num_large_diffs <= 5 {
                    println!(
                        "{}: large diff at batch {} index {}: cpu={:.6}, gpu={:.6}, diff={:.6}",
                        name, batch_idx, i, cpu_val, gpu_val, diff
                    );
                }
            }
        }

        println!(
            "{}: batch {} - max abs diff: {:.6}, max rel diff: {:.6}, large diffs: {}/{}",
            name, batch_idx, max_diff, max_rel_diff, num_large_diffs, cpu_emb.len()
        );

        assert!(
            max_diff < tolerance * 10.0,
            "{}: batch {} max difference {:.6} exceeds 10x tolerance {:.6}",
            name,
            batch_idx,
            max_diff,
            tolerance * 10.0
        );
    }
}

fn get_model_cache_dir(repo_id: &str) -> std::path::PathBuf {
    dirs::cache_dir()
        .unwrap()
        .join("edgegpt")
        .join(repo_id.replace('/', "_"))
}

#[tokio::test]
async fn test_cpu_gpu_parity_single_sentence() -> Result<()> {
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = get_model_cache_dir(model_repo);
    ensure_model_files(model_repo, &cache_dir).await?;

    let cpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Cpu,
        None,
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let gpu_context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Wgpu,
        Some(gpu_context),
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let sentence = "The quick brown fox jumps over the lazy dog.";

    let cpu_embedding = cpu_encoder.encode(sentence).await?;
    let cpu_norm: f32 = cpu_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("cpu norm: {:.6}, first 10: {:?}", cpu_norm, &cpu_embedding[..10]);

    let gpu_embedding = gpu_encoder.encode(sentence).await?;
    let gpu_norm: f32 = gpu_embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("gpu norm: {:.6}, first 10: {:?}", gpu_norm, &gpu_embedding[..10]);

    assert_embeddings_close(&[cpu_embedding], &[gpu_embedding], 1e-3, "single sentence");
    Ok(())
}

#[tokio::test]
async fn test_cpu_gpu_parity_batch() -> Result<()> {
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = get_model_cache_dir(model_repo);
    ensure_model_files(model_repo, &cache_dir).await?;

    let cpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Cpu,
        None,
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let gpu_context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Wgpu,
        Some(gpu_context),
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let sentences = &[
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "WGPU provides a modern graphics and compute API.",
        "This library aims for maximum performance.",
    ];

    let cpu_embeddings = cpu_encoder.encode_batch(sentences).await?;
    println!("cpu batch encoded: {} sentences", cpu_embeddings.len());

    let gpu_embeddings = gpu_encoder.encode_batch(sentences).await?;
    println!("gpu batch encoded: {} sentences", gpu_embeddings.len());

    assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "batch");
    Ok(())
}

#[tokio::test]
async fn test_cpu_gpu_parity_varied_lengths() -> Result<()> {
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = get_model_cache_dir(model_repo);
    ensure_model_files(model_repo, &cache_dir).await?;

    let cpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Cpu,
        None,
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let gpu_context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Wgpu,
        Some(gpu_context),
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let sentences = &[
        "Hi",
        "Hello world!",
        "This is a medium length sentence for testing.",
        "This is a much longer sentence that should test the model's ability to handle various input lengths and still produce consistent embeddings between CPU and GPU implementations.",
    ];

    let cpu_embeddings = cpu_encoder.encode_batch(sentences).await?;
    let gpu_embeddings = gpu_encoder.encode_batch(sentences).await?;

    assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "varied lengths");
    Ok(())
}

#[tokio::test]
async fn test_cpu_gpu_parity_large_batch() -> Result<()> {
    let model_repo = "sentence-transformers/all-MiniLM-L6-v2";
    let cache_dir = get_model_cache_dir(model_repo);
    ensure_model_files(model_repo, &cache_dir).await?;

    let cpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Cpu,
        None,
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let gpu_context = WgpuContext::new().await?;
    let gpu_encoder = SentenceEncoder::from_pretrained(
        &cache_dir,
        Device::Wgpu,
        Some(gpu_context),
        None,
        Some(ModelType::MiniLML6V2),
    )?;

    let base_sentences = vec![
        "The quick brown fox jumps over the lazy dog.",
        "Rust is a systems programming language.",
        "WGPU provides a modern graphics and compute API.",
        "This library aims for maximum performance.",
    ];

    let mut sentences = Vec::new();
    for i in 0..4 {
        for s in &base_sentences {
            sentences.push(format!("{} Iteration {}.", s, i));
        }
    }
    let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();

    let cpu_embeddings = cpu_encoder.encode_batch(&sentence_refs).await?;
    let gpu_embeddings = gpu_encoder.encode_batch(&sentence_refs).await?;

    assert_embeddings_close(&cpu_embeddings, &gpu_embeddings, 1e-3, "large batch");
    Ok(())
}