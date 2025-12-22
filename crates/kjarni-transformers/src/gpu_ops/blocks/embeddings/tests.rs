use super::*;
use crate::activations::Activation;
use crate::embeddings::Embeddings;
use crate::gpu_ops::blocks::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu_ops::{GpuFrameContext, GpuTensor};
use crate::traits::LanguageModelConfig;
use crate::traits::TransformerConfig;
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array2, Array3};
use std::any::Any;
/// Mock config for testing embeddings
struct MockEmbedConfig {
    hidden_size: usize,
    vocab_size: usize,
    max_position: usize,
    scale: bool,
}

impl TransformerConfig for MockEmbedConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        4
    }
    fn num_hidden_layers(&self) -> usize {
        1
    }
    fn layer_norm_eps(&self) -> f32 {
        1e-5
    }
    fn is_causal(&self) -> bool {
        false
    }
    fn is_prenorm(&self) -> bool {
        false
    }
}

impl LanguageModelConfig for MockEmbedConfig {
    fn vocab_size(&self) -> usize {
        self.vocab_size
    }
    fn max_position_embeddings(&self) -> usize {
        self.max_position
    }
    fn intermediate_size(&self) -> usize {
        self.hidden_size * 4
    }

    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        ("word", "pos", None)
    }

    fn scale_embeddings(&self) -> bool {
        self.scale
    }

    fn activation_function(&self) -> Activation {
        Activation::Gelu
    }
    fn decoder_start_token_id(&self) -> u32 {
        0
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn assert_close(cpu: &Array3<f32>, gpu: &Array3<f32>, atol: f32, name: &str) {
    assert_eq!(cpu.shape(), gpu.shape(), "{} shape mismatch", name);
    let max_diff = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("[{}] Max diff: {:.6}", name, max_diff);
    println!(
        "  CPU first 5: {:?}",
        cpu.iter().take(5).collect::<Vec<_>>()
    );
    println!(
        "  GPU first 5: {:?}",
        gpu.iter().take(5).collect::<Vec<_>>()
    );

    if max_diff > atol {
        panic!("[FAIL] {} - max_diff {} > atol {}", name, max_diff, atol);
    }
    println!("[PASS] {}\n", name);
}

fn make_embeddings(
    vocab_size: usize,
    max_pos: usize,
    hidden_size: usize,
) -> (Array2<f32>, Array2<f32>) {
    let word_emb = Array2::from_shape_fn((vocab_size, hidden_size), |(i, j)| {
        ((i * hidden_size + j) % 1000) as f32 * 0.001 - 0.5
    });
    let pos_emb = Array2::from_shape_fn((max_pos, hidden_size), |(i, j)| {
        ((i * hidden_size + j + 5000) % 1000) as f32 * 0.001 - 0.5
    });
    (word_emb, pos_emb)
}

#[tokio::test]
async fn test_word_embedding_lookup_only() -> Result<()> {
    let ctx = Arc::new(WgpuContext::new().await?);

    let vocab_size = 1000;
    let hidden_size = 64;
    let batch_size = 2;
    let seq_len = 6;

    let (word_emb, _) = make_embeddings(vocab_size, 128, hidden_size);

    // CPU

    let cpu_embed = Embeddings::new(
        crate::embeddings::EmbeddingData::F32(word_emb.clone()),
        None,
        None,
    );

    // GPU
    let gpu_weights = GpuEmbeddingWeights {
        word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
        position_embeddings: None,
        token_type_embeddings: None,
    };
    let gpu_embed = GpuEmbeddings::new(&ctx)?;

    let config = MockEmbedConfig {
        hidden_size,
        vocab_size,
        max_position: 128,
        scale: false,
    };

    let input_ids: Vec<u32> = vec![1, 50, 100, 200, 500, 999, 0, 10, 20, 30, 40, 50];
    let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

    let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, false);

    let pool = ctx.get_inference_pool();
    let mut pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        &config,
        pool_ref,
    )?;

    frame.finish();
    let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

    assert_close(&cpu_output, &gpu_output_cpu, 1e-5, "Word Embedding Lookup");
    Ok(())
}

#[tokio::test]
async fn test_word_plus_position_embeddings() -> Result<()> {
    let ctx = Arc::new(WgpuContext::new().await?);

    let vocab_size = 1000;
    let hidden_size = 64;
    let max_pos = 128;
    let batch_size = 2;
    let seq_len = 6;

    let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

    let cpu_embed = Embeddings::new(
        crate::embeddings::EmbeddingData::F32(word_emb.clone()),
        Some(pos_emb.clone()),
        None,
    );

    let gpu_weights = GpuEmbeddingWeights {
        word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
        position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
        token_type_embeddings: None,
    };
    let gpu_embed = GpuEmbeddings::new(&ctx)?;

    let config = MockEmbedConfig {
        hidden_size,
        vocab_size,
        max_position: max_pos,
        scale: false,
    };

    let input_ids: Vec<u32> = vec![1, 50, 100, 200, 500, 999, 0, 10, 20, 30, 40, 50];
    let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

    let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, false);

    let pool = ctx.get_inference_pool();
    let mut pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        &config,
        pool_ref,
    )?;

    frame.finish();
    let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

    assert_close(
        &cpu_output,
        &gpu_output_cpu,
        1e-5,
        "Word + Position Embeddings",
    );
    Ok(())
}

#[tokio::test]
async fn test_embeddings_with_position_offset() -> Result<()> {
    let ctx = Arc::new(WgpuContext::new().await?);

    let vocab_size = 1000;
    let hidden_size = 64;
    let max_pos = 128;
    let batch_size = 1;
    let seq_len = 4;
    let position_offset = 2; // BART style

    let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

    let cpu_embed = Embeddings::new(
        crate::embeddings::EmbeddingData::F32(word_emb.clone()),
        Some(pos_emb.clone()),
        None,
    );

    let gpu_weights = GpuEmbeddingWeights {
        word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
        position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
        token_type_embeddings: None,
    };
    let gpu_embed = GpuEmbeddings::new(&ctx)?;

    let config = MockEmbedConfig {
        hidden_size,
        vocab_size,
        max_position: max_pos,
        scale: false,
    };

    let input_ids: Vec<u32> = vec![1, 50, 100, 200];
    let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

    // CPU with offset
    let cpu_output = cpu_embed.forward(&input_ids_cpu, None, position_offset, false);

    let pool = ctx.get_inference_pool();
    let mut pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    // GPU with offset
    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        position_offset,
        &config,
        pool_ref,
    )?;

    frame.finish();
    let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

    assert_close(
        &cpu_output,
        &gpu_output_cpu,
        1e-5,
        "Embeddings with offset=2",
    );
    Ok(())
}

#[tokio::test]
async fn test_embeddings_with_scaling() -> Result<()> {
    let ctx = Arc::new(WgpuContext::new().await?);

    let vocab_size = 1000;
    let hidden_size = 64;
    let max_pos = 128;
    let batch_size = 1;
    let seq_len = 4;

    let (word_emb, pos_emb) = make_embeddings(vocab_size, max_pos, hidden_size);

    let cpu_embed = Embeddings::new(
        crate::embeddings::EmbeddingData::F32(word_emb.clone()),
        Some(pos_emb.clone()),
        None,
    );

    let gpu_weights = GpuEmbeddingWeights {
        word_embeddings: GpuTensor::from_ndarray(&ctx, &word_emb)?,
        position_embeddings: Some(GpuTensor::from_ndarray(&ctx, &pos_emb)?),
        token_type_embeddings: None,
    };
    let gpu_embed = GpuEmbeddings::new(&ctx)?;

    let config = MockEmbedConfig {
        hidden_size,
        vocab_size,
        max_position: max_pos,
        scale: true, // Enable scaling
    };

    let input_ids: Vec<u32> = vec![1, 50, 100, 200];
    let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

    // CPU with scaling
    let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, true);

    let pool = ctx.get_inference_pool();
    let mut pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    // GPU with scaling
    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        &config,
        pool_ref,
    )?;

    frame.finish();
    let gpu_output_cpu = gpu_output.to_ndarray_3d::<f32>().await?;

    assert_close(
        &cpu_output,
        &gpu_output_cpu,
        1e-4,
        "Embeddings with scaling",
    );
    Ok(())
}
