use super::*;
use crate::Embeddings;
use crate::gpu::embeddings::{GpuEmbeddingWeights, GpuEmbeddings};
use crate::gpu::{GpuFrameContext, GpuTensor};
use crate::traits::{AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelLayout};
use crate::WgpuContext;
use anyhow::Result;
use ndarray::{Array2, Array3};

struct MockEmbedConfig {
    hidden_size: usize,
    vocab_size: usize,
    max_position: usize,
    scale: bool,
}

impl crate::traits::ModelConfig for MockEmbedConfig {
    fn model_type(&self) -> &str {
        "mock_embed"
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> crate::traits::ModelMetadata {
        crate::traits::ModelMetadata {
            decoder_layers: None,
            hidden_size: self.hidden_size,
            vocab_size: self.vocab_size,
            max_seq_len: self.max_position,
            scale_embeddings: self.scale,
            normalize_embedding: false,
            intermediate_size: 0,
            num_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: 4,
            head_dim: self.hidden_size / 4,
            norm_eps: 1e-5,
            activation: crate::activations::Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,

            extra_pos_embeddings: 0,
            is_prenorm: false,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            problem_type: None,
            normalization_strategy: crate::traits::NormalizationStrategy::LayerNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        let decoder_layer = DecoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "layer.{}.q.weight".to_string(),
                q_bias: Some("layer.{}.q.bias".to_string()),
                k_weight: "layer.{}.k.weight".to_string(),
                k_bias: Some("layer.{}.k.bias".to_string()),
                v_weight: "layer.{}.v.weight".to_string(),
                v_bias: Some("layer.{}.v.bias".to_string()),
                o_weight: "layer.{}.o.weight".to_string(),
                o_bias: Some("layer.{}.o.bias".to_string()),
                norm_weight: "layer.{}.attn_ln.weight".to_string(),
                norm_bias: Some("layer.{}.attn_ln.bias".to_string()),
            },
            cross_attn: None,
            ffn: FeedForwardLayout {
                up_weight: "layer.{}.up.weight".to_string(),
                up_bias: Some("layer.{}.up.bias".to_string()),
                down_weight: "layer.{}.down.weight".to_string(),
                down_bias: Some("layer.{}.down.bias".to_string()),
                gate_weight: None,
                gate_bias: None,
                norm_weight: "layer.{}.ffn_ln.weight".to_string(),
                norm_bias: Some("layer.{}.ffn_ln.bias".to_string()),
            },
        };

        ModelLayout {
            token_embedding: "word".to_string(),
            lm_head: "lm_head.weight".to_string(),
            encoder: None,
            decoder: Some(DecoderLayout {
                position_embedding: Some("pos".to_string()),
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some("final_norm.weight".to_string()),
                final_norm_bias: None,
                layer: decoder_layer,
            }),
        }
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
        crate::EmbeddingData::F32(Arc::new(word_emb.clone())),
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

    let input_ids: Vec<u32> = vec![1, 50, 100, 200, 500, 999, 0, 10, 20, 30, 40, 50];
    let input_ids_cpu = Array2::from_shape_vec((batch_size, seq_len), input_ids.clone())?;
    let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids_cpu)?;

    let cpu_output = cpu_embed.forward(&input_ids_cpu, None, 0, false);

    let pool = ctx.get_inference_pool();
    let pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        hidden_size,
        0,
        false,
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
        crate::EmbeddingData::F32(Arc::new(word_emb.clone())),
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
    let pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        hidden_size,
        0,
        false,
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
        crate::EmbeddingData::F32(Arc::new(word_emb.clone())),
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
    let pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    // GPU with offset
    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        position_offset,
        hidden_size,
        0,
        false,
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
        crate::EmbeddingData::F32(Arc::new(word_emb.clone())),
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
    let pool_guard = pool.lock().await;
    let mut frame = GpuFrameContext::new(&ctx, pool_guard);
    let (encoder, pool_ref) = frame.resources();

    // GPU with scaling
    let gpu_output = gpu_embed.encode(
        encoder,
        &gpu_weights,
        &input_ids_gpu,
        None,
        0,
        hidden_size,
        0,
        config.scale,
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
