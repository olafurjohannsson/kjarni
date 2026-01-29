use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use ndarray::{s, Array1, Array2, Array3, Array4};

use kjarni_transformers::common::{DecodingStrategy, GenerationConfig};
use kjarni_transformers::decoder::prelude::*;
use kjarni_transformers::gpu::normalization::{GpuRMSNorm, GpuRMSNormWeights};
use kjarni_transformers::gpu_ops::{GpuFrameContext, GpuTensor, Kernel};
use kjarni_transformers::models::base::{ModelInput, ModelLoadConfig};
use kjarni_transformers::models::ModelType;
use kjarni_transformers::normalization::RMSNorm;
use kjarni_transformers::rope::RoPE;
use kjarni_transformers::tensor::DType;
use kjarni_transformers::traits::{Device, ModelConfig};
use kjarni_transformers::{DecoderPipeline, WgpuContext};

use crate::models::gpt2::Gpt2Model;
use crate::models::llama::cpu_decoder::LlamaCpuDecoder;
use crate::models::llama::gpu_decoder::LlamaGpuDecoder;
use crate::models::llama::LlamaModel;

#[tokio::test]
async fn test_full_text_generation_parity() -> Result<()> {
    let model_type = ModelType::DistilGpt2;
    let prompt = "Alan Turing was a";
    let config = GenerationConfig {
        max_new_tokens: Some(3),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };
    {
        let cpu_generator =
            Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;
        let cpu_gen = DecoderGenerator::new(Arc::new(cpu_generator))?;
        let cpu_generated_text = cpu_gen.generate(prompt, &config, None).await?;
        let context = WgpuContext::new().await?;
        let gpu_generator =
            Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(context), None).await?;
        let gpu_gen = DecoderGenerator::new(Arc::new(gpu_generator))?;
        let gpu_generated_text = gpu_gen.generate(prompt, &config, None).await?;
        assert_eq!(
            cpu_generated_text, gpu_generated_text,
            "cpu and gpu generated text did not match"
        );
    }
    kjarni_transformers::weights::clear_mmap_cache();

    Ok(())
}

fn assert_close_4d(cpu: &Array4<f32>, gpu: &Array4<f32>, atol: f32, name: &str) {
    assert_eq!(cpu.shape(), gpu.shape(), "[{}] shape mismatch", name);

    let max_diff = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff <= atol,
        "[{}] max_diff {:.6} > atol {}",
        name,
        max_diff,
        atol
    );
}

fn assert_close_3d(cpu: &Array3<f32>, gpu: &Array3<f32>, atol: f32, name: &str) {
    assert_eq!(
        cpu.shape(),
        gpu.shape(),
        "[{}] shape mismatch: {:?} vs {:?}",
        name,
        cpu.shape(),
        gpu.shape()
    );

    let max_diff = cpu
        .iter()
        .zip(gpu.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff <= atol,
        "[{}] max_diff {:.6} > atol {}",
        name,
        max_diff,
        atol
    );
}

#[tokio::test]
async fn test_rms_norm_isolated_parity() -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let batch = 1;
    let seq_len = 6;
    let hidden_size = 2048;
    let eps = 1e-5;

    let input_cpu = Array3::<f32>::from_shape_fn((batch, seq_len, hidden_size), |(b, s, h)| {
        let idx = (b * seq_len * hidden_size + s * hidden_size + h) as f32;
        (idx * 0.001).sin() * 0.5
    });

    let gamma_cpu = Array1::<f32>::from_shape_fn(hidden_size, |i| {
        1.0 + (i as f32 * 0.0001).sin() * 0.1
    });

    let cpu_norm = RMSNorm::new(gamma_cpu.clone(), eps);
    let cpu_output = cpu_norm.forward_3d(&input_cpu);

    let input_gpu = GpuTensor::from_ndarray(&ctx, &input_cpu)?;
    let gamma_gpu = GpuTensor::from_ndarray(&ctx, &gamma_cpu)?;

    let gpu_norm = GpuRMSNorm::new(&ctx, eps);
    let gpu_weights = GpuRMSNormWeights::new(gamma_gpu)?;

    let pool = ctx.get_inference_pool();
    let gpu_output = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(input_gpu.shape().to_vec());
        gpu_norm.encode(enc, &gpu_weights, &input_gpu, &out);

        frame.finish();
        out.to_ndarray_3d::<f32>().await?
    };

    assert_close_3d(&cpu_output, &gpu_output, 1e-4, "RMSNorm (synthetic)");

    let input_large = Array3::<f32>::from_shape_fn((batch, seq_len, hidden_size), |(_, s, h)| {
        ((s * hidden_size + h) as f32) * 0.01 - 10.0
    });

    let cpu_output_large = cpu_norm.forward_3d(&input_large);

    let input_large_gpu = GpuTensor::from_ndarray(&ctx, &input_large)?;
    let gpu_output_large = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(input_large_gpu.shape().to_vec());
        gpu_norm.encode(enc, &gpu_weights, &input_large_gpu, &out);

        frame.finish();
        out.to_ndarray_3d::<f32>().await?
    };

    assert_close_3d(
        &cpu_output_large,
        &gpu_output_large,
        1e-4,
        "RMSNorm (large values)",
    );

    Ok(())
}

#[tokio::test]
async fn test_rms_norm_bf16_gamma_parity() -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let batch = 1;
    let seq_len = 6;
    let hidden_size = 2048;
    let eps = 1e-5;

    let input_cpu = Array3::<f32>::from_shape_fn((batch, seq_len, hidden_size), |(_, s, h)| {
        (s * hidden_size + h) as f32 * 0.001 - 0.5
    });

    let gamma_cpu =
        Array1::<f32>::from_shape_fn(hidden_size, |i| 1.0 + (i as f32 * 0.0001).sin() * 0.1);

    let cpu_norm = RMSNorm::new(gamma_cpu.clone(), eps);
    let cpu_output = cpu_norm.forward_3d(&input_cpu);

    let input_gpu = GpuTensor::from_ndarray(&ctx, &input_cpu)?;

    let gamma_bf16: Vec<half::bf16> = gamma_cpu.iter().map(|&v| half::bf16::from_f32(v)).collect();
    let gamma_bf16_bytes: &[u8] = bytemuck::cast_slice(&gamma_bf16);

    let gamma_gpu = GpuTensor::from_bytes(
        &ctx,
        gamma_bf16_bytes,
        vec![hidden_size],
        DType::BF16,
        "gamma_bf16",
    )?;

    let gpu_norm = GpuRMSNorm::new(&ctx, eps);
    let gpu_weights = GpuRMSNormWeights::new(gamma_gpu)?;

    let pool = ctx.get_inference_pool();
    let gpu_output = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(input_gpu.shape().to_vec());
        gpu_norm.encode(enc, &gpu_weights, &input_gpu, &out);

        frame.finish();
        out.to_ndarray_3d::<f32>().await?
    };

    let max_diff = cpu_output
        .iter()
        .zip(gpu_output.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff <= 0.01,
        "bf16 gamma test failed: max_diff {:.6}",
        max_diff
    );

    Ok(())
}

async fn test_layer0_attention_vs_ffn_isolation(dtype: DType) -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let config = ModelLoadConfig {
        target_dtype: Some(dtype),
        ..Default::default()
    };
    {
        let cpu_model = LlamaModel::from_pretrained(
            Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
            Device::Cpu,
            None,
            Some(config),
            None,
        )?;

        let gpu_model = LlamaModel::from_pretrained(
            Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
            Device::Wgpu,
            Some(ctx.clone()),
            Some(config),
            None,
        )?;

        let cpu_pipeline: &DecoderPipeline = cpu_model.pipeline();
        let gpu_pipeline: &DecoderPipeline = gpu_model.pipeline();

        let cpu_decoder = cpu_pipeline
            .cpu_decoder()
            .expect("no cpu decoder")
            .as_any()
            .downcast_ref::<LlamaCpuDecoder>()
            .expect("failed to downcast cpu decoder");
        let gpu_decoder = gpu_pipeline
            .gpu_decoder()
            .expect("no gpu decoder")
            .as_any()
            .downcast_ref::<LlamaGpuDecoder>()
            .expect("failed to downcast gpu decoder");

        let config = cpu_model.config();
        let meta = config.metadata();

        let input_tokens: Vec<u32> = vec![1, 15043, 29892, 590, 1024, 338];
        let input_ids = Array2::from_shape_vec((1, input_tokens.len()), input_tokens.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids)?;

        let seq_len = input_tokens.len();
        let attention_mask = Array2::<f32>::ones((1, seq_len));
        let attention_mask_gpu = GpuTensor::from_ndarray(&ctx, &attention_mask)?;

        let position_offset = 0usize;
        let pool = ctx.get_inference_pool();

        let ops = cpu_model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow::anyhow!("model does not support cpu execution"))?;

        let cpu_embeddings = ops.embed(&input_ids, position_offset)?;

        let gpu_embeddings = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let emb = gpu_decoder.embed(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                position_offset,
            )?;

            frame.finish();
            emb.to_ndarray_3d::<f32>().await?
        };

        assert_close_3d(&cpu_embeddings, &gpu_embeddings, 1e-4, "Embeddings");

        let cpu_layer0 = &cpu_decoder.layers[0];
        let gpu_layer0 = &gpu_decoder.layers[0];

        let layer_input = cpu_embeddings.clone();
        let layer_input_gpu = GpuTensor::from_ndarray(&ctx, &layer_input)?;

        let cpu_attn_block_out = {
            let residual = &layer_input;
            let kv_dim = meta.num_kv_heads * meta.head_dim;
            let normed = cpu_layer0.attention_norm.forward(residual);
            let (b, s, _) = normed.dim();
            let mut temp_k = Array3::<f32>::zeros((b, s, kv_dim));
            let mut temp_v = Array3::<f32>::zeros((b, s, kv_dim));
            let attn_out = cpu_layer0.attention.forward(
                &normed,
                Some(&attention_mask),
                temp_k.view_mut(),
                temp_v.view_mut(),
                0,
                Some(&cpu_layer0.rope),
            )?;

            residual + &attn_out
        };

        let gpu_attn_block_out = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let residual = &layer_input_gpu;

            let normed = pool_ref.get(residual.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                residual,
                &normed,
            );

            let attn_output = gpu_layer0.self_attn.forward(
                enc,
                &normed,
                &gpu_layer0.self_attn_weights,
                &gpu_decoder.gpu_rope,
                &attention_mask_gpu,
                None,
                position_offset,
                pool_ref,
            )?;

            let out = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .add
                .encode(enc, &[residual, &attn_output.hidden_states], &out);

            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        let attn_diff = cpu_attn_block_out
            .iter()
            .zip(gpu_attn_block_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(attn_diff <= 0.01, "attention block diverges: {:.6}", attn_diff);

        let ffn_input = cpu_attn_block_out.clone();

        let cpu_ffn_block_out = {
            let residual = &ffn_input;
            let normed = cpu_layer0.ffn_norm.forward(residual);
            let ffn_out = cpu_layer0.feed_forward.forward(&normed)?;
            residual + &ffn_out
        };

        let ffn_input_gpu = GpuTensor::from_ndarray(&ctx, &ffn_input)?;

        let gpu_ffn_block_out = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let residual = &ffn_input_gpu;

            let normed = pool_ref.get(residual.shape().to_vec());
            gpu_layer0
                .ffn_norm
                .encode(enc, &gpu_layer0.ffn_norm_weights, residual, &normed);

            let (b, s, h) = normed.dims3();
            let normed_2d = normed.view(vec![b * s, h]);
            let ffn_out_2d = pool_ref.get(vec![b * s, h]);

            gpu_layer0.feedforward.encode(
                enc,
                &gpu_layer0.ff_weights,
                &normed_2d,
                &ffn_out_2d,
                pool_ref,
            );

            let ffn_out = ffn_out_2d.view(vec![b, s, h]);

            let out = pool_ref.get(residual.shape().to_vec());
            gpu_layer0.add.encode(enc, &[residual, &ffn_out], &out);

            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        let ffn_diff = cpu_ffn_block_out
            .iter()
            .zip(gpu_ffn_block_out.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(ffn_diff <= 0.01, "ffn block diverges: {:.6}", ffn_diff);
    }
    kjarni_transformers::weights::clear_mmap_cache();

    Ok(())
}

#[tokio::test]
async fn test_layer0_attention_vs_ffn_isolation_bf16() -> Result<()> {
    test_layer0_attention_vs_ffn_isolation(DType::BF16).await
}

#[tokio::test]
async fn test_llama_cpu_gpu_step_by_step_parity_bf16() -> Result<()> {
    test_llama_cpu_gpu_step_by_step_parity(DType::BF16).await
}

#[tokio::test]
async fn test_llama_cpu_gpu_step_by_step_parity_f32() -> Result<()> {
    test_llama_cpu_gpu_step_by_step_parity(DType::F32).await
}

async fn test_llama_cpu_gpu_step_by_step_parity(dtype: DType) -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let config = ModelLoadConfig {
        target_dtype: Some(dtype),
        ..Default::default()
    };
    {
        let cpu_model = LlamaModel::from_pretrained(
            Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
            Device::Cpu,
            None,
            Some(config),
            None,
        )?;

        let gpu_model = LlamaModel::from_pretrained(
            Path::new("/home/olafurj/.cache/kjarni/meta-llama_Llama-3.2-1B"),
            Device::Wgpu,
            Some(ctx.clone()),
            Some(config),
            None,
        )?;

        let cpu_pipeline: &DecoderPipeline = cpu_model.pipeline();
        let gpu_pipeline: &DecoderPipeline = gpu_model.pipeline();
        let config = cpu_model.config();

        let cpu_decoder = cpu_pipeline
            .cpu_decoder()
            .expect("no cpu decoder")
            .as_any()
            .downcast_ref::<LlamaCpuDecoder>()
            .expect("failed to downcast cpu decoder");
        let gpu_decoder = gpu_pipeline
            .gpu_decoder()
            .expect("no gpu decoder")
            .as_any()
            .downcast_ref::<LlamaGpuDecoder>()
            .expect("failed to downcast gpu decoder");

        let meta = config.metadata();

        let input_tokens: Vec<u32> = vec![1, 15043, 29892, 590, 1024, 338];
        let input_ids = Array2::from_shape_vec((1, input_tokens.len()), input_tokens.clone())?;
        let input_ids_gpu = GpuTensor::from_ndarray(&ctx, &input_ids)?;

        let seq_len = input_tokens.len();
        let attention_mask = Array2::<f32>::ones((1, seq_len));
        let attention_mask_gpu = GpuTensor::from_ndarray(&ctx, &attention_mask)?;

        let position_offset = 0usize;
        let pool = ctx.get_inference_pool();

        let ops = cpu_model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow::anyhow!("model does not support cpu execution"))?;

        let cpu_embeddings = ops.embed(&input_ids, position_offset)?;

        let gpu_embeddings = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let emb = gpu_decoder.embed(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                position_offset,
            )?;

            frame.finish();
            emb.to_ndarray_3d::<f32>().await?
        };

        assert_close_3d(&cpu_embeddings, &gpu_embeddings, 1e-4, "Embeddings");

        let cpu_layer0 = &cpu_decoder.layers[0];
        let gpu_layer0 = &gpu_decoder.layers[0];

        let layer_input = cpu_embeddings.clone();
        let layer_input_gpu = GpuTensor::from_ndarray(&ctx, &layer_input)?;

        let cpu_rms_out = cpu_layer0.attention_norm.forward(&layer_input);

        let gpu_rms_out = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let out = pool_ref.get(layer_input_gpu.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &layer_input_gpu,
                &out,
            );

            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        assert_close_3d(&cpu_rms_out, &gpu_rms_out, 1e-4, "RMSNorm Pre-Attention");

        let hidden_size = meta.hidden_size;
        let num_heads = meta.num_attention_heads;
        let num_kv_heads = meta.num_kv_heads;
        let head_dim = meta.head_dim;

        let rms_2d = cpu_rms_out
            .view()
            .into_shape_with_order((seq_len, hidden_size))
            .unwrap();

        let cpu_q = cpu_layer0.attention.q_proj.matmul(&rms_2d);
        let cpu_k = cpu_layer0.attention.k_proj.matmul(&rms_2d);
        let cpu_v = cpu_layer0.attention.v_proj.matmul(&rms_2d);

        let (gpu_q, gpu_k, gpu_v) = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let rms_out = pool_ref.get(layer_input_gpu.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &layer_input_gpu,
                &rms_out,
            );

            let q = gpu_layer0.self_attn.ops().project(
                enc,
                &rms_out,
                &gpu_layer0.self_attn_weights.q_weight,
                &gpu_layer0.self_attn_weights.q_bias,
                pool_ref,
            );
            let k = gpu_layer0.self_attn.ops().project(
                enc,
                &rms_out,
                &gpu_layer0.self_attn_weights.k_weight,
                &gpu_layer0.self_attn_weights.k_bias,
                pool_ref,
            );
            let v = gpu_layer0.self_attn.ops().project(
                enc,
                &rms_out,
                &gpu_layer0.self_attn_weights.v_weight,
                &gpu_layer0.self_attn_weights.v_bias,
                pool_ref,
            );

            frame.finish();

            (
                q.to_ndarray_3d::<f32>().await?,
                k.to_ndarray_3d::<f32>().await?,
                v.to_ndarray_3d::<f32>().await?,
            )
        };

        let cpu_q_3d = cpu_q
            .clone()
            .into_shape_with_order((1, seq_len, num_heads * head_dim))
            .unwrap();
        let cpu_k_3d = cpu_k
            .clone()
            .into_shape_with_order((1, seq_len, num_kv_heads * head_dim))
            .unwrap();
        let cpu_v_3d = cpu_v
            .clone()
            .into_shape_with_order((1, seq_len, num_kv_heads * head_dim))
            .unwrap();

        assert_close_3d(&cpu_q_3d, &gpu_q, 1e-3, "Q Projection");
        assert_close_3d(&cpu_k_3d, &gpu_k, 1e-3, "K Projection");
        assert_close_3d(&cpu_v_3d, &gpu_v, 1e-3, "V Projection");

        let cpu_q_heads = cpu_q
            .clone()
            .into_shape_with_order((1, seq_len, num_heads, head_dim))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        let cpu_k_heads = cpu_k
            .clone()
            .into_shape_with_order((1, seq_len, num_kv_heads, head_dim))
            .unwrap()
            .permuted_axes([0, 2, 1, 3])
            .to_owned();

        let cpu_rope = &cpu_layer0.rope;
        let cpu_q_rotated = cpu_rope.rotate_4d(&cpu_q_heads, position_offset);
        let cpu_k_rotated = cpu_rope.rotate_4d(&cpu_k_heads, position_offset);

        let (gpu_q_rotated, gpu_k_rotated) = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let rms_out = pool_ref.get(layer_input_gpu.shape().to_vec());
            gpu_layer0.self_attn_norm.encode(
                enc,
                &gpu_layer0.self_attn_norm_weights,
                &layer_input_gpu,
                &rms_out,
            );

            let q_proj = gpu_layer0.self_attn.ops().project(
                enc,
                &rms_out,
                &gpu_layer0.self_attn_weights.q_weight,
                &gpu_layer0.self_attn_weights.q_bias,
                pool_ref,
            );
            let k_proj = gpu_layer0.self_attn.ops().project(
                enc,
                &rms_out,
                &gpu_layer0.self_attn_weights.k_weight,
                &gpu_layer0.self_attn_weights.k_bias,
                pool_ref,
            );

            let q_split = gpu_layer0
                .self_attn
                .ops()
                .split_heads(enc, &q_proj, pool_ref);
            let k_split = gpu_layer0
                .self_attn
                .ops()
                .split_heads(enc, &k_proj, pool_ref);

            let q_rotated = pool_ref.get(q_split.shape().to_vec());
            let k_rotated = pool_ref.get(k_split.shape().to_vec());

            gpu_decoder
                .gpu_rope
                .encode(enc, &q_split, &q_rotated, position_offset);
            gpu_decoder
                .gpu_rope
                .encode(enc, &k_split, &k_rotated, position_offset);

            frame.finish();

            (
                q_rotated.to_ndarray_4d::<f32>().await?,
                k_rotated.to_ndarray_4d::<f32>().await?,
            )
        };

        assert_close_4d(&cpu_q_rotated, &gpu_q_rotated, 1e-3, "Q after RoPE");
        assert_close_4d(&cpu_k_rotated, &gpu_k_rotated, 1e-3, "K after RoPE");

        let kv_dim = cpu_layer0.attention.num_kv_heads * cpu_layer0.attention.head_dim;
        let (_, _, _) = layer_input.dim();

        let (b, s, _) = layer_input.dim();
        let mut temp_k = Array3::<f32>::zeros((b, s, kv_dim));
        let mut temp_v = Array3::<f32>::zeros((b, s, kv_dim));
        let cpu_layer0_out = cpu_layer0.forward(
            &layer_input,
            &attention_mask,
            position_offset,
            temp_k.view_mut(),
            temp_v.view_mut(),
        )?;

        let gpu_layer0_out = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let out = gpu_layer0.forward(
                enc,
                &layer_input_gpu,
                &attention_mask_gpu,
                0,
                position_offset,
                None,
                pool_ref,
                &gpu_decoder.gpu_rope,
            )?;

            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        assert_close_3d(&cpu_layer0_out, &gpu_layer0_out, 1e-2, "Full Layer 0");

        let num_layers = cpu_decoder.num_layers();
        let test_layers = num_layers.min(6);

        for n in 1..=test_layers {
            let cpu_out = cpu_decoder.forward_layers(
                &layer_input,
                &attention_mask,
                position_offset,
                None,
                0,
                n,
            )?;

            let gpu_out = {
                let pool_guard = pool.lock().await;
                let mut frame = GpuFrameContext::new(&ctx, pool_guard);
                let (enc, pool_ref) = frame.resources();

                let out = gpu_decoder.forward_layers(
                    enc,
                    pool_ref,
                    &layer_input_gpu,
                    &attention_mask_gpu,
                    position_offset,
                    None,
                    0,
                    n,
                )?;

                frame.finish();
                out.to_ndarray_3d::<f32>().await?
            };

            let max_diff = cpu_out
                .iter()
                .zip(gpu_out.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            assert!(max_diff <= 0.1, "divergence at layer {}: {:.6}", n, max_diff);
        }

        let ops = cpu_model
            .decoder_cpu_ops()
            .ok_or_else(|| anyhow::anyhow!("model does not support cpu execution"))?;

        let cpu_embeddings = ops.embed(&input_ids, position_offset)?;

        let cpu_full =
            cpu_decoder.forward(&cpu_embeddings, &attention_mask, position_offset, None)?;

        let gpu_full = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);
            let (enc, pool_ref) = frame.resources();

            let out = gpu_decoder.forward(
                enc,
                pool_ref,
                ModelInput::TokensGpu(&input_ids_gpu),
                &attention_mask_gpu,
                position_offset,
                None,
                None,
            )?;

            frame.finish();
            out.to_ndarray_3d::<f32>().await?
        };

        assert_close_3d(&cpu_full, &gpu_full, 0.1, "Full Forward");

        let cpu_logits = cpu_model
            .decoder_cpu_ops()
            .unwrap()
            .project_to_logits(&cpu_full)?;

        let cpu_argmax = cpu_logits
            .slice(s![0, -1, ..])
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let gpu_logits = {
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&ctx, pool_guard);

            let gpu_full_tensor = GpuTensor::from_ndarray(&ctx, &cpu_full)?;
            let logits = gpu_model
                .decoder_gpu_ops()
                .unwrap()
                .project_to_logits(&mut frame, &gpu_full_tensor)?;

            frame.finish();
            logits.to_ndarray_3d::<f32>().await?
        };

        let gpu_argmax = gpu_logits
            .slice(s![0, -1, ..])
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        assert_eq!(cpu_argmax, gpu_argmax, "token prediction mismatch");
    }
    kjarni_transformers::weights::clear_mmap_cache();
    Ok(())
}

#[tokio::test]
async fn test_rope_cpu_gpu_parity() -> Result<()> {
    use kjarni_transformers::gpu_ops::blocks::rope::GpuRoPE;

    let ctx = WgpuContext::new().await?;

    let batch = 1;
    let num_heads = 32;
    let seq_len = 8;
    let head_dim = 128;
    let max_seq_len = 8192;
    let rope_theta = 500000.0;
    let position_offset = 0;

    let cpu_input =
        Array4::<f32>::from_shape_fn((batch, num_heads, seq_len, head_dim), |(b, h, s, d)| {
            ((b * 1000 + h * 100 + s * 10 + d) as f32) * 0.01
        });

    let cpu_rope = RoPE::new(head_dim, max_seq_len, rope_theta);
    let gpu_rope = GpuRoPE::from_cpu_rope(&ctx, &cpu_rope)?;

    let cpu_output = cpu_rope.rotate_4d(&cpu_input, position_offset);

    let gpu_input = GpuTensor::from_ndarray(&ctx, &cpu_input)?;
    let pool = ctx.get_inference_pool();

    let gpu_output = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(gpu_input.shape().to_vec());
        gpu_rope.encode(enc, &gpu_input, &out, position_offset);

        frame.finish();
        out.to_ndarray_4d::<f32>().await?
    };

    assert_close_4d(&cpu_output, &gpu_output, 1e-5, "RoPE");

    let position_offset = 100;

    let cpu_output_offset = cpu_rope.rotate_4d(&cpu_input, position_offset);

    let gpu_output_offset = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(gpu_input.shape().to_vec());
        gpu_rope.encode(enc, &gpu_input, &out, position_offset);

        frame.finish();
        out.to_ndarray_4d::<f32>().await?
    };

    assert_close_4d(
        &cpu_output_offset,
        &gpu_output_offset,
        1e-5,
        "RoPE with offset",
    );

    Ok(())
}

#[tokio::test]
async fn test_gqa_expansion_parity() -> Result<()> {
    let ctx = WgpuContext::new().await?;

    let batch = 1;
    let num_heads = 32;
    let num_kv_heads = 8;
    let seq_len = 4;
    let head_dim = 64;

    let kv_input =
        Array4::<f32>::from_shape_fn((batch, num_kv_heads, seq_len, head_dim), |(b, h, s, d)| {
            ((b * 1000 + h * 100 + s * 10 + d) as f32) * 0.001
        });

    let gqa_ratio = num_heads / num_kv_heads;
    let mut cpu_expanded = Array4::<f32>::zeros((batch, num_heads, seq_len, head_dim));
    for b in 0..batch {
        for h in 0..num_heads {
            let kv_h = h / gqa_ratio;
            for s in 0..seq_len {
                for d in 0..head_dim {
                    cpu_expanded[[b, h, s, d]] = kv_input[[b, kv_h, s, d]];
                }
            }
        }
    }

    let gpu_input = GpuTensor::from_ndarray(&ctx, &kv_input)?;
    let pool = ctx.get_inference_pool();

    let gpu_expanded = {
        let pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (enc, pool_ref) = frame.resources();

        let out = pool_ref.get(vec![batch, num_heads, seq_len, head_dim]);

        let repeat_kv = kjarni_transformers::gpu_ops::primitives::repeat_kv::GpuRepeatKV::new(&ctx);
        repeat_kv.encode(enc, &gpu_input, &out);

        frame.finish();
        out.to_ndarray_4d::<f32>().await?
    };

    assert_close_4d(&cpu_expanded, &gpu_expanded, 1e-6, "GQA Expansion");

    Ok(())
}