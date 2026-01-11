use crate::attention::MultiHeadAttention;
use crate::decoder::prelude::*;
use crate::feedforward::{FeedForward, SwiGluFeedForward};
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights, GpuRMSNorm,
    GpuRMSNormWeights, GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
};
use crate::gpu_ops::{GpuTensor, GpuTensorPool, Kernel};
use crate::linear_layer::LinearLayer;
use crate::normalization::{Normalization, RMSNorm};
use crate::WgpuContext;

use crate::activations::Activation;
use crate::rope::RoPE as CpuRoPE;
use crate::traits::{
    AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelConfig,
    ModelLayout, ModelMetadata,
};
// New Traits
use anyhow::Result;
use common::{assert_tensors_are_close, assert_tensors_are_close_4d, get_test_context};
use ndarray::{Array, Array1, Array2, Array3, Array4};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::sync::Arc;

#[path = "../../../tests/common.rs"]
mod common;

// --- Mock Llama Config for testing ---
struct TestLlamaConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
}

// Replaces all previous trait implementations
impl ModelConfig for TestLlamaConfig {
    fn model_type(&self) -> &str {
        "llama"
    }
fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn metadata(&self) -> ModelMetadata {
        ModelMetadata {
            hidden_size: self.hidden_size,
            num_layers: 1,
            decoder_layers: None,
            num_attention_heads: self.num_attention_heads,
            num_kv_heads: self.num_key_value_heads,
            head_dim: self.hidden_size / self.num_attention_heads,
            vocab_size: 32000,
            max_seq_len: 2048,
            norm_eps: 1e-5,
            activation: crate::activations::Activation::SilU,
            rope_theta: Some(10000.0),
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: crate::traits::NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
        }
    }

    fn layout(&self) -> ModelLayout {
        // --- Define the Decoder's Layer Structure for the test ---
        let decoder_layer = DecoderLayerLayout {
            self_attn: AttentionLayout {
                q_weight: "model.layers.{}.self_attn.q_proj.weight".to_string(),
                q_bias: None,
                k_weight: "model.layers.{}.self_attn.k_proj.weight".to_string(),
                k_bias: None,
                v_weight: "model.layers.{}.self_attn.v_proj.weight".to_string(),
                v_bias: None,
                o_weight: "model.layers.{}.self_attn.o_proj.weight".to_string(),
                o_bias: None,
                norm_weight: "model.layers.{}.input_layernorm.weight".to_string(),
                norm_bias: None,
            },
            cross_attn: None, // No cross-attention in this test model
            ffn: FeedForwardLayout {
                up_weight: "model.layers.{}.mlp.up_proj.weight".to_string(),
                up_bias: None,
                down_weight: "model.layers.{}.mlp.down_proj.weight".to_string(),
                down_bias: None,
                gate_weight: Some("model.layers.{}.mlp.gate_proj.weight".to_string()),
                gate_bias: None,
                norm_weight: "model.layers.{}.post_attention_layernorm.weight".to_string(),
                norm_bias: None,
            },
        };

        // --- Assemble the final ModelLayout ---
        ModelLayout {
            token_embedding: "dummy.wte".to_string(),
            lm_head: "lm_head.weight".to_string(),
            encoder: None, // This is a decoder-only test model
            decoder: Some(DecoderLayout {
                position_embedding: None,
                token_type_embedding: None,
                embedding_norm_weight: None,
                embedding_norm_bias: None,
                final_norm_weight: Some("norm.weight".to_string()),
                final_norm_bias: None,
                layer: decoder_layer,
            }),
        }
    }
}
async fn create_test_layer_pair(
    context: &Arc<WgpuContext>,
    config: &TestLlamaConfig,
) -> Result<(
    GpuPreNormDecoderLayer,
    crate::decoder::prelude::DecoderLayer,
)> {
    let head_dim = config.hidden_size / config.num_attention_heads;
    let kv_dim = config.num_key_value_heads * head_dim;

    // --- Weights: [out_features, in_features] ---
    let q_w = LinearLayer::from(Array::random(
        (config.hidden_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    ));
    let k_w = LinearLayer::from(Array::random(
        (kv_dim, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    ));
    let v_w = LinearLayer::from(Array::random(
        (kv_dim, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    ));
    let o_w = LinearLayer::from(Array::random(
        (config.hidden_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    ));
    let gate_w = Array::random(
        (config.intermediate_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    );
    let up_w = Array::random(
        (config.intermediate_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    );
    let down_w = Array::random(
        (config.hidden_size, config.intermediate_size),
        Uniform::new(-0.1, 0.1),
    );

    let attn_norm_w = Array::random(config.hidden_size, Uniform::new(0.9, 1.1));
    let ffn_norm_w = Array::random(config.hidden_size, Uniform::new(0.9, 1.1));

    let to_gpu_native = |arr: &Array2<f32>| -> Result<GpuTensor> {
        GpuTensor::from_ndarray::<f32, _>(context, arr)
    };

    // --- Build GPU Layer ---
    let gpu_attn_weights = GpuAttentionWeights::new(
        q_w.to_gpu(context)?,
        Some(GpuTensor::from_ndarray::<f32, _>(
            context,
            &Array1::zeros(config.hidden_size),
        )?),
        k_w.to_gpu(context)?,
        Some(GpuTensor::from_ndarray::<f32, _>(
            context,
            &Array1::zeros(kv_dim),
        )?),
        v_w.to_gpu(context)?,
        Some(GpuTensor::from_ndarray::<f32, _>(
            context,
            &Array1::zeros(kv_dim),
        )?),
        o_w.to_gpu(context)?,
        Some(GpuTensor::from_ndarray::<f32, _>(
            context,
            &Array1::zeros(config.hidden_size),
        )?),
    )?;

    let gpu_attn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, 1e-5));
    let gpu_attn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
        GpuTensor::from_ndarray(context, &attn_norm_w)?,
    )?);

    let gpu_ffn_weights = GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(
        to_gpu_native(&gate_w)?,
        to_gpu_native(&up_w)?,
        to_gpu_native(&down_w)?,
    )?);
    let gpu_ffn = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(context)?);
    let gpu_ffn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, 1e-5));
    let gpu_ffn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
        GpuTensor::from_ndarray(context, &ffn_norm_w)?,
    )?);
    // ffn_norm_weights: GpuNormalizationWeights,
    // hidden_size: usize,
    // num_heads: usize,
    // num_kv_heads: usize,
    let gpu_layer = GpuPreNormDecoderLayer::new(
        context,
        gpu_attn_weights,
        gpu_attn_norm,
        gpu_attn_norm_weights,
        gpu_ffn,
        gpu_ffn_weights,
        gpu_ffn_norm,
        gpu_ffn_norm_weights,
        config.hidden_size,
        config.num_attention_heads,
        config.num_key_value_heads,
    )?;

    // --- Build CPU Layer ---
    let cpu_attn = DecoderAttention::new(
        config.hidden_size,
        config.num_attention_heads,
        q_w,
        k_w,
        v_w,
        o_w,
        Some(config.num_key_value_heads),
    );
    let cpu_attn_norm = Normalization::RMSNorm(RMSNorm::new(attn_norm_w, 1e-5));
    let cpu_ffn = FeedForward::SwiGLU(SwiGluFeedForward::new(gate_w, up_w, down_w, Activation::SilU));
    let cpu_ffn_norm = Normalization::RMSNorm(RMSNorm::new(ffn_norm_w, 1e-5));
    let cpu_rope = Arc::new(CpuRoPE::new(head_dim, 1024, 10000.0));

    let cpu_layer = crate::decoder::prelude::DecoderLayer {
        self_attn: cpu_attn,
        self_attn_layer_norm: cpu_attn_norm,
        feedforward: cpu_ffn,
        ffn_layer_norm: cpu_ffn_norm,
        is_prenorm: true,
        rope: Some(cpu_rope),
    };

    Ok((gpu_layer, cpu_layer))
}

#[test]
fn test_decoder_attention_matches_multihead_attention() -> Result<()> {
    let hidden_size = 64;
    let num_heads = 4;
    let batch_size = 2;
    let seq_len = 10;
    let head_dim = hidden_size / num_heads;
    let weight_data: Vec<f32> = (0..hidden_size * hidden_size)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();

    let weight_in_out = Array2::from_shape_vec((hidden_size, hidden_size), weight_data.clone())?;
    let weight_out_in = weight_in_out.t().as_standard_layout().to_owned();
    let bias = Array1::<f32>::zeros(hidden_size);

    let mha = MultiHeadAttention::new(
        hidden_size,
        num_heads,
        weight_in_out.clone(),
        bias.clone(),
        weight_in_out.clone(),
        bias.clone(),
        weight_in_out.clone(),
        bias.clone(),
        weight_in_out.clone(),
        bias.clone(),
        None,
    );

    let da = DecoderAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        Some(num_heads),
    );

    let input: Array3<f32> =
        Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
            ((b * 100 + s * 10 + h) % 23) as f32 * 0.1 - 1.0
        });

    let mask = Array2::<f32>::ones((batch_size, seq_len));

    let (mha_output, _, _) = mha.forward_with_cache(&input, None, Some(&mask), true, None, None)?;

    // Allocate cache for DecoderAttention
    let kv_dim = num_heads * head_dim;
    let mut k_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));
    let mut v_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));

    let da_output = da.forward(
        &input,
        Some(&mask),
        k_cache.view_mut(),
        v_cache.view_mut(),
        0,
        None,
    )?;

    let diff = (&mha_output - &da_output).mapv(|x| x.abs());
    let max_diff = diff.iter().cloned().fold(0.0f32, f32::max);
    assert!(max_diff < 1e-5);

    Ok(())
}

#[tokio::test]
async fn test_swiglu_ffn_parity() -> Result<()> {
    let context = get_test_context().await;
    let hidden_size = 128;
    let intermediate_size = 256;
    let batch_size = 1;
    let seq_len = 7;

    let gate_w_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1));
    let up_w_cpu = Array::random((intermediate_size, hidden_size), Uniform::new(-0.1, 0.1));
    let down_w_cpu = Array::random((hidden_size, intermediate_size), Uniform::new(-0.1, 0.1));

    let cpu_ffn = SwiGluFeedForward::new(gate_w_cpu.clone(), up_w_cpu.clone(), down_w_cpu.clone(), Activation::SilU);

    let gpu_ffn_weights = GpuSwiGLUFFNWeights::new(
        GpuTensor::from_ndarray(&context, &gate_w_cpu)?,
        GpuTensor::from_ndarray(&context, &up_w_cpu)?,
        GpuTensor::from_ndarray(&context, &down_w_cpu)?,
    )?;

    let gpu_ffn_block = GpuSwiGLUFFN::new(&context)?;
    let input_cpu = Array::random((batch_size, seq_len, hidden_size), Uniform::new(-1.0, 1.0));
    let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;

    let expected_output_cpu = cpu_ffn.forward(&input_cpu)?;

    let mut encoder = context.device.create_command_encoder(&Default::default());
    let mut temp = GpuTensorPool::new(context.clone());

    let (b, s, h) = input_gpu.dims3();
    let input_gpu_2d = input_gpu.view(vec![b * s, h]);
    let output_gpu_2d = temp.get(vec![b * s, h]);

    gpu_ffn_block.encode(
        &mut encoder,
        &gpu_ffn_weights,
        &input_gpu_2d,
        &output_gpu_2d,
        &mut temp,
    );
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }

    let output_gpu_3d = output_gpu_2d.view(vec![b, s, h]);

    assert_tensors_are_close(
        &expected_output_cpu,
        &output_gpu_3d,
        "SwiGLU FFN Parity",
        1e-4,
    )
        .await;

    Ok(())
}

#[tokio::test]
async fn test_llama_layer_step_by_step() -> Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let context = get_test_context().await;

    let config = TestLlamaConfig {
        hidden_size: 128,
        num_attention_heads: 8,
        num_key_value_heads: 4,
        intermediate_size: 256,
    };
    let batch_size = 1;
    let seq_len = 7;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let position_offset = 0;

    let (gpu_layer, cpu_layer) = create_test_layer_pair(&context, &config).await?;

    let hidden_states_cpu = Array::random(
        (batch_size, seq_len, config.hidden_size),
        Uniform::new(-1.0, 1.0),
    );
    let attention_mask_cpu = Array2::ones((batch_size, seq_len));
    let hidden_states_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &hidden_states_cpu)?;
    let attention_mask_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &attention_mask_cpu)?;

    let cpu_rope_instance = CpuRoPE::new(head_dim, 1024, 10000.0);
    let gpu_rope_instance = GpuRoPE::from_cpu_rope(&context, &cpu_rope_instance)?;

    let mut pool = GpuTensorPool::new(context.clone());

    forward_llama_with_debug(
        &context,
        &gpu_layer,
        &cpu_layer,
        &hidden_states_gpu,
        &hidden_states_cpu,
        &attention_mask_gpu,
        &attention_mask_cpu,
        position_offset,
        &mut pool,
        Some(&gpu_rope_instance),
    )
        .await?;

    Ok(())
}
pub async fn forward_llama_with_debug(
    // MODIFIED: Takes context directly, manages its own encoder
    context: &Arc<WgpuContext>,
    gpu_layer: &GpuPreNormDecoderLayer,
    cpu_layer: &crate::decoder::prelude::DecoderLayer, // Pass the CPU layer for comparison
    hidden_states_gpu: &GpuTensor,
    hidden_states_cpu: &Array3<f32>, // Pass the initial CPU hidden states
    attention_mask_gpu: &GpuTensor,
    attention_mask_cpu: &Array2<f32>, // Pass the CPU attention mask
    position_offset: usize,
    temp: &mut GpuTensorPool,
    rope: Option<&GpuRoPE>,
) -> Result<GpuTensor> {
    let tolerance = 1e-4; // Set a reasonable tolerance for float comparisons

    // --- Ground Truth CPU Calculation ---
    let cpu_residual_1 = hidden_states_cpu.clone();
    let cpu_ln1_out = cpu_layer.self_attn_layer_norm.forward(&cpu_residual_1);

    // Allocate cache for CPU attention
    let (batch, seq_len, _) = cpu_ln1_out.dim();
    let kv_dim = cpu_layer.self_attn.num_kv_heads * cpu_layer.self_attn.head_dim;
    let mut cpu_k_cache = Array3::<f32>::zeros((batch, seq_len, kv_dim));
    let mut cpu_v_cache = Array3::<f32>::zeros((batch, seq_len, kv_dim));

    let cpu_attn_out = cpu_layer.self_attn.forward(
        &cpu_ln1_out,
        Some(attention_mask_cpu),
        cpu_k_cache.view_mut(),
        cpu_v_cache.view_mut(),
        0,
        cpu_layer.rope.as_deref(),
    )?;
    let cpu_attn_block_output = &cpu_residual_1 + &cpu_attn_out;
    let cpu_residual_2 = cpu_attn_block_output.clone();
    let cpu_ln2_out = cpu_layer.ffn_layer_norm.forward(&cpu_residual_2);
    let cpu_ffn_out = cpu_layer.feedforward.forward(&cpu_ln2_out)?;
    let cpu_final_output = &cpu_residual_2 + &cpu_ffn_out;
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let residual_gpu = hidden_states_gpu;
    let ln1_out_gpu = temp.get(hidden_states_gpu.shape().to_vec());
    gpu_layer.self_attn_norm.encode(
        &mut encoder,
        &gpu_layer.self_attn_norm_weights,
        residual_gpu,
        &ln1_out_gpu,
    );
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(&cpu_ln1_out, &ln1_out_gpu, "ln1_out", tolerance).await;
    let q_proj = gpu_layer.self_attn.project(
        &mut encoder,
        &ln1_out_gpu,
        &gpu_layer.self_attn_weights.q_weight,
        &gpu_layer.self_attn_weights.q_bias,
        temp,
    );
    let k_proj = gpu_layer.self_attn.project(
        &mut encoder,
        &ln1_out_gpu,
        &gpu_layer.self_attn_weights.k_weight,
        &gpu_layer.self_attn_weights.k_bias,
        temp,
    );
    let v_proj = gpu_layer.self_attn.project(
        &mut encoder,
        &ln1_out_gpu,
        &gpu_layer.self_attn_weights.v_weight,
        &gpu_layer.self_attn_weights.v_bias,
        temp,
    );
    let q_split = gpu_layer.self_attn.split_heads(&mut encoder, &q_proj, temp);
    let k_split = gpu_layer.self_attn.split_heads(&mut encoder, &k_proj, temp);
    let v_split = gpu_layer.self_attn.split_heads(&mut encoder, &v_proj, temp);
    let q_rotated = temp.get(q_split.shape().to_vec());
    let k_rotated = temp.get(k_split.shape().to_vec());
    rope.unwrap()
        .encode(&mut encoder, &q_split, &q_rotated, position_offset);
    rope.unwrap()
        .encode(&mut encoder, &k_split, &k_rotated, position_offset);
    let attn_out_gpu = gpu_layer.self_attn.llama_attention(
        &mut encoder,
        &q_rotated,
        &k_rotated,
        &v_split,
        attention_mask_gpu,
        position_offset,
        temp,
        &gpu_layer.self_attn_weights,
    );
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(&cpu_attn_out, &attn_out_gpu, "attn_out", tolerance).await;
    let attn_block_output_gpu = temp.get(hidden_states_gpu.shape().to_vec());
    gpu_layer.add.encode(
        &mut encoder,
        &[residual_gpu, &attn_out_gpu],
        &attn_block_output_gpu,
    );
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(
        &cpu_attn_block_output,
        &attn_block_output_gpu,
        "attn_block_output",
        tolerance,
    )
        .await;
    let residual_2_gpu = &attn_block_output_gpu;
    let ln2_out_gpu = temp.get(residual_2_gpu.shape().to_vec());
    gpu_layer.ffn_norm.encode(
        &mut encoder,
        &gpu_layer.ffn_norm_weights,
        residual_2_gpu,
        &ln2_out_gpu,
    );
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(&cpu_ln2_out, &ln2_out_gpu, "ln2_out", tolerance).await;
    log::info!("✓ Step 4: Pre-FFN Norm output matches CPU.");

    // --- 5. FFN ---
    log::debug!("--- Step 5: FFN Block ---");
    let (b, s, h) = ln2_out_gpu.dims3();
    let ln2_out_2d_gpu = ln2_out_gpu.view(vec![b * s, h]);
    let ffn_out_2d_gpu = temp.get(vec![b * s, h]);
    gpu_layer.feedforward.encode(
        &mut encoder,
        &gpu_layer.ff_weights,
        &ln2_out_2d_gpu,
        &ffn_out_2d_gpu,
        temp,
    );
    let ffn_out_gpu = ffn_out_2d_gpu.view(vec![b, s, h]);
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(&cpu_ffn_out, &ffn_out_gpu, "ffn_out", tolerance).await;
    log::info!("✓ Step 5: FFN output matches CPU.");

    // --- 6. Second Residual Connection ---
    log::debug!("--- Step 6: Second Residual Connection ---");
    let final_output_gpu = temp.get(residual_2_gpu.shape().to_vec());
    gpu_layer.add.encode(
        &mut encoder,
        &[residual_2_gpu, &ffn_out_gpu],
        &final_output_gpu,
    );
    context.queue.submit(Some(encoder.finish()));
    // No need to create a new encoder after the last step
    assert_tensors_are_close(
        &cpu_final_output,
        &final_output_gpu,
        "final_output",
        tolerance,
    )
        .await;
    log::info!("✓ Step 6: Final layer output matches CPU.");

    Ok(final_output_gpu)
}

#[test]
fn test_decoder_attention_with_rope_matches_mha() -> Result<()> {
    // --- Test Setup ---
    let hidden_size = 64;
    let num_heads = 4;
    let batch_size = 2;
    let seq_len = 10;
    let head_dim = hidden_size / num_heads;

    // --- Create Identical Weights (same as before) ---
    // ... copy the weight creation logic from the previous test ...
    let weight_data: Vec<f32> = (0..hidden_size * hidden_size)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
        .collect();
    let weight_in_out = Array2::from_shape_vec((hidden_size, hidden_size), weight_data.clone())?;
    let weight_out_in = weight_in_out.t().as_standard_layout().to_owned();
    let bias = Array1::<f32>::zeros(hidden_size);

    // --- Create a RoPE instance ---
    let rope = Arc::new(crate::rope::RoPE::new(head_dim, 128, 10000.0));

    // --- 1. Configure MultiHeadAttention ---
    let mha = MultiHeadAttention::new(
        hidden_size,
        num_heads,
        weight_in_out.clone(),
        bias.clone(), // Q
        weight_in_out.clone(),
        bias.clone(), // K
        weight_in_out.clone(),
        bias.clone(), // V
        weight_in_out.clone(),
        bias.clone(), // O
        None,         // num_kv_heads = None (no GQA)
    );

    // --- 2. Configure DecoderAttention ---
    let da = DecoderAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(weight_out_in.clone(), Some(bias.clone())),
        Some(num_heads), // num_kv_heads = num_heads (no GQA)
    );

    // --- 3. Create Input & Mask (same as before) ---
    let input = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
        ((b * 100 + s * 10 + h) % 23) as f32 * 0.1 - 1.0
    });
    let mask = Array2::<f32>::ones((batch_size, seq_len));

    // --- 4. Run Both, but THIS TIME PASS THE ROPE INSTANCE ---
    let (mha_output, _, _) =
        mha.forward_with_cache(&input, None, Some(&mask), true, None, Some(rope.as_ref()))?;

    // Allocate cache for DecoderAttention
    let kv_dim = num_heads * head_dim;
    let mut k_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));
    let mut v_cache = Array3::<f32>::zeros((batch_size, seq_len, kv_dim));

    let da_output = da.forward(
        &input,
        Some(&mask),
        k_cache.view_mut(),
        v_cache.view_mut(),
        0,
        Some(rope.as_ref()),
    )?;

    // --- 5. Compare (same as before) ---
    let max_diff = (&mha_output - &da_output)
        .mapv(|x| x.abs())
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "Outputs with RoPE differ! Max diff: {}",
        max_diff
    );
    println!("✓ DecoderAttention with RoPE successfully matches MultiHeadAttention");

    Ok(())
}
#[tokio::test]
async fn test_gpu_repeat_kv_kernel_parity() -> Result<()> {
    let context = get_test_context().await;

    // --- Test Setup ---
    let num_heads = 8;
    let num_kv_heads = 4; // n_rep = 2
    let batch_size = 2;
    let seq_len = 10;
    let head_dim = 64;

    // --- 1. Create CPU Input Tensor ---
    // This is the KV tensor with fewer heads: [B, H_kv, S, D]
    let kv_heads_cpu = Array::random(
        (batch_size, num_kv_heads, seq_len, head_dim),
        Uniform::new(-1.0, 1.0),
    );

    // --- 2. Calculate Ground Truth on CPU ---
    // This is the logic from your passing `test_decoder_attention_gqa_matches_manual_expansion`
    let n_rep = num_heads / num_kv_heads;
    let mut expected_expanded_cpu = Array4::zeros((batch_size, num_heads, seq_len, head_dim));
    for b in 0..batch_size {
        for s in 0..seq_len {
            for i in 0..num_kv_heads {
                let source_head = kv_heads_cpu.slice(ndarray::s![b, i, s, ..]);
                for g in 0..n_rep {
                    let target_head_idx = i * n_rep + g;
                    expected_expanded_cpu
                        .slice_mut(ndarray::s![b, target_head_idx, s, ..])
                        .assign(&source_head);
                }
            }
        }
    }

    // --- 3. Run the GPU Kernel ---
    let gpu_attention = GpuAttention::new(
        &context,
        (num_heads * head_dim) as u32,
        num_heads as u32,
        num_kv_heads as u32,
    );
    let mut encoder = context.device.create_command_encoder(&Default::default());

    // Upload the smaller KV tensor to the GPU
    let kv_heads_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &kv_heads_cpu)?;

    // Prepare the output tensor
    let expanded_gpu_out = GpuTensor::uninitialized(
        &context,
        vec![batch_size, num_heads, seq_len, head_dim],
        crate::gpu_ops::DType::F32,
        "Expanded KV Output",
    );

    // Run the kernel
    gpu_attention
        .repeat_kv
        .encode(&mut encoder, &kv_heads_gpu, &expanded_gpu_out);

    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    assert_tensors_are_close_4d(
        &expected_expanded_cpu,
        &expanded_gpu_out,
        "GpuRepeatKV Kernel vs CPU Logic",
        1e-6, // Use a very small tolerance
    )
        .await;

    println!("GpuRepeatKV kernel is correct!");

    Ok(())
}
#[test]
fn test_decoder_attention_gqa_matches_manual_expansion() -> Result<()> {
    // --- Test Setup ---
    let hidden_size = 64;
    let num_heads = 4;
    let num_kv_heads = 2; // GQA with n_rep = 2
    let head_dim = hidden_size / num_heads;
    let kv_dim = num_kv_heads * head_dim; // 32
    let batch_size = 2;
    let seq_len = 10;

    // --- Create Weights ---
    // Q is full size, K and V are smaller
    let q_weight_out_in = Array2::<f32>::from_shape_fn((hidden_size, hidden_size), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.001).sin()
    });
    let k_weight_out_in = Array2::<f32>::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.002).cos()
    });
    let v_weight_out_in = Array2::<f32>::from_shape_fn((kv_dim, hidden_size), |(i, j)| {
        (i as f32 * 0.01 + j as f32 * 0.003).sin()
    });
    let o_weight_out_in = q_weight_out_in.clone(); // Reuse for simplicity
    let bias = Array1::<f32>::zeros(hidden_size);
    let kv_bias = Array1::<f32>::zeros(kv_dim);

    // --- 1. Configure DecoderAttention with GQA ---
    let da_gqa = DecoderAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::new_f32(q_weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(k_weight_out_in.clone(), Some(kv_bias.clone())),
        LinearLayer::new_f32(v_weight_out_in.clone(), Some(kv_bias.clone())),
        LinearLayer::new_f32(o_weight_out_in.clone(), Some(bias.clone())),
        Some(num_kv_heads), // Enable GQA
    );

    // --- 2. Configure a Standard MHA by MANUALLY expanding the K/V weights ---
    // This simulates what GQA does internally.
    let n_rep = num_heads / num_kv_heads;
    let mut expanded_k_weight = Array2::zeros((hidden_size, hidden_size));
    let mut expanded_v_weight = Array2::zeros((hidden_size, hidden_size));
    for i in 0..num_kv_heads {
        for g in 0..n_rep {
            let target_head_idx = i * n_rep + g;
            let target_slice = ndarray::s![
                target_head_idx * head_dim..(target_head_idx + 1) * head_dim,
                ..
            ];
            let source_slice = ndarray::s![i * head_dim..(i + 1) * head_dim, ..];
            expanded_k_weight
                .slice_mut(target_slice)
                .assign(&k_weight_out_in.slice(source_slice));
            expanded_v_weight
                .slice_mut(target_slice)
                .assign(&v_weight_out_in.slice(source_slice));
        }
    }

    // Create a standard MHA with these expanded weights
    let da_mha = DecoderAttention::new(
        hidden_size,
        num_heads,
        LinearLayer::new_f32(q_weight_out_in.clone(), Some(bias.clone())),
        LinearLayer::new_f32(expanded_k_weight, Some(bias.clone())), // Use expanded K
        LinearLayer::new_f32(expanded_v_weight, Some(bias.clone())), // Use expanded V
        LinearLayer::new_f32(o_weight_out_in.clone(), Some(bias.clone())),
        Some(num_heads), // No GQA, it's now standard MHA
    );

    // --- 3. Run Both (No RoPE to isolate GQA) ---
    let input = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
        ((b * 100 + s * 10 + h) % 19) as f32 * 0.1 - 0.8
    });
    let (b, s, _) = input.dim();
    let mask = Array2::<f32>::ones((batch_size, seq_len));

    // --- FIX IS HERE ---

    // 1. Run GQA (Expects compressed KV dim = 32)
    let mut temp_k_gqa = Array3::<f32>::zeros((b, s, kv_dim));
    let mut temp_v_gqa = Array3::<f32>::zeros((b, s, kv_dim));

    let gqa_output = da_gqa.forward(
        &input,
        Some(&mask),
        temp_k_gqa.view_mut(),
        temp_v_gqa.view_mut(),
        0,
        None,
    )?;

    // 2. Run Manual MHA (Expects full hidden size = 64)
    // We cannot reuse the GQA buffer because this "Manual Expansion" model
    // acts like a standard MHA model, outputting full-sized heads.
    let mut temp_k_mha = Array3::<f32>::zeros((b, s, hidden_size));
    let mut temp_v_mha = Array3::<f32>::zeros((b, s, hidden_size));

    let manual_mha_output = da_mha.forward(
        &input,
        Some(&mask),
        temp_k_mha.view_mut(),
        temp_v_mha.view_mut(),
        0,
        None,
    )?;

    // --- 4. Compare ---
    let max_diff = (&gqa_output - &manual_mha_output)
        .mapv(|x| x.abs())
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-5,
        "GQA logic does not match manual expansion! Max diff: {}",
        max_diff
    );
    println!("✓ DecoderAttention GQA logic is correct on CPU");

    Ok(())
}

#[tokio::test]
async fn test_rope_parity_single_token() -> Result<()> {
    let context = get_test_context().await;
    let head_dim = 64;
    let max_seq_len = 128;
    let theta = 10000.0;
    let batch_size = 1;
    let num_heads = 4;
    let seq_len = 1; // Test single token generation
    let position_offset = 5; // Test with a non-zero offset

    // 1. Create CPU and GPU RoPE instances
    let cpu_rope = CpuRoPE::new(head_dim, max_seq_len, theta);
    let gpu_rope = GpuRoPE::from_cpu_rope(&context, &cpu_rope)?;

    // 2. Create identical input tensors
    let input_cpu = Array::random(
        (batch_size, num_heads, seq_len, head_dim),
        Uniform::new(-1.0, 1.0),
    );
    let input_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &input_cpu)?;

    // 3. Run CPU RoPE to get ground truth
    let expected_rotated_cpu = cpu_rope.rotate_4d(&input_cpu, position_offset);

    // 4. Run GPU RoPE
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let output_gpu = GpuTensor::uninitialized(
        &context,
        input_gpu.shape().to_vec(),
        crate::gpu_ops::DType::F32,
        "RoPE Output",
    );

    // Use a dummy tensor for K since we only care about Q for this test
    gpu_rope.encode(&mut encoder, &input_gpu, &output_gpu, position_offset);
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }

    // 5. Compare results
    assert_tensors_are_close_4d(
        &expected_rotated_cpu,
        &output_gpu,
        "RoPE single token",
        1e-5,
    )
        .await;

    Ok(())
}

#[tokio::test]
async fn test_rope_parity_multiple_tokens() -> Result<()> {
    let context = get_test_context().await;
    let head_dim = 128;
    let max_seq_len = 256;
    let theta = 10000.0;
    let batch_size = 1;
    let num_heads = 2;
    let seq_len = 10; // Test a sequence of tokens (prefill)
    let position_offset = 0; // Test prefill case

    // 1. Create CPU and GPU RoPE instances
    let cpu_rope = CpuRoPE::new(head_dim, max_seq_len, theta);
    let gpu_rope = GpuRoPE::from_cpu_rope(&context, &cpu_rope)?;

    // 2. Create identical input tensors for Q and K
    let q_cpu = Array::random(
        (batch_size, num_heads, seq_len, head_dim),
        Uniform::new(-1.0, 1.0),
    );
    let k_cpu = Array::random(
        (batch_size, num_heads, seq_len, head_dim),
        Uniform::new(-1.0, 1.0),
    );
    let q_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &q_cpu)?;
    let k_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &k_cpu)?;

    // 3. Run CPU RoPE to get ground truth
    let (expected_q_rot, expected_k_rot) = cpu_rope.apply_4d(&q_cpu, &k_cpu, position_offset);

    // 4. Run GPU RoPE
    let mut encoder = context.device.create_command_encoder(&Default::default());
    let q_rot_gpu = GpuTensor::uninitialized(
        &context,
        q_gpu.shape().to_vec(),
        crate::gpu_ops::DType::F32,
        "Q Rot",
    );
    let k_rot_gpu = GpuTensor::uninitialized(
        &context,
        k_gpu.shape().to_vec(),
        crate::gpu_ops::DType::F32,
        "K Rot",
    );

    gpu_rope.encode(&mut encoder, &q_gpu, &q_rot_gpu, position_offset);
    gpu_rope.encode(&mut encoder, &k_gpu, &k_rot_gpu, position_offset);
    context.queue.submit(Some(encoder.finish()));
    match context.device.poll(wgpu::PollType::wait_indefinitely()) {
        Ok(status) => println!("GPU Poll OK: {:?}", status),
        Err(e) => panic!("GPU Poll Failed: {:?}", e),
    }
    assert_tensors_are_close_4d(&expected_q_rot, &q_rot_gpu, "RoPE multi-token Q", 1e-5).await;
    assert_tensors_are_close_4d(&expected_k_rot, &k_rot_gpu, "RoPE multi-token K", 1e-5).await;

    Ok(())
}
