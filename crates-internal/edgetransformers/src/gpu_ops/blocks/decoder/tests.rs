use super::*;
use crate::Device::Cpu;
use crate::activations::Activation::{Gelu, SilU};
use crate::attention::MultiHeadAttention;
use crate::decoder_layer::DecoderLayer;
use crate::feedforward::{FeedForward, SwiGluFeedForward};
use crate::gpu_context::WgpuContext;
use crate::gpu_ops::blocks::attention::{GpuAttentionWeights, TempStorage};
use crate::gpu_ops::blocks::rope::GpuRoPE;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights, GpuRMSNorm,
    GpuRMSNormWeights, GpuSwiGLUFFN, GpuSwiGLUFFNWeights,
};
use crate::gpu_ops::{DType, GpuFrameContext, GpuTensor};
use crate::normalization::{Normalization, RMSNorm};
use crate::rope::RoPE as CpuRoPE; // Import your CPU implementation
use anyhow::Result;
use common::{
    assert_tensors_are_close, assert_tensors_are_close_4d, get_test_context, read_gpu_tensor_to_vec,
};
use ndarray::{Array, Array1, Array2, Array3, Array4, Ix4};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use std::sync::Arc;
#[path = "../../../tests/common.rs"]
mod common;
use crate::traits::{
    DecoderArchitecture, LanguageModelConfig, LayerAttentionNames, LayerDecoderAttentionNames,
    LayerFeedForwardNames, TransformerConfig,
};

// --- Mock Llama Config for testing ---
struct TestLlamaConfig {
    hidden_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    intermediate_size: usize,
}

// Implement the base traits first
impl TransformerConfig for TestLlamaConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn num_hidden_layers(&self) -> usize {
        1
    }
    fn layer_norm_eps(&self) -> f32 {
        1e-5
    }
    fn is_causal(&self) -> bool {
        true
    }
    fn is_prenorm(&self) -> bool {
        true
    }
}

impl LanguageModelConfig for TestLlamaConfig {
    fn vocab_size(&self) -> usize {
        32000
    }
    fn decoder_start_token_id(&self) -> u32 {
        // For a decoder-only model, the "start token" for generation
        // is the Beginning-Of-Sequence token.
        self.bos_token_id().unwrap()
    }
    fn max_position_embeddings(&self) -> usize {
        2048
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads
    }
    fn get_embedding_weight_names(&self) -> (&str, &str, Option<&str>) {
        ("dummy.wte", "", None)
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn activation_function(&self) -> crate::activations::Activation {
        crate::activations::Activation::SilU
    }
}

// Now, implement the specific trait required by the layer constructor
impl DecoderArchitecture for TestLlamaConfig {
    fn get_final_layer_norm_names(&self) -> (&str, &str) {
        ("norm.weight", "")
    }
    fn get_lm_head_name(&self) -> &str {
        "lm_head.weight"
    }

    // This is for GPT-2 style combined QKV, return empty strings for Llama
    fn get_attention_names(&self, _layer_index: usize) -> LayerDecoderAttentionNames {
        LayerDecoderAttentionNames {
            qkv_weight: "".to_string(),
            qkv_bias: "".to_string(),
            output_weight: "".to_string(),
            output_bias: "".to_string(),
            norm_weight: "".to_string(),
            norm_bias: "".to_string(),
        }
    }

    // This is for Llama style separate QKV
    fn get_layer_attention_names(&self, layer_index: usize) -> LayerAttentionNames {
        // These names don't matter for the test since we build weights manually,
        // but they need to be valid strings.
        LayerAttentionNames {
            q_weight: format!("model.layers.{}.self_attn.q_proj.weight", layer_index),
            q_bias: "".to_string(),
            k_weight: format!("model.layers.{}.self_attn.k_proj.weight", layer_index),
            k_bias: "".to_string(),
            v_weight: format!("model.layers.{}.self_attn.v_proj.weight", layer_index),
            v_bias: "".to_string(),
            output_weight: format!("model.layers.{}.self_attn.o_proj.weight", layer_index),
            output_bias: "".to_string(),
            norm_weight: format!("model.layers.{}.input_layernorm.weight", layer_index),
            norm_bias: "".to_string(),
        }
    }

    fn get_feed_forward_names(&self, layer_index: usize) -> LayerFeedForwardNames {
        LayerFeedForwardNames {
            gate_weight: Some(format!("model.layers.{}.mlp.gate_proj.weight", layer_index)),
            intermediate_weight: format!("model.layers.{}.mlp.up_proj.weight", layer_index),
            intermediate_bias: "".to_string(),
            output_weight: format!("model.layers.{}.mlp.down_proj.weight", layer_index),
            output_bias: "".to_string(),
            norm_weight: format!(
                "model.layers.{}.post_attention_layernorm.weight",
                layer_index
            ),
            norm_bias: "".to_string(),
        }
    }
}
async fn create_test_layer_pair(
    context: &Arc<WgpuContext>,
    config: &TestLlamaConfig,
) -> Result<(GpuPreNormDecoderLayer, DecoderLayer)> {
    let head_dim = config.hidden_size / config.num_attention_heads;

    // --- Common Random Weights (ndarray) ---
    let q_w = Array::random(
        (config.hidden_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    );
    let k_w = Array::random(
        (config.hidden_size, config.kv_dim()),
        Uniform::new(-0.1, 0.1),
    );
    let v_w = Array::random(
        (config.hidden_size, config.kv_dim()),
        Uniform::new(-0.1, 0.1),
    );
    let o_w = Array::random(
        (config.hidden_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    );
    let gate_w = Array::random(
        (config.hidden_size, config.intermediate_size),
        Uniform::new(-0.1, 0.1),
    );
    let up_w = Array::random(
        (config.hidden_size, config.intermediate_size),
        Uniform::new(-0.1, 0.1),
    );
    let down_w = Array::random(
        (config.intermediate_size, config.hidden_size),
        Uniform::new(-0.1, 0.1),
    );
    let attn_norm_w = Array::random(config.hidden_size, Uniform::new(0.9, 1.1));
    let ffn_norm_w = Array::random(config.hidden_size, Uniform::new(0.9, 1.1));
    let q = GpuTensor::from_ndarray::<f32, _>(context, &Array1::zeros(config.hidden_size()))?;

    // --- Build GPU Layer ---
    let gpu_attn_weights = GpuAttentionWeights::new(
        GpuTensor::from_ndarray::<f32, _>(context, &q_w)?,
        q,
        GpuTensor::from_ndarray::<f32, _>(context, &k_w)?,
        GpuTensor::from_ndarray::<f32, _>(context, &Array1::zeros(config.kv_dim()))?,
        GpuTensor::from_ndarray::<f32, _>(context, &v_w)?,
        GpuTensor::from_ndarray::<f32, _>(context, &Array1::zeros(config.kv_dim()))?,
        GpuTensor::from_ndarray::<f32, _>(context, &o_w)?,
        GpuTensor::from_ndarray::<f32, _>(context, &Array1::zeros(config.hidden_size()))?,
    )?;
    let gpu_attn_norm =
        GpuNormalization::RMSNorm(GpuRMSNorm::new(context, config.layer_norm_eps()));
    let gpu_attn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
        GpuTensor::from_ndarray(context, &attn_norm_w)?,
    )?);
    let gpu_ffn_weights = GpuFeedForwardWeights::SwiGLU(GpuSwiGLUFFNWeights::new(
        GpuTensor::from_ndarray(context, &gate_w)?,
        GpuTensor::from_ndarray(context, &up_w)?,
        GpuTensor::from_ndarray(context, &down_w)?,
    )?);
    let gpu_ffn = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(context)?);
    let gpu_ffn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, config.layer_norm_eps()));
    let gpu_ffn_norm_weights = GpuNormalizationWeights::RMSNorm(GpuRMSNormWeights::new(
        GpuTensor::from_ndarray(context, &ffn_norm_w)?,
    )?);

    let gpu_layer = GpuPreNormDecoderLayer::new(
        context,
        gpu_attn_weights,
        gpu_attn_norm,
        gpu_attn_norm_weights,
        gpu_ffn,
        gpu_ffn_weights,
        gpu_ffn_norm,
        gpu_ffn_norm_weights,
        Arc::new(TestLlamaConfig {
            // Pass a new Arc'd config
            hidden_size: config.hidden_size,
            num_attention_heads: config.num_attention_heads,
            num_key_value_heads: config.num_key_value_heads,
            intermediate_size: config.intermediate_size,
        }),
        None,
    )?;

    // --- Build CPU Layer ---
    let cpu_attn = MultiHeadAttention::new(
        config.hidden_size,
        config.num_attention_heads,
        q_w,
        Array1::zeros(config.hidden_size),
        k_w,
        Array1::zeros(config.kv_dim()),
        v_w,
        Array1::zeros(config.kv_dim()),
        o_w,
        Array1::zeros(config.hidden_size),
        Some(config.num_key_value_heads),
    );
    let cpu_attn_norm = Normalization::RMSNorm(RMSNorm::new(attn_norm_w, config.layer_norm_eps()));
    let cpu_ffn = FeedForward::SwiGLU(SwiGluFeedForward::new(gate_w, up_w, down_w));
    let cpu_ffn_norm = Normalization::RMSNorm(RMSNorm::new(ffn_norm_w, config.layer_norm_eps()));
    let cpu_rope = Arc::new(CpuRoPE::new(head_dim, 1024, 10000.0));

    let cpu_layer = DecoderLayer {
        self_attn: cpu_attn,
        self_attn_layer_norm: cpu_attn_norm,
        feedforward: cpu_ffn,
        ffn_layer_norm: cpu_ffn_norm,
        is_prenorm: true,
        rope: Some(cpu_rope),
    };

    Ok((gpu_layer, cpu_layer))
}

#[tokio::test]
async fn test_llama_layer_step_by_step() -> Result<()> {
    let _ = env_logger::builder().is_test(true).try_init();
    let context = get_test_context().await;

    // --- Test Parameters ---
    let config = TestLlamaConfig {
        hidden_size: 128,
        num_attention_heads: 8,
        num_key_value_heads: 4, // GQA
        intermediate_size: 256,
    };
    let batch_size = 1;
    let seq_len = 7;
    let head_dim = config.hidden_size / config.num_attention_heads;
    let position_offset = 0;

    // --- Create Layers ---
    let (gpu_layer, cpu_layer) = create_test_layer_pair(&context, &config).await?;

    // --- Create Inputs ---
    let hidden_states_cpu = Array::random(
        (batch_size, seq_len, config.hidden_size),
        Uniform::new(-1.0, 1.0),
    );
    let attention_mask_cpu = Array2::ones((batch_size, seq_len));
    let hidden_states_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &hidden_states_cpu)?;
    let attention_mask_gpu = GpuTensor::from_ndarray::<f32, _>(&context, &attention_mask_cpu)?;

    // --- Create RoPE ---
    let cpu_rope_instance = CpuRoPE::new(head_dim, 1024, 10000.0);
    let gpu_rope_instance = GpuRoPE::from_cpu_rope(&context, &cpu_rope_instance)?;

    let mut pool = GpuTensorPool::new(context.clone());

    let _ = forward_llama_with_debug(
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

    log::info!(
        "✅✅✅ All intermediate steps in the Llama layer match the CPU implementation! ✅✅✅"
    );

    Ok(())
}
pub async fn forward_llama_with_debug(
    // MODIFIED: Takes context directly, manages its own encoder
    context: &Arc<WgpuContext>,
    gpu_layer: &GpuPreNormDecoderLayer,
    cpu_layer: &DecoderLayer, // Pass the CPU layer for comparison
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
    let (cpu_attn_out, _, _) = cpu_layer.self_attn.forward_with_cache(
        &cpu_ln1_out,
        None,
        Some(attention_mask_cpu),
        true,
        None,
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
    context.device.poll(wgpu::PollType::wait_indefinitely());
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
    context.device.poll(wgpu::PollType::wait_indefinitely());
    encoder = context.device.create_command_encoder(&Default::default());
    assert_tensors_are_close(&cpu_attn_out, &attn_out_gpu, "attn_out", tolerance).await;
    let attn_block_output_gpu = temp.get(hidden_states_gpu.shape().to_vec());
    gpu_layer.add.encode(
        &mut encoder,
        &[residual_gpu, &attn_out_gpu],
        &attn_block_output_gpu,
    );
    context.queue.submit(Some(encoder.finish()));
    context.device.poll(wgpu::PollType::wait_indefinitely());
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
    context.device.poll(wgpu::PollType::wait_indefinitely());
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
    context.device.poll(wgpu::PollType::wait_indefinitely());
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
    context.device.poll(wgpu::PollType::wait_indefinitely());

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
    context.device.poll(wgpu::PollType::wait_indefinitely());

    // 5. Compare results
    assert_tensors_are_close_4d(&expected_q_rot, &q_rot_gpu, "RoPE multi-token Q", 1e-5).await;
    assert_tensors_are_close_4d(&expected_k_rot, &k_rot_gpu, "RoPE multi-token K", 1e-5).await;

    Ok(())
}
