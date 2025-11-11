use edgetransformers::prelude::*;
use edgetransformers::decoder::{CpuTransformerDecoder, GpuTransformerDecoder, TransformerDecoder};
use edgetransformers::models::llama::LlamaConfig;
use edgetransformers::rope::RoPE;
use edgetransformers::weights::{ModelWeights, WeightTensor};
use ndarray::{Array, Array2, Array3, Ix3};
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use std::collections::HashMap;
use std::sync::Arc;
use anyhow::Result;

// ... include assert_tensors_approx_equal helper for 3D tensors ...

fn create_dummy_llama_weights(config: &LlamaConfig) -> ModelWeights {
    let mut tensors: HashMap<String, WeightTensor> = HashMap::new();
    let h = config.hidden_size();
    let i = config.intermediate_size();
    let v = config.vocab_size();
    let kv_dim = config.kv_dim();

    // Embeddings
    tensors.insert("model.embed_tokens.weight".to_string(), WeightTensor::new(Array::random((v, h), Uniform::new(-1.0, 1.0))));

    for l in 0..config.num_hidden_layers() {
        // Attention
        tensors.insert(format!("model.layers.{}.self_attn.q_proj.weight", l), WeightTensor::new(Array::random((h, h), Uniform::new(-1.0, 1.0))));
        tensors.insert(format!("model.layers.{}.self_attn.k_proj.weight", l), WeightTensor::new(Array::random((h, kv_dim), Uniform::new(-1.0, 1.0))));
        tensors.insert(format!("model.layers.{}.self_attn.v_proj.weight", l), WeightTensor::new(Array::random((h, kv_dim), Uniform::new(-1.0, 1.0))));
        tensors.insert(format!("model.layers.{}.self_attn.o_proj.weight", l), WeightTensor::new(Array::random((h, h), Uniform::new(-1.0, 1.0))));
        
        // Norms
        tensors.insert(format!("model.layers.{}.input_layernorm.weight", l), WeightTensor::new(Array::random(h, Uniform::new(0.5, 1.5))));
        tensors.insert(format!("model.layers.{}.post_attention_layernorm.weight", l), WeightTensor::new(Array::random(h, Uniform::new(0.5, 1.5))));

        // FFN
        tensors.insert(format!("model.layers.{}.mlp.gate_proj.weight", l), WeightTensor::new(Array::random((h, i), Uniform::new(-1.0, 1.0))));
        tensors.insert(format!("model.layers.{}.mlp.up_proj.weight", l), WeightTensor::new(Array::random((h, i), Uniform::new(-1.0, 1.0))));
        tensors.insert(format!("model.layers.{}.mlp.down_proj.weight", l), WeightTensor::new(Array::random((i, h), Uniform::new(-1.0, 1.0))));
    }

    // Final Norm
    tensors.insert("model.norm.weight".to_string(), WeightTensor::new(Array::random(h, Uniform::new(0.5, 1.5))));

    ModelWeights { tensors, config_json: "".to_string() }
}


#[tokio::test]
async fn test_full_llama_decoder_parity() -> Result<()> {
    // --- 1. Arrange ---
    let config = Arc::new(LlamaConfig {
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2, // Enable GQA
        intermediate_size: 128,
        vocab_size: 1000,
        max_position_embeddings: 64,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        // ... other fields can be default ...
        ..LlamaConfig::llama_3_2_1b()
    });
    
    let weights = create_dummy_llama_weights(&config);
    let context = Arc::new(WgpuContext::new().await?);
    let rope = Arc::new(RoPE::new(config.head_dim(), config.max_position_embeddings(), config.rope_theta));

    // --- Build CPU and GPU Decoders ---
    let cpu_decoder = CpuTransformerDecoder::new(&weights, config.clone(), Some(rope.clone()))?;
    // TODO: You will need to adapt GpuTransformerDecoder::new to accept a GpuRoPE instance
    // and build the correct GpuSwiGLUFFN and GpuAttention with GQA.
    // let gpu_decoder = GpuTransformerDecoder::new(&weights, config.clone(), context.clone(), Some(gpu_rope))?;

    let (batch_size, seq_len) = (1, 8);
    let tolerance = 1e-4;

    let input_ids = Array::random((batch_size, seq_len), Uniform::new(0., config.vocab_size as f32))
        .mapv(|v: f32| v.floor() as f32);
    let attention_mask = Array2::ones((batch_size, seq_len));

    // --- 2. Act & Assert ---
    let cpu_output = cpu_decoder.forward(&input_ids, &attention_mask, None).await?;
    // let gpu_output = gpu_decoder.forward(&input_ids, &attention_mask, None).await?;
    
    // --- 3. Assert ---
    // assert_tensors_approx_equal(&cpu_output.last_hidden_state, &gpu_output.last_hidden_state, tolerance).await;

    println!("✅ Full Llama decoder parity test is structured and ready for GPU implementation.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*; // Imports GpuAttention, GpuAttentionWeights
    use crate::attention::MultiHeadAttention as CpuAttention; // Import CPU impl
    use crate::rope::RoPE as CpuRoPE;
    use crate::gpu_ops::blocks::rope::GpuRoPE;
    use ndarray::{Array3, Array4, Ix4};
    use ndarray_rand::{rand_distr::Uniform, RandomExt};
    use anyhow::Result;
    // ... include assert_tensors_are_close helper for 4D tensors ...

    #[tokio::test]
    async fn test_gpu_gqa_parity_with_rope() -> Result<()> {
        // --- 1. Arrange ---
        let context = Arc::new(WgpuContext::new().await?);
        let (b, s, h) = (1, 8, 256); // Batch, SeqLen, HiddenSize
        let num_q_heads = 8;
        let num_kv_heads = 2; // GQA Ratio: 4
        let head_dim = h / num_q_heads; // 32
        let kv_dim = num_kv_heads * head_dim; // 64
        let position_offset = 16;

        // --- Create CPU and GPU RoPE Modules ---
        let cpu_rope = CpuRoPE::new(head_dim, 128, 10000.0);
        // TODO: You will need to create GpuRoPE for this to compile.
        let gpu_rope = GpuRoPE::new(&context, &cpu_rope.cos_cache, &cpu_rope.sin_cache)?;
        
        // --- Create CPU and GPU Attention Modules ---
        // TODO: You will need to adapt GpuAttention::new to accept num_kv_heads
        let gpu_attention = GpuAttention::new(&context, h as u32, num_q_heads as u32, num_kv_heads as u32);
        
        // --- Create identical random weights ---
        let q_w = Array::random((h, h), Uniform::new(-1.0, 1.0));
        let k_w = Array::random((h, kv_dim), Uniform::new(-1.0, 1.0));
        let v_w = Array::random((h, kv_dim), Uniform::new(-1.0, 1.0));
        let o_w = Array::random((h, h), Uniform::new(-1.0, 1.0));
        // Llama has no biases, so we'll use zeros
        let zero_bias_h = Array1::zeros(h);
        let zero_bias_kv = Array1::zeros(kv_dim);

        let cpu_attention = CpuAttention::new(h, num_q_heads, q_w.clone(), zero_bias_h.clone(), k_w.clone(), zero_bias_kv.clone(), v_w.clone(), zero_bias_kv.clone(), o_w.clone(), zero_bias_h.clone(), Some(num_kv_heads));
        
        let weights_gpu = GpuAttentionWeights::new(
            GpuTensor::from_ndarray(&context, &q_w)?, GpuTensor::from_ndarray(&context, &zero_bias_h)?,
            GpuTensor::from_ndarray(&context, &k_w)?, GpuTensor::from_ndarray(&context, &zero_bias_kv)?,
            GpuTensor::from_ndarray(&context, &v_w)?, GpuTensor::from_ndarray(&context, &zero_bias_kv)?,
            GpuTensor::from_ndarray(&context, &o_w)?, GpuTensor::from_ndarray(&context, &zero_bias_h)?
        )?;

        // --- Create identical inputs ---
        let input_cpu = Array::random((b, s, h), Uniform::new(-1.0, 1.0));
        let input_gpu = GpuTensor::from_ndarray(&context, &input_cpu)?;
        let output_gpu = GpuTensor::uninitialized(&context, input_cpu.shape().to_vec(), input_gpu.dtype(), "GQA Output");
        // No mask for this test to keep it simple
        let mask = Array2::ones((b, s));
        let mask_gpu = GpuTensor::from_ndarray(&context, &mask)?;

        // --- 2. CPU Ground Truth ---
        let (k_proj, v_proj) = cpu_attention.project_kv(&input_cpu);
        let expected_cpu = cpu_attention.attend(&input_cpu, &k_proj, &v_proj, Some(&mask), false, position_offset, Some(&cpu_rope))?;

        // --- 3. GPU Execution ---
        // TODO: You will need a GpuAttention orchestrator function that applies RoPE, GQA, etc.
        // This test assumes a high-level `forward` on GpuAttention that handles this.
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut temp = TempStorage::new(context.clone());
        // gpu_attention.full_forward(&mut encoder, &input_gpu, &weights_gpu, &mask_gpu, &gpu_rope, position_offset, &output_gpu, &mut temp);
        // NOTE: The above line is a placeholder for how you might orchestrate the GPU call.

        // --- 4. Assert ---
        // For now, this test serves as a structural guide. You'll fill in the GPU execution part.
        // assert_tensors_are_close_3d(&expected_cpu, &output_gpu, "GQA Output", 1e-4).await;
        
        println!("✅ GpuAttention GQA test structure is ready.");
        Ok(())
    }
}