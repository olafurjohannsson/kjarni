use anyhow::Result;
use std::sync::Arc;

use crate::WgpuContext;
use crate::activations;
use crate::gpu_ops::blocks::attention::{GpuAttention, GpuAttentionWeights};
use crate::gpu_ops::blocks::ffn::{GpuFeedForward, GpuFeedForwardWeights};
use crate::gpu_ops::blocks::layer_norm::GpuLayerNorm;
use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
use crate::gpu_ops::primitives::add::GpuAdd;
use crate::gpu_ops::{GpuTensor, GpuTensorPool, Kernel};
use crate::traits::ModelMetadata;

pub struct GpuEncoderLayer {
    self_attn: GpuAttention,
    self_attn_weights: GpuAttentionWeights,
    self_attn_layer_norm: GpuLayerNorm,
    self_attn_ln_weights: GpuLayerNormWeights,

    feedforward: GpuFeedForward,
    ff_weights: GpuFeedForwardWeights,
    ffn_layer_norm: GpuLayerNorm,
    ffn_ln_weights: GpuLayerNormWeights,

    add: GpuAdd,
}

impl GpuEncoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        self_attn_weights: GpuAttentionWeights,
        self_attn_ln_weights: GpuLayerNormWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_ln_weights: GpuLayerNormWeights,
        activation: activations::Activation,
        meta: &ModelMetadata,
    ) -> Result<Self> {
        let hidden_size = meta.hidden_size as u32;
        let num_heads = meta.num_attention_heads as u32;

        // In Encoders, KV heads usually equal Attention heads.
        // Metadata provides this logic (falling back to num_heads if not specified).
        let num_kv_heads = meta.num_kv_heads as u32;

        // 1. Initialize Attention Engine
        let self_attn = GpuAttention::new(context, hidden_size, num_heads, num_kv_heads);

        // 2. Initialize LayerNorm Engines (using Metadata eps)
        let self_attn_layer_norm = GpuLayerNorm::new(context, meta.norm_eps);
        let ffn_layer_norm = GpuLayerNorm::new(context, meta.norm_eps);

        // 3. Initialize FFN Engine
        let feedforward = GpuFeedForward::new(context, activation)?;

        // 4. Initialize Math Engine
        let add = GpuAdd::new(context);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_layer_norm,
            self_attn_ln_weights,
            feedforward,
            ff_weights,
            ffn_layer_norm,
            ffn_ln_weights,
            add,
        })
    }

    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        meta: &ModelMetadata, // Use Metadata struct
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        // Dispatch based on architectural style defined in metadata
        // BERT/RoBERTa = Post-Norm (false)
        // Llama/Modern Encoders = Pre-Norm (true)
        if meta.is_prenorm {
            self.forward_prenorm(encoder, hidden_states, attention_mask, pool)
        } else {
            self.forward_postnorm(encoder, hidden_states, attention_mask, pool)
        }
    }

    fn forward_prenorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let residual = hidden_states;

        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(
            encoder,
            &self.self_attn_ln_weights,
            hidden_states,
            &ln1_out,
        );

        let (new_k, new_v) =
            self.self_attn
                .project_kv(encoder, &ln1_out, &self.self_attn_weights, 0, pool, None);
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

        let attn_out = self.self_attn.attend(
            encoder,
            &ln1_out, // Query
            &self.self_attn_weights,
            attention_mask,
            false,                        // is_causal is false for encoders
            (&new_k_split, &new_v_split), // K and V are from the input itself
            0,                            // No position offset
            pool,
        );

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add
            .encode(encoder, &[residual, &attn_out], &attn_block_output);

        let residual_2 = &attn_block_output;

        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_layer_norm
            .encode(encoder, &self.ffn_ln_weights, residual_2, &ln2_out);

        let ffn_out = self
            .feedforward
            .encode(encoder, &ln2_out, &self.ff_weights, pool);

        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }

    /// The forward pass logic for a Post-Normalization architecture (e.g., BERT style).
    pub fn forward_postnorm(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        pool: &mut GpuTensorPool,
    ) -> Result<GpuTensor> {
        let residual = hidden_states;

        let (new_k, new_v) = self.self_attn.project_kv(
            encoder,
            hidden_states,
            &self.self_attn_weights,
            0,
            pool,
            None,
        );
        let new_k_split = self.self_attn.split_heads(encoder, &new_k, pool);
        let new_v_split = self.self_attn.split_heads(encoder, &new_v, pool);

        let attn_out = self.self_attn.attend(
            encoder,
            hidden_states, // Query
            &self.self_attn_weights,
            attention_mask,
            false, // is_causal
            (&new_k_split, &new_v_split),
            0,
            pool,
        );

        let add_1_out = pool.get(hidden_states.shape().to_vec());
        self.add.encode(encoder, &[residual, &attn_out], &add_1_out);

        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.self_attn_layer_norm.encode(
            encoder,
            &self.self_attn_ln_weights,
            &add_1_out,
            &attn_block_output,
        );

        let residual_2 = &attn_block_output;

        let ffn_out = self
            .feedforward
            .encode(encoder, residual_2, &self.ff_weights, pool);

        let add_2_out = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &add_2_out);

        let final_output = pool.get(residual_2.shape().to_vec());
        self.ffn_layer_norm
            .encode(encoder, &self.ffn_ln_weights, &add_2_out, &final_output);

        Ok(final_output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::WgpuContext;
    use crate::activations::Activation;
    use crate::encoder::encoder_layer::EncoderLayer;
    use crate::encoder::encoder_self_attention::EncoderSelfAttention;
    use crate::feedforward::{FeedForward, StdFeedForward};
    use crate::gpu_ops::blocks::GpuFeedForwardWeightsStd;
    use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
    use crate::gpu_ops::blocks::encoder::GpuEncoderLayer;
    use crate::gpu_ops::blocks::layer_norm::GpuLayerNormWeights;
    use crate::gpu_ops::{GpuFrameContext, GpuTensor};
    use crate::linear_layer::LinearLayer;
    use crate::normalization::LayerNorm;
    use crate::traits::{AttentionLayout, DecoderLayerLayout, DecoderLayout, FeedForwardLayout, ModelLayout};
    use anyhow::Result;
    use ndarray::{Array1, Array2, Array3};
    use std::sync::Arc;

    async fn get_test_context() -> Arc<WgpuContext> {
        WgpuContext::new().await.unwrap()
    }
    #[tokio::test]
    async fn test_bart_weights_loading_parity() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        // Load actual BART weights
        let model_path =
            std::path::Path::new("/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6");
        let weights = crate::weights::ModelWeights::new(model_path)?;

        let prefix = "model.encoder.layers.0";

        // Load Q weight two ways:
        // 1. via get_raw -> from_raw (what BartGpuEncoder does)
        let q_raw = weights.get_raw(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let q_gpu_from_raw = GpuTensor::from_raw(&ctx, &q_raw, "q_raw")?;

        // 2. via get_array2 -> from_ndarray (what works in test)
        let q_arr = weights.get_array2(&format!("{}.self_attn.q_proj.weight", prefix))?;
        let q_gpu_from_ndarray = GpuTensor::from_ndarray(&ctx, &q_arr)?;

        // Compare shapes
        println!("from_raw shape: {:?}", q_gpu_from_raw.shape());
        println!("from_ndarray shape: {:?}", q_gpu_from_ndarray.shape());

        // Compare actual values
        let raw_cpu = q_gpu_from_raw.to_ndarray_2d::<f32>().await?;
        let ndarray_cpu = q_gpu_from_ndarray.to_ndarray_2d::<f32>().await?;

        println!(
            "from_raw first 5: {:?}",
            raw_cpu.iter().take(5).collect::<Vec<_>>()
        );
        println!(
            "from_ndarray first 5: {:?}",
            ndarray_cpu.iter().take(5).collect::<Vec<_>>()
        );

        let max_diff = raw_cpu
            .iter()
            .zip(ndarray_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        println!("Max diff between loading methods: {}", max_diff);

        assert!(max_diff < 1e-6, "Weights differ based on loading method!");

        Ok(())
    }
    fn assert_close(cpu: &Array3<f32>, gpu: &Array3<f32>, atol: f32, name: &str) {
        assert_eq!(cpu.shape(), gpu.shape(), "{} shape mismatch", name);
        let max_diff = cpu
            .iter()
            .zip(gpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        if max_diff > atol {
            println!("[FAIL] {} - Max diff: {}", name, max_diff);
            println!(
                "  CPU first 10: {:?}",
                cpu.iter().take(10).collect::<Vec<_>>()
            );
            println!(
                "  GPU first 10: {:?}",
                gpu.iter().take(10).collect::<Vec<_>>()
            );
            panic!("{} mismatch: max_diff={}", name, max_diff);
        } else {
            println!("[PASS] {} - Max diff: {}", name, max_diff);
        }
    }

    /// Create deterministic "random" weights for reproducibility
    fn make_weight(rows: usize, cols: usize, seed: usize) -> Array2<f32> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            let idx = seed * 10000 + i * cols + j;
            ((idx % 1000) as f32 - 500.0) * 0.001
        })
    }

    fn make_bias(size: usize, seed: usize) -> Array1<f32> {
        Array1::from_shape_fn(size, |i| {
            let idx = seed * 1000 + i;
            ((idx % 100) as f32 - 50.0) * 0.001
        })
    }

    #[tokio::test]
    async fn test_encoder_layer_cpu_vs_gpu_parity() -> Result<()> {
        let ctx = Arc::new(WgpuContext::new().await?);

        let hidden_size = 256;
        let intermediate_size = 512;
        let num_heads = 4;
        let batch_size = 2;
        let seq_len = 8;
        let eps = 1e-5;

        // === Create weights (shared between CPU and GPU) ===
        // All weights in [Out, In] layout

        // Attention weights [hidden, hidden]
        let q_w = make_weight(hidden_size, hidden_size, 1);
        let k_w = make_weight(hidden_size, hidden_size, 2);
        let v_w = make_weight(hidden_size, hidden_size, 3);
        let o_w = make_weight(hidden_size, hidden_size, 4);
        let q_b = make_bias(hidden_size, 1);
        let k_b = make_bias(hidden_size, 2);
        let v_b = make_bias(hidden_size, 3);
        let o_b = make_bias(hidden_size, 4);

        // FFN weights [Out, In]
        let fc1_w = make_weight(intermediate_size, hidden_size, 5); // [512, 256]
        let fc2_w = make_weight(hidden_size, intermediate_size, 6); // [256, 512]
        let fc1_b = make_bias(intermediate_size, 5);
        let fc2_b = make_bias(hidden_size, 6);

        // LayerNorm weights
        let attn_ln_w = Array1::from_elem(hidden_size, 1.0f32);
        let attn_ln_b = Array1::zeros(hidden_size);
        let ffn_ln_w = Array1::from_elem(hidden_size, 1.0f32);
        let ffn_ln_b = Array1::zeros(hidden_size);

        // === Build CPU EncoderLayer ===
        let cpu_layer = {
            let self_attn = EncoderSelfAttention::new(
                hidden_size,
                num_heads,
                LinearLayer::new_f32(q_w.clone(), Some(q_b.clone())),
                LinearLayer::new_f32(k_w.clone(), Some(k_b.clone())),
                LinearLayer::new_f32(v_w.clone(), Some(v_b.clone())),
                LinearLayer::new_f32(o_w.clone(), Some(o_b.clone())),
            );
            let self_attn_ln = LayerNorm::new(attn_ln_w.clone(), attn_ln_b.clone(), eps);

            let ffn = FeedForward::Standard(StdFeedForward::new(
                fc1_w.clone(),
                fc1_b.clone(),
                fc2_w.clone(),
                fc2_b.clone(),
                Activation::Gelu,
            ));
            let ffn_ln = LayerNorm::new(ffn_ln_w.clone(), ffn_ln_b.clone(), eps);

            EncoderLayer::new(self_attn, self_attn_ln, ffn, ffn_ln)
        };

        // === Build GPU EncoderLayer ===
        let gpu_layer = {
            let self_attn_weights = GpuAttentionWeights::new(
                GpuTensor::from_ndarray(&ctx, &q_w)?,
                GpuTensor::from_ndarray(&ctx, &q_b)?,
                GpuTensor::from_ndarray(&ctx, &k_w)?,
                GpuTensor::from_ndarray(&ctx, &k_b)?,
                GpuTensor::from_ndarray(&ctx, &v_w)?,
                GpuTensor::from_ndarray(&ctx, &v_b)?,
                GpuTensor::from_ndarray(&ctx, &o_w)?,
                GpuTensor::from_ndarray(&ctx, &o_b)?,
            )?;

            let self_attn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&ctx, &attn_ln_w)?,
                GpuTensor::from_ndarray(&ctx, &attn_ln_b)?,
            )?;

            let ff_weights = GpuFeedForwardWeightsStd::new(
                GpuTensor::from_ndarray(&ctx, &fc1_w)?,
                GpuTensor::from_ndarray(&ctx, &fc1_b)?,
                GpuTensor::from_ndarray(&ctx, &fc2_w)?,
                GpuTensor::from_ndarray(&ctx, &fc2_b)?,
            )?;

            let ffn_ln_weights = GpuLayerNormWeights::new(
                GpuTensor::from_ndarray(&ctx, &ffn_ln_w)?,
                GpuTensor::from_ndarray(&ctx, &ffn_ln_b)?,
            )?;
            let mock_meta = ModelMetadata {
                hidden_size,
                num_layers: 1,
                num_attention_heads: num_heads,
                num_kv_heads: num_heads,
                head_dim: hidden_size / num_heads,
                vocab_size: 32000,
                max_seq_len: 512,
                norm_eps: eps,
                activation: Activation::Gelu,
                rope_theta: None,
                rope_scaling: None,
                scale_embeddings: false,
                extra_pos_embeddings: 0,
                is_prenorm: false, // BERT/Mock style
                transpose_ffn_weights: false,
                transpose_attention_weights: false,
            };
            GpuEncoderLayer::new(
                &ctx,
                self_attn_weights,
                self_attn_ln_weights,
                ff_weights,
                ffn_ln_weights,
                Activation::Gelu,
                &mock_meta,
            )?
        };

        // === Create test input ===
        let input = Array3::from_shape_fn((batch_size, seq_len, hidden_size), |(b, s, h)| {
            ((b * 1000 + s * 100 + h) % 200) as f32 * 0.01 - 1.0
        });
        let mask = Array2::<f32>::ones((batch_size, seq_len));

        // === Run CPU ===
        let cpu_output = cpu_layer.forward(input.clone(), &mask, None, false)?; // post-norm

        // === Run GPU ===
        let input_gpu = GpuTensor::from_ndarray(&ctx, &input)?;
        let mask_gpu = GpuTensor::from_ndarray(&ctx, &mask)?;

        let pool = ctx.get_inference_pool();
        let mut pool_guard = pool.lock().await;
        let mut frame = GpuFrameContext::new(&ctx, pool_guard);
        let (encoder_cmd, pool_ref) = frame.resources();

        let gpu_output_tensor =
            gpu_layer.forward_postnorm(encoder_cmd, &input_gpu, &mask_gpu, pool_ref)?;

        frame.finish();

        let gpu_output = gpu_output_tensor.to_ndarray_3d::<f32>().await?;

        // === Compare ===
        assert_close(&cpu_output, &gpu_output, 1e-4, "EncoderLayer PostNorm");

        println!("✓ CPU vs GPU EncoderLayer parity test passed!");
        Ok(())
    }

    // Mock config for the GPU layer
    struct MockConfig {
        hidden_size: usize,
        num_heads: usize,
        eps: f32,
    }

    impl crate::traits::ModelConfig for MockConfig {
        fn model_type(&self) -> &str {
            "mock"
        }

        fn metadata(&self) -> crate::traits::ModelMetadata {
            crate::traits::ModelMetadata {
                hidden_size: self.hidden_size,
                num_layers: 1,
                num_attention_heads: self.num_heads,
                num_kv_heads: self.num_heads,
                head_dim: self.hidden_size / self.num_heads,
                vocab_size: 1000,
                max_seq_len: 512,
                norm_eps: self.eps,
                activation: crate::activations::Activation::SilU,
                rope_theta: None,
                rope_scaling: None,
                scale_embeddings: false,
                extra_pos_embeddings: 0,
                is_prenorm: false, // Legacy style post-norm
                transpose_ffn_weights: false,
                transpose_attention_weights: false,
            }
        }

        fn layout(&self) -> ModelLayout {
            // --- Define the Decoder's Layer Structure for the test ---
            let decoder_layer = DecoderLayerLayout {
                self_attn: AttentionLayout {
                    q_weight: "layer.{}.attn.q.weight".to_string(),
                    q_bias: Some("layer.{}.attn.q.bias".to_string()),
                    k_weight: "layer.{}.attn.k.weight".to_string(),
                    k_bias: Some("layer.{}.attn.k.bias".to_string()),
                    v_weight: "layer.{}.attn.v.weight".to_string(),
                    v_bias: Some("layer.{}.attn.v.bias".to_string()),
                    o_weight: "layer.{}.attn.o.weight".to_string(),
                    o_bias: Some("layer.{}.attn.o.bias".to_string()),
                    norm_weight: "layer.{}.attn_ln.weight".to_string(),
                    norm_bias: Some("layer.{}.attn_ln.bias".to_string()),
                },
                cross_attn: None, // No cross-attention in this test model
                ffn: FeedForwardLayout {
                    up_weight: "layer.{}.ffn.up.weight".to_string(),
                    up_bias: Some("layer.{}.ffn.up.bias".to_string()),
                    down_weight: "layer.{}.ffn.down.weight".to_string(),
                    down_bias: Some("layer.{}.ffn.down.bias".to_string()),
                    gate_weight: None, // Standard FFN for this test
                    norm_weight: "layer.{}.ffn_ln.weight".to_string(),
                    norm_bias: Some("layer.{}.ffn_ln.bias".to_string()),
                },
            };

            // --- Assemble the final ModelLayout ---
            ModelLayout {
                token_embedding: "embeddings.word_embeddings.weight".to_string(),
                lm_head: "lm_head.weight".to_string(),
                encoder: None, // This is a decoder-only test model
                decoder: Some(DecoderLayout {
                    position_embedding: Some("embeddings.position_embeddings.weight".to_string()),
                    token_type_embedding: None,
                    embedding_norm_weight: Some("embeddings.LayerNorm.weight".to_string()),
                    embedding_norm_bias: Some("embeddings.LayerNorm.bias".to_string()),
                    final_norm_weight: Some("norm.weight".to_string()),
                    final_norm_bias: None,
                    layer: decoder_layer,
                }),
            }
        }
    }
}
