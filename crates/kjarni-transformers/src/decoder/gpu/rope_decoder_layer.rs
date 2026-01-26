use crate::{
    WgpuContext,
    cache::GpuKVCache,
    decoder::{gpu::GpuRoPEAttention, prelude::*},
    gpu_ops::{
        GpuTensor, GpuTensorPool, Kernel,
        blocks::{
            GpuFeedForward, GpuFeedForwardWeights, GpuNormalization, GpuNormalizationWeights,
            GpuSwiGLUFFN, attention::GpuAttentionWeights, rms_norm::GpuRMSNorm, rope::GpuRoPE,
        },
        primitives::add::GpuAdd,
    },
};
use anyhow::Result;
use std::sync::Arc;
pub struct GpuRoPEDecoderLayer {
    pub self_attn: GpuRoPEAttention,
    pub self_attn_weights: GpuAttentionWeights,
    pub self_attn_norm: GpuNormalization,
    pub self_attn_norm_weights: GpuNormalizationWeights,
    pub feedforward: GpuFeedForward,
    pub ff_weights: GpuFeedForwardWeights,
    pub ffn_norm: GpuNormalization,
    pub ffn_norm_weights: GpuNormalizationWeights,
    pub add: GpuAdd,
}

impl GpuRoPEDecoderLayer {
    pub fn new(
        context: &Arc<WgpuContext>,
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        self_attn_weights: GpuAttentionWeights,
        self_attn_norm_weights: GpuNormalizationWeights,
        ff_weights: GpuFeedForwardWeights,
        ffn_norm_weights: GpuNormalizationWeights,
        norm_eps: f32,
    ) -> Result<Self> {
        let self_attn = GpuRoPEAttention::new(
            context,
            hidden_size as u32,
            num_heads as u32,
            num_kv_heads as u32,
        );
        let add = GpuAdd::new(context);

        // Llama specific blocks
        let self_attn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let ffn_norm = GpuNormalization::RMSNorm(GpuRMSNorm::new(context, norm_eps));
        let feedforward = GpuFeedForward::SwiGLU(GpuSwiGLUFFN::new(context)?);

        Ok(Self {
            self_attn,
            self_attn_weights,
            self_attn_norm,
            self_attn_norm_weights,
            feedforward,
            ff_weights,
            ffn_norm,
            ffn_norm_weights,
            add,
        })
    }
    pub fn forward(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        hidden_states: &GpuTensor,
        attention_mask: &GpuTensor,
        layer_idx: usize,
        position_offset: usize,
        gpu_cache: Option<&mut GpuKVCache>,
        pool: &mut GpuTensorPool,
        rope: &GpuRoPE,
    ) -> Result<GpuTensor> {
        // --- 1. Self-Attention Block (Pre-Norm) ---
        let residual = hidden_states;
        let ln1_out = pool.get(hidden_states.shape().to_vec());
        self.self_attn_norm.encode(
            encoder,
            &self.self_attn_norm_weights,
            hidden_states,
            &ln1_out,
        );

        // Get cached KV if available (returns Option<(&GpuTensor, &GpuTensor)>)
        let cached_tensors = gpu_cache.as_ref().and_then(|c| c.get(layer_idx));

        let cached_kv: Option<(&GpuTensor, &GpuTensor)> =
            cached_tensors.as_ref().map(|(k, v)| (k, v));

        // Single forward call handles: Q/K/V projection, RoPE, GQA, attention, output projection
        let attn_output = self.self_attn.forward(
            encoder,
            &ln1_out,
            &self.self_attn_weights,
            rope,
            attention_mask,
            cached_kv,
            position_offset,
            pool,
        )?;

        // Update cache with new K/V (need mutable borrow now)
        if let Some(cache) = gpu_cache {
            cache.update(
                encoder,
                layer_idx,
                &attn_output.new_k,
                &attn_output.new_v,
                position_offset,
            )?;
        }

        // Residual add
        let attn_block_output = pool.get(hidden_states.shape().to_vec());
        self.add.encode(
            encoder,
            &[residual, &attn_output.hidden_states],
            &attn_block_output,
        );

        // --- 2. Feed-Forward Block (Pre-Norm) ---
        let residual_2 = &attn_block_output;
        let ln2_out = pool.get(residual_2.shape().to_vec());
        self.ffn_norm
            .encode(encoder, &self.ffn_norm_weights, residual_2, &ln2_out);

        // FFN (needs 2D input)
        let (b, s, h) = ln2_out.dims3();
        let ln2_out_2d = ln2_out.view(vec![b * s, h]);
        let ffn_out_2d = pool.get(vec![b * s, h]);

        self.feedforward
            .encode(encoder, &self.ff_weights, &ln2_out_2d, &ffn_out_2d, pool);
        let ffn_out = ffn_out_2d.view(vec![b, s, h]);

        // Residual add
        let final_output = pool.get(residual_2.shape().to_vec());
        self.add
            .encode(encoder, &[residual_2, &ffn_out], &final_output);

        Ok(final_output)
    }
}

#[cfg(test)]
mod rope_decoder_gpu_test {

    use super::*;
    use crate::cpu::decoder::{CpuRoPEDecoderLayer, DecoderAttention};
    use crate::feedforward::SwiGluFeedForward;
    use crate::gpu_ops::blocks::{GpuFeedForwardWeights, GpuRMSNormWeights, GpuSwiGLUFFNWeights};
    use crate::gpu_ops::{GpuTensor, GpuTensorPool};
    use crate::linear_layer::LinearLayer;
    use crate::cpu::normalization::RMSNorm as CpuRMSNorm;
    use crate::rope::RoPE;
    use crate::{Normalization, WgpuContext};
    use anyhow::Result;
    use ndarray::{Array, Array1, Array2, Array3};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    use std::sync::Arc;

    // =========================================================================
    //  Helpers
    // =========================================================================

    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] Shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }
        let mut max_diff = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let diff = (a_val - b_val).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            let tolerance = atol + rtol * b_val.abs();
            if diff > tolerance {
                panic!(
                    "[{}] Mismatch: {} vs {} (diff {})",
                    context, a_val, b_val, diff
                );
            }
        }
        println!("[{}] Passed. Max diff: {:.6e}", context, max_diff);
    }

    fn create_mock_cpu_layer(
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        intermediate: usize,
    ) -> CpuRoPEDecoderLayer {
        let gen_w = |shape, scale| Array2::from_shape_fn(shape, |(i, j)| ((i + j) as f32 * scale));
        // Llama uses RMSNorm (no bias usually, but we mock weights)
        let gen_norm = |size| Array1::from_elem(size, 1.0);

        let attention = DecoderAttention::new(
            hidden,
            heads,
            LinearLayer::from(gen_w((hidden, hidden), 0.001)), // Q
            LinearLayer::from(gen_w((hidden / heads * kv_heads, hidden), 0.002)), // K
            LinearLayer::from(gen_w((hidden / heads * kv_heads, hidden), 0.003)), // V
            LinearLayer::from(gen_w((hidden, hidden), 0.004)), // O
            Some(kv_heads),
        );

        let feed_forward = SwiGluFeedForward::new(
            LinearLayer::from(gen_w((intermediate, hidden), 0.01)), // Gate
            LinearLayer::from(gen_w((intermediate, hidden), 0.02)), // Up
            LinearLayer::from(gen_w((hidden, intermediate), 0.03)), // Down
            crate::activations::Activation::SilU,
        );

        let rope = Arc::new(RoPE::new(hidden / heads, 1024, 10000.0));

        CpuRoPEDecoderLayer {
            attention,
            feed_forward,
            attention_norm: Normalization::RMSNorm(CpuRMSNorm::new(gen_norm(hidden), 1e-5)),
            ffn_norm: Normalization::RMSNorm(CpuRMSNorm::new(gen_norm(hidden), 1e-5)),
            rope,
        }
    }

    fn create_gpu_layer_from_cpu(
        ctx: &Arc<WgpuContext>,
        cpu: &CpuRoPEDecoderLayer,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
    ) -> Result<GpuRoPEDecoderLayer> {
        let load = |l: &LinearLayer| l.to_gpu(ctx);

        // 1. Attention Weights
        let sa_weights = GpuAttentionWeights::new(
            load(&cpu.attention.q_proj)?,
            None,
            load(&cpu.attention.k_proj)?,
            None,
            load(&cpu.attention.v_proj)?,
            None,
            load(&cpu.attention.o_proj)?,
            None,
        )?;

        // 2. Norm Weights
        let load_norm = |n: &Normalization| -> Result<GpuNormalizationWeights> {
            match n {
                Normalization::RMSNorm(rms) => {
                    let w = GpuTensor::from_ndarray(ctx, &rms.weight)?;
                    let ww = GpuRMSNormWeights::new(w)?;
                    Ok(GpuNormalizationWeights::RMSNorm(ww))
                }
                _ => panic!("Expected RMSNorm"),
            }
        };
        let sa_norm_w = load_norm(&cpu.attention_norm)?;
        let ffn_norm_w = load_norm(&cpu.ffn_norm)?;

        // 3. FFN Weights
        let ff_weights = {
            let gate = load(&cpu.feed_forward.gate)?;
            let up = load(&cpu.feed_forward.up)?;
            let down = load(&cpu.feed_forward.down)?;
            
            let w = GpuSwiGLUFFNWeights::new(gate, up, down)?;
            GpuFeedForwardWeights::SwiGLU(w)
        };

        GpuRoPEDecoderLayer::new(
            ctx, hidden, heads, kv_heads, sa_weights, sa_norm_w, ff_weights, ffn_norm_w, 1e-5,
        )
    }

    // =========================================================================
    //  End-to-End Parity Test
    // =========================================================================

    #[tokio::test]
    async fn test_rope_decoder_layer_parity() -> Result<()> {
        // 1. Setup
        let context = WgpuContext::new().await?;
        let (batch, seq_len, hidden, heads, kv_heads, inter) = (1, 3, 64, 4, 2, 128); // GQA 4:2
        let head_dim = hidden / heads;

        // 2. Create Layers
        let cpu_layer = create_mock_cpu_layer(hidden, heads, kv_heads, inter);
        let gpu_layer = create_gpu_layer_from_cpu(&context, &cpu_layer, hidden, heads, kv_heads)?;
        let gpu_rope =
            crate::gpu_ops::blocks::rope::GpuRoPE::from_cpu_rope(&context, &cpu_layer.rope)?;

        // 3. Create Inputs
        let hidden_cpu = Array::random((batch, seq_len, hidden), Uniform::new(-0.5, 0.5));
        let mask_cpu = Array2::ones((batch, seq_len)); // Full mask for simplicity

        let hidden_gpu = GpuTensor::from_ndarray(&context, &hidden_cpu)?;
        let mask_gpu = GpuTensor::from_ndarray(&context, &mask_cpu)?;

        // 4. Create Cache (Required for forward)
        // CPU Cache: simple arrays
        let mut k_cache_cpu = Array3::<f32>::zeros((batch, seq_len, kv_heads * head_dim));
        let mut v_cache_cpu = Array3::<f32>::zeros((batch, seq_len, kv_heads * head_dim));

        // GPU Cache: GpuKVCache
        let mut gpu_cache =
            crate::cache::GpuKVCache::new(&context, 1, batch, kv_heads, head_dim, seq_len)?;

        // 5. Run CPU Forward
        let cpu_out = cpu_layer.forward(
            &hidden_cpu,
            &mask_cpu,
            0, // offset
            k_cache_cpu.view_mut(),
            v_cache_cpu.view_mut(),
        )?;

        // 6. Run GPU Forward
        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let gpu_out_t = gpu_layer.forward(
            &mut encoder,
            &hidden_gpu,
            &mask_gpu,
            0, // layer idx
            0, // offset
            Some(&mut gpu_cache),
            &mut pool,
            &gpu_rope,
        )?;

        context.queue.submit(Some(encoder.finish()));
        let gpu_out = gpu_out_t.to_ndarray_3d().await?;

        // 7. Verify Output
        assert_all_close(&cpu_out, &gpu_out, 1e-3, 1e-4, "Decoder Output");

        // 8. Verify Cache
        // Note: CPU cache writes are usually [batch, seq, head*dim]
        // GPU cache writes are [batch, heads, seq, dim]
        // We need to fetch and reshape GPU cache to compare.
        let (k_gpu_t, v_gpu_t) = gpu_cache.get(0).unwrap();
        let k_gpu_raw = k_gpu_t.to_ndarray_4d::<f32>().await?; // [1, 2, 3, 16]

        let p = k_gpu_raw.permuted_axes([0, 2, 1, 3]);

        // Reshape GPU [b, h, s, d] -> [b, s, h, d] -> [b, s, h*d]
        let k_gpu_reshaped = p
            .as_standard_layout()
            .into_shape_with_order((batch, seq_len, kv_heads * head_dim))?;

        assert_all_close(&k_cache_cpu, &k_gpu_reshaped.to_owned(), 1e-3, 1e-4, "K Cache");

        Ok(())
    }

    // =========================================================================
    //  Subcomponent Parity Test (Step-by-Step)
    // =========================================================================

    #[tokio::test]
    async fn test_rope_layer_subcomponents() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (batch, seq, hidden, heads, kv_heads, inter) = (1, 1, 32, 4, 4, 64);

        let cpu = create_mock_cpu_layer(hidden, heads, kv_heads, inter);
        let gpu = create_gpu_layer_from_cpu(&context, &cpu, hidden, heads, kv_heads)?;
        let rope = crate::gpu_ops::blocks::rope::GpuRoPE::from_cpu_rope(&context, &cpu.rope)?;

        let input = Array::random((batch, seq, hidden), Uniform::new(-1.0, 1.0));
        let mask = Array2::ones((batch, seq));

        let input_g = GpuTensor::from_ndarray(&context, &input)?;
        let mask_g = GpuTensor::from_ndarray(&context, &mask)?;

        let mut enc = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        // --- Step 1: Norm 1 ---
        let cpu_norm1 = cpu.attention_norm.forward(&input);
        let gpu_norm1 = pool.get(input_g.shape().to_vec());
        gpu.self_attn_norm
            .encode(&mut enc, &gpu.self_attn_norm_weights, &input_g, &gpu_norm1);

        // --- Step 2: Attention (No Cache) ---
        // CPU: Mock cache just for the call
        let mut k_tmp = Array3::zeros((batch, seq, hidden));
        let mut v_tmp = Array3::zeros((batch, seq, hidden));
        let cpu_attn = cpu.attention.forward(
            &cpu_norm1,
            Some(&mask),
            k_tmp.view_mut(),
            v_tmp.view_mut(),
            0,
            Some(&cpu.rope),
        )?;

        let gpu_attn_out = gpu.self_attn.forward(
            &mut enc,
            &gpu_norm1,
            &gpu.self_attn_weights,
            &rope,
            &mask_g,
            None,
            0,
            &mut pool,
        )?;

        // --- Step 3: Residual 1 ---
        let cpu_res1 = &input + &cpu_attn;
        // GPU: Implicit in full forward, but we can check attn output

        // Verify Steps so far
        context.queue.submit(Some(enc.finish()));

        let gpu_norm1_v = gpu_norm1.to_ndarray_3d().await?;
        assert_all_close(&cpu_norm1, &gpu_norm1_v, 1e-4, 1e-5, "Norm 1");

        let gpu_attn_v = gpu_attn_out.hidden_states.to_ndarray_3d().await?;
        assert_all_close(&cpu_attn, &gpu_attn_v, 1e-3, 1e-4, "Attention Output");

        Ok(())
    }
}
