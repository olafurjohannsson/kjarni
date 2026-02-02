use std::sync::Arc;

use anyhow::Result;
use ndarray::{Array, Array1, Array2, Array3};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

use super::*;
use crate::cpu::encoder_decoder::decoder_cross_attn::DecoderCrossAttention;
use crate::cpu::encoder_decoder::decoder_cross_attn_layer::CrossDecoderLayer as CpuDecoderLayer;
use crate::cpu::normalization::LayerNorm as CpuLayerNorm;
use crate::encoder_decoder::decoder_self_attn::DecoderSelfAttention;
use crate::feedforward::{FeedForward as CpuFf, LegacyFeedForward as CpuStdFf};
use crate::gpu::normalization::{
    GpuLayerNorm, GpuLayerNormWeights, GpuNormalization, GpuNormalizationWeights,
};
use crate::gpu_ops::blocks::attention::GpuAttentionWeights;
use crate::gpu_ops::blocks::{
    GpuFeedForward, GpuFeedForwardStd, GpuFeedForwardWeights, GpuFeedForwardWeightsStd,
};
use crate::gpu::{GpuTensor, GpuTensorPool, Kernel};
use crate::linear_layer::LinearLayer;
use crate::{Normalization, WgpuContext};

mod gpu_cross_decoder_test {
    use super::*;

    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let abs_diff = (a_val - b_val).abs();
            let tolerance = atol + rtol * b_val.abs();
            if abs_diff > tolerance {
                panic!(
                    "[{}] arrays not close at a={}, b={}, diff {} > tolerance {}",
                    context, a_val, b_val, abs_diff, tolerance
                );
            }
        }
    }

    fn create_mock_cpu_layer(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
    ) -> CpuDecoderLayer {
        let gen_weight =
            |shape, scale| Array2::from_shape_fn(shape, |(i, j)| ((i + j) as f32 * scale));
        let gen_bias = |size, val| Array1::from_elem(size, val);

        let self_attn = DecoderSelfAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.001)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.002)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.003)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), -0.001)),
        );
        let self_attn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
            gen_bias(hidden_size, 1.0),
            gen_bias(hidden_size, 0.0),
            1e-5,
        ));

        let cross_attn = DecoderCrossAttention::new(
            hidden_size,
            num_heads,
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.004)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.005)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), 0.006)),
            LinearLayer::from(gen_weight((hidden_size, hidden_size), -0.002)),
        );
        let cross_attn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
            gen_bias(hidden_size, 1.0),
            gen_bias(hidden_size, 0.0),
            1e-5,
        ));

        let feedforward = CpuFf::Legacy(CpuStdFf::new(
            gen_weight((hidden_size, intermediate_size), 0.01),
            gen_bias(intermediate_size, 0.0),
            gen_weight((intermediate_size, hidden_size), -0.01),
            gen_bias(hidden_size, 0.0),
            crate::activations::Activation::Gelu,
        ));
        let ffn_layer_norm = Normalization::LayerNorm(CpuLayerNorm::new(
            gen_bias(hidden_size, 1.0),
            gen_bias(hidden_size, 0.0),
            1e-5,
        ));

        CpuDecoderLayer {
            self_attn,
            self_attn_layer_norm,
            cross_attn,
            cross_attn_layer_norm,
            feedforward,
            ffn_layer_norm,
            pre_norm: false,
        }
    }

    fn create_gpu_layer_from_cpu(
        context: &Arc<WgpuContext>,
        cpu_layer: &CpuDecoderLayer,
        hidden_size: u32,
        num_heads: u32,
    ) -> Result<GpuCrossDecoderLayer> {
        let load_linear = |layer: &LinearLayer| -> Result<(GpuTensor, GpuTensor)> {
            let w_gpu = layer.to_gpu(context)?;

            let b_gpu = if let Some(bias) = &layer.bias {
                GpuTensor::from_ndarray(context, bias)?
            } else {
                let out_features = layer.out_features();
                let zeros = ndarray::Array1::<f32>::zeros(out_features);
                GpuTensor::from_ndarray(context, &zeros)?
            };

            Ok((w_gpu, b_gpu))
        };

        let (qw, qb) = load_linear(&cpu_layer.self_attn.q_proj)?;
        let (kw, kb) = load_linear(&cpu_layer.self_attn.k_proj)?;
        let (vw, vb) = load_linear(&cpu_layer.self_attn.v_proj)?;
        let (ow, ob) = load_linear(&cpu_layer.self_attn.o_proj)?;

        let self_attn_weights =
            GpuAttentionWeights::new(qw, Some(qb), kw, Some(kb), vw, Some(vb), ow, Some(ob))?;

        let self_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(
                context,
                &cpu_layer
                    .self_attn_layer_norm
                    .as_layer_norm()
                    .unwrap()
                    .weight,
            )?,
            GpuTensor::from_ndarray(
                context,
                &cpu_layer.self_attn_layer_norm.as_layer_norm().unwrap().bias,
            )?,
        )?);

        let (xqw, xqb) = load_linear(&cpu_layer.cross_attn.q_proj)?;
        let (xkw, xkb) = load_linear(&cpu_layer.cross_attn.k_proj)?;
        let (xvw, xvb) = load_linear(&cpu_layer.cross_attn.v_proj)?;
        let (xow, xob) = load_linear(&cpu_layer.cross_attn.o_proj)?;

        let cross_attn_weights = GpuAttentionWeights::new(
            xqw,
            Some(xqb),
            xkw,
            Some(xkb),
            xvw,
            Some(xvb),
            xow,
            Some(xob),
        )?;

        let cross_attn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(
                context,
                &cpu_layer
                    .cross_attn_layer_norm
                    .as_layer_norm()
                    .unwrap()
                    .weight,
            )?,
            GpuTensor::from_ndarray(
                context,
                &cpu_layer
                    .cross_attn_layer_norm
                    .as_layer_norm()
                    .unwrap()
                    .bias,
            )?,
        )?);

        let ff_weights = if let crate::feedforward::FeedForward::Legacy(ff) = &cpu_layer.feedforward
        {
            let weights_std = GpuFeedForwardWeightsStd::from_ndarrays(
                context,
                &ff.dense1_weight,
                &ff.dense1_bias,
                &ff.dense2_weight,
                &ff.dense2_bias,
            )?;
            GpuFeedForwardWeights::Standard(weights_std)
        } else {
            panic!("expected standard feedforward layer");
        };

        let ffn_norm_weights = GpuNormalizationWeights::LayerNorm(GpuLayerNormWeights::new(
            GpuTensor::from_ndarray(
                context,
                &cpu_layer.ffn_layer_norm.as_layer_norm().unwrap().weight,
            )?,
            GpuTensor::from_ndarray(
                context,
                &cpu_layer.ffn_layer_norm.as_layer_norm().unwrap().bias,
            )?,
        )?);

        Ok(GpuCrossDecoderLayer {
            self_attn: GpuDecoderSelfAttention::new(context, hidden_size, num_heads),
            self_attn_weights,
            self_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            self_attn_norm_weights,
            cross_attn: GpuCrossAttention::new(context, hidden_size, num_heads),
            cross_attn_weights,
            cross_attn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            cross_attn_norm_weights,
            feedforward: GpuFeedForward::Standard(GpuFeedForwardStd::new(
                context,
                crate::activations::Activation::Gelu,
            )?),
            ff_weights,
            ffn_norm: GpuNormalization::LayerNorm(GpuLayerNorm::new(context, 1e-5)),
            ffn_norm_weights,
            add: crate::gpu_ops::primitives::add::GpuAdd::new(context),
        })
    }

    #[tokio::test]
    async fn test_gpu_cpu_layer_consistency() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

        let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
        let gpu_layer =
            create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

        let cpu_decoder_hs = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_decoder_mask = Array2::ones((batch, dec_len));
        let cpu_encoder_mask = Array2::ones((batch, enc_len));

        let cpu_cross_kv = cpu_layer.precompute_cross_kv(&cpu_encoder_hs)?;

        let (cpu_output, (cpu_k, cpu_v)) = cpu_layer.forward(
            &cpu_decoder_hs,
            &cpu_encoder_hs,
            Some(&cpu_decoder_mask),
            Some(&cpu_encoder_mask),
            None,
            Some(&cpu_cross_kv),
            None,
        )?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let gpu_decoder_hs = GpuTensor::from_ndarray(&context, &cpu_decoder_hs)?;
        let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
        let gpu_decoder_mask = GpuTensor::from_ndarray(&context, &cpu_decoder_mask)?;
        let gpu_encoder_mask = GpuTensor::from_ndarray(&context, &cpu_encoder_mask)?;

        let gpu_cross_kv = gpu_layer.precompute_cross_kv(&mut encoder, &gpu_encoder_hs, &mut pool);

        let (gpu_output_t, gpu_k_t, gpu_v_t) = gpu_layer.forward(
            &mut encoder,
            &gpu_decoder_hs,
            &gpu_cross_kv,
            &gpu_decoder_mask,
            Some(&gpu_encoder_mask),
            None,
            0,
            &mut pool,
        )?;

        context.queue.submit(Some(encoder.finish()));
        pool.next_frame();

        let gpu_output = gpu_output_t.to_ndarray_3d().await?;
        let gpu_k = gpu_k_t.to_ndarray_3d().await?;
        let gpu_v = gpu_v_t.to_ndarray_3d().await?;

        let rtol = 1e-3;
        let atol = 1e-4;
        assert_all_close(&cpu_output, &gpu_output, rtol, atol, "final output");
        assert_all_close(&cpu_k, &gpu_k, rtol, atol, "new K value");
        assert_all_close(&cpu_v, &gpu_v, rtol, atol, "new V value");

        Ok(())
    }

    #[tokio::test]
    async fn test_layer_subcomponent_parity() -> Result<()> {
        let context = WgpuContext::new().await?;
        let (batch, dec_len, enc_len, hidden, inter, heads) = (1, 1, 93, 1024, 4096, 16);

        let cpu_layer = create_mock_cpu_layer(hidden, inter, heads);
        let gpu_layer =
            create_gpu_layer_from_cpu(&context, &cpu_layer, hidden as u32, heads as u32)?;

        let cpu_hidden = Array::random((batch, dec_len, hidden), Uniform::new(-1.0, 1.0));
        let cpu_encoder_hs = Array::random((batch, enc_len, hidden), Uniform::new(-1.0, 1.0));

        let cpu_dec_mask = Array2::ones((batch, dec_len));
        let cpu_enc_mask = Array2::ones((batch, enc_len));

        let gpu_hidden = GpuTensor::from_ndarray(&context, &cpu_hidden)?;
        let gpu_encoder_hs = GpuTensor::from_ndarray(&context, &cpu_encoder_hs)?;
        let gpu_dec_mask = GpuTensor::from_ndarray(&context, &cpu_dec_mask)?;
        let gpu_enc_mask = GpuTensor::from_ndarray(&context, &cpu_enc_mask)?;

        let mut encoder = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());

        let (attn_out, new_k, new_v) =
            cpu_layer
                .self_attn
                .forward(&cpu_hidden, Some(&cpu_dec_mask), None, None)?;
        let hidden_states_after_add = &cpu_hidden + &attn_out;
        let final_output = cpu_layer
            .self_attn_layer_norm
            .forward(&hidden_states_after_add);
        let (cpu_sa_out, (_cpu_k, _cpu_v)) = (final_output, (new_k, new_v));

        let o = gpu_layer.self_attn.forward(
            &mut encoder,
            &gpu_hidden,
            &gpu_layer.self_attn_weights,
            &gpu_dec_mask,
            None,
            0,
            &mut pool,
        )?;
        let gpu_sa_attn_out = o.hidden_states;
        let _gpu_k = o.new_k;
        let _gpu_v = o.new_v;

        let gpu_sa_add = pool.get(gpu_hidden.shape().to_vec());
        gpu_layer
            .add
            .encode(&mut encoder, &[&gpu_hidden, &gpu_sa_attn_out], &gpu_sa_add);

        let gpu_sa_out = pool.get(gpu_sa_add.shape().to_vec());
        gpu_layer.self_attn_norm.encode(
            &mut encoder,
            &gpu_layer.self_attn_norm_weights,
            &gpu_sa_add,
            &gpu_sa_out,
        );

        context.queue.submit(Some(encoder.finish()));
        let gpu_sa_out_cpu = gpu_sa_out.to_ndarray_3d().await?;

        assert_all_close(
            &cpu_sa_out,
            &gpu_sa_out_cpu,
            1e-3,
            1e-4,
            "self-attention block",
        );

        let input_for_step_2 = cpu_sa_out.clone();
        let gpu_input_for_step_2 = GpuTensor::from_ndarray(&context, &input_for_step_2)?;

        encoder = context.device.create_command_encoder(&Default::default());
        pool.next_frame();

        let (k, v) = cpu_layer
            .cross_attn
            .precompute_encoder_kv(&cpu_encoder_hs)?;
        let cross_attn_output =
            cpu_layer
                .cross_attn
                .forward(&input_for_step_2, &k, &v, Some(&cpu_enc_mask));
        let hidden_states_after_add = &input_for_step_2 + &cross_attn_output?;
        let final_output = cpu_layer
            .cross_attn_layer_norm
            .forward(&hidden_states_after_add);
        let cpu_ca_out = final_output;

        let gpu_cross_kv = gpu_layer.precompute_cross_kv(&mut encoder, &gpu_encoder_hs, &mut pool);
        let gpu_ca_attn_out = gpu_layer.cross_attn.forward(
            &mut encoder,
            &gpu_input_for_step_2,
            &gpu_cross_kv,
            &gpu_layer.cross_attn_weights,
            Some(&gpu_enc_mask),
            &mut pool,
        );

        let gpu_ca_add = pool.get(gpu_input_for_step_2.shape().to_vec());
        gpu_layer.add.encode(
            &mut encoder,
            &[&gpu_input_for_step_2, &gpu_ca_attn_out],
            &gpu_ca_add,
        );

        let gpu_ca_out = pool.get(gpu_ca_add.shape().to_vec());
        gpu_layer.cross_attn_norm.encode(
            &mut encoder,
            &gpu_layer.cross_attn_norm_weights,
            &gpu_ca_add,
            &gpu_ca_out,
        );

        context.queue.submit(Some(encoder.finish()));
        let gpu_ca_out_cpu = gpu_ca_out.to_ndarray_3d().await?;

        assert_all_close(
            &cpu_ca_out,
            &gpu_ca_out_cpu,
            1e-3,
            1e-4,
            "cross-attention block",
        );

        let input_for_step_3 = cpu_ca_out.clone();
        let gpu_input_for_step_3 = GpuTensor::from_ndarray(&context, &input_for_step_3)?;

        encoder = context.device.create_command_encoder(&Default::default());
        pool.next_frame();

        let cpu_ffn_out = cpu_layer.feed_forward(&input_for_step_3)?;

        let gpu_ffn_inner_out = pool.get(gpu_input_for_step_3.shape().to_vec());
        gpu_layer.feedforward.encode(
            &mut encoder,
            &gpu_layer.ff_weights,
            &gpu_input_for_step_3,
            &gpu_ffn_inner_out,
            &mut pool,
        );

        let gpu_ffn_add = pool.get(gpu_input_for_step_3.shape().to_vec());
        gpu_layer.add.encode(
            &mut encoder,
            &[&gpu_input_for_step_3, &gpu_ffn_inner_out],
            &gpu_ffn_add,
        );

        let gpu_ffn_out = pool.get(gpu_ffn_add.shape().to_vec());
        gpu_layer.ffn_norm.encode(
            &mut encoder,
            &gpu_layer.ffn_norm_weights,
            &gpu_ffn_add,
            &gpu_ffn_out,
        );

        context.queue.submit(Some(encoder.finish()));
        let gpu_ffn_out_cpu = gpu_ffn_out.to_ndarray_3d().await?;

        assert_all_close(
            &cpu_ffn_out,
            &gpu_ffn_out_cpu,
            1e-3,
            1e-4,
            "feed-forward block",
        );

        Ok(())
    }
}
