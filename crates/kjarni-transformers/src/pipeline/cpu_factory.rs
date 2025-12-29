use crate::decoder::prelude::DecoderAttention;
use crate::feedforward::{FeedForward, LegacyFeedForward, SwiGluFeedForward};
use crate::linear_layer::{F32MatmulStrategy, LinearLayer};
use crate::normalization::{LayerNorm, Normalization, RMSNorm};
use crate::tensor::DType;
use crate::traits::{AttentionLayout, FeedForwardLayout, ModelLayout, ModelMetadata};
use crate::weights::ModelWeights;
use anyhow::{Context, Result, anyhow};

pub struct CpuLayerFactory;

impl CpuLayerFactory {
    /// Builds a Normalization brick.
    /// Logic: If bias template exists -> LayerNorm, else -> RMSNorm.
    pub fn build_norm(
        weights: &ModelWeights,
        w_template: &String,
        b_template: &Option<String>,
        eps: f32,
        idx: usize,
    ) -> Result<Normalization> {
        let i_str = idx.to_string();
        let w_name = w_template.replace("{}", &i_str);

        let weight = weights.get_array1(&w_name)?;

        if let Some(b_tmpl) = b_template {
            let bias = weights.get_array1(&b_tmpl.replace("{}", &i_str))?;
            Ok(Normalization::LayerNorm(LayerNorm::new(weight, bias, eps)))
        } else {
            Ok(Normalization::RMSNorm(RMSNorm::new(weight, eps)))
        }
    }
    pub fn build_swiglu_ffn(
        weights: &ModelWeights,
        layout: &FeedForwardLayout,
        idx: usize,
        target_dtype: Option<DType>,
    ) -> Result<SwiGluFeedForward> {
        let i_str = idx.to_string();
        let strategy = Some(F32MatmulStrategy::CustomSimd);

        // Helper to resolve template strings
        let name = |t: &String| t.replace("{}", &i_str);

        // 1. Get the Gate name (SwiGLU requires this)
        let gate_name = layout.gate_weight.as_ref().ok_or_else(|| {
            anyhow!("Architecture layout missing required gate_weight for SwiGLU")
        })?;

        // 2. Load the 3 Linear Layers
        // SwiGLU variants typically do not use biases, so we pass None
        let gate =
            LinearLayer::from_weights(weights, &name(gate_name), None, target_dtype, strategy)?;

        let up = LinearLayer::from_weights(
            weights,
            &name(&layout.up_weight),
            None,
            target_dtype,
            strategy,
        )?;

        let down = LinearLayer::from_weights(
            weights,
            &name(&layout.down_weight),
            None,
            target_dtype,
            strategy,
        )?;

        // 3. Construct the SwiGluFeedForward brick
        Ok(SwiGluFeedForward::new(gate, up, down))
    }
    pub fn build_decoder_attention(
        weights: &ModelWeights,
        meta: &ModelMetadata,
        layout: &AttentionLayout,
        idx: usize,
        target_dt: Option<DType>,
    ) -> Result<DecoderAttention> {
        let i_str = idx.to_string();
        let strategy = Some(F32MatmulStrategy::CustomSimd);

        // Helper to resolve template strings
        let name = |t: &String| t.replace("{}", &i_str);
        let opt_name = |t: &Option<String>| t.as_ref().map(|s| s.replace("{}", &i_str));

        // 1. Load the 4 Linear Layers
        // LinearLayer::from_weights handles the bias automatically if opt_name returns Some
        let q = LinearLayer::from_weights(
            weights,
            &name(&layout.q_weight),
            opt_name(&layout.q_bias).as_deref(),
            target_dt,
            strategy,
        )?;

        let k = LinearLayer::from_weights(
            weights,
            &name(&layout.k_weight),
            opt_name(&layout.k_bias).as_deref(),
            target_dt,
            strategy,
        )?;

        let v = LinearLayer::from_weights(
            weights,
            &name(&layout.v_weight),
            opt_name(&layout.v_bias).as_deref(),
            target_dt,
            strategy,
        )?;

        let o = LinearLayer::from_weights(
            weights,
            &name(&layout.o_weight),
            opt_name(&layout.o_bias).as_deref(),
            target_dt,
            strategy,
        )?;

        // 2. Construct the DecoderAttention brick
        Ok(DecoderAttention::new(
            meta.hidden_size,
            meta.num_attention_heads,
            q,
            k,
            v,
            o,
            Some(meta.num_kv_heads),
        ))
    }
}
