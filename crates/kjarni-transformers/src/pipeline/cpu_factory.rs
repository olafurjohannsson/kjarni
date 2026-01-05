use crate::decoder::prelude::DecoderAttention;
use crate::feedforward::SwiGluFeedForward;
use crate::linear_layer::{F32MatmulStrategy, LinearLayer};
use crate::normalization::{LayerNorm, Normalization, RMSNorm};
use crate::tensor::DType;
use crate::traits::{AttentionLayout, FeedForwardLayout, ModelMetadata};
use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};

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
        let gate = LinearLayer::builder(weights, &name(gate_name))
            .with_target_dtype(target_dtype)
            .with_f32_strategy(strategy)
            .with_optional_bias(None)
            .build()?;

        let up = LinearLayer::builder(weights, &name(&layout.up_weight))
            .with_target_dtype(target_dtype)
            .with_f32_strategy(strategy)
            .with_optional_bias(None)
            .build()?;

        let down = LinearLayer::builder(weights, &name(&layout.down_weight))
            .with_target_dtype(target_dtype)
            .with_f32_strategy(strategy)
            .with_optional_bias(None)
            .build()?;

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
        let q = LinearLayer::builder(weights, &name(&layout.q_weight))
            .with_optional_bias(opt_name(&layout.q_bias).as_deref())
            .with_target_dtype(target_dt)
            .with_f32_strategy(strategy)
            .build()?;

        let k = LinearLayer::builder(weights, &name(&layout.k_weight))
            .with_optional_bias(opt_name(&layout.k_bias).as_deref())
            .with_target_dtype(target_dt)
            .with_f32_strategy(strategy)
            .build()?;

        let v = LinearLayer::builder(weights, &name(&layout.v_weight))
            .with_optional_bias(opt_name(&layout.v_bias).as_deref())
            .with_target_dtype(target_dt)
            .with_f32_strategy(strategy)
            .build()?;

        let o = LinearLayer::builder(weights, &name(&layout.o_weight))
            .with_optional_bias(opt_name(&layout.o_bias).as_deref())
            .with_target_dtype(target_dt)
            .with_f32_strategy(strategy)
            .build()?;

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::traits::{ModelMetadata, NormalizationStrategy};
    use safetensors::serialize;
    use safetensors::tensor::{Dtype, TensorView};
    use std::io::Write;
    use tempfile::NamedTempFile;
    use tempfile::TempDir; // Use TempDir instead of NamedTempFile

    fn create_dummy_weights(tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) -> (TempDir, ModelWeights) {
        // 1. Create a temporary directory
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.safetensors");
        let config_path = dir.path().join("config.json");

        // 2. Create byte buffers and data map
        let mut buffers: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
            .into_iter()
            .map(|(name, data, shape)| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                (name.to_string(), bytes, shape)
            })
            .collect();

        let mut data_map = std::collections::HashMap::new();
        for (name, bytes, shape) in &buffers {
            let view = TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap();
            data_map.insert(name.clone(), view);
        }

        // 3. Write model.safetensors
        let serialized = serialize(&data_map, &None).unwrap();
        std::fs::write(&model_path, &serialized).unwrap();

        // 4. Write a dummy config.json (Required by ModelWeights::new for directory)
        // We put minimal valid JSON here.
        let config_json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4,
        "vocab_size": 100
    }"#;
        std::fs::write(&config_path, config_json).unwrap();

        // 5. Load from the directory
        let weights = ModelWeights::new(dir.path()).unwrap();
        (dir, weights)
    }

    fn dummy_metadata() -> ModelMetadata {
        ModelMetadata {
            hidden_size: 4,
            num_layers: 1,
            num_attention_heads: 2,
            num_kv_heads: 1,
            head_dim: 2,
            vocab_size: 10,
            max_seq_len: 128,
            norm_eps: 1e-5,
            activation: Activation::Gelu,
            rope_theta: None,
            rope_scaling: None,
            scale_embeddings: false,
            normalize_embedding: false,
            extra_pos_embeddings: 0,
            is_prenorm: true,
            transpose_ffn_weights: false,
            transpose_attention_weights: false,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
        }
    }

    // =========================================================================
    //  Normalization Tests
    // =========================================================================

    #[test]
    fn test_build_rmsnorm() {
        let (_file, weights) =
            create_dummy_weights(vec![("layer.0.norm.weight", vec![1.0; 4], vec![4])]);

        let norm = CpuLayerFactory::build_norm(
            &weights,
            &"layer.{}.norm.weight".to_string(),
            &None, // No bias -> RMSNorm
            1e-5,
            0,
        )
        .unwrap();

        match norm {
            Normalization::RMSNorm(rms) => {
                assert_eq!(rms.eps, 1e-5);
                assert_eq!(rms.weight.len(), 4);
            }
            _ => panic!("Expected RMSNorm"),
        }
    }

    #[test]
    fn test_build_layernorm() {
        let (_file, weights) = create_dummy_weights(vec![
            ("layer.0.ln.weight", vec![1.0; 4], vec![4]),
            ("layer.0.ln.bias", vec![0.0; 4], vec![4]),
        ]);

        let norm = CpuLayerFactory::build_norm(
            &weights,
            &"layer.{}.ln.weight".to_string(),
            &Some("layer.{}.ln.bias".to_string()), // Bias -> LayerNorm
            1e-5,
            0,
        )
        .unwrap();

        match norm {
            Normalization::LayerNorm(ln) => {
                assert_eq!(ln.eps, 1e-5);
                assert_eq!(ln.weight.len(), 4);
                assert_eq!(ln.bias.len(), 4);
            }
            _ => panic!("Expected LayerNorm"),
        }
    }

    // =========================================================================
    //  FFN Tests
    // =========================================================================

    #[test]
    fn test_build_swiglu_ffn() {
        // Hidden=4, Intermediate=8
        // Weights are [Out, In]
        let (_file, weights) = create_dummy_weights(vec![
            ("layer.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.up.weight", vec![0.2; 32], vec![8, 4]),
            ("layer.0.down.weight", vec![0.3; 32], vec![4, 8]),
        ]);

        let layout = FeedForwardLayout {
            gate_weight: Some("layer.{}.gate.weight".to_string()),
            gate_bias: None,
            up_weight: "layer.{}.up.weight".to_string(),
            down_weight: "layer.{}.down.weight".to_string(),
            up_bias: None,
            down_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let ffn = CpuLayerFactory::build_swiglu_ffn(&weights, &layout, 0, None).unwrap();

        assert_eq!(ffn.gate.shape(), [8, 4]);
        assert_eq!(ffn.up.shape(), [8, 4]);
        assert_eq!(ffn.down.shape(), [4, 8]);
    }

    #[test]
    fn test_build_swiglu_ffn_missing_gate() {
        let (_file, weights) = create_dummy_weights(vec![]);

        let layout = FeedForwardLayout {
            gate_weight: None, // Missing gate!
            gate_bias: None,
            up_weight: "up".to_string(),
            down_weight: "down".to_string(),
            up_bias: None,
            down_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let result = CpuLayerFactory::build_swiglu_ffn(&weights, &layout, 0, None);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "Architecture layout missing required gate_weight for SwiGLU"
        );
    }

    // =========================================================================
    //  Attention Tests
    // =========================================================================

    #[test]
    fn test_build_decoder_attention() {
        // Hidden=4, Heads=2, HeadDim=2
        // Q, K, V, O weights
        let (_file, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.q.bias", vec![0.0; 4], vec![4]), // Optional bias
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: Some("l.{}.q.bias".to_string()),
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();

        let attn =
            CpuLayerFactory::build_decoder_attention(&weights, &meta, &layout, 0, None).unwrap();

        assert!(attn.q_proj.has_bias());
        assert!(!attn.k_proj.has_bias());
        assert_eq!(attn.q_proj.shape(), [4, 4]);
    }
}
