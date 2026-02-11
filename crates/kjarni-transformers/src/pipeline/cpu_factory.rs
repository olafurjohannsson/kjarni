use crate::activations::Activation;
use crate::cpu::feedforward::SwiGluFeedForward;
use crate::cpu::normalization::{LayerNorm, Normalization, RMSNorm};
use crate::decoder::prelude::DecoderAttention;
use crate::linear_layer::{F32MatmulStrategy, LinearLayer};
use crate::models::base::ModelLoadConfig;
use crate::tensor::DType;
use crate::traits::{AttentionLayout, FeedForwardLayout, ModelMetadata};
use crate::weights::ModelWeights;
use anyhow::{Result, anyhow};

pub struct CpuLayerFactory<'a> {
    weights: &'a ModelWeights,
    target_dtype: Option<DType>,
    f32_strategy: Option<F32MatmulStrategy>,
}

impl<'a> CpuLayerFactory<'a> {
    pub fn new(weights: &'a ModelWeights) -> Self {
        Self {
            weights,
            target_dtype: None,
            f32_strategy: Some(F32MatmulStrategy::CustomSimd),
        }
    }

    pub fn with_load_config(mut self, config: &ModelLoadConfig) -> Self {
        self.target_dtype = config.target_dtype;
        self
    }

    pub fn with_target_dtype(mut self, dtype: Option<DType>) -> Self {
        self.target_dtype = dtype;
        self
    }

    fn resolve(template: &str, layer_idx: usize) -> String {
        template.replace("{}", &layer_idx.to_string())
    }

    pub fn build_linear(
        &self,
        weight_template: &str,
        bias_template: Option<&str>,
        layer_idx: usize,
    ) -> Result<LinearLayer> {
        let weight_name = Self::resolve(weight_template, layer_idx);
        let bias_name = bias_template.map(|t| Self::resolve(t, layer_idx));

        LinearLayer::builder(self.weights, &weight_name)
            .with_optional_bias(bias_name.as_deref())
            .with_target_dtype(self.target_dtype)
            .with_f32_strategy(self.f32_strategy)
            .build()
    }

    pub fn build_norm(
        &self,
        w_template: &String,
        b_template: &Option<String>,
        eps: f32,
        idx: usize,
    ) -> Result<Normalization> {
        let i_str = idx.to_string();
        let w_name = w_template.replace("{}", &i_str);

        let weight = self.weights.get_array1(&w_name)?;

        if let Some(b_tmpl) = b_template {
            let bias = self.weights.get_array1(&b_tmpl.replace("{}", &i_str))?;
            Ok(Normalization::LayerNorm(LayerNorm::new(weight, bias, eps)))
        } else {
            Ok(Normalization::RMSNorm(RMSNorm::new(weight, eps)))
        }
    }
    pub fn build_swiglu_ffn(
        &self,
        layout: &FeedForwardLayout,
        activation: Activation,
        idx: usize,
    ) -> Result<SwiGluFeedForward> {
        let i_str = idx.to_string();
        let gate_template = layout
            .gate_weight
            .as_ref()
            .ok_or_else(|| anyhow!("SwiGLU requires gate_weight"))?;
        let gate = Self::build_linear(&self, gate_template, layout.gate_bias.as_deref(), idx)?;
        let up = Self::build_linear(&self, &layout.up_weight, layout.up_bias.as_deref(), idx)?;
        let down =
            Self::build_linear(&self, &layout.down_weight, layout.down_bias.as_deref(), idx)?;

        Ok(SwiGluFeedForward::new(gate, up, down, activation))
    }
    pub fn build_decoder_attention(
        &self,
        meta: &ModelMetadata,
        layout: &AttentionLayout,
        idx: usize,
    ) -> Result<DecoderAttention> {
        let i_str = idx.to_string();
        let opt_name = |t: &Option<String>| t.as_ref().map(|s| s.replace("{}", &i_str));

        let q = Self::build_linear(
            &self,
            &layout.q_weight,
            opt_name(&layout.q_bias).as_deref(),
            idx,
        )?;

        let k = Self::build_linear(
            &self,
            &layout.k_weight,
            opt_name(&layout.k_bias).as_deref(),
            idx,
        )?;

        let v = Self::build_linear(
            &self,
            &layout.v_weight,
            opt_name(&layout.v_bias).as_deref(),
            idx,
        )?;

        let o = Self::build_linear(
            &self,
            &layout.o_weight,
            opt_name(&layout.o_bias).as_deref(),
            idx,
        )?;

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
    
    use tempfile::TempDir;
    fn create_dummy_weights(tensors: Vec<(&str, Vec<f32>, Vec<usize>)>) -> (TempDir, ModelWeights) {
        let dir = TempDir::new().unwrap();
        let model_path = dir.path().join("model.safetensors");
        let config_path = dir.path().join("config.json");

        let buffers: Vec<(String, Vec<u8>, Vec<usize>)> = tensors
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

        let serialized = serialize(&data_map, &None).unwrap();
        std::fs::write(&model_path, &serialized).unwrap();

        let config_json = r#"{
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4,
        "vocab_size": 100
    }"#;
        std::fs::write(&config_path, config_json).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        (dir, weights)
    }

    fn dummy_metadata() -> ModelMetadata {
        ModelMetadata {
            decoder_layers: None,
            hidden_size: 4,
            intermediate_size: 0,
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
            problem_type: None,
            normalization_strategy: NormalizationStrategy::RMSNorm,
            no_scale_qk: false,
        }
    }

    #[test]
    fn test_build_rmsnorm() {
        let (_file, weights) =
            create_dummy_weights(vec![("layer.0.norm.weight", vec![1.0; 4], vec![4])]);

        let factory = CpuLayerFactory::new(&weights);

        let norm = factory
            .build_norm(
                &"layer.{}.norm.weight".to_string(),
                &None,
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

        let factory = CpuLayerFactory::new(&weights);

        let norm = factory
            .build_norm(
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

    #[test]
    fn test_build_swiglu_ffn() {
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

        let factory = CpuLayerFactory::new(&weights);

        let ffn = factory
            .build_swiglu_ffn(&layout, crate::activations::Activation::SilU, 0)
            .unwrap();

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
        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_swiglu_ffn(&layout, crate::activations::Activation::SilU, 0);
        assert!(result.is_err());
        assert_eq!(
            result.err().unwrap().to_string(),
            "Architecture layout missing required gate_weight for SwiGLU"
        );
    }

    #[test]
    fn test_build_decoder_attention() {
        let (_file, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.q.bias", vec![0.0; 4], vec![4]),
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
        let factory = CpuLayerFactory::new(&weights);
        let attn = factory.build_decoder_attention(&meta, &layout, 0).unwrap();

        assert!(attn.q_proj.has_bias());
        assert!(!attn.k_proj.has_bias());
        assert_eq!(attn.q_proj.shape(), [4, 4]);
    }
    #[test]
    fn test_factory_new() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        let factory = CpuLayerFactory::new(&weights);
        
        assert!(factory.target_dtype.is_none());
        assert!(matches!(factory.f32_strategy, Some(F32MatmulStrategy::CustomSimd)));
    }

    #[test]
    fn test_factory_with_target_dtype() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        let factory = CpuLayerFactory::new(&weights)
            .with_target_dtype(Some(DType::F32));
        
        assert_eq!(factory.target_dtype, Some(DType::F32));
    }

    #[test]
    fn test_factory_with_target_dtype_none() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        let factory = CpuLayerFactory::new(&weights)
            .with_target_dtype(None);
        
        assert!(factory.target_dtype.is_none());
    }

    #[test]
    fn test_factory_with_load_config() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        let mut load_config = ModelLoadConfig::default();
        load_config.target_dtype = Some(DType::F32);
        
        let factory = CpuLayerFactory::new(&weights)
            .with_load_config(&load_config);
        
        assert_eq!(factory.target_dtype, Some(DType::F32));
    }

    #[test]
    fn test_resolve_single_placeholder() {
        let result = CpuLayerFactory::resolve("layer.{}.weight", 5);
        assert_eq!(result, "layer.5.weight");
    }

    #[test]
    fn test_resolve_multiple_placeholders() {
        let result = CpuLayerFactory::resolve("model.layer.{}.attn.{}.weight", 3);
        assert_eq!(result, "model.layer.3.attn.3.weight");
    }

    #[test]
    fn test_resolve_no_placeholder() {
        let result = CpuLayerFactory::resolve("model.weight", 0);
        assert_eq!(result, "model.weight");
    }

    #[test]
    fn test_resolve_zero_index() {
        let result = CpuLayerFactory::resolve("layer.{}.norm", 0);
        assert_eq!(result, "layer.0.norm");
    }

    #[test]
    fn test_resolve_large_index() {
        let result = CpuLayerFactory::resolve("layer.{}.weight", 999);
        assert_eq!(result, "layer.999.weight");
    }

    #[test]
    fn test_build_linear_weight_only() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.weight", vec![0.1; 16], vec![4, 4]),
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        let linear = factory.build_linear("layer.{}.weight", None, 0).unwrap();
        
        assert_eq!(linear.shape(), [4, 4]);
        assert!(!linear.has_bias());
    }

    #[test]
    fn test_build_linear_with_bias() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.weight", vec![0.1; 16], vec![4, 4]),
            ("layer.0.bias", vec![0.0; 4], vec![4]),
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        let linear = factory.build_linear(
            "layer.{}.weight",
            Some("layer.{}.bias"),
            0
        ).unwrap();
        
        assert_eq!(linear.shape(), [4, 4]);
        assert!(linear.has_bias());
    }

    #[test]
    fn test_build_linear_different_layer_indices() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.weight", vec![0.1; 16], vec![4, 4]),
            ("layer.1.weight", vec![0.2; 16], vec![4, 4]),
            ("layer.2.weight", vec![0.3; 16], vec![4, 4]),
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        
        let linear0 = factory.build_linear("layer.{}.weight", None, 0).unwrap();
        let linear1 = factory.build_linear("layer.{}.weight", None, 1).unwrap();
        let linear2 = factory.build_linear("layer.{}.weight", None, 2).unwrap();
        
        assert_eq!(linear0.shape(), [4, 4]);
        assert_eq!(linear1.shape(), [4, 4]);
        assert_eq!(linear2.shape(), [4, 4]);
    }

    #[test]
    fn test_build_linear_missing_weight() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        
        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_linear("nonexistent.{}.weight", None, 0);
        
        assert!(result.is_err());
    }

    #[test]
    fn test_build_linear_non_square() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("proj.0.weight", vec![0.1; 32], vec![8, 4]), // [out, in]
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        let linear = factory.build_linear("proj.{}.weight", None, 0).unwrap();
        
        assert_eq!(linear.shape(), [8, 4]);
    }

    #[test]
    fn test_build_norm_different_eps() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("norm.0.weight", vec![1.0; 8], vec![8]),
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        
        let norm1 = factory.build_norm(
            &"norm.{}.weight".to_string(),
            &None,
            1e-5,
            0
        ).unwrap();
        
        let norm2 = factory.build_norm(
            &"norm.{}.weight".to_string(),
            &None,
            1e-6,
            0
        ).unwrap();
        
        match (norm1, norm2) {
            (Normalization::RMSNorm(rms1), Normalization::RMSNorm(rms2)) => {
                assert_eq!(rms1.eps, 1e-5);
                assert_eq!(rms2.eps, 1e-6);
            }
            _ => panic!("Expected RMSNorm"),
        }
    }

    #[test]
    fn test_build_norm_different_layers() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.norm.weight", vec![1.0; 4], vec![4]),
            ("layer.1.norm.weight", vec![2.0; 4], vec![4]),
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        
        let norm0 = factory.build_norm(
            &"layer.{}.norm.weight".to_string(),
            &None,
            1e-5,
            0
        ).unwrap();
        
        let norm1 = factory.build_norm(
            &"layer.{}.norm.weight".to_string(),
            &None,
            1e-5,
            1
        ).unwrap();
        
        match (norm0, norm1) {
            (Normalization::RMSNorm(rms0), Normalization::RMSNorm(rms1)) => {
                assert_eq!(rms0.weight[0], 1.0);
                assert_eq!(rms1.weight[0], 2.0);
            }
            _ => panic!("Expected RMSNorm"),
        }
    }

    #[test]
    fn test_build_norm_missing_weight() {
        let (_dir, weights) = create_dummy_weights(vec![]);
        
        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_norm(
            &"nonexistent.{}.weight".to_string(),
            &None,
            1e-5,
            0
        );
        
        assert!(result.is_err());
    }

    #[test]
    fn test_build_layernorm_missing_bias() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.ln.weight", vec![1.0; 4], vec![4]),
            // Missing bias!
        ]);
        
        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_norm(
            &"layer.{}.ln.weight".to_string(),
            &Some("layer.{}.ln.bias".to_string()),
            1e-5,
            0
        );
        
        assert!(result.is_err());
    }
    #[test]
    fn test_build_decoder_attention_all_biases() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.q.bias", vec![0.0; 4], vec![4]),
            ("l.0.k.bias", vec![0.0; 4], vec![4]),
            ("l.0.v.bias", vec![0.0; 4], vec![4]),
            ("l.0.o.bias", vec![0.0; 4], vec![4]),
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: Some("l.{}.q.bias".to_string()),
            k_bias: Some("l.{}.k.bias".to_string()),
            v_bias: Some("l.{}.v.bias".to_string()),
            o_bias: Some("l.{}.o.bias".to_string()),
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();
        let factory = CpuLayerFactory::new(&weights);
        let attn = factory.build_decoder_attention(&meta, &layout, 0).unwrap();

        assert!(attn.q_proj.has_bias());
        assert!(attn.k_proj.has_bias());
        assert!(attn.v_proj.has_bias());
        assert!(attn.o_proj.has_bias());
    }

    #[test]
    fn test_build_decoder_attention_no_biases() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();
        let factory = CpuLayerFactory::new(&weights);
        let attn = factory.build_decoder_attention(&meta, &layout, 0).unwrap();

        assert!(!attn.q_proj.has_bias());
        assert!(!attn.k_proj.has_bias());
        assert!(!attn.v_proj.has_bias());
        assert!(!attn.o_proj.has_bias());
    }

    #[test]
    fn test_build_decoder_attention_missing_weight() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            // Missing v.weight and o.weight!
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();
        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_decoder_attention(&meta, &layout, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_build_decoder_attention_different_layers() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
            ("l.1.q.weight", vec![0.2; 16], vec![4, 4]),
            ("l.1.k.weight", vec![0.2; 16], vec![4, 4]),
            ("l.1.v.weight", vec![0.2; 16], vec![4, 4]),
            ("l.1.o.weight", vec![0.2; 16], vec![4, 4]),
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();
        let factory = CpuLayerFactory::new(&weights);
        
        let attn0 = factory.build_decoder_attention(&meta, &layout, 0).unwrap();
        let attn1 = factory.build_decoder_attention(&meta, &layout, 1).unwrap();

        assert_eq!(attn0.q_proj.shape(), [4, 4]);
        assert_eq!(attn1.q_proj.shape(), [4, 4]);
    }

    #[test]
    fn test_build_swiglu_ffn_with_biases() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.gate.bias", vec![0.0; 8], vec![8]),
            ("layer.0.up.weight", vec![0.2; 32], vec![8, 4]),
            ("layer.0.up.bias", vec![0.0; 8], vec![8]),
            ("layer.0.down.weight", vec![0.3; 32], vec![4, 8]),
            ("layer.0.down.bias", vec![0.0; 4], vec![4]),
        ]);

        let layout = FeedForwardLayout {
            gate_weight: Some("layer.{}.gate.weight".to_string()),
            gate_bias: Some("layer.{}.gate.bias".to_string()),
            up_weight: "layer.{}.up.weight".to_string(),
            up_bias: Some("layer.{}.up.bias".to_string()),
            down_weight: "layer.{}.down.weight".to_string(),
            down_bias: Some("layer.{}.down.bias".to_string()),
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let factory = CpuLayerFactory::new(&weights);
        let ffn = factory.build_swiglu_ffn(&layout, Activation::SilU, 0).unwrap();

        assert!(ffn.gate.has_bias());
        assert!(ffn.up.has_bias());
        assert!(ffn.down.has_bias());
    }

    #[test]
    fn test_build_swiglu_ffn_different_activations() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.up.weight", vec![0.2; 32], vec![8, 4]),
            ("layer.0.down.weight", vec![0.3; 32], vec![4, 8]),
        ]);

        let layout = FeedForwardLayout {
            gate_weight: Some("layer.{}.gate.weight".to_string()),
            gate_bias: None,
            up_weight: "layer.{}.up.weight".to_string(),
            up_bias: None,
            down_weight: "layer.{}.down.weight".to_string(),
            down_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let factory = CpuLayerFactory::new(&weights);
        
        let ffn_silu = factory.build_swiglu_ffn(&layout, Activation::SilU, 0).unwrap();
        assert_eq!(ffn_silu.activation, Activation::SilU);
        
        let ffn_gelu = factory.build_swiglu_ffn(&layout, Activation::Gelu, 0).unwrap();
        assert_eq!(ffn_gelu.activation, Activation::Gelu);
    }

    #[test]
    fn test_build_swiglu_ffn_different_layers() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.up.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.down.weight", vec![0.1; 32], vec![4, 8]),
            ("layer.1.gate.weight", vec![0.2; 32], vec![8, 4]),
            ("layer.1.up.weight", vec![0.2; 32], vec![8, 4]),
            ("layer.1.down.weight", vec![0.2; 32], vec![4, 8]),
        ]);

        let layout = FeedForwardLayout {
            gate_weight: Some("layer.{}.gate.weight".to_string()),
            gate_bias: None,
            up_weight: "layer.{}.up.weight".to_string(),
            up_bias: None,
            down_weight: "layer.{}.down.weight".to_string(),
            down_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let factory = CpuLayerFactory::new(&weights);
        
        let ffn0 = factory.build_swiglu_ffn(&layout, Activation::SilU, 0).unwrap();
        let ffn1 = factory.build_swiglu_ffn(&layout, Activation::SilU, 1).unwrap();

        assert_eq!(ffn0.gate.shape(), [8, 4]);
        assert_eq!(ffn1.gate.shape(), [8, 4]);
    }

    #[test]
    fn test_build_swiglu_ffn_missing_up_weight() {
        let (_dir, weights) = create_dummy_weights(vec![
            ("layer.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("layer.0.down.weight", vec![0.3; 32], vec![4, 8]),
        ]);

        let layout = FeedForwardLayout {
            gate_weight: Some("layer.{}.gate.weight".to_string()),
            gate_bias: None,
            up_weight: "layer.{}.up.weight".to_string(),
            up_bias: None,
            down_weight: "layer.{}.down.weight".to_string(),
            down_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let factory = CpuLayerFactory::new(&weights);
        let result = factory.build_swiglu_ffn(&layout, Activation::SilU, 0);

        assert!(result.is_err());
    }

    #[test]
    fn test_build_attention_gqa_config() {
        // GQA: num_kv_heads < num_attention_heads
        let (_dir, weights) = create_dummy_weights(vec![
            ("l.0.q.weight", vec![0.1; 64], vec![16, 4]), // 4 heads * head_dim=4
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),  // 1 kv head * head_dim=4
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),  // 1 kv head * head_dim=4
            ("l.0.o.weight", vec![0.1; 64], vec![4, 16]),
        ]);

        let layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "".to_string(),
            norm_bias: None,
        };

        let mut meta = dummy_metadata();
        meta.hidden_size = 4;
        meta.num_attention_heads = 4;
        meta.num_kv_heads = 1; // GQA
        meta.head_dim = 4;

        let factory = CpuLayerFactory::new(&weights);
        let attn = factory.build_decoder_attention(&meta, &layout, 0).unwrap();

        assert_eq!(attn.q_proj.shape(), [16, 4]);
        assert_eq!(attn.k_proj.shape(), [4, 4]);
    }

    #[test]
    fn test_build_full_layer_components() {
        let (_dir, weights) = create_dummy_weights(vec![
            // Attention
            ("l.0.q.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.k.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.v.weight", vec![0.1; 16], vec![4, 4]),
            ("l.0.o.weight", vec![0.1; 16], vec![4, 4]),
            // FFN
            ("l.0.gate.weight", vec![0.1; 32], vec![8, 4]),
            ("l.0.up.weight", vec![0.1; 32], vec![8, 4]),
            ("l.0.down.weight", vec![0.1; 32], vec![4, 8]),
            // Norms
            ("l.0.attn_norm.weight", vec![1.0; 4], vec![4]),
            ("l.0.ffn_norm.weight", vec![1.0; 4], vec![4]),
        ]);

        let attn_layout = AttentionLayout {
            q_weight: "l.{}.q.weight".to_string(),
            k_weight: "l.{}.k.weight".to_string(),
            v_weight: "l.{}.v.weight".to_string(),
            o_weight: "l.{}.o.weight".to_string(),
            q_bias: None,
            k_bias: None,
            v_bias: None,
            o_bias: None,
            norm_weight: "l.{}.attn_norm.weight".to_string(),
            norm_bias: None,
        };

        let ffn_layout = FeedForwardLayout {
            gate_weight: Some("l.{}.gate.weight".to_string()),
            gate_bias: None,
            up_weight: "l.{}.up.weight".to_string(),
            up_bias: None,
            down_weight: "l.{}.down.weight".to_string(),
            down_bias: None,
            norm_weight: "l.{}.ffn_norm.weight".to_string(),
            norm_bias: None,
        };

        let meta = dummy_metadata();
        let factory = CpuLayerFactory::new(&weights);

        let attn = factory.build_decoder_attention(&meta, &attn_layout, 0).unwrap();
        let ffn = factory.build_swiglu_ffn(&ffn_layout, Activation::SilU, 0).unwrap();
        let attn_norm = factory.build_norm(
            &attn_layout.norm_weight,
            &attn_layout.norm_bias,
            meta.norm_eps,
            0
        ).unwrap();
        let ffn_norm = factory.build_norm(
            &ffn_layout.norm_weight,
            &ffn_layout.norm_bias,
            meta.norm_eps,
            0
        ).unwrap();

        assert_eq!(attn.q_proj.shape(), [4, 4]);
        assert_eq!(ffn.gate.shape(), [8, 4]);
        assert!(matches!(attn_norm, Normalization::RMSNorm(_)));
        assert!(matches!(ffn_norm, Normalization::RMSNorm(_)));
    }
}
