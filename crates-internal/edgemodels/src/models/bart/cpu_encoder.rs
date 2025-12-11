use crate::models::bart::config::BartConfig;
use anyhow::Result;
use async_trait::async_trait;
use edgetransformers::activations::Activation;
use edgetransformers::embeddings::Embeddings;
use edgetransformers::encoder::encoder_self_attention::EncoderSelfAttention;
use edgetransformers::encoder::traits::CpuEncoder;
use edgetransformers::encoder_layer::EncoderLayer;
use edgetransformers::feedforward::{FeedForward, StdFeedForward};
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::normalization::LayerNorm;
use edgetransformers::prelude::*;
use edgetransformers::traits::TransformerConfig;
use edgetransformers::traits::{Encoder, TransformerModel};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3};
use std::sync::Arc;

pub struct BartCpuEncoder {
    embeddings: Embeddings,
    embed_layer_norm: LayerNorm,
    pub layers: Vec<EncoderLayer>,
    config: Arc<BartConfig>,
}

impl BartCpuEncoder {
    pub fn new(weights: &ModelWeights, config: Arc<BartConfig>) -> Result<Self> {
        // 1. Embeddings
        let embeddings = Embeddings::new(
            weights.get_array2("model.shared.weight")?,
            Some(weights.get_array2("model.encoder.embed_positions.weight")?),
            None,
        );

        // 2. LayerNorm
        let embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.encoder.layernorm_embedding.weight")?,
            weights.get_array1("model.encoder.layernorm_embedding.bias")?,
            config.layer_norm_eps,
        );

        // 3. Layers
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            let prefix = format!("model.encoder.layers.{}", i);

            // 1. Load LinearLayers externally
            let q_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.q_proj.weight", prefix),
                None,
            )?;
            let k_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.k_proj.weight", prefix),
                None,
            )?;
            let v_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.v_proj.weight", prefix),
                None,
            )?;
            let out_proj = LinearLayer::from_weights(
                weights,
                &format!("{}.self_attn.out_proj.weight", prefix),
                None,
            )?;

            // 2. Pass them to the constructor
            let self_attn = EncoderSelfAttention::new(
                config.d_model,
                config.encoder_attention_heads,
                q_proj,
                k_proj,
                v_proj,
                out_proj,
            );

            let self_attn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
                config.layer_norm_eps,
            );

            // B. FeedForward
            let fc1 = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
            let fc2 = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

            // let feedforward = FeedForward::Legacy(LegacyFeedForward::new(
            //     fc1.t().as_standard_layout().to_owned(),
            //     weights.get_array1(&format!("{}.fc1.bias", prefix))?,
            //     fc2.t().as_standard_layout().to_owned(),
            //     weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            //     Activation::Gelu,
            // ));
            // In BartCpuEncoder::new()
            let feedforward = FeedForward::Standard(StdFeedForward::new(
                fc1, // Keep as [Out, In] - no transpose
                weights.get_array1(&format!("{}.fc1.bias", prefix))?,
                fc2, // Keep as [Out, In] - no transpose
                weights.get_array1(&format!("{}.fc2.bias", prefix))?,
                Activation::Gelu,
            ));

            let ffn_layer_norm = LayerNorm::new(
                weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
                weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
                config.layer_norm_eps,
            );

            layers.push(EncoderLayer {
                self_attn,
                self_attn_layer_norm,
                feedforward,
                ffn_layer_norm,
            });
        }

        Ok(Self {
            embeddings,
            embed_layer_norm,
            layers,
            config,
        })
    }
}

impl TransformerModel for BartCpuEncoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn context(&self) -> Option<Arc<edgetransformers::gpu_context::WgpuContext>> {
        None
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl CpuEncoder for BartCpuEncoder {
    // type Input = Array2<u32>;
    // type Output = EncoderOutput;

    /// Compute embeddings only (word + position + token_type)
    fn embed(&self, input_ids: &Array2<u32>, token_type_ids: Option<&Array2<u32>>) -> Array3<f32> {
        self.embeddings.forward(input_ids, token_type_ids, 2, false)
    }

    /// Compute embeddings + initial normalization
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        token_type_ids: Option<&Array2<u32>>,
    ) -> Array3<f32> {
        let hidden = self.embed(input_ids, token_type_ids);
        self.embed_layer_norm.forward_3d(&hidden)
    }

    /// Run layers [start_layer, end_layer) on pre-computed hidden states
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        attention_mask: &Array2<f32>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<Array3<f32>> {
        let mut hidden = hidden_states.clone();

        for layer in self.layers.iter().take(end_layer).skip(start_layer) {
            hidden = layer.forward(hidden, attention_mask, None, false)?;
        }

        Ok(hidden)
    }

    /// Number of encoder layers
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Hidden size (needed for projection heads)
    fn hidden_size(&self) -> usize {
        self.config.hidden_size()
    }

    // async fn forward(
    //     &self,
    //     input_ids: &Self::Input,
    //     attention_mask: &Array2<f32>,
    //     _token_type_ids: Option<&Array2<u32>>,
    // ) -> Result<Self::Output> {
    //     // BART logic: Offset positions by 2
    //     let mut hidden_states = self.embeddings.forward(input_ids, None, 2, false);
    //     hidden_states = self.embed_layer_norm.forward_3d(&hidden_states);

    //     for layer in &self.layers {
    //         hidden_states = layer.forward(hidden_states, attention_mask, None, false)?;
    //     }

    //     Ok(EncoderOutput {
    //         last_hidden_state: hidden_states,
    //     })
    // }

    // async fn get_hidden_states(
    //     &self,
    //     input: &Self::Input,
    //     mask: &Array2<f32>,
    //     _: Option<&Array2<u32>>,
    // ) -> Result<Array3<f32>> {
    //     Ok(self.forward(input, mask, None).await?.last_hidden_state)
    // }
}


#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;
    use ndarray::{s, Array2};
    use std::path::Path;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_bart_encoder_layer0_golden() -> Result<()> {
        // 1. Setup
        let path_str = "/home/olafurj/.cache/edgetransformers/olafuraron_distilbart-cnn-12-6/";
        let path = Path::new(path_str);
        if !path.exists() {
            println!("SKIPPING TEST: Weights not found.");
            return Ok(());
        }
        let weights = ModelWeights::new(path)?;

        let config_json = std::fs::read_to_string(path.join("config.json"))?;
        let config: Arc<BartConfig> = Arc::new(serde_json::from_str(&config_json)?);

        // 2. Create Encoder
        let encoder = BartCpuEncoder::new(&weights, config.clone())?;

        // 3. Prepare Input IDs
        let input_ids_vec = vec![0, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6];
        let input_ids = Array2::from_shape_vec((1, 10), input_ids_vec)?;

        // 4. Run Embeddings + LayerNorm
        // This validates "embed_and_normalize" vs Python "Layer 0 Input"
        let hidden = encoder.embed_and_normalize(&input_ids, None);

        // --- CHECKPOINT A: Input to Layer 0 ---
        // Paste the output from the Python script here!
        let expected_input: Vec<f32> = vec![
            -0.012427182, -0.1763359, 0.028129267, -0.010629091, 0.015348487,
            0.00571412, 0.020377142, -0.07212893, -0.012256589, -0.07150629
        ];

        // If this assertion fails, your `embed_layer_norm` or `embed_and_normalize` logic is wrong.
        let actual_input_slice = hidden.slice(s![0, 0, 0..10]);
        for (i, &expected) in expected_input.iter().enumerate() {
            let actual = actual_input_slice[i];
            assert!((actual - expected).abs() < 1e-5, "Layer 0 Input Mismatch at {}: expected {}, got {}", i, expected, actual);
        }
        println!("✅ Checkpoint A Passed: Layer 0 Input matches.");

        // 5. Run Layer 0
        let mask = Array2::<f32>::ones((1, 10));
        // Note: passing None for attention_mask because we are testing unmasked
        // If your layer logic requires it, pass &mask
        let layer0_out = encoder.layers[0].forward(hidden.clone(), &mask, None, false)?;

        // --- CHECKPOINT B: Output of Layer 0 ---
        // --- CHECKPOINT B: Layer 0 Output ---
        let expected_output: Vec<f32> = vec![
            -0.009041301, -0.054615177, -0.012196737, -0.009672836, 0.02251066,
            0.021573834, 0.0050799875, -0.0085836, 0.03302486, -0.033092678
        ];

        let actual_output_slice = layer0_out.slice(s![0, 0, 0..10]);
        println!("Layer 0 Actual: {:?}", actual_output_slice);

        for (i, &expected) in expected_output.iter().enumerate() {
            let actual = actual_output_slice[i];
            assert!((actual - expected).abs() < 1e-4, "Layer 0 Output Mismatch at {}: expected {}, got {}", i, expected, actual);
        }

        println!("✅ Checkpoint B Passed: Layer 0 Output matches.");
        Ok(())
    }
}