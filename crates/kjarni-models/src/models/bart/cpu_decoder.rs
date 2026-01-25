use crate::models::bart::config::BartConfig;
use anyhow::Result;
use async_trait::async_trait;

use kjarni_transformers::{
    Normalization,
    activations::Activation,
    cache::{Cache, CpuBeamKVCache},
    cpu::encoder_decoder::{CrossDecoderLayer, DecoderCrossAttention},
    embeddings::Embeddings,
    encoder_decoder::{
        DecoderSelfAttention,
        traits::{CpuCrossAttentionKVCache, CpuCrossDecoder, CpuCrossDecoderOutput},
    },
    feedforward::{FeedForward, LegacyFeedForward},
    linear_layer::LinearLayer,
    models::base::ModelLoadConfig,
    normalization::LayerNorm,
    traits::{Device, InferenceModel, ModelConfig, ModelMetadata},
    weights::ModelWeights,
};

use ndarray::{Array2, Array3};
use std::sync::Arc;

pub struct BartCpuDecoder {
    pub embeddings: Embeddings,
    pub layers: Vec<CrossDecoderLayer>,
    pub embed_layer_norm: LayerNorm,
    pub config: Arc<BartConfig>,
    pub meta: ModelMetadata,
}

impl BartCpuDecoder {
    pub fn new(
        weights: &ModelWeights,
        config: Arc<BartConfig>,
        _load_config: ModelLoadConfig,
    ) -> Result<Self> {
        let meta = config.metadata();
        let layout = config.layout();

        // 1. Word Embeddings (Logic preserved: Force dequantize to F32)
        let word_embeddings = weights.get_array2(&layout.token_embedding)?;
        let embed = kjarni_transformers::embeddings::EmbeddingData::F32(Arc::new(word_embeddings));

        // 2. Decoder-Specific Embeddings (Preserving BART-specific naming logic)
        let embeddings = Embeddings::new(
            embed,
            Some(weights.get_array2("model.decoder.embed_positions.weight")?),
            None, // No token_type_embeddings in BART
        );

        // 3. Decoder-Specific Embedding Norm
        let embed_layer_norm = LayerNorm::new(
            weights.get_array1("model.decoder.layernorm_embedding.weight")?,
            weights.get_array1("model.decoder.layernorm_embedding.bias")?,
            meta.norm_eps,
        );

        // 4. Build decoder layers (Using config.decoder_layers as per BART spec)
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            // Logic preserved: continues to use the existing load_layer method
            layers.push(Self::load_layer(weights, &config, i)?);
        }

        Ok(Self {
            embeddings,
            layers,
            embed_layer_norm,
            config,
            meta,
        })
    }

    fn load_layer(
        weights: &ModelWeights,
        config: &BartConfig,
        i: usize,
    ) -> Result<CrossDecoderLayer> {
        let prefix = format!("model.decoder.layers.{}", i);
        let dtype = None; // Add BF16 support here later if needed

        // A. Self Attention
        let self_attn = DecoderSelfAttention::new(
            config.d_model,
            config.decoder_attention_heads,
            LinearLayer::builder(weights, &format!("{}.self_attn.q_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.self_attn.k_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.self_attn.v_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.self_attn.out_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
        );
        let self_attn_norm = Normalization::LayerNorm(LayerNorm::new(
            weights.get_array1(&format!("{}.self_attn_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.self_attn_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        ));

        // B. Cross Attention
        let cross_attn = DecoderCrossAttention::new(
            config.d_model,
            config.decoder_attention_heads,
            LinearLayer::builder(weights, &format!("{}.encoder_attn.q_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.encoder_attn.k_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.encoder_attn.v_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
            LinearLayer::builder(weights, &format!("{}.encoder_attn.out_proj.weight", prefix))
                .with_optional_bias(None)
                .with_target_dtype(dtype)
                .build()?,
        );
        let cross_attn_norm = Normalization::LayerNorm(LayerNorm::new(
            weights.get_array1(&format!("{}.encoder_attn_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.encoder_attn_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        ));

        // C. Feed Forward (FC1 -> FC2)
        // Note: Using raw arrays for StdFeedForward until it supports LinearLayer
        let fc1 = weights.get_array2(&format!("{}.fc1.weight", prefix))?;
        let fc2 = weights.get_array2(&format!("{}.fc2.weight", prefix))?;

        let ffn = FeedForward::Legacy(LegacyFeedForward::new(
            fc1.t().as_standard_layout().to_owned(), // Transpose
            weights.get_array1(&format!("{}.fc1.bias", prefix))?,
            fc2.t().as_standard_layout().to_owned(), // Transpose
            weights.get_array1(&format!("{}.fc2.bias", prefix))?,
            Activation::Gelu, // Hardcoded for BART
        ));
        let ffn_norm = Normalization::LayerNorm(LayerNorm::new(
            weights.get_array1(&format!("{}.final_layer_norm.weight", prefix))?,
            weights.get_array1(&format!("{}.final_layer_norm.bias", prefix))?,
            config.layer_norm_eps,
        ));

        Ok(CrossDecoderLayer {
            self_attn,
            self_attn_layer_norm: self_attn_norm,
            cross_attn,
            cross_attn_layer_norm: cross_attn_norm,
            feedforward: ffn,
            ffn_layer_norm: ffn_norm,
            pre_norm: false,
        })
    }
}

impl InferenceModel for BartCpuDecoder {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[async_trait]
impl CpuCrossDecoder for BartCpuDecoder {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }
    fn layers(&self) -> &Vec<CrossDecoderLayer> {
        &self.layers
    }
    fn hidden_size(&self) -> usize {
        self.config.d_model
    }

    fn precompute_cross_attention_kv(
        &self,
        encoder_hidden_states: &Array3<f32>,
    ) -> Result<CpuCrossAttentionKVCache> {
        let mut cache_vec = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            // Your Cpu EncoderLayer must expose a method to do this projection.
            // This is new logic you need to add to your EncoderLayer.
            let (k, v) = layer.precompute_cross_kv(encoder_hidden_states)?;
            cache_vec.push((k, v));
        }
        Ok(CpuCrossAttentionKVCache(cache_vec))
    }
    fn embed(&self, decoder_input_ids: &Array2<u32>, position_offset: usize) -> Array3<f32> {
        self.embeddings.forward(
            decoder_input_ids,
            None,
            position_offset + 2,
            self.meta.scale_embeddings,
        )
    }
    fn embed_and_normalize(
        &self,
        input_ids: &Array2<u32>,
        position_offset: usize,
    ) -> Result<Array3<f32>> {
        // 1. Raw Embeddings
        let hidden = self.embed(input_ids, position_offset);
        // 2. LayerNorm
        Ok(self.embed_layer_norm.forward_3d(&hidden))
    }
    fn forward_layers(
        &self,
        hidden_states: &Array3<f32>,
        encoder_hidden_states: &Array3<f32>,
        decoder_attention_mask: Option<&Array2<f32>>,
        cache: Option<&mut dyn Cache>,
        cross_kv_cache: Option<&CpuCrossAttentionKVCache>,
        start_layer: usize,
        end_layer: usize,
    ) -> Result<CpuCrossDecoderOutput> {
        let cpu_cache = cache.and_then(|c| c.as_any().downcast_ref::<CpuBeamKVCache>());

        let mut current_hidden_states = hidden_states.clone();
        log::error!("embed sum: {:?}", current_hidden_states.sum());
        let mut new_self_attn_kvs = Vec::with_capacity(end_layer - start_layer);

        if start_layer >= self.layers.len() || end_layer > self.layers.len() {
            return Err(anyhow::anyhow!("Layer indices out of bounds"));
        }

        for i in start_layer..end_layer {
            let layer = &self.layers[i];

            // Get self-attention cache (mutable during generation)
            let self_attn_past_kv = cpu_cache.and_then(|c| c.get(i));
            let self_attn_past_kv_views = self_attn_past_kv
                .as_ref()
                .map(|(k, v)| (k.view(), v.view()));

            // Get pre-computed cross-attention cache (static during generation)
            let cross_kv_for_layer = cross_kv_cache.and_then(|c| c.0.get(i));

            let (new_hidden, (new_k, new_v)) = layer.forward(
                &current_hidden_states,
                encoder_hidden_states,
                decoder_attention_mask,
                None,
                self_attn_past_kv_views,
                cross_kv_for_layer,
                None,
            )?;

            current_hidden_states = new_hidden;
            new_self_attn_kvs.push((new_k, new_v));

            log::debug!("layer {} output sum: {:?}", i, current_hidden_states.sum());
        }

        Ok(CpuCrossDecoderOutput {
            last_hidden_state: current_hidden_states,
            new_self_attn_kv: new_self_attn_kvs,
        })
    }
}

// In cpu_decoder.rs, add tests module

#[cfg(test)]
mod bart_cpu_cross_decoder_tests {
    use super::*;
    use crate::models::bart::cpu_encoder::BartCpuEncoder;
    use anyhow::Result;
    use kjarni_transformers::{
        cpu::encoder::traits::CpuEncoder, embeddings::EmbeddingData, traits::CpuTransformerCore,
    };
    use ndarray::{Array2, s};
    use std::path::Path;

    const DISTILBART_PATH: &str = "/home/olafurj/.cache/kjarni/olafuraron_distilbart-cnn-12-6/";

    mod golden {
        // Encoder output for 10-token input
        pub const ENCODER_HIDDEN: [f32; 10] = [
            -0.000564052,
            0.013685571,
            0.000094226,
            0.001022849,
            -0.000851990,
            0.009666207,
            -0.003442739,
            0.010040940,
            -0.003043500,
            0.001472963,
        ];

        // Decoder step 0 (BOS token only)
        pub const DECODER_HIDDEN_STEP0: [f32; 10] = [
            0.8607765,
            -0.058337964,
            -0.14485303,
            0.41463178,
            0.1469638,
            0.044795237,
            -0.56972146,
            0.97622347,
            0.5656951,
            -0.41660827,
        ];

        pub const LOGITS_STEP0: [f32; 10] = [
            7.259005,
            -0.44914877,
            7.331996,
            -0.83050156,
            6.1281047,
            7.0948877,
            7.206973,
            4.9941998,
            6.0572596,
            5.520153,
        ];

        pub const ARGMAX_TOKEN_STEP0: u32 = 46541;
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32, name: &str) {
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "{} mismatch at {}: expected {}, got {} (diff: {})",
                name,
                i,
                e,
                a,
                (a - e).abs()
            );
        }
    }

    fn setup() -> Result<(
        BartCpuEncoder,
        BartCpuDecoder,
        Arc<BartConfig>,
        Embeddings,
        ModelMetadata,
    )> {
        let path = Path::new(DISTILBART_PATH);
        if !path.exists() {
            anyhow::bail!("Weights not found");
        }

        let weights = ModelWeights::new(path)?;
        let config_json = std::fs::read_to_string(path.join("config.json"))?;
        let config: Arc<BartConfig> = Arc::new(serde_json::from_str(&config_json)?);
        let meta = config.metadata();
        let layout = config.layout();

        // 1. Word Embeddings (Logic preserved: Force dequantize to F32)
        let word_embeddings = weights.get_array2(&layout.token_embedding)?;
        let embed = kjarni_transformers::embeddings::EmbeddingData::F32(Arc::new(word_embeddings));

        let encoder = BartCpuEncoder::new(&weights, config.clone(), ModelLoadConfig::default())?;
        let decoder = BartCpuDecoder::new(&weights, config.clone(), ModelLoadConfig::default())?;
        let embeddings = Embeddings::new(
            embed,
            Some(weights.get_array2("model.encoder.embed_positions.weight")?),
            None, // No token_type_embeddings in BART
        );
        Ok((encoder, decoder, config, embeddings, meta))
    }

    #[tokio::test]
    async fn test_beam_search_step0() -> Result<()> {
        // Setup same as greedy test
        let (encoder, decoder, _config, embeddings, modelMetadata) = setup()?;
        let weights = ModelWeights::new(Path::new(DISTILBART_PATH))?;

        let lm_head = LinearLayer::builder(&weights, "model.shared.weight").build()?;
        let final_logits_bias = weights.get_array2("final_logits_bias")?.row(0).to_owned();

        // Encode
        let input_ids = Array2::from_shape_vec(
            (1, 10),
            vec![0u32, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6],
        )?;
        let mask = Array2::<f32>::ones((1, 10));
        let hidden_states = embeddings.forward(&input_ids, None, 2, modelMetadata.scale_embeddings);

        let normalzied = encoder.embed_norm(&hidden_states)?;

        let encoder_output = encoder.forward(&normalzied, &mask)?;

        // Step 0: BOS token
        let dec_input = Array2::from_shape_vec((1, 1), vec![2u32])?;
        let decoder_output = decoder.forward(
            &dec_input,
            &encoder_output.last_hidden_state,
            None,
            None,
            None,
        )?;

        let hidden = &decoder_output.last_hidden_state;
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((1, hidden.shape()[2]))?;
        let mut logits = lm_head.matmul(&hidden_2d);
        logits += &final_logits_bias;

        // Get top-4 tokens (simulating num_beams=4)
        let mut indexed: Vec<(usize, f32)> = logits
            .row(0)
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("Top 4 tokens at step 0:");
        for (i, (token, score)) in indexed.iter().take(4).enumerate() {
            println!("  Beam {}: token {} score {}", i, token, score);
        }

        // Best token should be 46541 (Rust)
        assert_eq!(indexed[0].0 as u32, 46541, "Top token should be 'Rust'");
        println!("✅ Top token is correct");

        Ok(())
    }
    #[tokio::test]
    async fn test_greedy_generation_step_by_step() -> Result<()> {
        let (encoder, decoder, _config, embeddings, modelMetadata) = setup()?;
        let weights = ModelWeights::new(Path::new(DISTILBART_PATH))?;

        let lm_head = LinearLayer::builder(&weights, "model.shared.weight")
            .with_optional_bias(None)
            .build()?;
        let final_logits_bias = weights.get_array2("final_logits_bias")?.row(0).to_owned();

        // 1. Encode
        let input_ids = Array2::from_shape_vec(
            (1, 10),
            vec![0u32, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6],
        )?;
        let mask = Array2::<f32>::ones((1, 10));
        let hidden_states = embeddings.forward(&input_ids, None, 2, modelMetadata.scale_embeddings);

        let normalzied = encoder.embed_norm(&hidden_states)?;

        let encoder_output = encoder.forward(&normalzied, &mask)?;

        // 2. Greedy decode (no beam search, no cache for simplicity)
        let mut decoder_ids = vec![2u32]; // BOS

        // Expected from Python: [46541, 16, 10, 3228, 12, 5489, 625, 35045, 6, 3228]
        let expected_tokens: [u32; 10] = [46541, 16, 10, 3228, 12, 5489, 625, 35045, 6, 3228];

        for step in 0..10 {
            let dec_input = Array2::from_shape_vec((1, decoder_ids.len()), decoder_ids.clone())?;

            let decoder_output = decoder.forward(
                &dec_input,
                &encoder_output.last_hidden_state,
                None,
                None,
                None,
            )?;

            // Get last position logits
            let hidden = &decoder_output.last_hidden_state;
            let last_hidden = hidden.slice(s![0, -1.., ..]).to_owned();
            let last_hidden_2d = last_hidden.into_shape_with_order((1, hidden.shape()[2]))?;

            let mut logits = lm_head.matmul(&last_hidden_2d.view());
            logits += &final_logits_bias;

            let next_token = logits
                .row(0)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();

            println!(
                "Step {}: token {} (expected {})",
                step, next_token, expected_tokens[step]
            );
            assert_eq!(
                next_token, expected_tokens[step],
                "Token mismatch at step {}",
                step
            );

            decoder_ids.push(next_token);
        }

        println!("✅ All greedy steps match!");
        Ok(())
    }
    #[tokio::test]
    async fn test_generation_step_by_step_vs_python() -> Result<()> {
        let (encoder, decoder, _config, embeddings, modelMetadata) = setup()?;
        let weights = ModelWeights::new(Path::new(DISTILBART_PATH))?;

        let lm_head = LinearLayer::builder(&weights, "model.shared.weight")
            .with_optional_bias(None)
            .build()?;
        let final_logits_bias = weights.get_array2("final_logits_bias")?.row(0).to_owned();

        // Full input (93 tokens as Python used)
        let text = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without using a garbage collector. To simultaneously enforce memory safety and prevent data races, its \"borrow checker\" tracks the object lifetime of all references in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang.";

        // Tokenize
        let tokenizer =
            tokenizers::Tokenizer::from_file(format!("{}/tokenizer.json", DISTILBART_PATH))
                .map_err(|e| anyhow::anyhow!("{}", e))?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
        let input_ids_vec: Vec<u32> = encoding.get_ids().to_vec();
        println!("Input length: {} tokens", input_ids_vec.len());

        // Verify tokenization matches Python
        let expected_input_ids: Vec<u32> = vec![
            0, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6, 937, 12, 25064, 8326, 2777, 14, 27995,
            819, 6, 1907, 1078, 6, 8, 10146, 37079, 4, 85, 1177, 34532, 3783, 1078, 578, 24872, 14,
            70, 13115, 477, 7, 8218, 3783, 578, 19010, 634, 10, 11671, 22779, 4, 598, 11586, 10914,
            3783, 1078, 8, 2097, 414, 4694, 6, 63, 22, 12514, 4610, 1649, 254, 113, 5297, 5, 7626,
            7370, 9, 70, 13115, 11, 10, 586, 148, 32245, 4, 23083, 21, 11359, 30, 11991, 101, 230,
            42964, 6, 38592, 6, 8, 4594, 32373, 4, 2,
        ];
        assert_eq!(input_ids_vec, expected_input_ids, "Tokenization mismatch");

        let input_ids = Array2::from_shape_vec((1, input_ids_vec.len()), input_ids_vec)?;
        let mask = Array2::<f32>::ones(input_ids.dim());
        let hidden_states = embeddings.forward(&input_ids, None, 2, false);
        
        let normalzied = encoder.embed_norm(&hidden_states)?;
        
        let encoder_output = encoder.forward(&normalzied, &mask)?;

        // Golden tokens from Python step-by-step greedy generation (first 30 steps)
        let golden_tokens: Vec<u32> = vec![
            23083, 16, 10, 3228, 12, 5489, 625, 35045, 6, 937, // Steps 0-9
            12, 25064, 8326, 2777, 479, 85, 27995, 819, 6, 1907, // Steps 10-19
            1078, 6, 8, 10146, 37079, 479, 85, 1177, 34532, 3783, // Steps 20-29
        ];

        // Greedy decode (no cache, to match Python step-by-step)
        let mut decoder_ids = vec![2u32]; // BOS
        let mut generated_tokens: Vec<u32> = Vec::new();

        println!("\nStep-by-step greedy generation:");
        for step in 0..30 {
            let dec_input = Array2::from_shape_vec((1, decoder_ids.len()), decoder_ids.clone())?;

            let decoder_output = decoder.forward(
                &dec_input,
                &encoder_output.last_hidden_state,
                None,
                None,
                None,
            )?;

            let hidden = &decoder_output.last_hidden_state;
            let last_hidden = hidden.slice(s![0, -1.., ..]).to_owned();
            let last_hidden_2d = last_hidden.into_shape_with_order((1, hidden.shape()[2]))?;

            let mut logits = lm_head.matmul(&last_hidden_2d.view());
            logits += &final_logits_bias;

            let next_token = logits
                .row(0)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i as u32)
                .unwrap();

            let token_text = tokenizer.decode(&[next_token], true).unwrap_or_default();
            println!(
                "Step {}: token {} = '{}' (expected: {})",
                step, next_token, token_text, golden_tokens[step]
            );

            // Assert token matches golden value
            assert_eq!(
                next_token, golden_tokens[step],
                "Token mismatch at step {}: got {}, expected {}",
                step, next_token, golden_tokens[step]
            );

            generated_tokens.push(next_token);
            decoder_ids.push(next_token);

            if next_token == 2 {
                // EOS
                break;
            }
        }

        println!(
            "\n✅ All {} tokens match Python golden values!",
            generated_tokens.len()
        );

        // Decode final output
        let output_text = tokenizer
            .decode(&generated_tokens, true)
            .unwrap_or_default();
        println!("Generated: {}", output_text);

        Ok(())
    }

    #[tokio::test]
    async fn test_decoder_step0_hidden() -> Result<()> {
        let (encoder, decoder, _config, embeddings, modelMetadata) = setup()?;

        // 1. Encode
        let input_ids = Array2::from_shape_vec(
            (1, 10),
            vec![0u32, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6],
        )?;
        let mask = Array2::<f32>::ones((1, 10));

        let hidden_states = embeddings.forward(&input_ids, None, 2, modelMetadata.scale_embeddings);

        let normalzied = encoder.embed_norm(&hidden_states)?;

        let encoder_output = encoder.forward(&normalzied, &mask)?;

        // Verify encoder output
        let enc_actual: Vec<f32> = encoder_output
            .last_hidden_state
            .slice(s![0, 0, 0..10])
            .to_vec();
        println!("Encoder hidden: {:?}", enc_actual);
        assert_close(&enc_actual, &golden::ENCODER_HIDDEN, 1e-4, "Encoder Hidden");
        println!("✅ Encoder hidden matches");

        // 2. Decode step 0 (BOS token = 2)
        let decoder_input_ids = Array2::from_shape_vec((1, 1), vec![2u32])?;
        let decoder_output = decoder.forward(
            &decoder_input_ids,
            &encoder_output.last_hidden_state,
            None,
            None,
            None,
        )?;

        let dec_actual: Vec<f32> = decoder_output
            .last_hidden_state
            .slice(s![0, 0, 0..10])
            .to_vec();
        println!("Decoder hidden: {:?}", dec_actual);
        assert_close(
            &dec_actual,
            &golden::DECODER_HIDDEN_STEP0,
            1e-4,
            "Decoder Hidden Step0",
        );
        println!("✅ Decoder hidden step0 matches");

        Ok(())
    }

    #[tokio::test]
    async fn test_decoder_step0_logits() -> Result<()> {
        let (encoder, decoder, _config, embeddings, modelMetadata) = setup()?;
        let weights = ModelWeights::new(Path::new(DISTILBART_PATH))?;

        // Load LM head
        let lm_head = LinearLayer::builder(&weights, "model.shared.weight")
            .with_optional_bias(None)
            .build()?;
        // final_logits_bias is [1, vocab_size], squeeze to 1D
        let final_logits_bias = weights.get_array2("final_logits_bias")?.row(0).to_owned();

        // 1. Encode
        let input_ids = Array2::from_shape_vec(
            (1, 10),
            vec![0u32, 46541, 16, 10, 3228, 12, 5489, 625, 35045, 6],
        )?;
        let mask = Array2::<f32>::ones((1, 10));
        let hidden_states = embeddings.forward(&input_ids, None, 2, modelMetadata.scale_embeddings);
        let normalized = encoder.embed_norm(&hidden_states)?;
        let encoder_output = encoder.forward(&normalized, &mask)?;

        // 2. Decode step 0
        let decoder_input_ids = Array2::from_shape_vec((1, 1), vec![2u32])?;
        let decoder_output = decoder.forward(
            &decoder_input_ids,
            &encoder_output.last_hidden_state,
            None,
            None,
            None,
        )?;

        // 3. LM head + bias
        let hidden = &decoder_output.last_hidden_state;
        let (batch, seq, hidden_dim) = hidden.dim();
        let hidden_2d = hidden
            .view()
            .into_shape_with_order((batch * seq, hidden_dim))?;
        let mut logits = lm_head.matmul(&hidden_2d);
        logits += &final_logits_bias;

        let logits_actual: Vec<f32> = logits.slice(s![0, 0..10]).to_vec();
        println!("Logits: {:?}", logits_actual);
        assert_close(&logits_actual, &golden::LOGITS_STEP0, 1e-3, "Logits Step0");
        println!("✅ Logits step0 matches");

        // Check argmax
        let argmax = logits
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        println!("Argmax token: {}", argmax);
        assert_eq!(argmax, golden::ARGMAX_TOKEN_STEP0, "Argmax mismatch");
        println!("✅ Argmax matches");

        Ok(())
    }
}
