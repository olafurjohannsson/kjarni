//! GPT-2 style decoder-only language model.
//!
//! This module provides the `Gpt2Model`, a model container responsible for loading
//! weights and configuration for models like GPT-2, DistilGPT2, etc.
//!
//! The actual text generation is handled by the generic `Generator` struct,
//! which can operate on any model that implements the `DecoderLanguageModel` trait.

use crate::models::gpt2::config::Gpt2Config;
use crate::models::gpt2::gpu_decoder::Gpt2GpuDecoder;
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::models::base::AutoregressiveLoop;
use edgetransformers::models::base::GpuDecoder;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{DecoderLanguageModel, LanguageModel, ModelArchitecture, ModelType};
use edgetransformers::prelude::*;
use edgetransformers::traits::{Decoder, DecoderArchitecture, DecoderOutput, LanguageModelConfig};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use edgetransformers::models::base::DecoderLoadConfig;
/// A model container for GPT-2 and its variants (e.g., DistilGPT2).
///
/// This struct holds the model's components (decoder, tokenizer, config) but
/// delegates the actual text generation task to the `Generator`.
pub struct Gpt2Model {
    decoder: Option<TransformerDecoder>,
    gpu_decoder: Option<Gpt2GpuDecoder>,
    tokenizer: Tokenizer,
    config: Arc<Gpt2Config>,
    // lm_head: Array2<f32>,
    lm_head: LinearLayer,
    gpu_lm_head_transposed: Option<GpuTensor>,
    device: Device,
    context: Option<Arc<WgpuContext>>,
}

impl Gpt2Model {
    /// A list of the specific model types supported by this implementation.
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::DistilGpt2,
        ModelType::Gpt2,
        ModelType::Gpt2Medium,
        ModelType::Gpt2Large,
        ModelType::Gpt2XL,
    ];
    pub fn concrete_config(&self) -> &Arc<Gpt2Config> {
        &self.config
    }
    /// Creates a `Gpt2Model` from the HuggingFace model registry.
    ///
    /// This will download the model files to a local cache directory if they
    /// are not already present.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<DecoderLoadConfig>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Unsupported GPT-2 model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::Decoder {
            return Err(anyhow!("Model {:?} is not a decoder model.", model_type));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        download_model_files(&model_dir, &model_type.info().paths).await?;
        Self::from_pretrained(&model_dir, model_type, device, context, decoder_config)
    }

    /// Creates a `Gpt2Model` from a local directory containing the model files.
    pub fn from_pretrained(
        model_path: &Path,
        model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
        decoder_config: Option<DecoderLoadConfig>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        // The DecoderArchitecture trait is key here. It provides the generic
        // TransformerDecoder with the specific tensor names for GPT-2.
        let config_arc: Arc<dyn DecoderArchitecture + Send + Sync> = {
            let mut cfg = serde_json::from_str::<Gpt2Config>(&weights.config_json)?;
            if model_type == ModelType::DistilGpt2 {
                // Special handling for DistilGPT2's unique weight naming convention.
                cfg.set_model_type("distilgpt2".to_string());
            };
            Arc::new(cfg)
        };

        // GPT-2 shares weights between embeddings and the final layer.
        // We load them and transpose for the matmul in the projection step.
        // let lm_head = weights
        //     .get_array2(config_arc.get_lm_head_name())?
        //     .t()
        //     .to_owned();
        let lm_head = LinearLayer::from_weights(
            &weights, 
            config_arc.get_lm_head_name(), 
            decoder_config.unwrap_or_default().target_dtype
        )?;

        let config = config_arc
            .as_any()
            .downcast_ref::<Gpt2Config>()
            .cloned()
            .map(Arc::new)
            .ok_or_else(|| anyhow!("Failed to downcast config to Gpt2Config"))?;

        // Split CPU/GPU paths
        match device {
            Device::Cpu => {
                log::info!("Building CPU decoder...");
                let decoder = TransformerDecoder::new(
                    &weights, config_arc, device, None, None, None, // No RoPE for GPT-2
                )?;

                Ok(Self {
                    decoder: Some(decoder),
                    gpu_decoder: None,
                    tokenizer,
                    config,
                    lm_head,
                    gpu_lm_head_transposed: None,
                    device,
                    context: None,
                })
            }
            Device::Wgpu => {
                log::info!("Building GPU decoder...");
                let ctx = context.ok_or_else(|| anyhow!("GPU device requires context"))?;

                ctx.print_memory_usage();

                let gpu_decoder = Gpt2GpuDecoder::new(&ctx, &weights, config.clone(), decoder_config.unwrap_or_default())?;

                // lm_head already transposed
                // let gpu_lm_head_transposed =
                //     Some(GpuTensor::from_ndarray(&ctx, &lm_head)?);
                let gpu_lm_head_transposed =
                    Some(GpuTensor::from_ndarray(&ctx, &lm_head.to_f32_transposed())?);

                log::info!("✓ GPU model loaded successfully");
                ctx.print_memory_usage();

                Ok(Self {
                    decoder: None,
                    gpu_decoder: Some(gpu_decoder),
                    tokenizer,
                    config,
                    lm_head,
                    gpu_lm_head_transposed,
                    device,
                    context: Some(ctx),
                })
            }
        }
    }
}
// --- Trait Implementations ---
// These implementations make `Gpt2Model` compatible with the generic `Generator`.

impl TransformerModel for Gpt2Model {
    fn device(&self) -> Device {
        self.device
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        let ctx = self.context.clone();
        ctx
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl LanguageModel for Gpt2Model {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }
    fn bos_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn pad_token_id(&self) -> Option<u32> {
        Some(50256)
    }
    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        _num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        Ok(match self.device() {
            Device::Cpu => Box::new(CpuKVCache::new(
                self.num_layers(),
                batch_size,
                max_len,
                self.hidden_size(),
            )),
            Device::Wgpu => {
                let context = self
                    .context()
                    .ok_or_else(|| anyhow!("GPU model missing context"))?;
                let head_dim = self.hidden_size() / self.num_heads();
                Box::new(GpuKVCache::new(
                    &context,
                    self.num_layers(),
                    batch_size,
                    self.num_heads(),
                    head_dim,
                    max_len,
                )?)
            }
        })
    }
}

use edgetransformers::utils::linear_algebra::matmul_2d;

#[async_trait]
impl DecoderLanguageModel for Gpt2Model {
    fn decoder(&self) -> &dyn Decoder<Input = Array2<u32>, Output = DecoderOutput> {
        self.decoder
            .as_ref()
            .expect("CPU decoder not initialized - use Device::Cpu")
    }
    fn lm_head(&self) -> &Array2<f32> {
        self.lm_head.as_f32().expect("lm_head not F32")
    }
    fn autoregressive_loop(&self) -> AutoregressiveLoop {
        AutoregressiveLoop::Legacy
    }
    fn gpu_decoder(&self) -> Result<&(dyn GpuDecoder + Send + Sync)> {
        self.gpu_decoder
            .as_ref()
            .map(|d| d as &(dyn GpuDecoder + Send + Sync))
            .ok_or_else(|| anyhow!("Not a GPU model - use Device::Wgpu"))
    }

    fn gpu_lm_head_transposed(&self) -> Result<&GpuTensor> {
        self.gpu_lm_head_transposed
            .as_ref()
            .ok_or_else(|| anyhow!("Not a GPU model"))
    }
    
fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden) = hidden_states.dim();
        let hidden_2d = hidden_states.view().into_shape((batch * seq, hidden))?;

        // FAST PATH: Uses LinearLayer (matmul_2d_transposed or bf16_mixed)
        // This reads memory contiguously!
        let logits_2d = self.lm_head.matmul(&hidden_2d);

        logits_2d
            .into_shape_with_order((batch, seq, self.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generation::{DecodingStrategy, GenerationConfig};
    use crate::generation::decoder::{CpuDecoderBackend, Generator};
    
    use edgetransformers::prelude::{DecoderLanguageModel, LanguageModel};
    use std::sync::Arc;

    /// Helper function to load the DistilGPT2 model for testing.
    async fn load_distilgpt2_for_test() -> Result<Gpt2Model> {
        Gpt2Model::from_registry(ModelType::DistilGpt2, None, Device::Cpu, None, None).await
    }

    #[tokio::test]
    async fn test_distilgpt2_generation_parity() -> Result<()> {
        // This test verifies deterministic (greedy) generation output.
        let model = load_distilgpt2_for_test().await?;
        let generator = Generator::new(Box::new(model))?;

        let prompt = "The field of Artificial Intelligence has seen a lot of progress";
        let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the past few years, but it is still not clear how much improvement will be made.";

        let config = GenerationConfig {
            max_new_tokens: Some(20),
            add_bos_token: false,
            strategy: DecodingStrategy::Greedy,
            repetition_penalty: 1.1,
            ..Default::default()
        };

        let generated_text = generator.generate(prompt, &config).await?;
        assert_eq!(generated_text.trim(), expected_output.trim());

        Ok(())
    }

    #[tokio::test]
    async fn test_distilgpt2_architectural_properties() -> Result<()> {
        // 1. Arrange: Load the model.
        let model = load_distilgpt2_for_test().await?;
        let config = model.concrete_config();

        // 2. Assert: Check architectural values directly from the config struct.
        assert_eq!(config.vocab_size, 50257);
        assert_eq!(config.n_embd, 768); // hidden size
        assert_eq!(config.n_layer, 6);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.n_ctx, 1024); // max_position_embeddings

        // 3. Assert: Check that the trait implementations correctly expose these values.
        assert_eq!(model.vocab_size(), 50257);
        assert_eq!(model.hidden_size(), 768);
        assert_eq!(model.num_layers(), 6);
        assert_eq!(model.num_heads(), 12);
        assert_eq!(model.max_length(), 1024);

        // Check token IDs. GPT-2 uses the same ID for BOS, EOS, and PAD.
        assert_eq!(model.bos_token_id(), Some(50256));
        assert_eq!(model.eos_token_id(), Some(50256));
        assert_eq!(model.pad_token_id(), Some(50256));

        Ok(())
    }

    #[tokio::test]
    async fn test_distilgpt2_generation_parity_2() -> Result<()> {
        // This test verifies that our implementation produces the exact same output
        // as the HuggingFace transformers library for a deterministic (greedy) generation task.

        // 1. Setup: Define the exact same configuration as the Python script and the working main.rs.
        let model_type = ModelType::DistilGpt2;
        let prompt = "The field of Artificial Intelligence has seen a lot of progress";

        // The "golden" output string from the Python reference script and your now-working Rust implementation.
        let expected_output = "The field of Artificial Intelligence has seen a lot of progress in the past few years, but it is still not clear how much improvement will be made.";

        // Create a config that perfectly matches the reference implementations.
        let config = GenerationConfig {
            max_new_tokens: Some(20),
            strategy: DecodingStrategy::Greedy,
            repetition_penalty: 1.1,
            add_bos_token: false, // CRITICAL for parity with default Hugging Face behavior
            ..Default::default()
        };

        // 2. Load model and create the generator.
        //    We run on CPU to match the Python script's environment.
        let gpt2_model = Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

        let generator = Generator::new(Box::new(gpt2_model))?;

        // 3. Execute the generation. We use the non-streaming `generate` for a simple string comparison.
        let generated_text = generator.generate(prompt, &config).await?;

        // 4. Assert that the generated output is bit-for-bit identical to the golden value.
        //    We trim both strings to avoid any potential whitespace differences at the end.
        assert_eq!(generated_text.trim(), expected_output.trim());

        Ok(())
    }

    #[tokio::test]
    async fn test_distilgpt2_generation_parity_cpu_gpu() -> Result<()> {
        // This test verifies that our implementation produces the exact same output
        // as the HuggingFace transformers library for a deterministic (greedy) generation task.

        // 1. Setup: Define the exact same configuration as the Python script and the working main.rs.
        let model_type = ModelType::DistilGpt2;
        let prompt = "The field of Artificial Intelligence has seen a lot of progress";

        // Create a config that perfectly matches the reference implementations.
        let config = GenerationConfig {
            max_new_tokens: Some(5),
            strategy: DecodingStrategy::Greedy,
            repetition_penalty: 1.1,
            add_bos_token: false, // CRITICAL for parity with default Hugging Face behavior
            ..Default::default()
        };

        // 2. Load model and create the generator.
        //    We run on CPU to match the Python script's environment.
        let gpt2_model = Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;

        let generator = Generator::new(Box::new(gpt2_model))?;

        // 3. Execute the generation. We use the non-streaming `generate` for a simple string comparison.
        let generated_text = generator.generate(prompt, &config).await?;

        let ctx = WgpuContext::new().await?;
        let gpt2_model_2 =
            Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(ctx), None).await?;
        let generator_2 = Generator::new(Box::new(gpt2_model_2))?;
        let generated_text_2 = generator_2.generate(prompt, &config).await?;

        // 4. Assert that the generated output is bit-for-bit identical to the golden value.
        //    We trim both strings to avoid any potential whitespace differences at the end.
        assert_eq!(generated_text.trim(), generated_text_2.trim());

        Ok(())
    }
}
