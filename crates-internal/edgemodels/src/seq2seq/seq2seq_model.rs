use crate::seq2seq::bart_configs::{BartConfig, BartLikeConfig};
use anyhow::{Result, anyhow};
use async_trait::async_trait;
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{
    ModelArchitecture, ModelType,
    base::{BeamSearchParams, DecodingStrategy, GenerationConfig},
};
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    CrossAttentionDecoder, DecoderOutput, Encoder, EncoderDecoderArchitecture, EncoderOutput,
    LanguageModelConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, Array2, Array3};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;

pub enum AnySeq2SeqModel {
    Bart(Seq2SeqModel<BartConfig>),
    // T5(Seq2SeqModel<T5Config>), // add t5
}

pub struct Seq2SeqModel<C: EncoderDecoderArchitecture + Send + Sync> {
    model: TransformerEncoderDecoder,
    tokenizer: Tokenizer,
    config: Arc<C>,
    lm_head: Array2<f32>, // [vocab_size, hidden_size]
    final_logits_bias: Option<Array1<f32>>,
}
impl<C: EncoderDecoderArchitecture + Send + Sync> Seq2SeqModel<C> {
    pub fn concrete_config(&self) -> &Arc<C> {
        &self.config
    }
}
impl AnySeq2SeqModel {
    /// Supported seq2seq model types
    const SUPPORTED_MODELS: &'static [ModelType] = &[
        ModelType::BartLargeCnn,
        ModelType::DistilBartCnn,
        ModelType::T5Small,
        ModelType::MarianEnIs,
    ];
    /// Create seq2seq model from HuggingFace model registry.
    pub async fn from_registry(
        model_type: ModelType,
        cache_dir: Option<PathBuf>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        if !Self::SUPPORTED_MODELS.contains(&model_type) {
            return Err(anyhow!("Seq2seq: Unsupported model type: {:?}", model_type));
        }
        if model_type.info().architecture != ModelArchitecture::EncoderDecoder {
            return Err(anyhow!(
                "Model {:?} is not an encoder-decoder model.",
                model_type
            ));
        }

        let cache_dir = cache_dir.unwrap_or_else(|| {
            dirs::cache_dir()
                .expect("No cache directory found")
                .join("edgetransformers")
        });
        let model_dir = cache_dir.join(model_type.repo_id().replace('/', "_"));

        download_model_files(&model_dir, &model_type.info().paths).await?;

        match model_type {
            ModelType::BartLargeCnn | ModelType::DistilBartCnn => {
                let model =
                    Seq2SeqModel::<BartConfig>::from_pretrained(&model_dir, device, context)?;

                Ok(AnySeq2SeqModel::Bart(model))
            }
            ModelType::T5Small => {
                unimplemented!("T5 model support is not yet implemented.")
            }
            ModelType::MarianEnIs => {
                unimplemented!("Marian model support is not yet implemented.")
            }
            _ => Err(anyhow!(
                "Unsupported or unknown seq2seq model type: {:?}",
                model_type
            )),
        }
    }
}
impl<C> Seq2SeqModel<C>
where
    C: EncoderDecoderArchitecture + Send + Sync + for<'de> serde::Deserialize<'de>,
{
    pub fn from_pretrained(
        model_path: &Path,
        // model_type: ModelType,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<Self> {
        let weights = ModelWeights::new(model_path)?;
        let tokenizer =
            Tokenizer::from_file(model_path.join("tokenizer.json")).map_err(|e| anyhow!(e))?;

        // let config: Arc<BartConfig> = Arc::new(serde_json::from_str(&weights.config_json)?);
        let config: Arc<C> = Arc::new(serde_json::from_str(&weights.config_json)?);
        let model = TransformerEncoderDecoder::new(&weights, config.clone(), device, context)?;

        // Load the LM head WITHOUT transposing. Shape is [vocab_size, hidden_size].
        let lm_head = weights.get_array2(config.get_lm_head_name())?;

        let final_logits_bias = config
            .get_final_logits_bias_name()
            .and_then(|name| weights.get_array1(name).ok());

        Ok(Self {
            model,
            tokenizer,
            config,
            lm_head,
            final_logits_bias,
        })
    }
    // pub fn config(&self) -> &Arc<BartConfig> {
    //     &self.config
    // }
}

// --- TRAIT IMPLEMENTATIONS ---

impl<C: EncoderDecoderArchitecture + Send + Sync> TransformerModel for Seq2SeqModel<C> {
    fn device(&self) -> Device {
        self.model.device()
    }
    fn context(&self) -> Option<Arc<WgpuContext>> {
        self.model.context()
    }
}

impl<C: EncoderDecoderArchitecture + Send + Sync> LanguageModel for Seq2SeqModel<C> {
    fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    fn config(&self) -> &dyn LanguageModelConfig {
        self.config.as_ref()
    }

    fn new_cache(&self, batch_size: usize, max_len: usize) -> Result<Box<dyn Cache>> {
        println!("Inside new cache!");
        // Use the device() method from the TransformerModel trait to decide which cache to create.
        match self.model.device() {
            // Assuming `self.model` holds the actual EncoderDecoder
            Device::Cpu => {
                println!("Cpu");
                let head_dim = self.config.hidden_size() / self.config.num_attention_heads();
                Ok(Box::new(CpuKVCache::new(
                    self.config.num_decoder_layers(),
                    batch_size,
                    max_len,
                    self.config.hidden_size(),
                )))
            }
            Device::Wgpu => {
                println!("Gpu");
                // Get the context from the model.
                let context = self
                    .model
                    .context()
                    .ok_or_else(|| anyhow!("GPU context not available for GPU model"))?;
                println!("context");
                let head_dim = self.config.hidden_size() / self.config.num_attention_heads();
                println!("head dim {}", head_dim);
                Ok(Box::new(GpuKVCache::new(
                    &context,
                    self.config.num_decoder_layers(),
                    batch_size,
                    self.config.num_attention_heads(),
                    head_dim,
                    max_len,
                )?))
            }
        }
    }
}

#[async_trait]
impl<C> EncoderDecoderLanguageModel for Seq2SeqModel<C>
where
    C: EncoderDecoderArchitecture + Send + Sync + for<'de> serde::Deserialize<'de> + BartLikeConfig,
{
    fn encoder(&self) -> &dyn Encoder<Input = Array2<u32>, Output = EncoderOutput> {
        self.model.encoder()
    }
    fn lm_head(&self) -> &Array2<f32> {
        &self.lm_head
    }

    fn final_logits_bias(&self) -> Option<&Array1<f32>> {
        self.final_logits_bias.as_ref()
    }
    fn decoder(&self) -> &dyn CrossAttentionDecoder<Input = Array2<u32>, Output = DecoderOutput> {
        self.model.decoder()
    }
    fn get_default_generation_config(&self) -> GenerationConfig {
        // This logic is now cleanly encapsulated here.
        if let Some(params) = &self.config.task_specific_params() {
            let summary_params = &params.summarization;
            return GenerationConfig {
                max_length: summary_params.max_length,
                min_length: summary_params.min_length,
                no_repeat_ngram_size: summary_params.no_repeat_ngram_size,
                repetition_penalty: 1.0, // For some reason it has to default to this to achieve parity
                max_new_tokens: None,
                add_bos_token: false, // BART handles this via decoder_start_token_id
                strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                    num_beams: summary_params.num_beams,
                    length_penalty: summary_params.length_penalty,
                    early_stopping: summary_params.early_stopping,
                }),
            };
        }

        // Fallback to a generic default if task-specific params are missing
        println!("Warning: No task-specific params found. Using generic beam search config.");
        GenerationConfig {
            max_length: self.config.max_position_embeddings(), // Use model's own limit
            min_length: 0,
            no_repeat_ngram_size: 0,
            repetition_penalty: 1.1,
            max_new_tokens: None,
            add_bos_token: false,
            strategy: DecodingStrategy::BeamSearch(BeamSearchParams {
                num_beams: 1,          // A beam size of 1 is equivalent to greedy search. SAFE default.
                length_penalty: 1.0,   // No penalty.
                early_stopping: false, // Don't stop early unless requested.
            }),
        }
    }

    fn decoder_start_token_id(&self) -> u32 {
        self.config.decoder_start_token_id()
    }

    fn project_to_logits(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch, seq, hidden) = hidden_states.dim();

        // Ensure our matrix dimensions align before the operation.
        // self.lm_head is [vocab_size, hidden_size]
        assert_eq!(
            self.lm_head.shape()[1],
            hidden,
            "LM head and hidden state dimensions do not match"
        );

        // Reshape for the operation. Shape becomes [batch * seq, hidden_size]
        let hidden_2d = hidden_states.to_shape((batch * seq, hidden))?;

        // Perform the multiplication: [vocab, hidden] @ [hidden, batch*seq] -> [vocab, batch*seq]
        // This is the same mathematical operation as the OLD code, just batched.
        let logits_2d_transposed = self.lm_head.dot(&hidden_2d.t());

        // Transpose the result back to [batch*seq, vocab]
        let logits_2d = logits_2d_transposed.t().as_standard_layout().to_owned();

        let final_logits_2d = if let Some(bias) = &self.final_logits_bias {
            logits_2d + bias
        } else {
            logits_2d
        };

        // Reshape the result back to the original 3D shape.
        final_logits_2d
            .into_shape_with_order((batch, seq, self.config.vocab_size()))
            .map_err(|e| anyhow!(e))
    }
}

mod tests {
    use super::*;
    use edgetransformers::TransformerConfig;
    use edgetransformers::models::base::{DecodingStrategy, EncoderDecoderLanguageModel};
    use edgetransformers::prelude::LanguageModel;

    /// Helper function to load the DistilBART model for testing,
    /// reducing code duplication in the tests below.
    async fn load_distilbart_for_test() -> Result<Seq2SeqModel<BartConfig>> {
        let any_model =
            AnySeq2SeqModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None)
                .await?;

        match any_model {
            AnySeq2SeqModel::Bart(m) => Ok(m),
            // Add other arms if you test other model types
        }
    }

    #[tokio::test]
    async fn test_distilbart_default_generation_config() -> Result<()> {
        // 1. Arrange: Load the model using the helper.
        let model = load_distilbart_for_test().await?;

        // 2. Act: Get the default generation config provided by the model.
        let gen_config = model.get_default_generation_config();

        // 3. Assert: Check that the generation parameters match the config.json file.
        assert_eq!(gen_config.max_length, 142);
        assert_eq!(gen_config.min_length, 56);
        assert_eq!(gen_config.no_repeat_ngram_size, 3);

        if let DecodingStrategy::BeamSearch(params) = gen_config.strategy {
            assert_eq!(params.num_beams, 4);
            assert_eq!(params.length_penalty, 2.0);
            assert!(params.early_stopping);
        } else {
            panic!("Expected BeamSearch strategy for BART summarization model");
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_distilbart_architectural_properties() -> Result<()> {
        // 1. Arrange: Load the model.
        let model = load_distilbart_for_test().await?;
        let config: &Arc<BartConfig> = model.concrete_config(); // Get the concrete config for direct checks.

        // 2. Assert: Check architectural values directly from the config struct.
        assert_eq!(config.vocab_size, 50264);
        assert_eq!(config.d_model, 1024);
        assert_eq!(config.encoder_layers, 12);
        assert_eq!(config.decoder_layers, 6);
        assert!(!config.scale_embedding);

        // 3. Assert: Check that the trait implementations correctly expose these values.
        // This is crucial for verifying your abstractions are working.
        assert_eq!(model.vocab_size(), 50264);
        assert_eq!(model.hidden_size(), 1024);

        // Check token IDs exposed via traits
        assert_eq!(model.decoder_start_token_id(), 2);

        assert_eq!(model.eos_token_id(), Some(2));
        assert_eq!(model.bos_token_id(), Some(0));
        assert_eq!(model.pad_token_id(), Some(1));

        // The default `eos_token_id()` method on the LanguageModel trait should
        // find the "</s>" token from the tokenizer, which has ID 2 for BART.
        assert_eq!(config.eos_token_id(), Some(2));
        assert_eq!(config.bos_token_id(), Some(0));
        assert_eq!(config.pad_token_id(), Some(1));
        assert_eq!(config.extra_pos_embeddings(), 2);
        assert_eq!(config.is_encoder_decoder(), Some(true));
        assert_eq!(config.model_type(), Some("bart".to_string()));

        Ok(())
    }

    #[tokio::test]
    async fn test_model_config() -> Result<()> {
        let any_model =
            AnySeq2SeqModel::from_registry(ModelType::DistilBartCnn, None, Device::Cpu, None)
                .await?;
        let model = match any_model {
            AnySeq2SeqModel::Bart(m) => m,
            // When you add T5, you'll have another match arm here.
        };

        // 2. Act: Get the default generation config provided by the model
        let gen_config = model.get_default_generation_config();

        // 3. Assert: Check that the parameters match the config.json file

        // Assert common parameters
        assert_eq!(gen_config.max_length, 142);
        assert_eq!(gen_config.min_length, 56);
        assert_eq!(gen_config.no_repeat_ngram_size, 3);

        // Use a match to safely access and assert strategy-specific parameters
        if let DecodingStrategy::BeamSearch(params) = gen_config.strategy {
            assert_eq!(params.num_beams, 4);
            assert_eq!(params.length_penalty, 2.0);
            assert_eq!(params.early_stopping, true);
        } else {
            panic!("Expected BeamSearch strategy for BART summarization model");
        }

        Ok(())
    }
}
