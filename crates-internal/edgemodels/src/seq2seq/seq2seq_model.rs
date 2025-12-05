use crate::models::bart::config::{BartConfig, BartLikeConfig};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use edgetransformers::cache::{Cache, CpuBeamKVCache, GpuBeamKVCache};
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::gpu_ops::blocks::GpuCrossAttentionDecoder;
use edgetransformers::gpu_ops::GpuTensor;
use edgetransformers::linear_layer::LinearLayer;
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{
    base::{BeamSearchParams, DecodingStrategy, GenerationConfig}, ModelArchitecture,
    ModelType,
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
    // lm_head: Array2<f32>, // [vocab_size, hidden_size]
    lm_head: LinearLayer,
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
        // let lm_head = weights.get_array2(config.get_lm_head_name())?;
        let lm_head = LinearLayer::from_weights(&weights, config.get_lm_head_name(), None)?;

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
    pub fn gpu_decoder(&self) -> Option<&GpuCrossAttentionDecoder> {
        match &self.model {
            // Assuming self.model.model is the Cpu/Gpu enum
            TransformerEncoderDecoder::Cpu(_) => None,
            TransformerEncoderDecoder::Gpu(gpu_model) => Some(gpu_model.decoder()),
        }
    }
    // pub fn config(&self) -> &Arc<BartConfig> {
    //     &self.config
    // }
}

impl<C: EncoderDecoderArchitecture + Send + Sync> TransformerModel for Seq2SeqModel<C> {
    fn device(&self) -> Device {
        self.model.device()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self // Simply return a reference to self as a `&dyn Any`
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

    fn new_cache(
        &self,
        batch_size: usize,
        max_len: usize,
        num_beams: usize,
    ) -> Result<Box<dyn Cache>> {
        match self.model.device() {
            Device::Cpu => {
                // The batch size for beam search is the number of beams.
                let effective_batch_size = if num_beams > 0 { num_beams } else { batch_size };
                Ok(Box::new(CpuBeamKVCache::new(
                    self.config.num_decoder_layers(),
                    effective_batch_size,
                    max_len,
                    self.config.hidden_size(),
                )))
            }
            Device::Wgpu => {
                let context = self
                    .model
                    .context()
                    .ok_or_else(|| anyhow!("GPU context missing"))?;
                let head_dim = self.config.hidden_size() / self.config.num_attention_heads();
                Ok(Box::new(GpuBeamKVCache::new(
                    &context,
                    self.config.num_decoder_layers(),
                    num_beams, // <-- Use num_beams
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
    fn encoder(&self) -> &dyn Encoder<Input=Array2<u32>, Output=EncoderOutput> {
        self.model.encoder()
    }
    fn lm_head(&self) -> &Array2<f32> {
        self.lm_head.as_f32().expect("Lm head expected")
    }
    fn lm_head_layer(&self) -> &LinearLayer {
        &self.lm_head
    }
    fn final_logits_bias(&self) -> Option<&Array1<f32>> {
        self.final_logits_bias.as_ref()
    }
    fn decoder(
        &self,
    ) -> &dyn CrossAttentionDecoder<
        TokenInput=Array2<u32>,
        EncoderStateInput=Array3<f32>,
        MaskInput=Array2<f32>,
        Output=DecoderOutput,
    > {
        self.model.decoder()
    }

    fn gpu_decoder(
        &self,
    ) -> &dyn CrossAttentionDecoder<
        TokenInput=GpuTensor,
        EncoderStateInput=GpuTensor,
        MaskInput=GpuTensor,
        Output=DecoderOutput,
    > {
        self.model.gpu_decoder().unwrap() // TODO: will crash if nothing is there
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
            // bos_token_id: 0,
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
        // let logits_2d_transposed = self.lm_head.dot(&hidden_2d.t());

        // Transpose the result back to [batch*seq, vocab]
        // let logits_2d = logits_2d_transposed.t().as_standard_layout().to_owned();

        let logits_2d = self.lm_head.matmul(&hidden_2d.view());

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
