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
use ndarray::{Array1, Array2, Array3, s};
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
    // Make sure to adjust these `use` paths to match your project structure
    use edgetransformers::attention::MultiHeadAttention as CpuMha;
    use edgetransformers::decoder_cross_attn_layer::DecoderCrossAttentionLayer as CpuDecoderLayer;
    use edgetransformers::feedforward::{FeedForward as CpuFf, StdFeedForward as CpuStdFf};
    use edgetransformers::normalization::LayerNorm as CpuLayerNorm;
    use edgetransformers::encoder_decoder::{GpuTransformerEncoderDecoder, CpuTransformerEncoderDecoder};
    use edgetransformers::traits::{EncoderDecoderArchitecture, CrossAttentionDecoder as CrossAttentionDecoderTrait};
    use edgetransformers::gpu_ops::GpuTensorPool;
    use edgetransformers::gpu_ops::GpuTensor;
    use edgetransformers::gpu_ops::GpuFrameContext;
    use tokio::sync::Mutex;
    use edgetransformers::gpu_ops::blocks::GpuCrossAttentionDecoder;
    use ndarray::{Array, Array1, Array2, Array3};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;
    
    fn assert_all_close_2d(a: &Array2<f32>, b: &Array2<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] Shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }

        let mut max_abs_diff = 0.0;
        let mut max_rel_diff = 0.0;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let abs_diff = (a_val - b_val).abs();
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }

            // The check: absolute difference must be within the combined tolerance
            let tolerance = atol + rtol * b_val.abs();
            if abs_diff > tolerance {
                panic!(
                    "[{}] Arrays are not close. Failed at values a={}, b={}. \
                 Absolute difference {} is greater than tolerance {}",
                    context, a_val, b_val, abs_diff, tolerance
                );
            }

            if b_val.abs() > 1e-8 {
                // Avoid division by zero
                let rel_diff = abs_diff / b_val.abs();
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
        }
        println!(
            "[{}] Check passed. Max absolute difference: {:.6e}, Max relative difference: {:.6e}",
            context, max_abs_diff, max_rel_diff
        );
    }
    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!(
                "[{}] Shape mismatch: {:?} vs {:?}",
                context,
                a.shape(),
                b.shape()
            );
        }

        let mut max_abs_diff = 0.0;
        let mut max_rel_diff = 0.0;

        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let abs_diff = (a_val - b_val).abs();
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }

            // The check: absolute difference must be within the combined tolerance
            let tolerance = atol + rtol * b_val.abs();
            if abs_diff > tolerance {
                panic!(
                    "[{}] Arrays are not close. Failed at values a={}, b={}. \
                 Absolute difference {} is greater than tolerance {}",
                    context, a_val, b_val, abs_diff, tolerance
                );
            }

            if b_val.abs() > 1e-8 {
                // Avoid division by zero
                let rel_diff = abs_diff / b_val.abs();
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
        }
        println!(
            "[{}] Check passed. Max absolute difference: {:.6e}, Max relative difference: {:.6e}",
            context, max_abs_diff, max_rel_diff
        );
    }
    /// Helper function to load the DistilBART model for testing,
    /// reducing code duplication in the tests below.
    async fn load_distilbart_for_test() -> Result<Seq2SeqModel<BartConfig>> {
        let context = Arc::new(WgpuContext::new().await?);
        let any_model =
            AnySeq2SeqModel::from_registry(ModelType::DistilBartCnn, None, Device::Wgpu, Some(context))
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
    // #[tokio::test]
    // async fn test_distilbart_gpu_generation_parity() -> Result<()> {
    //     // 1. Setup Context
    //     let context = Arc::new(WgpuContext::new().await?);

    //     // 2. Load Model on GPU
    //     let any_model = AnySeq2SeqModel::from_registry(
    //         ModelType::DistilBartCnn,
    //         None,
    //         Device::Wgpu,
    //         Some(context.clone()),
    //     ).await?;

    //     let model = match any_model {
    //         AnySeq2SeqModel::Bart(m) => m,
    //     };

    //     // 3. Create Generator
    //     let generator = crate::generation::seq2seq::Seq2SeqGenerator::new(Box::new(model));

    //     // 4. Define Input and Config
    //     let article = "Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, \
    //     type safety, and concurrency. It enforces memory safety—meaning that all references point to valid memory—without \
    //     using a garbage collector.";
        
    //     // Use Greedy for deterministic comparison
    //     let mut config = generator.model.get_default_generation_config();
    //     config.strategy = DecodingStrategy::Greedy; 
    //     config.max_new_tokens = Some(20);
    //     config.repetition_penalty = 1.0; // Disable penalty to match raw model output
    //     config.no_repeat_ngram_size = 0; // Disable ngram blocking

    //     // 5. Generate
    //     let summary = generator.generate(article, &config).await?;
    //     println!("GPU Summary: {}", summary);

    //     // 6. Assert
    //     // The expected output should match what you got on CPU with the same config.
    //     // Based on your previous logs, CPU produced a good summary.
    //     // If GPU produces "CNN CNN...", this test will fail (or we can assert against expected string).
        
    //     assert!(!summary.contains("CNN CNN"), "Model is hallucinating repetition");
    //     assert!(summary.contains("Rust"), "Summary should mention the subject");

    //     Ok(())
    // }
    use std::any::Any; // Needed for downcasting
use edgetransformers::traits::{TransformerModel, CrossAttentionDecoder};
    #[tokio::test]
    async fn test_embedding_no_layer_norm() -> Result<()> {
        // 1. SETUP: Load the real model for both CPU and GPU
        let context = Arc::new(WgpuContext::new().await?);
        let model_type = ModelType::DistilBartCnn; // Or your desired model

        // --- Load and Downcast CPU Model ---
        let cpu_model_any = AnySeq2SeqModel::from_registry(model_type, None, Device::Cpu, None).await?;
        let cpu_model = if let AnySeq2SeqModel::Bart(m) = cpu_model_any { m } else { panic!("Expected BART model"); };
        // Downcast the `Box<dyn CrossAttentionDecoder>` to its concrete type to access its fields
        let cpu_decoder = cpu_model.decoder().as_any().downcast_ref::<CpuTransformerEncoderDecoder>().expect("Failed to downcast CPU decoder");

        // --- Load and Downcast GPU Model ---
        let gpu_model_any = AnySeq2SeqModel::from_registry(model_type, None, Device::Wgpu, Some(context.clone())).await?;
        let gpu_model = if let AnySeq2SeqModel::Bart(m) = gpu_model_any { m } else { panic!("Expected BART model"); };
        // Downcast the `Box<dyn CrossAttentionDecoder>` to its concrete type
        let gpu_decoder = gpu_model.decoder().as_any().downcast_ref::<GpuCrossAttentionDecoder>().expect("Failed to downcast to GpuCrossAttentionDecoder");
        // Step 2: Now that we have the concrete "Car", we can access its "Engine".
        let cpu_pos_embeddings = cpu_decoder.decoder_embeddings.position_embeddings.as_ref().unwrap();

        // 2. Get the GPU positional embeddings
        let gpu_pos_embeddings_tensor = gpu_decoder.embedding_weights().position_embeddings.as_ref().unwrap();

        // 3. Copy the GPU tensor back to the CPU for comparison
        let gpu_pos_embeddings_ndarray = gpu_pos_embeddings_tensor.to_ndarray_2d().await?;

        // 4. Print and compare
        println!("[CPU] Positional Embeddings: {:?}", cpu_pos_embeddings.slice(s![0..4, 0..8]));
        println!("[GPU] Positional Embeddings: {:?}", gpu_pos_embeddings_ndarray.slice(s![0..4, 0..8]));

        // You can add an assertion here as well
        assert_all_close_2d(cpu_pos_embeddings, &gpu_pos_embeddings_ndarray, 1e-6, 1e-6, "Positional Embeddings");
        
        let config = gpu_decoder.config.clone();
        println!("--- Testing Embedding Stage Consistency ---");

        // 2. CREATE IDENTICAL INPUTS
        let batch_size = 1;
        let seq_len = 1;
        let position_offset = 0;
        let decoder_start_token_id = config.decoder_start_token_id();
        assert_eq!(decoder_start_token_id, 2, "invalid start token id");
        let cpu_input_ids = Array2::from_elem((batch_size, seq_len), decoder_start_token_id as u32);

        // 3. RUN CPU PATH (using the downcasted concrete type)
        let cpu_output = cpu_decoder.decoder_embeddings.forward(
            &cpu_input_ids,
            None,
            position_offset + config.extra_pos_embeddings(),
            config.scale_embeddings(),
        );

        // 4. RUN GPU PATH
        let gpu_output = {
            let pool = Mutex::new(GpuTensorPool::new(context.clone()));
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (encoder, pool) = frame.resources();

            let gpu_input_ids = GpuTensor::from_ndarray(&context, &cpu_input_ids)?;
            
            let gpu_after_embed = gpu_decoder.embeddings.encode(
                encoder,
                &gpu_decoder.embedding_weights,
                &gpu_input_ids,
                None,
                position_offset,
                config.as_ref(),
                pool,
            )?;

            frame.finish();
            gpu_after_embed.to_ndarray_3d().await?
        };

        // 5. COMPARE RESULTS
        println!("[CPU] Embedding Stage Output: {:?}", cpu_output.slice(s![0, 0, 0..8]));
        println!("[GPU] Embedding Stage Output: {:?}", gpu_output.slice(s![0, 0, 0..8]));

        let rtol = 1e-4;
        let atol = 1e-5;
        assert_all_close(&cpu_output, &gpu_output, rtol, atol, "Embedding Stage Output");

        println!("✅ Embedding stage is consistent!");

        Ok(())
    }
    #[tokio::test]
    async fn test_embedding_stage_consistency_2() -> Result<()> {
        // 1. SETUP: Load the real model for both CPU and GPU
        let context = Arc::new(WgpuContext::new().await?);
        let model_type = ModelType::DistilBartCnn; // Or your desired model

        // --- Load and Downcast CPU Model ---
        let cpu_model_any = AnySeq2SeqModel::from_registry(model_type, None, Device::Cpu, None).await?;
        let cpu_model = if let AnySeq2SeqModel::Bart(m) = cpu_model_any { m } else { panic!("Expected BART model"); };
        // Downcast the `Box<dyn CrossAttentionDecoder>` to its concrete type to access its fields
        let cpu_decoder = cpu_model.decoder().as_any().downcast_ref::<CpuTransformerEncoderDecoder>().expect("Failed to downcast CPU decoder");

        // --- Load and Downcast GPU Model ---
        let gpu_model_any = AnySeq2SeqModel::from_registry(model_type, None, Device::Wgpu, Some(context.clone())).await?;
        let gpu_model = if let AnySeq2SeqModel::Bart(m) = gpu_model_any { m } else { panic!("Expected BART model"); };
        // Downcast the `Box<dyn CrossAttentionDecoder>` to its concrete type
        let gpu_decoder = gpu_model.decoder().as_any().downcast_ref::<GpuCrossAttentionDecoder>().expect("Failed to downcast to GpuCrossAttentionDecoder");
        // Step 2: Now that we have the concrete "Car", we can access its "Engine".
        
        
        let config = gpu_decoder.config.clone();
        println!("--- Testing Embedding Stage Consistency ---");

        // 2. CREATE IDENTICAL INPUTS
        let batch_size = 1;
        let seq_len = 1;
        let position_offset = 0;
        let decoder_start_token_id = config.decoder_start_token_id();
        assert_eq!(decoder_start_token_id, 2, "invalid start token id");
        let cpu_input_ids = Array2::from_elem((batch_size, seq_len), decoder_start_token_id as u32);

        // 3. RUN CPU PATH (using the downcasted concrete type)
        let cpu_after_embed = cpu_decoder.decoder_embeddings.forward(
            &cpu_input_ids,
            None,
            position_offset + config.extra_pos_embeddings(),
            config.scale_embeddings(),
        );
        let cpu_output = cpu_decoder.decoder_embed_layer_norm.forward_3d(&cpu_after_embed);

        // 4. RUN GPU PATH
        let gpu_output = {
            let pool = Mutex::new(GpuTensorPool::new(context.clone()));
            let pool_guard = pool.lock().await;
            let mut frame = GpuFrameContext::new(&context, pool_guard);
            let (encoder, pool) = frame.resources();

            let gpu_input_ids = GpuTensor::from_ndarray(&context, &cpu_input_ids)?;
            
            let gpu_after_embed = gpu_decoder.embeddings.encode(
                encoder,
                &gpu_decoder.embedding_weights,
                &gpu_input_ids,
                None,
                position_offset,
                config.as_ref(),
                pool,
            )?;
            
            let gpu_output_t = pool.get(gpu_after_embed.shape().to_vec());
            gpu_decoder.embed_layer_norm.encode(encoder, &gpu_decoder.embed_ln_weights, &gpu_after_embed, &gpu_output_t);

            frame.finish();
            gpu_output_t.to_ndarray_3d().await?
        };

        // 5. COMPARE RESULTS
        println!("[CPU] Embedding Stage Output: {:?}", cpu_output.slice(s![0, 0, 0..8]));
        println!("[GPU] Embedding Stage Output: {:?}", gpu_output.slice(s![0, 0, 0..8]));

        let rtol = 1e-4;
        let atol = 1e-5;
        assert_all_close(&cpu_output, &gpu_output, rtol, atol, "Embedding Stage Output");

        println!("✅ Embedding stage is consistent!");

        Ok(())
    }
}


