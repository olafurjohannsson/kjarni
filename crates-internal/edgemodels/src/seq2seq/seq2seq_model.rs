use crate::seq2seq::bart_configs::{BartConfig, BartLikeConfig};
use anyhow::{anyhow, Result};
use async_trait::async_trait;
use edgetransformers::cache::{Cache, GpuBeamKVCache};
use edgetransformers::encoder_decoder::TransformerEncoderDecoder;
use edgetransformers::gpu_ops::blocks::GpuCrossAttentionDecoder;
use edgetransformers::gpu_ops::{GpuTensor, GpuFrameContext, GpuTensorPool};
use edgetransformers::models::base::EncoderDecoderLanguageModel;
use edgetransformers::models::download_model_files;
use edgetransformers::models::{
    base::{BeamSearchParams, DecodingStrategy, GenerationConfig}, ModelArchitecture,
    ModelType,
};
use tokio::sync::Mutex;
use edgetransformers::gpu_ops::Kernel;
use edgetransformers::encoder_decoder::CpuTransformerEncoderDecoder;
use edgetransformers::prelude::*;
use edgetransformers::traits::{
    CrossAttentionDecoder, DecoderOutput, Encoder, EncoderDecoderArchitecture, EncoderOutput,
    LanguageModelConfig, TransformerModel,
};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array1, s, Array2, Array3};
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
    pub fn gpu_decoder(&self) -> Option<&GpuCrossAttentionDecoder> {
        match &self.model { // Assuming self.model.model is the Cpu/Gpu enum
            TransformerEncoderDecoder::Cpu(_) => None,
            TransformerEncoderDecoder::Gpu(gpu_model) => Some(gpu_model.decoder()),
        }
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

    fn new_cache(&self, batch_size: usize, max_len: usize, num_beams: usize) -> Result<Box<dyn Cache>> {
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
                let context = self.model.context().ok_or_else(|| anyhow!("GPU context missing"))?;
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
        &self.lm_head
    }

    fn final_logits_bias(&self) -> Option<&Array1<f32>> {
        self.final_logits_bias.as_ref()
    }
    fn decoder(&self) -> &dyn CrossAttentionDecoder<
        TokenInput=Array2<u32>,
        EncoderStateInput=Array3<f32>,
        MaskInput=Array2<f32>,
        Output=DecoderOutput> {
        self.model.decoder()
    }

    fn gpu_decoder(&self) -> &dyn CrossAttentionDecoder<
        TokenInput=GpuTensor,
        EncoderStateInput=GpuTensor,
        MaskInput=GpuTensor,
        Output=DecoderOutput> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use edgetransformers::models::base::{DecodingStrategy, EncoderDecoderLanguageModel};
    use edgetransformers::prelude::LanguageModel;
    use edgetransformers::TransformerConfig;
    use ndarray::{Array2, Array3};
    use ndarray_rand::RandomExt;
    use ndarray_rand::rand_distr::Uniform;

    #[tokio::test]
    async fn test_cache_basic_update_and_retrieval() -> Result<()> {
        println!("\n=== Testing Basic Cache Update and Retrieval ===\n");
        
        let context = Arc::new(WgpuContext::new().await?);
        let num_layers = 6;
        let num_beams = 4;
        let num_heads = 16;
        let head_dim = 64;
        let capacity = 100;
        
        // Create cache
        let mut cache = GpuBeamKVCache::new(
            &context,
            num_layers,
            num_beams,
            num_heads,
            head_dim,
            capacity,
        )?;
        
        // Test 1: Initial state
        assert_eq!(cache.get_seq_length(), 0, "Cache should start empty");
        
        // Test 2: Add first token
        let new_k = Array3::from_shape_fn((num_beams, 1, num_heads * head_dim), 
            |(b, _, i)| (b * 1000 + i) as f32 * 0.001);
        let new_v = Array3::from_shape_fn((num_beams, 1, num_heads * head_dim),
            |(b, _, i)| (b * 2000 + i) as f32 * 0.001);
        
        let gpu_k = GpuTensor::from_ndarray(&context, &new_k)?;
        let gpu_v = GpuTensor::from_ndarray(&context, &new_v)?;
        
        // Update layer 0
        let mut encoder = context.device.create_command_encoder(&Default::default());
        cache.update(&mut encoder, 0, &gpu_k, &gpu_v)?;
        context.queue.submit(Some(encoder.finish()));
        
        // Increment length AFTER update
        cache.increment_len(1);
        assert_eq!(cache.get_seq_length(), 1, "Cache length should be 1 after first update");
        
        // Test 3: Retrieve and verify
        let (cached_k, cached_v) = cache.get_layer_tensors(0).unwrap();
        let retrieved_k = cached_k.to_ndarray_4d::<f32>().await?;
        let retrieved_v = cached_v.to_ndarray_4d::<f32>().await?;
        
        // Check first position has our data
        for beam in 0..num_beams {
            for head in 0..num_heads {
                for dim in 0..head_dim {
                    let expected_k = (beam * 1000 + head * head_dim + dim) as f32 * 0.001;
                    let expected_v = (beam * 2000 + head * head_dim + dim) as f32 * 0.001;
                    
                    let actual_k = retrieved_k[[beam, head, 0, dim]];
                    let actual_v = retrieved_v[[beam, head, 0, dim]];
                    
                    assert!((actual_k - expected_k).abs() < 1e-5, 
                        "K mismatch at beam={}, head={}, dim={}: {} vs {}", 
                        beam, head, dim, actual_k, expected_k);
                    assert!((actual_v - expected_v).abs() < 1e-5,
                        "V mismatch at beam={}, head={}, dim={}: {} vs {}",
                        beam, head, dim, actual_v, expected_v);
                }
            }
        }
        
        println!("✅ Basic cache update and retrieval works!");
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_reordering() -> Result<()> {
        println!("\n=== Testing Cache Reordering ===\n");
        
        let context = Arc::new(WgpuContext::new().await?);
        let num_layers = 2;
        let num_beams = 4;
        let num_heads = 4;
        let head_dim = 8;
        let capacity = 10;
        
        let mut cache = GpuBeamKVCache::new(
            &context,
            num_layers,
            num_beams,
            num_heads,
            head_dim,
            capacity,
        )?;
        
        // Add 3 tokens with distinct patterns
        for step in 0..3 {
            let new_k = Array3::from_shape_fn((num_beams, 1, num_heads * head_dim),
                |(b, _, i)| (b * 100 + step * 10 + i) as f32);
            let new_v = Array3::from_shape_fn((num_beams, 1, num_heads * head_dim),
                |(b, _, i)| (b * 200 + step * 20 + i) as f32);
            
            let gpu_k = GpuTensor::from_ndarray(&context, &new_k)?;
            let gpu_v = GpuTensor::from_ndarray(&context, &new_v)?;
            
            let mut encoder = context.device.create_command_encoder(&Default::default());
            for layer in 0..num_layers {
                cache.update(&mut encoder, layer, &gpu_k, &gpu_v)?;
            }
            context.queue.submit(Some(encoder.finish()));
            cache.increment_len(1);
        }
        
        assert_eq!(cache.get_seq_length(), 3, "Should have 3 tokens");
        
        // Now reorder: [0, 1, 2, 3] -> [2, 0, 2, 1]
        let parent_indices = Array1::from_vec(vec![2u32, 0, 2, 1]);
        let gpu_indices = GpuTensor::from_ndarray(&context, &parent_indices)?;
        
        let mut encoder = context.device.create_command_encoder(&Default::default());
        cache.reorder(&mut encoder, &gpu_indices);
        context.queue.submit(Some(encoder.finish()));
        
        // Verify reordering worked
        let (reordered_k, _) = cache.get_layer_tensors(0).unwrap();
        let k_data = reordered_k.to_ndarray_4d::<f32>().await?;
        
        // Check beam 0 now has data from original beam 2
        let beam_0_token_0 = k_data[[0, 0, 0, 0]];
        let expected = (2 * 100 + 0 * 10 + 0) as f32; // beam=2, step=0, i=0
        assert!((beam_0_token_0 - expected).abs() < 1e-5,
            "Beam 0 should have beam 2's data: {} vs {}", beam_0_token_0, expected);
        
        // Check beam 1 now has data from original beam 0
        let beam_1_token_0 = k_data[[1, 0, 0, 0]];
        let expected = (0 * 100 + 0 * 10 + 0) as f32; // beam=0, step=0, i=0
        assert!((beam_1_token_0 - expected).abs() < 1e-5,
            "Beam 1 should have beam 0's data: {} vs {}", beam_1_token_0, expected);
        
        println!("✅ Cache reordering works correctly!");
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_sequence_building() -> Result<()> {
        println!("\n=== Testing Cache Sequence Building ===\n");
        
        let context = Arc::new(WgpuContext::new().await?);
        let num_beams = 2;
        let num_heads = 2;
        let head_dim = 4;
        
        let mut cache = GpuBeamKVCache::new(&context, 1, num_beams, num_heads, head_dim, 50)?;
        
        // Simulate 5 generation steps
        for step in 0..5 {
            println!("Step {}: cache_len={}", step, cache.get_seq_length());
            
            // Generate unique values for this step
            let new_k = Array3::from_shape_fn((num_beams, 1, num_heads * head_dim),
                |(b, _, i)| (step * 1000 + b * 100 + i) as f32);
            let new_v = new_k.clone() + 5000.0;
            
            let gpu_k = GpuTensor::from_ndarray(&context, &new_k)?;
            let gpu_v = GpuTensor::from_ndarray(&context, &new_v)?;
            
            // Update cache
            let mut encoder = context.device.create_command_encoder(&Default::default());
            cache.update(&mut encoder, 0, &gpu_k, &gpu_v)?;
            context.queue.submit(Some(encoder.finish()));
            
            // Simulate reordering (even steps only)
            if step % 2 == 0 && step > 0 {
                let indices = if step == 2 {
                    Array1::from_vec(vec![1u32, 0]) // Swap beams
                } else {
                    Array1::from_vec(vec![0u32, 1]) // Keep same
                };
                
                let gpu_indices = GpuTensor::from_ndarray(&context, &indices)?;
                let mut encoder = context.device.create_command_encoder(&Default::default());
                cache.reorder(&mut encoder, &gpu_indices);
                context.queue.submit(Some(encoder.finish()));
                println!("  Reordered with indices: {:?}", indices);
            }
            
            // Increment after everything
            cache.increment_len(1);
            
            // Verify cache integrity
            let (k_tensor, _) = cache.get_layer_tensors(0).unwrap();
            let k_data = k_tensor.to_ndarray_4d::<f32>().await?;
            
            // Check all positions up to current length
            for pos in 0..cache.get_seq_length() {
                let val = k_data[[0, 0, pos, 0]];
                assert!(!val.is_nan(), "Found NaN at position {}", pos);
                assert!(val.abs() < 1e10, "Found huge value at position {}: {}", pos, val);
            }
        }
        
        assert_eq!(cache.get_seq_length(), 5, "Should have 5 tokens total");
        println!("✅ Cache sequence building works!");
        Ok(())
    }

    #[tokio::test]
    async fn test_cache_boundary_conditions() -> Result<()> {
        println!("\n=== Testing Cache Boundary Conditions ===\n");
        
        let context = Arc::new(WgpuContext::new().await?);
        
        // Test 1: Single beam
        let mut cache = GpuBeamKVCache::new(&context, 1, 1, 4, 8, 10)?;
        assert_eq!(cache.get_seq_length(), 0);
        
        // Test 2: Fill to capacity
        let capacity = 10;
        let mut cache = GpuBeamKVCache::new(&context, 1, 2, 4, 8, capacity)?;
        
        for i in 0..capacity {
            let k: Array3<f32> = Array3::ones((2, 1, 32)) * i as f32;
            let v = k.clone() + 100.0;
            
            let gpu_k = GpuTensor::from_ndarray(&context, &k)?;
            let gpu_v = GpuTensor::from_ndarray(&context, &v)?;
            
            let mut encoder = context.device.create_command_encoder(&Default::default());
            cache.update(&mut encoder, 0, &gpu_k, &gpu_v)?;
            context.queue.submit(Some(encoder.finish()));
            cache.increment_len(1);
        }
        
        assert_eq!(cache.get_seq_length(), capacity);
        
        // Test 3: Verify we can't exceed capacity (this should panic or error in real impl)
        // For now just verify we're at capacity
        println!("✅ Cache boundary conditions handled!");
        Ok(())
    }

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

    // Needed for downcasting
    use edgetransformers::traits::{CrossAttentionDecoder, TransformerModel};
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
        let gpu_decoder = gpu_model.gpu_decoder().unwrap(); //.as_any().downcast_ref::<GpuCrossAttentionDecoder>().expect("Failed to downcast to GpuCrossAttentionDecoder");
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
        let gpu_decoder = gpu_model.gpu_decoder().unwrap(); //.as_any().downcast_ref::<GpuCrossAttentionDecoder>().expect("Failed to downcast to GpuCrossAttentionDecoder");
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


    #[tokio::test]
async fn test_cross_attention_with_real_model() -> Result<()> {
    use ndarray::{s, Array2, Array3};
    
    // Helper function for assertions
    fn assert_all_close(a: &Array3<f32>, b: &Array3<f32>, rtol: f32, atol: f32, context: &str) {
        if a.shape() != b.shape() {
            panic!("[{}] Shape mismatch: {:?} vs {:?}", context, a.shape(), b.shape());
        }
        
        let mut max_abs_diff: f32 = 0.0;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let abs_diff: f32 = (a_val - b_val).abs();
            max_abs_diff = max_abs_diff.max(abs_diff as f32);
            
            let tolerance = atol + rtol * b_val.abs();
            if abs_diff > tolerance {
                panic!(
                    "[{}] Arrays not close. a={}, b={}, diff={}, tol={}",
                    context, a_val, b_val, abs_diff, tolerance
                );
            }
        }
        println!("[{}] Check passed. Max diff: {:.6e}", context, max_abs_diff);
    }
    
    println!("\n=== Testing Cross-Attention with Real BART Model ===\n");
    
    // 1. SETUP
    let context = Arc::new(WgpuContext::new().await?);
    let model_type = ModelType::DistilBartCnn;
    
    // 2. LOAD MODELS
    println!("Loading CPU and GPU models...");
    
    // Load CPU Model
    let cpu_model_any = AnySeq2SeqModel::from_registry(
        model_type, None, Device::Cpu, None
    ).await?;
    let cpu_model = if let AnySeq2SeqModel::Bart(m) = cpu_model_any { 
        m 
    } else { 
        panic!("Expected BART model") 
    };
    
    // Load GPU Model
    let gpu_model_any = AnySeq2SeqModel::from_registry(
        model_type, None, Device::Wgpu, Some(context.clone())
    ).await?;
    let gpu_model = if let AnySeq2SeqModel::Bart(m) = gpu_model_any { 
        m 
    } else { 
        panic!("Expected BART model") 
    };
    
    // 3. PREPARE INPUT FOR ENCODER
    let batch_size = 1;
    let encoder_seq_len = 10;  // Short sequence for testing
    let decoder_seq_len = 1;   // Single token generation step
    
    // Create realistic input tokens (not random)
    let input_ids = Array2::from_shape_vec(
        (batch_size, encoder_seq_len),
        vec![0, 23083, 21, 10, 3231, 6251, 1012, 2003, 999, 2]  // Example tokens
    )?;
    let encoder_mask = Array2::ones((batch_size, encoder_seq_len));
    
    // 4. RUN ENCODER TO GET REAL ENCODER HIDDEN STATES
    println!("\nRunning encoder forward pass...");
    
    // CPU Encoder
    let cpu_encoder_output = cpu_model.encoder()
        .forward(&input_ids, &encoder_mask, None)
        .await?;
    println!("CPU encoder output shape: {:?}", cpu_encoder_output.last_hidden_state.shape());
    
    // GPU Encoder  
    let gpu_encoder_output = gpu_model.encoder()
        .forward(&input_ids, &encoder_mask, None)
        .await?;
    println!("GPU encoder output shape: {:?}", gpu_encoder_output.last_hidden_state.shape());
    
    // Verify encoder outputs match
    assert_all_close(
        &cpu_encoder_output.last_hidden_state,
        &gpu_encoder_output.last_hidden_state,
        1e-3, 1e-4, "Encoder Outputs"
    );
    
    // 5. PREPARE DECODER INPUT
    let decoder_start_token_id = cpu_model.config().decoder_start_token_id();
    let decoder_input_ids = Array2::from_elem(
        (batch_size, decoder_seq_len), 
        decoder_start_token_id
    );
    let decoder_mask: Array2<f32> = Array2::ones((batch_size, decoder_seq_len));
    
    // 6. GET DECODER HIDDEN STATES (after embedding + first self-attention)
    println!("\nPreparing decoder hidden states...");
    
    // Downcast decoders to access internals
    let cpu_decoder = cpu_model.decoder()
        .as_any()
        .downcast_ref::<CpuTransformerEncoderDecoder>()
        .expect("Failed to downcast CPU decoder");
    
    let gpu_decoder = gpu_model.gpu_decoder().unwrap()
        .as_any()
        .downcast_ref::<GpuCrossAttentionDecoder>()
        .expect("Failed to downcast GPU decoder");
    
    // Run embeddings on decoder input
    let cpu_decoder_hidden = cpu_decoder.decoder_embeddings.forward(
        &decoder_input_ids,
        None,
        0, // position_offset
        cpu_model.config().scale_embeddings(),
    );
    
    // Normalize if needed (BART uses post-norm)
    let cpu_decoder_hidden = if !cpu_model.config().is_prenorm() {
        cpu_decoder.decoder_embed_layer_norm.forward_3d(&cpu_decoder_hidden)
    } else {
        cpu_decoder_hidden
    };
    
    println!("Decoder hidden state shape: {:?}", cpu_decoder_hidden.shape());
    
    // 7. TEST CROSS-ATTENTION SPECIFICALLY
    println!("\n--- Testing Cross-Attention Component ---");
    
    // Get the first decoder layer for testing
    let cpu_layer = &cpu_decoder.decoder_layers[0];
    let gpu_layer = &gpu_decoder.layers[0];
    
    // CPU Cross-Attention
    println!("Running CPU cross-attention...");
    let cpu_cross_attn_out = cpu_layer.cross_attention_block(
        &cpu_decoder_hidden,
        &cpu_encoder_output.last_hidden_state,
        Some(&encoder_mask),
    )?;
    
    println!("CPU cross-attention output sample: {:?}", 
        cpu_cross_attn_out.slice(s![0, 0, 0..5]));
    
    // GPU Cross-Attention
    println!("Running GPU cross-attention...");
    let gpu_cross_attn_out = {
        let mut encoder_cmd = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());
        
        // Upload tensors
        let gpu_decoder_hidden = GpuTensor::from_ndarray(&context, &cpu_decoder_hidden)?;
        let gpu_encoder_hidden = GpuTensor::from_ndarray(&context, &cpu_encoder_output.last_hidden_state)?;
        let gpu_encoder_mask = GpuTensor::from_ndarray(&context, &encoder_mask)?;
        
        // Run cross-attention
        let cross_attn_output = gpu_layer.cross_attn.forward_cross(
            &mut encoder_cmd,
            &gpu_decoder_hidden,  // Query
            &gpu_encoder_hidden,  // Key/Value  
            &gpu_layer.cross_attn_weights,
            Some(&gpu_encoder_mask),
            &mut pool,
        );
        
        // Add residual connection
        let residual_output = pool.get(gpu_decoder_hidden.shape().to_vec());
        gpu_layer.add.encode(
            &mut encoder_cmd,
            &[&gpu_decoder_hidden, &cross_attn_output],
            &residual_output,
        );
        
        // Apply layer norm
        let normed_output = pool.get(residual_output.shape().to_vec());
        gpu_layer.cross_attn_norm.encode(
            &mut encoder_cmd,
            &gpu_layer.cross_attn_norm_weights,
            &residual_output,
            &normed_output,
        );
        
        context.queue.submit(Some(encoder_cmd.finish()));
        normed_output.to_ndarray_3d().await?
    };
    
    println!("GPU cross-attention output sample: {:?}", 
        gpu_cross_attn_out.slice(s![0, 0, 0..5]));
    
    // 8. COMPARE CROSS-ATTENTION OUTPUTS
    assert_all_close(
        &cpu_cross_attn_out,
        &gpu_cross_attn_out,
        1e-3, 1e-4,
        "Cross-Attention Outputs"
    );
    
    // 9. ANALYSIS - Check that cross-attention is actually working
    println!("\n--- Cross-Attention Analysis ---");
    
    // Check output statistics
    let cpu_mean = cpu_cross_attn_out.mean().unwrap();
    let cpu_std = cpu_cross_attn_out.std(0.0);
    let gpu_mean = gpu_cross_attn_out.mean().unwrap();
    let gpu_std = gpu_cross_attn_out.std(0.0);
    
    println!("CPU output - Mean: {:.6}, Std: {:.6}", cpu_mean, cpu_std);
    println!("GPU output - Mean: {:.6}, Std: {:.6}", gpu_mean, gpu_std);
    
    // Verify outputs are not degenerate
    assert!(cpu_std > 1e-4, "CPU cross-attention output has suspiciously low variance");
    assert!(gpu_std > 1e-4, "GPU cross-attention output has suspiciously low variance");
    
    // Check that cross-attention actually changed the input
    let input_output_diff = (&cpu_cross_attn_out - &cpu_decoder_hidden)
        .mapv(f32::abs)
        .mean()
        .unwrap();
    
    println!("Mean absolute difference from input: {:.6}", input_output_diff);
    assert!(
        input_output_diff > 1e-3, 
        "Cross-attention output is too similar to input - it may not be working!"
    );
    
    println!("\n✅ Cross-attention with real model test PASSED!");
    
    Ok(())
}

#[tokio::test] 
async fn test_cross_attention_attention_pattern() -> Result<()> {
    
    
    println!("\n=== Testing Cross-Attention Pattern Analysis ===\n");
    
    // 1. Setup
    let context = Arc::new(WgpuContext::new().await?);
    let model_type = ModelType::DistilBartCnn;
    
    // 2. Load GPU model
    let gpu_model_any = AnySeq2SeqModel::from_registry(
        model_type, None, Device::Wgpu, Some(context.clone())
    ).await?;
    let gpu_model = if let AnySeq2SeqModel::Bart(m) = gpu_model_any { 
        m 
    } else { 
        panic!("Expected BART model") 
    };
    
    // 3. Create encoder input with known pattern
    let batch_size = 1;
    let encoder_seq_len = 8;
    
    // Create input where first and last tokens are special
    let input_ids = Array2::from_shape_vec(
        (batch_size, encoder_seq_len),
        vec![0, 100, 200, 300, 400, 500, 600, 2]  // BOS, content, EOS
    )?;
    let encoder_mask = Array2::ones((batch_size, encoder_seq_len));
    
    // 4. Get encoder output
    let encoder_output = gpu_model.encoder()
        .forward(&input_ids, &encoder_mask, None)
        .await?;
    
    // 5. Create decoder query
    let decoder_start_token_id = gpu_model.config().decoder_start_token_id();
    let decoder_hidden: Array3<f32> = Array3::ones((batch_size, 1, gpu_model.config().hidden_size())) * 0.1;
    
    // 6. Compute attention scores (before softmax)
    let gpu_decoder = gpu_model.gpu_decoder().unwrap()
        .as_any()
        .downcast_ref::<GpuCrossAttentionDecoder>()
        .expect("Failed to downcast GPU decoder");
    
    let gpu_layer = &gpu_decoder.layers[0];
    
    let (attention_scores, attention_weights) = {
        let mut encoder_cmd = context.device.create_command_encoder(&Default::default());
        let mut pool = GpuTensorPool::new(context.clone());
        
        let gpu_decoder_hidden = GpuTensor::from_ndarray(&context, &decoder_hidden)?;
        let gpu_encoder_hidden = GpuTensor::from_ndarray(&context, &encoder_output.last_hidden_state)?;
        
        // Project Q and K
        let q_proj = gpu_layer.cross_attn.project(
            &mut encoder_cmd,
            &gpu_decoder_hidden,
            &gpu_layer.cross_attn_weights.q_weight,
            &gpu_layer.cross_attn_weights.q_bias,
            &mut pool,
        );
        
        let k_proj = gpu_layer.cross_attn.project(
            &mut encoder_cmd,
            &gpu_encoder_hidden,
            &gpu_layer.cross_attn_weights.k_weight,
            &gpu_layer.cross_attn_weights.k_bias,
            &mut pool,
        );
        
        // Split heads
        let q_heads = gpu_layer.cross_attn.split_heads(&mut encoder_cmd, &q_proj, &mut pool);
        let k_heads = gpu_layer.cross_attn.split_heads(&mut encoder_cmd, &k_proj, &mut pool);
        
        // Compute raw scores
        let k_transposed = k_heads.permute(
            &mut encoder_cmd, 
            &gpu_layer.cross_attn.permute,
            &[0, 1, 3, 2]
        );
        let scores = gpu_layer.cross_attn.bmm_4d(
            &mut encoder_cmd,
            &q_heads,
            &k_transposed,
            &mut pool
        );
        
        // Also compute after softmax for comparison
        let scores_copy = scores.clone();
        gpu_layer.cross_attn.softmax.encode(
            &mut encoder_cmd,
            &scores_copy,
            gpu_layer.cross_attn.scale_factor
        );
        
        context.queue.submit(Some(encoder_cmd.finish()));
        
        (scores.to_ndarray_4d::<f32>().await?, 
         scores_copy.to_ndarray_4d::<f32>().await?)
    };
    
    // 7. Analyze attention pattern
    println!("Attention scores shape: {:?}", attention_scores.shape());
    
    // Look at first head's attention pattern
    let first_head_scores = attention_scores.slice(s![0, 0, 0, ..]);
    let first_head_weights = attention_weights.slice(s![0, 0, 0, ..]);
    
    println!("\nAttention pattern (first head, decoder->encoder):");
    println!("Raw scores: {:?}", first_head_scores);
    println!("After softmax: {:?}", first_head_weights);
    
    // Check that attention weights sum to 1
    let weight_sum: f32 = first_head_weights.sum();
    println!("Attention weight sum: {:.6}", weight_sum);
    assert!((weight_sum - 1.0).abs() < 1e-4, "Attention weights don't sum to 1!");
    
    // Check entropy of attention (lower = more focused)
    let entropy: f32 = -first_head_weights.iter()
        .map(|&w| if w > 1e-10 { w * w.ln() } else { 0.0 })
        .sum::<f32>();
    println!("Attention entropy: {:.4}", entropy);
    
    println!("\n✅ Cross-attention pattern analysis complete!");
    
    Ok(())
}



}


