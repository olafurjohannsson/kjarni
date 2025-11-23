use crate::models::gpt2::{Gpt2Config, Gpt2Model};
use crate::generation::Generator;
use anyhow::Result;
use edgetransformers::cache::{CpuKVCache, GpuKVCache};
use edgetransformers::decoder::TransformerDecoder;
use edgetransformers::decoder::{CpuTransformerDecoder, GpuTransformerDecoder};
use edgetransformers::gpu_context::WgpuContext;
use edgetransformers::models::ModelType;
use edgetransformers::models::base::{GenerationConfig, DecodingStrategy};
use edgetransformers::models::{DecoderLanguageModel};
use edgetransformers::traits::Device;
use edgetransformers::traits::{Decoder, DecoderArchitecture};
use edgetransformers::weights::ModelWeights;
use ndarray::{Array, Array2, Array3, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;
use std::path::PathBuf;
use std::sync::Arc;


#[tokio::test]
async fn test_full_text_generation_parity() -> Result<()> {
    let model_type = ModelType::DistilGpt2;
    let prompt = "Alan Turing was a";
    let config = GenerationConfig {
        max_new_tokens: Some(3),
        strategy: DecodingStrategy::Greedy,
        ..Default::default()
    };
    let cpu_generator = Gpt2Model::from_registry(model_type, None, Device::Cpu, None).await?;
    let cpu_gen = Generator::new(Box::new(cpu_generator));
    let cpu_generated_text = cpu_gen.generate(prompt, &config).await?;
    let context = Arc::new(edgetransformers::WgpuContext::new().await?);
    let gpu_generator =
        Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(context)).await?;
    let gpu_gen = Generator::new(Box::new(gpu_generator));
    let gpu_generated_text = gpu_gen.generate(prompt, &config).await?;
    assert_eq!(
        cpu_generated_text, gpu_generated_text,
        "The final generated text from CPU and GPU backends did not match!"
    );

    Ok(())
}

/// A helper function to compare two tensors for approximate equality.
fn assert_tensors_approx_equal(a: &Array3<f32>, b: &Array3<f32>, tolerance: f32) {
    assert_eq!(a.shape(), b.shape(), "Tensor shapes do not match");
    for (i, (val_a, val_b)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (val_a - val_b).abs() < tolerance,
            "Tensor values differ at index {}: a={}, b={}",
            i,
            val_a,
            val_b
        );
    }
    println!("✓ Tensors are approximately equal.");
}

#[tokio::test]
async fn test_full_forward_pass_parity() -> Result<()> {
    let model_dir = PathBuf::from("/home/olafurj/.cache/edgetransformers/distilgpt2_resolve/");
    let weights = ModelWeights::new(&model_dir)?;
    let mut config1 = serde_json::from_str::<Gpt2Config>(&weights.config_json)?;
    config1.set_model_type("distilgpt2".to_string());
    let config: Arc<dyn DecoderArchitecture + Send + Sync> = Arc::new(config1);
    let context = Arc::new(WgpuContext::new().await?);
    let cpu_decoder =
        TransformerDecoder::Cpu(CpuTransformerDecoder::new(&weights, config.clone(), None)?);
    let gpu_decoder = TransformerDecoder::Gpu(GpuTransformerDecoder::new(
        &weights,
        config.clone(),
        context.clone(),
        None,
    )?);
    let batch_size = 1;
    let prompt_len = 11;
    let max_len = 32;
    let tolerance = 1e-3;
    let input_ids: Array2<u32> = Array::random((batch_size, prompt_len), Uniform::new(0, 50256));
    let mut full_attention_mask = Array2::zeros((batch_size, max_len));
    full_attention_mask
        .slice_mut(s![.., 0..prompt_len])
        .fill(1.0);
    let sliced_mask_priming = full_attention_mask.slice(s![.., 0..prompt_len]).to_owned();

    let cpu_output_priming = cpu_decoder
        .forward(&input_ids, &sliced_mask_priming, None)
        .await?;
    let gpu_output_priming = gpu_decoder
        .forward(&input_ids, &sliced_mask_priming, None)
        .await?;

    assert_tensors_approx_equal(
        &cpu_output_priming.last_hidden_state,
        &gpu_output_priming.last_hidden_state,
        tolerance,
    );
    let mut cpu_cache = CpuKVCache::new(
        config.num_hidden_layers(),
        batch_size,
        max_len,
        config.hidden_size(),
    );
    let mut gpu_cache = GpuKVCache::new(
        &context,
        config.num_hidden_layers(),
        batch_size,
        config.num_attention_heads(),
        config.hidden_size() / config.num_attention_heads(),
        max_len,
    )?;
    cpu_decoder
        .forward(&input_ids, &sliced_mask_priming, Some(&mut cpu_cache))
        .await?;
    gpu_decoder
        .forward(&input_ids, &full_attention_mask, Some(&mut gpu_cache))
        .await?;

    let next_token_id = Array2::from_elem((batch_size, 1), 500);
    let current_len = prompt_len;
    full_attention_mask[[0, current_len]] = 1.0;
    
    let cpu_mask_gen = full_attention_mask
        .slice(s![.., 0..current_len + 1])
        .to_owned();
    let gpu_mask_gen = full_attention_mask.clone();

    let cpu_output_gen = cpu_decoder
        .forward(&next_token_id, &cpu_mask_gen, Some(&mut cpu_cache))
        .await?;
    let gpu_output_gen = gpu_decoder
        .forward(&next_token_id, &gpu_mask_gen, Some(&mut gpu_cache))
        .await?;

    assert_tensors_approx_equal(
        &cpu_output_gen.last_hidden_state,
        &gpu_output_gen.last_hidden_state,
        tolerance,
    );

    Ok(())
}
