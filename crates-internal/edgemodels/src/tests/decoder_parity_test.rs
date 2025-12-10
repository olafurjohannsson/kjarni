use crate::models::gpt2::{Gpt2Config, Gpt2Model};
// use crate::generation::Generator;
use crate::generation::decoder::{CpuDecoderBackend, GpuDecoderBackend, Generator};
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
    let cpu_generator = Gpt2Model::from_registry(model_type, None, Device::Cpu, None, None).await?;
    let cpu_gen = Generator::new(Box::new(cpu_generator))?;
    let cpu_generated_text = cpu_gen.generate(prompt, &config).await?;
    let context = edgetransformers::WgpuContext::new().await?;
    let gpu_generator =
        Gpt2Model::from_registry(model_type, None, Device::Wgpu, Some(context), None).await?;
    let gpu_gen = Generator::new(Box::new(gpu_generator))?;
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
