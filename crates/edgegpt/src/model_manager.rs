//! Internal model manager for lazy loading

use anyhow::{anyhow, Result};
use edgemodels::cross_encoder::CrossEncoder;
use edgemodels::sentence_encoder::SentenceEncoder;
// use edgemodels::seq2seq::{Seq2SeqModel, AnySeq2SeqModel};
use edgemodels::generation::{Generator, encoder_decoder::Seq2SeqGenerator};
use edgemodels::models::{
    gpt2::Gpt2Model,
    llama::model::LlamaModel
};
use edgetransformers::models::ModelType;
use edgetransformers::encoder_decoder::traits::EncoderDecoderLanguageModel;
use edgetransformers::models::base::{DecoderLanguageModel, EncoderLanguageModel};
use edgetransformers::prelude::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default)]
pub(crate) struct ModelManager {
    pub cross_encoder: Mutex<Option<CrossEncoder>>,
    pub sentence_encoder: Mutex<Option<SentenceEncoder>>,
    pub text_generator: Mutex<Option<Generator>>,
    pub seq2seq_generator: Mutex<Option<Seq2SeqGenerator>>,
}

impl ModelManager {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load sentence encoder if not already loaded
    pub async fn get_or_load_sentence_encoder(
        &self,
        model_type: ModelType,
        cache_dir: Option<&str>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<()> {
        let mut guard = self.sentence_encoder.lock().await;

        let dir = if cache_dir.is_some() {
            Some(PathBuf::from_str(cache_dir.unwrap())?)
        } else {
            None
        };

        if guard.is_none() {
            *guard = Some(SentenceEncoder::from_registry(model_type, dir, device, context).await?);
        }

        Ok(())
    }

    /// Load cross encoder if not already loaded
    pub async fn get_or_load_cross_encoder(
        &self,
        model_type: ModelType,
        cache_dir: Option<&str>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<()> {
        let mut guard = self.cross_encoder.lock().await;

        let dir = if cache_dir.is_some() {
            Some(PathBuf::from_str(cache_dir.unwrap())?)
        } else {
            None
        };
        if guard.is_none() {
            *guard = Some(CrossEncoder::from_registry(model_type, dir, device, context).await?);
        }

        Ok(())
    }

    pub async fn get_or_load_text_generator(
        &self,
        model_type: ModelType,
        cache_dir: Option<&str>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<()> {
        let mut guard = self.text_generator.lock().await;
        if guard.is_some() { return Ok(()); }

        let dir = Self::resolve_cache_dir(cache_dir)?;
        
        // Factory logic for Decoder-only models
        let model: Box<dyn DecoderLanguageModel> = match model_type {
            ModelType::Llama3_2_1B => {
                Box::new(LlamaModel::from_registry(model_type, Some(dir), device, context, None).await?)
            }
            ModelType::Gpt2 | ModelType::DistilGpt2 | ModelType::Gpt2Medium | ModelType::Gpt2Large | ModelType::Gpt2XL => {
                Box::new(Gpt2Model::from_registry(model_type, Some(dir), device, context, None).await?)
            }
            _ => return Err(anyhow!("Unsupported text generation model: {:?}", model_type)),
        };

        *guard = Some(Generator::new(model));
        Ok(())
    }

    pub async fn get_or_load_seq2seq_generator(
        &self,
        model_type: ModelType,
        cache_dir: Option<&str>,
        device: Device,
        context: Option<Arc<WgpuContext>>,
    ) -> Result<()> {
        unimplemented!()
        // let mut guard = self.seq2seq_generator.lock().await;
        // if guard.is_some() { return Ok(()); }

        // let dir = Self::resolve_cache_dir(cache_dir)?;

        // // Factory logic for Seq2Seq models
        // let any_model = AnySeq2SeqModel::from_registry(model_type, Some(dir), device, context).await?;
        
        // // Unwrap the specific model type (currently only BART supported)
        // let model: Box<dyn EncoderDecoderLanguageModel> = match any_model {
        //     AnySeq2SeqModel::Bart(m) => Box::new(m),
        // };
        // let generator = Seq2SeqGenerator::new(model)?;

        // *guard = Some(generator);
        // Ok(())
    }

    /// Unload sentence encoder to free memory
    pub async fn unload_sentence_encoder(&self) {
        *self.sentence_encoder.lock().await = None;
    }

    /// Unload cross encoder to free memory
    pub async fn unload_cross_encoder(&self) {
        *self.cross_encoder.lock().await = None;
    }

    pub async fn unload_text_generator(&self) { *self.text_generator.lock().await = None; }
    pub async fn unload_seq2seq_generator(&self) { *self.seq2seq_generator.lock().await = None; }

    pub async fn unload_all(&self) {
        self.unload_sentence_encoder().await;
        self.unload_cross_encoder().await;
        self.unload_text_generator().await;
        self.unload_seq2seq_generator().await;
    }


    fn resolve_cache_dir(cache_dir: Option<&str>) -> Result<PathBuf> {
        if let Some(d) = cache_dir {
            Ok(PathBuf::from(d))
        } else {
            Ok(dirs::cache_dir().ok_or(anyhow!("No cache dir"))?.join("edgetransformers"))
        }
    }
}
