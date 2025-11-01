//! Internal model manager for lazy loading

use anyhow::Result;
use edgemodels::cross_encoder::CrossEncoder;
use edgemodels::sentence_encoder::SentenceEncoder;
use edgetransformers::models::ModelType;
use edgetransformers::prelude::*;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokio::sync::Mutex;

#[derive(Default)]
pub(crate) struct ModelManager {
    pub cross_encoder: Mutex<Option<CrossEncoder>>,
    pub sentence_encoder: Mutex<Option<SentenceEncoder>>,
    // Future: text generator, etc.
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

    /// Unload sentence encoder to free memory
    pub async fn unload_sentence_encoder(&self) {
        *self.sentence_encoder.lock().await = None;
    }

    /// Unload cross encoder to free memory
    pub async fn unload_cross_encoder(&self) {
        *self.cross_encoder.lock().await = None;
    }

    /// Unload all models
    pub async fn unload_all(&self) {
        self.unload_sentence_encoder().await;
        self.unload_cross_encoder().await;
    }
}
