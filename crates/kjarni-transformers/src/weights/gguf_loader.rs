use crate::tensor::RawTensor;
use crate::weights::WeightLoader;
use anyhow::{anyhow, Result};
use std::path::Path;

/// STUBBED loader for the .gguf format.
pub struct GgufLoader;

impl GgufLoader {
    pub fn new(path: &Path) -> Result<Self> {
        // In the future, you would parse the GGUF file header here.
        // For example:
        // let gguf_ctx = gguf_rs::gguf::GgufContext::load_from_file(path)?;
        log::warn!("GGUF loader is a stub and does not actually load weights yet.");
        Ok(Self)
    }
}

impl WeightLoader for GgufLoader {
    fn get_raw(&self, name: &str) -> Result<RawTensor<'_>> {
        unimplemented!("GGUF tensor loading for '{}' is not yet implemented.", name)
    }

    fn contains(&self, name: &str) -> bool {
        // In the future, you would check if the tensor exists in the GGUF context.
        false
    }
}