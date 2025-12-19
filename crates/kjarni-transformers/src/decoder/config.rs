use crate::tensor::DType;


#[derive(Debug, Clone, Copy)]
pub struct DecoderLoadConfig {
    /// If true, embedding weights are kept in system RAM and lookup happens on CPU.
    /// Saves VRAM (~1GB for Llama 1B).
    pub offload_embeddings: bool,

    /// If true, the LM Head (final projection) is kept in system RAM.
    /// Saves VRAM (~1GB for Llama 1B).
    pub offload_lm_head: bool,

    /// Number of layers to run on GPU. If None, all layers are on GPU.
    /// Useful for partial offloading.
    pub gpu_layers: Option<usize>,

    pub target_dtype: Option<DType>,
}

impl Default for DecoderLoadConfig {
    fn default() -> Self {
        Self {
            offload_embeddings: false, // Default to Performance (Pure GPU)
            offload_lm_head: false,
            gpu_layers: None,
            target_dtype: None,
        }
    }
}
