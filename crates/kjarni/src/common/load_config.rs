
//! Model loading configuration.

use kjarni_transformers::{models::base::ModelLoadConfig, tensor::DType};

/// Wrapper around ModelLoadConfig for the high-level API.
#[derive(Debug, Clone)]
pub struct LoadConfig {
    pub(crate) inner: ModelLoadConfig,
}

impl LoadConfig {
    /// Create a new default LoadConfig.
    pub fn new() -> Self {
        Self {
            inner: ModelLoadConfig::default(),
        }
    }

    /// Get the inner ModelLoadConfig.
    pub fn into_inner(self) -> ModelLoadConfig {
        self.inner
    }

    /// Get a reference to the inner config.
    pub fn as_inner(&self) -> &ModelLoadConfig {
        &self.inner
    }
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl From<ModelLoadConfig> for LoadConfig {
    fn from(inner: ModelLoadConfig) -> Self {
        Self { inner }
    }
}

/// Builder for LoadConfig.
#[derive(Debug, Clone, Default)]
pub struct LoadConfigBuilder {
    inner: ModelLoadConfig,
}

impl LoadConfigBuilder {
    /// Create a new builder with defaults.
    pub fn new() -> Self {
        Self {
            inner: ModelLoadConfig::default(),
        }
    }

    /// Create a builder from an existing LoadConfig.
    pub fn from_config(config: LoadConfig) -> Self {
        Self {
            inner: config.inner,
        }
    }

    /// Create a builder from an existing ModelLoadConfig.
    pub fn from_inner(inner: ModelLoadConfig) -> Self {
        Self { inner }
    }

    /// Keep embedding layer on CPU to save VRAM.
    pub fn offload_embeddings(mut self, offload: bool) -> Self {
        self.inner.offload_embeddings = offload;
        self
    }

    /// Keep language model head on CPU to save VRAM.
    pub fn offload_lm_head(mut self, offload: bool) -> Self {
        self.inner.offload_lm_head = offload;
        self
    }

    /// Set target dtype for weights.
    pub fn dtype(mut self, dtype: DType) -> Self {
        self.inner.target_dtype = Some(dtype);
        self
    }

    /// Use F16 precision.
    pub fn f16(self) -> Self {
        self.dtype(DType::F16)
    }

    /// Use BF16 precision.
    pub fn bf16(self) -> Self {
        self.dtype(DType::BF16)
    }

    /// Use F32 precision (default).
    pub fn f32(self) -> Self {
        self.dtype(DType::F32)
    }

    /// Quantize the LM head to specified dtype.
    pub fn quantize_lm_head(mut self, dtype: DType) -> Self {
        self.inner.quantize_lm_head = Some(dtype);
        self
    }

    /// Quantize LM head to Q8_0.
    pub fn quantize_lm_head_q8(self) -> Self {
        self.quantize_lm_head(DType::Q8_0)
    }

    /// Set maximum batch size for KV cache pre-allocation.
    pub fn max_batch_size(mut self, size: usize) -> Self {
        self.inner.max_batch_size = Some(size);
        self
    }

    /// Set maximum sequence length for KV cache pre-allocation.
    pub fn max_sequence_length(mut self, length: usize) -> Self {
        self.inner.max_sequence_length = Some(length);
        self
    }

    /// Prefer GGUF format if available.
    pub fn prefer_gguf(mut self, prefer: bool) -> Self {
        self.inner.use_gguf = prefer;
        self
    }

    /// Build the LoadConfig.
    pub fn build(self) -> LoadConfig {
        LoadConfig { inner: self.inner }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_config_new() {
        let config = LoadConfig::new();
        let inner = config.as_inner();
        
        assert!(!inner.offload_embeddings);
        assert!(!inner.offload_lm_head);
        assert!(inner.target_dtype.is_none());
        assert!(inner.quantize_lm_head.is_none());
        assert!(inner.max_batch_size.is_none());
        assert!(inner.max_sequence_length.is_none());
        assert!(!inner.use_gguf);
    }
    #[test]
    fn test_load_config_default() {
        let config = LoadConfig::default();
        let inner = config.as_inner();
        
        assert!(!inner.offload_embeddings);
        assert!(!inner.offload_lm_head);
    }

    #[test]
    fn test_load_config_new_equals_default() {
        let new_config = LoadConfig::new();
        let default_config = LoadConfig::default();
        
        assert_eq!(
            new_config.as_inner().offload_embeddings,
            default_config.as_inner().offload_embeddings
        );
        assert_eq!(
            new_config.as_inner().offload_lm_head,
            default_config.as_inner().offload_lm_head
        );
        assert_eq!(
            new_config.as_inner().use_gguf,
            default_config.as_inner().use_gguf
        );
    }

    #[test]
    fn test_load_config_into_inner() {
        let config = LoadConfig::new();
        let inner: ModelLoadConfig = config.into_inner();
        
        assert!(!inner.offload_embeddings);
    }

    #[test]
    fn test_load_config_into_inner_consumes() {
        let config = LoadConfig::new();
        let _inner = config.into_inner();
    }

    #[test]
    fn test_load_config_as_inner() {
        let config = LoadConfig::new();
        let inner = config.as_inner();
        
        assert!(!inner.offload_embeddings);
        let _inner2 = config.as_inner();
    }

    #[test]
    fn test_load_config_as_inner_multiple_calls() {
        let config = LoadConfig::new();
        
        let inner1 = config.as_inner();
        let inner2 = config.as_inner();
        
        assert_eq!(inner1.offload_embeddings, inner2.offload_embeddings);
    }
    #[test]
    fn test_load_config_from_model_load_config() {
        let mut inner = ModelLoadConfig::default();
        inner.offload_embeddings = true;
        inner.use_gguf = true;
        
        let config: LoadConfig = inner.into();
        
        assert!(config.as_inner().offload_embeddings);
        assert!(config.as_inner().use_gguf);
    }

    #[test]
    fn test_load_config_from_preserves_all_fields() {
        let mut inner = ModelLoadConfig::default();
        inner.offload_embeddings = true;
        inner.offload_lm_head = true;
        inner.target_dtype = Some(DType::F16);
        inner.quantize_lm_head = Some(DType::Q8_0);
        inner.max_batch_size = Some(8);
        inner.max_sequence_length = Some(2048);
        inner.use_gguf = true;
        
        let config = LoadConfig::from(inner);
        let result = config.as_inner();
        
        assert!(result.offload_embeddings);
        assert!(result.offload_lm_head);
        assert_eq!(result.target_dtype, Some(DType::F16));
        assert_eq!(result.quantize_lm_head, Some(DType::Q8_0));
        assert_eq!(result.max_batch_size, Some(8));
        assert_eq!(result.max_sequence_length, Some(2048));
        assert!(result.use_gguf);
    }

    #[test]
    fn test_load_config_debug() {
        let config = LoadConfig::new();
        let debug = format!("{:?}", config);
        
        assert!(debug.contains("LoadConfig"));
    }

    #[test]
    fn test_load_config_clone() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .build();
        
        let cloned = config.clone();
        
        assert_eq!(
            config.as_inner().offload_embeddings,
            cloned.as_inner().offload_embeddings
        );
    }

    #[test]
    fn test_load_config_clone_independence() {
        let config = LoadConfig::new();
        let cloned = config.clone();
        
        let _inner1 = config.into_inner();
        let _inner2 = cloned.into_inner();
    }
    #[test]
    fn test_builder_new() {
        let builder = LoadConfigBuilder::new();
        let config = builder.build();
        
        assert!(!config.as_inner().offload_embeddings);
        assert!(!config.as_inner().offload_lm_head);
    }
    #[test]
    fn test_builder_default() {
        let builder = LoadConfigBuilder::default();
        let config = builder.build();
        
        assert!(!config.as_inner().offload_embeddings);
    }

    #[test]
    fn test_builder_new_equals_default() {
        let new_builder = LoadConfigBuilder::new();
        let default_builder = LoadConfigBuilder::default();
        
        let new_config = new_builder.build();
        let default_config = default_builder.build();
        
        assert_eq!(
            new_config.as_inner().offload_embeddings,
            default_config.as_inner().offload_embeddings
        );
    }

    #[test]
    fn test_builder_from_config() {
        let original = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .f16()
            .build();
        
        let builder = LoadConfigBuilder::from_config(original);
        let config = builder.build();
        
        assert!(config.as_inner().offload_embeddings);
        assert_eq!(config.as_inner().target_dtype, Some(DType::F16));
    }

    #[test]
    fn test_builder_from_config_can_modify() {
        let original = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .build();
        
        let config = LoadConfigBuilder::from_config(original)
            .offload_lm_head(true)
            .build();
        
        assert!(config.as_inner().offload_embeddings);
        assert!(config.as_inner().offload_lm_head);
    }

    #[test]
    fn test_builder_from_inner() {
        let mut inner = ModelLoadConfig::default();
        inner.use_gguf = true;
        
        let builder = LoadConfigBuilder::from_inner(inner);
        let config = builder.build();
        
        assert!(config.as_inner().use_gguf);
    }
    #[test]
    fn test_builder_offload_embeddings_true() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .build();
        
        assert!(config.as_inner().offload_embeddings);
    }

    #[test]
    fn test_builder_offload_embeddings_false() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(false)
            .build();
        
        assert!(!config.as_inner().offload_embeddings);
    }

    #[test]
    fn test_builder_offload_embeddings_override() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .offload_embeddings(false)
            .build();
        
        // Last value wins
        assert!(!config.as_inner().offload_embeddings);
    }
    #[test]
    fn test_builder_offload_lm_head_true() {
        let config = LoadConfigBuilder::new()
            .offload_lm_head(true)
            .build();
        
        assert!(config.as_inner().offload_lm_head);
    }

    #[test]
    fn test_builder_offload_lm_head_false() {
        let config = LoadConfigBuilder::new()
            .offload_lm_head(false)
            .build();
        
        assert!(!config.as_inner().offload_lm_head);
    }
    #[test]
    fn test_builder_dtype_f16() {
        let config = LoadConfigBuilder::new()
            .dtype(DType::F16)
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::F16));
    }

    #[test]
    fn test_builder_dtype_bf16() {
        let config = LoadConfigBuilder::new()
            .dtype(DType::BF16)
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::BF16));
    }

    #[test]
    fn test_builder_dtype_f32() {
        let config = LoadConfigBuilder::new()
            .dtype(DType::F32)
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::F32));
    }

    #[test]
    fn test_builder_dtype_override() {
        let config = LoadConfigBuilder::new()
            .dtype(DType::F16)
            .dtype(DType::BF16)
            .build();
        
        // Last value wins
        assert_eq!(config.as_inner().target_dtype, Some(DType::BF16));
    }
    #[test]
    fn test_builder_f16_method() {
        let config = LoadConfigBuilder::new()
            .f16()
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::F16));
    }

    #[test]
    fn test_builder_bf16_method() {
        let config = LoadConfigBuilder::new()
            .bf16()
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::BF16));
    }

    #[test]
    fn test_builder_f32_method() {
        let config = LoadConfigBuilder::new()
            .f32()
            .build();
        
        assert_eq!(config.as_inner().target_dtype, Some(DType::F32));
    }

    #[test]
    fn test_builder_dtype_methods_equivalent() {
        let via_method = LoadConfigBuilder::new().f16().build();
        let via_dtype = LoadConfigBuilder::new().dtype(DType::F16).build();
        
        assert_eq!(
            via_method.as_inner().target_dtype,
            via_dtype.as_inner().target_dtype
        );
    }
    #[test]
    fn test_builder_quantize_lm_head() {
        let config = LoadConfigBuilder::new()
            .quantize_lm_head(DType::Q8_0)
            .build();
        
        assert_eq!(config.as_inner().quantize_lm_head, Some(DType::Q8_0));
    }

    #[test]
    fn test_builder_quantize_lm_head_q8() {
        let config = LoadConfigBuilder::new()
            .quantize_lm_head_q8()
            .build();
        
        assert_eq!(config.as_inner().quantize_lm_head, Some(DType::Q8_0));
    }

    #[test]
    fn test_builder_quantize_lm_head_q8_equivalent() {
        let via_method = LoadConfigBuilder::new().quantize_lm_head_q8().build();
        let via_dtype = LoadConfigBuilder::new().quantize_lm_head(DType::Q8_0).build();
        
        assert_eq!(
            via_method.as_inner().quantize_lm_head,
            via_dtype.as_inner().quantize_lm_head
        );
    }
    #[test]
    fn test_builder_max_batch_size() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(8)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(8));
    }

    #[test]
    fn test_builder_max_batch_size_one() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(1)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(1));
    }

    #[test]
    fn test_builder_max_batch_size_large() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(128)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(128));
    }

    #[test]
    fn test_builder_max_batch_size_override() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(8)
            .max_batch_size(16)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(16));
    }

    #[test]
    fn test_builder_max_sequence_length() {
        let config = LoadConfigBuilder::new()
            .max_sequence_length(2048)
            .build();
        
        assert_eq!(config.as_inner().max_sequence_length, Some(2048));
    }

    #[test]
    fn test_builder_max_sequence_length_small() {
        let config = LoadConfigBuilder::new()
            .max_sequence_length(512)
            .build();
        
        assert_eq!(config.as_inner().max_sequence_length, Some(512));
    }

    #[test]
    fn test_builder_max_sequence_length_large() {
        let config = LoadConfigBuilder::new()
            .max_sequence_length(32768)
            .build();
        
        assert_eq!(config.as_inner().max_sequence_length, Some(32768));
    }

    #[test]
    fn test_builder_max_sequence_length_override() {
        let config = LoadConfigBuilder::new()
            .max_sequence_length(1024)
            .max_sequence_length(4096)
            .build();
        
        assert_eq!(config.as_inner().max_sequence_length, Some(4096));
    }
    #[test]
    fn test_builder_prefer_gguf_true() {
        let config = LoadConfigBuilder::new()
            .prefer_gguf(true)
            .build();
        
        assert!(config.as_inner().use_gguf);
    }

    #[test]
    fn test_builder_prefer_gguf_false() {
        let config = LoadConfigBuilder::new()
            .prefer_gguf(false)
            .build();
        
        assert!(!config.as_inner().use_gguf);
    }

    #[test]
    fn test_builder_prefer_gguf_override() {
        let config = LoadConfigBuilder::new()
            .prefer_gguf(true)
            .prefer_gguf(false)
            .build();
        
        assert!(!config.as_inner().use_gguf);
    }
    #[test]
    fn test_builder_chaining_all_options() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .offload_lm_head(true)
            .f16()
            .quantize_lm_head_q8()
            .max_batch_size(4)
            .max_sequence_length(4096)
            .prefer_gguf(true)
            .build();
        
        let inner = config.as_inner();
        assert!(inner.offload_embeddings);
        assert!(inner.offload_lm_head);
        assert_eq!(inner.target_dtype, Some(DType::F16));
        assert_eq!(inner.quantize_lm_head, Some(DType::Q8_0));
        assert_eq!(inner.max_batch_size, Some(4));
        assert_eq!(inner.max_sequence_length, Some(4096));
        assert!(inner.use_gguf);
    }

    #[test]
    fn test_builder_partial_configuration() {
        let config = LoadConfigBuilder::new()
            .f16()
            .max_batch_size(8)
            .build();
        
        let inner = config.as_inner();
        // Set values
        assert_eq!(inner.target_dtype, Some(DType::F16));
        assert_eq!(inner.max_batch_size, Some(8));
        // Default values
        assert!(!inner.offload_embeddings);
        assert!(!inner.offload_lm_head);
        assert!(inner.quantize_lm_head.is_none());
        assert!(inner.max_sequence_length.is_none());
        assert!(!inner.use_gguf);
    }

    #[test]
    fn test_builder_order_independence() {
        let config1 = LoadConfigBuilder::new()
            .f16()
            .offload_embeddings(true)
            .max_batch_size(8)
            .build();
        
        let config2 = LoadConfigBuilder::new()
            .max_batch_size(8)
            .offload_embeddings(true)
            .f16()
            .build();
        
        assert_eq!(
            config1.as_inner().target_dtype,
            config2.as_inner().target_dtype
        );
        assert_eq!(
            config1.as_inner().offload_embeddings,
            config2.as_inner().offload_embeddings
        );
        assert_eq!(
            config1.as_inner().max_batch_size,
            config2.as_inner().max_batch_size
        );
    }

    #[test]
    fn test_builder_debug() {
        let builder = LoadConfigBuilder::new().f16();
        let debug = format!("{:?}", builder);
        
        assert!(debug.contains("LoadConfigBuilder"));
    }

    #[test]
    fn test_builder_clone() {
        let builder = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .f16();
        
        let cloned = builder.clone();
        
        let config1 = builder.build();
        let config2 = cloned.build();
        
        assert_eq!(
            config1.as_inner().offload_embeddings,
            config2.as_inner().offload_embeddings
        );
        assert_eq!(
            config1.as_inner().target_dtype,
            config2.as_inner().target_dtype
        );
    }

    #[test]
    fn test_builder_clone_independence() {
        let builder = LoadConfigBuilder::new().f16();
        let cloned = builder.clone();
        
        let config1 = builder.bf16().build();
        let config2 = cloned.build();
        
        assert_eq!(config1.as_inner().target_dtype, Some(DType::BF16));
        assert_eq!(config2.as_inner().target_dtype, Some(DType::F16));
    }
    #[test]
    fn test_config_for_small_gpu() {
        let config = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .offload_lm_head(true)
            .f16()
            .quantize_lm_head_q8()
            .max_batch_size(1)
            .max_sequence_length(2048)
            .build();
        
        let inner = config.as_inner();
        assert!(inner.offload_embeddings);
        assert!(inner.offload_lm_head);
        assert_eq!(inner.target_dtype, Some(DType::F16));
        assert_eq!(inner.quantize_lm_head, Some(DType::Q8_0));
    }

    #[test]
    fn test_config_for_cpu() {
        let config = LoadConfigBuilder::new()
            .f32()
            .max_batch_size(1)
            .build();
        
        let inner = config.as_inner();
        assert_eq!(inner.target_dtype, Some(DType::F32));
        assert!(!inner.offload_embeddings);
    }

    #[test]
    fn test_config_for_large_gpu() {
        let config = LoadConfigBuilder::new()
            .bf16()
            .max_batch_size(32)
            .max_sequence_length(8192)
            .build();
        
        let inner = config.as_inner();
        assert_eq!(inner.target_dtype, Some(DType::BF16));
        assert_eq!(inner.max_batch_size, Some(32));
        assert_eq!(inner.max_sequence_length, Some(8192));
    }

    #[test]
    fn test_config_prefer_gguf_quantized() {
        let config = LoadConfigBuilder::new()
            .prefer_gguf(true)
            .build();
        
        assert!(config.as_inner().use_gguf);
    }

    #[test]
    fn test_builder_zero_batch_size() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(0)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(0));
    }

    #[test]
    fn test_builder_zero_sequence_length() {
        let config = LoadConfigBuilder::new()
            .max_sequence_length(0)
            .build();
        
        assert_eq!(config.as_inner().max_sequence_length, Some(0));
    }

    #[test]
    fn test_builder_max_values() {
        let config = LoadConfigBuilder::new()
            .max_batch_size(usize::MAX)
            .max_sequence_length(usize::MAX)
            .build();
        
        assert_eq!(config.as_inner().max_batch_size, Some(usize::MAX));
        assert_eq!(config.as_inner().max_sequence_length, Some(usize::MAX));
    }
    #[test]
    fn test_roundtrip_load_config() {
        let original = LoadConfigBuilder::new()
            .offload_embeddings(true)
            .f16()
            .max_batch_size(8)
            .build();
        
        let inner = original.into_inner();
        let restored = LoadConfig::from(inner);
        
        assert!(restored.as_inner().offload_embeddings);
        assert_eq!(restored.as_inner().target_dtype, Some(DType::F16));
        assert_eq!(restored.as_inner().max_batch_size, Some(8));
    }

    #[test]
    fn test_roundtrip_through_builder() {
        let original = LoadConfigBuilder::new()
            .bf16()
            .prefer_gguf(true)
            .build();
        
        let rebuilt = LoadConfigBuilder::from_config(original).build();
        
        assert_eq!(rebuilt.as_inner().target_dtype, Some(DType::BF16));
        assert!(rebuilt.as_inner().use_gguf);
    }

    #[test]
    fn test_load_config_size_reasonable() {
        let size = std::mem::size_of::<LoadConfig>();
        assert!(size < 256, "LoadConfig size {} is too large", size);
    }

    #[test]
    fn test_builder_size_reasonable() {
        let size = std::mem::size_of::<LoadConfigBuilder>();
        assert!(size < 256, "LoadConfigBuilder size {} is too large", size);
    }
}