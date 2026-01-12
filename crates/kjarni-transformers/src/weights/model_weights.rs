//! High-level interface for loading model weights.
//!
//! [`ModelWeights`] provides a unified API for accessing model weights regardless
//! of the underlying format (SafeTensors or GGUF). It handles:
//!
//! - Format auto-detection
//! - Config synthesis for GGUF files
//! - Type conversion helpers
//! - The callback pattern for safe tensor access
//!
//! # Memory-Efficient Access
//!
//! Prefer the callback-based `with_raw_tensor` method which keeps data mmap'd:
//!
//! ```ignore
//! // GOOD: Data stays on disk, only pages touched are loaded
//! weights.with_raw_tensor("layer.weight", |view| {
//!     // Process view.bytes directly
//!     build_linear_layer(view)
//! })?;
//! ```
//!
//! Avoid the `get_array*` methods which eagerly load into RAM:
//!
//! ```ignore
//! // BAD: Allocates full tensor in RAM
//! let array = weights.get_array2("layer.weight")?;
//! ```

use super::gguf_loader::GgufLoader;
use super::safetensors_loader::SafeTensorsLoader;
use super::WeightLoader;
use crate::tensor::raw_tensor::TensorView;
use crate::tensor::{CpuTensor, DType, QuantizedMatrix};
use crate::weights::raw_to_typed_gguf;
use anyhow::{anyhow, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use serde_json::json;
use std::path::Path;
use std::sync::Arc;

/// Attention layout information extracted from model weights.
///
/// Used for quantized weight reshaping in GGUF models.
#[derive(Debug, Clone, Copy)]
pub struct AttentionLayout {
    /// Number of attention heads
    pub n_heads: usize,
    /// Number of key-value heads (for GQA)
    pub n_kv_heads: usize,
    /// Dimension of each head
    pub head_dim: usize,
}

/// Internal storage for ModelWeights.
struct ModelWeightsInner {
    /// The underlying format-specific loader
    loader: Box<dyn WeightLoader + Send + Sync>,
    /// Model configuration JSON (loaded or synthesized)
    config_json: String,
    /// Whether this is a GGUF file (affects tensor conversion)
    is_gguf: bool,
}

/// High-level interface for loading model weights.
///
/// Provides format-agnostic access to model weights with:
/// - Automatic format detection (SafeTensors vs GGUF)
/// - Config synthesis for GGUF files
/// - Type conversion utilities
/// - Memory-efficient mmap-based access
///
/// # Thread Safety
///
/// `ModelWeights` is `Clone` and thread-safe. Multiple clones share the same
/// underlying mmap through `Arc`.
///
/// # Preferred Usage Pattern
///
/// Use `with_raw_tensor` for memory-efficient access:
///
/// ```ignore
/// weights.with_raw_tensor("embedding.weight", |view| {
///     // Build your layer directly from view.bytes
///     // No extra allocation needed
///     LinearLayer::from_view(view, dtype)
/// })?;
/// ```
pub struct ModelWeights {
    inner: Arc<ModelWeightsInner>,
}

impl Clone for ModelWeights {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl ModelWeights {
    /// Create a new ModelWeights from a path.
    ///
    /// Accepts:
    /// - A `.gguf` file path
    /// - A directory containing `model.safetensors` or `model.safetensors.index.json`
    /// - A directory containing a `.gguf` file
    ///
    /// # Arguments
    ///
    /// * `path` - Path to model file or directory
    ///
    /// # Returns
    ///
    /// A configured `ModelWeights` instance.
    ///
    /// # Errors
    ///
    /// - Path doesn't exist
    /// - No supported weight format found
    /// - Format-specific parse errors
    pub fn new(path: &Path) -> Result<Self> {
        // Case 1: Direct GGUF file
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Self::from_gguf_file(path);
        }

        // Case 2: Directory
        if path.is_dir() {
            // Try SafeTensors first
            if path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
            {
                let loader = Box::new(SafeTensorsLoader::new(path)?) as Box<dyn WeightLoader + Send + Sync>;
                let config_json = std::fs::read_to_string(path.join("config.json"))
                    .unwrap_or_else(|_| "{}".to_string());

                return Ok(Self {
                    inner: Arc::new(ModelWeightsInner {
                        loader,
                        config_json,
                        is_gguf: false,
                    }),
                });
            }

            // Try to find GGUF in directory
            let gguf_in_dir = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"));

            if let Some(entry) = gguf_in_dir {
                return Self::from_gguf_file(&entry.path());
            }
        }

        Err(anyhow!("No supported weight format found at {:?}", path))
    }

    /// Create ModelWeights from a specific file.
    ///
    /// More flexible than `new()` - accepts any file path and attempts
    /// to load it based on extension or content.
    pub fn from_file(path: &Path) -> Result<Self> {
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Self::from_gguf_file(path);
        }

        if path.is_dir() {
            return Self::new(path);
        }

        if path.exists() {
            // Assume SafeTensors for other files
            let loader = Box::new(SafeTensorsLoader::new(path)?) as Box<dyn WeightLoader + Send + Sync>;
            return Ok(Self {
                inner: Arc::new(ModelWeightsInner {
                    loader,
                    config_json: "{}".to_string(),
                    is_gguf: false,
                }),
            });
        }

        Self::new(path)
    }

    /// Load from a GGUF file, synthesizing config from metadata.
    fn from_gguf_file(path: &Path) -> Result<Self> {
        let loader = GgufLoader::new(path)?;
        
        if log::log_enabled!(log::Level::Debug) {
            loader.debug_print_tensors();
        }

        let config_json = Self::synthesize_config_from_gguf(&loader)?;

        Ok(Self {
            inner: Arc::new(ModelWeightsInner {
                loader: Box::new(loader) as Box<dyn WeightLoader + Send + Sync>,
                config_json,
                is_gguf: true,
            }),
        })
    }

    /// Synthesize a config.json from GGUF metadata.
    fn synthesize_config_from_gguf(loader: &GgufLoader) -> Result<String> {
        let arch = loader.get_string("general.architecture").unwrap_or("llama");

        // Build rope_scaling if present
        let rope_scaling = loader
            .get_string(&format!("{}.rope.scaling.type", arch))
            .map(|_| {
                json!({
                    "rope_type": loader.get_string(&format!("{}.rope.scaling.type", arch))
                        .unwrap_or("llama3"),
                    "factor": loader.get_f32(&format!("{}.rope.scaling.factor", arch))
                        .unwrap_or(32.0),
                    "low_freq_factor": loader.get_f32(&format!("{}.rope.scaling.low_freq_factor", arch))
                        .unwrap_or(1.0),
                    "high_freq_factor": loader.get_f32(&format!("{}.rope.scaling.high_freq_factor", arch))
                        .unwrap_or(4.0),
                    "original_max_position_embeddings": loader
                        .get_u32(&format!("{}.rope.scaling.orig_ctx_len", arch))
                        .unwrap_or(8192),
                })
            });

        let config = json!({
            "architecture": arch,
            "hidden_size": loader.get_u32(&format!("{}.embedding_length", arch)),
            "intermediate_size": loader.get_u32(&format!("{}.feed_forward_length", arch)),
            "num_attention_heads": loader.get_u32(&format!("{}.attention.head_count", arch)),
            "num_hidden_layers": loader.get_u32(&format!("{}.block_count", arch)),
            "num_key_value_heads": loader.get_u32(&format!("{}.attention.head_count_kv", arch)),
            "max_position_embeddings": loader.get_u32(&format!("{}.context_length", arch)),
            "rope_theta": loader.get_f32(&format!("{}.rope.freq_base", arch)).unwrap_or(10000.0),
            "rope_scaling": rope_scaling,
            "rms_norm_eps": loader
                .get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch))
                .unwrap_or(1e-5),
            "vocab_size": loader.get_u32(&format!("{}.vocab_size", arch)).unwrap_or(128256),
            "bos_token_id": loader.get_u32("tokenizer.ggml.bos_token_id").unwrap_or(128000),
            "eos_token_id": loader.get_u32("tokenizer.ggml.eos_token_id").unwrap_or(128001),
        });

        Ok(config.to_string())
    }

    // =========================================================================
    // Public Accessors
    // =========================================================================

    /// Get a reference to the underlying loader.
    ///
    /// Useful for format-specific operations.
    pub fn loader(&self) -> &(dyn WeightLoader + Send + Sync) {
        &*self.inner.loader
    }

    /// Get the model configuration JSON.
    ///
    /// For SafeTensors, this is loaded from `config.json`.
    /// For GGUF, this is synthesized from embedded metadata.
    pub fn config_json(&self) -> &str {
        &self.inner.config_json
    }

    /// Check if a tensor exists.
    pub fn contains(&self, name: &str) -> bool {
        self.inner.loader.contains(name)
    }

    /// Check if this is a GGUF file.
    pub fn is_gguf(&self) -> bool {
        self.inner.is_gguf
    }

    // =========================================================================
    // Model Type Detection
    // =========================================================================

    /// Get the model_type from config.
    pub fn model_type(&self) -> Option<String> {
        let v: serde_json::Value = serde_json::from_str(&self.inner.config_json).ok()?;
        v.get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    }

    /// Check if this is a BERT model.
    pub fn is_bert(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("bert"))
    }

    /// Check if this is a DistilBERT model.
    pub fn is_distilbert(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("distilbert"))
    }

    /// Check if this is a RoBERTa model.
    pub fn is_roberta(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("roberta"))
    }

    /// Check if this is a DistilRoBERTa model.
    pub fn is_distilroberta(&self) -> bool {
        matches!(self.model_type(), Some(mt) if mt.eq_ignore_ascii_case("distilroberta"))
    }

    // =========================================================================
    // Tensor Loading - Callback Pattern (PREFERRED)
    // =========================================================================

    /// Process a tensor's raw bytes through a callback.
    ///
    /// **This is the recommended way to access tensor data.** The callback pattern
    /// ensures that the `TensorView` is consumed immediately, preventing mmap
    /// pages from being held resident longer than necessary.
    ///
    /// # Memory Efficiency
    ///
    /// Data remains on disk until accessed. Only the pages touched during the
    /// callback are loaded into RAM, and they can be evicted after use.
    ///
    /// # Arguments
    ///
    /// * `name` - Tensor name in HuggingFace format
    /// * `f` - Closure that processes the tensor and returns a result
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Build a linear layer without extra allocation
    /// let layer = weights.with_raw_tensor("layer.weight", |view| {
    ///     LinearLayer::from_bytes(&view.bytes, view.shape, view.dtype)
    /// })?;
    /// ```
    pub fn with_raw_tensor<R, F>(&self, name: &str, f: F) -> Result<R>
    where
        F: FnOnce(TensorView<'_>) -> Result<R>,
    {
        let view = self.inner.loader.get_raw(name)?;
        f(view)
    }

    /// Resolve the dtype of a tensor, optionally overriding.
    ///
    /// If `override_dtype` is provided, returns it directly.
    /// Otherwise, reads the tensor's native dtype from metadata.
    ///
    /// This is efficient - it only reads the tensor header, not the data.
    pub fn resolve_dtype(&self, tensor_name: &str, override_dtype: Option<DType>) -> Result<DType> {
        if let Some(dt) = override_dtype {
            return Ok(dt);
        }
        
        self.with_raw_tensor(tensor_name, |view| Ok(view.dtype))
    }

    // =========================================================================
    // Tensor Loading - Eager (DEPRECATED)
    // =========================================================================

    /// Get a typed CPU tensor.
    ///
    /// # ⚠️ Memory Warning
    ///
    /// This method eagerly loads the entire tensor into RAM. For large models,
    /// prefer using `with_raw_tensor` with a callback that processes data
    /// incrementally or builds GPU buffers directly.
    ///
    /// # When to Use
    ///
    /// - Small tensors (biases, layer norms)
    /// - CPU-only inference where you need the full array
    /// - Debugging/inspection
    ///
    /// # When to Avoid
    ///
    /// - Large weight matrices (attention projections, embeddings)
    /// - When building GPU buffers (use `with_raw_tensor` → GPU upload)
    pub fn get_typed_tensor(&self, name: &str) -> Result<CpuTensor> {
        self.with_raw_tensor(name, |raw| {
            if self.inner.is_gguf {
                let attn = self
                    .inner
                    .loader
                    .as_any()
                    .downcast_ref::<GgufLoader>()
                    .and_then(|g| g.attention_layout());
                raw_to_typed_gguf(raw, attn)
            } else {
                raw_to_typed(raw)
            }
        })
    }

    /// Get a 1D f32 array.
    ///
    /// # ⚠️ Memory Warning
    ///
    /// Eagerly loads the tensor into RAM and converts to f32.
    /// Only use for small tensors like biases.
    #[deprecated(
        since = "0.2.0",
        note = "Eagerly loads tensor into RAM. Use `with_raw_tensor` for memory-efficient access."
    )]
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array1_f32()
            .map_err(|e| anyhow!("Failed to load '{}': {}", name, e))
    }

    /// Get a 2D f32 array.
    ///
    /// # ⚠️ Memory Warning
    ///
    /// Eagerly loads the tensor into RAM and converts to f32.
    /// For a 4096x4096 weight matrix, this allocates ~64MB.
    /// Prefer `with_raw_tensor` for large matrices.
    #[deprecated(
        since = "0.2.0",
        note = "Eagerly loads tensor into RAM. Use `with_raw_tensor` for memory-efficient access."
    )]
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array2_f32()
            .map_err(|e| anyhow!("Failed to load '{}': {}", name, e))
    }

    /// Get a 3D f32 array.
    ///
    /// # ⚠️ Memory Warning
    ///
    /// Eagerly loads the tensor into RAM and converts to f32.
    /// Prefer `with_raw_tensor` for large tensors.
    #[deprecated(
        since = "0.2.0",
        note = "Eagerly loads tensor into RAM. Use `with_raw_tensor` for memory-efficient access."
    )]
    pub fn get_array3(&self, name: &str) -> Result<Array3<f32>> {
        #[allow(deprecated)]
        self.get_typed_tensor(name)?
            .to_array3_f32()
            .map_err(|e| anyhow!("Failed to load '{}': {}", name, e))
    }

    // =========================================================================
    // Efficient Accessors (for small tensors)
    // =========================================================================

    /// Get tensor shape without loading data.
    ///
    /// This is efficient - only reads tensor metadata.
    pub fn tensor_shape(&self, name: &str) -> Result<Vec<usize>> {
        self.with_raw_tensor(name, |view| Ok(view.shape.clone()))
    }

    /// Get tensor dtype without loading data.
    ///
    /// This is efficient - only reads tensor metadata.
    pub fn tensor_dtype(&self, name: &str) -> Result<DType> {
        self.with_raw_tensor(name, |view| Ok(view.dtype))
    }

    /// Get tensor size in bytes.
    ///
    /// This is efficient - only reads tensor metadata.
    pub fn tensor_size_bytes(&self, name: &str) -> Result<usize> {
        self.with_raw_tensor(name, |view| Ok(view.bytes.len()))
    }
}

// =============================================================================
// Tensor Conversion Utilities
// =============================================================================

/// Copy bytes to a typed vector, handling alignment.
///
/// Tries zero-copy cast first, falls back to byte-by-byte copy if misaligned.
///
/// # ⚠️ Memory Warning
///
/// This allocates a new Vec with the full tensor data.
/// Use only when you truly need an owned copy.
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        // Misaligned - copy to aligned buffer first
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

/// Convert a TensorView to a typed CpuTensor.
///
/// # ⚠️ Memory Warning
///
/// This allocates a new array with the full tensor data.
/// Prefer processing `TensorView.bytes` directly when possible.
pub fn raw_to_typed(raw: TensorView<'_>) -> Result<CpuTensor> {
    match raw.dtype {
        DType::F32 => Ok(CpuTensor::F32(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),
        
        DType::F16 => Ok(CpuTensor::F16(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),
        
        DType::BF16 => Ok(CpuTensor::BF16(ArrayD::from_shape_vec(
            IxDyn(&raw.shape),
            cast_or_copy(&raw.bytes),
        )?)),
        
        DType::Q8_0 => Ok(CpuTensor::Q8_0(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),
        
        DType::Q4_K => Ok(CpuTensor::Q4_K(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),
        
        DType::Q6_K => Ok(CpuTensor::Q6_K(QuantizedMatrix {
            blocks: cast_or_copy(&raw.bytes),
            shape: [raw.shape[0], raw.shape[1]],
        })),
        
        _ => Err(anyhow!("Unsupported dtype for conversion: {:?}", raw.dtype)),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    /// Helper to create a minimal safetensors file for testing.
 /// Helper to create a minimal safetensors file for testing.
    fn create_test_safetensors(dir: &TempDir, tensors: &[(&str, Vec<f32>, Vec<usize>)]) -> Result<()> {
        use safetensors::tensor::{Dtype, TensorView as StTensorView};
        use std::collections::HashMap;

        // Convert tensors to the format safetensors expects
        let stored: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(name, values, shape)| {
                let bytes: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();
                (name.to_string(), shape.clone(), bytes)
            })
            .collect();

        let mut tensor_map = HashMap::new();
        for (name, shape, bytes) in &stored {
            tensor_map.insert(
                name.clone(),
                StTensorView::new(Dtype::F32, shape.clone(), bytes)?,
            );
        }

        let file_path = dir.path().join("model.safetensors");
        safetensors::serialize_to_file(&tensor_map, &None, &file_path)?;

        // Write config.json
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"model_type": "bert", "hidden_size": 4}"#,
        )?;

        Ok(())
    }

    #[test]
    fn test_model_weights_new_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("test.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        assert!(!weights.is_gguf());
        assert!(weights.contains("test.weight"));
        assert!(!weights.contains("nonexistent"));
    }

    #[test]
    fn test_model_weights_with_raw_tensor() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("layer.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();

        let result = weights.with_raw_tensor("layer.weight", |view| {
            assert_eq!(view.shape, vec![2, 2]);
            assert_eq!(view.dtype, DType::F32);
            assert_eq!(view.bytes.len(), 16); // 4 floats * 4 bytes
            Ok(view.shape.iter().product::<usize>())
        }).unwrap();

        assert_eq!(result, 4);
    }

    #[test]
    fn test_tensor_shape_without_loading() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("big.weight", vec![0.0; 1000], vec![10, 100]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        
        // This should be cheap - doesn't load the data
        let shape = weights.tensor_shape("big.weight").unwrap();
        assert_eq!(shape, vec![10, 100]);
    }

    #[test]
    fn test_tensor_dtype() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("test.weight", vec![1.0], vec![1]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        let dtype = weights.tensor_dtype("test.weight").unwrap();
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_model_type_detection() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("test.weight", vec![1.0], vec![1]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        assert!(weights.is_bert());
        assert!(!weights.is_distilbert());
        assert!(!weights.is_roberta());
    }

    #[test]
    fn test_resolve_dtype_with_override() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("test.weight", vec![1.0], vec![1]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();

        // With override
        let dtype = weights.resolve_dtype("test.weight", Some(DType::F16)).unwrap();
        assert_eq!(dtype, DType::F16);

        // Without override - reads from tensor
        let dtype = weights.resolve_dtype("test.weight", None).unwrap();
        assert_eq!(dtype, DType::F32);
    }

    #[test]
    fn test_missing_tensor_error() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("exists.weight", vec![1.0], vec![1]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        
        let result = weights.with_raw_tensor("does_not_exist", |_| Ok(()));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_config_json_loaded() {
        let dir = tempfile::tempdir().unwrap();
        create_test_safetensors(&dir, &[
            ("test.weight", vec![1.0], vec![1]),
        ]).unwrap();

        let weights = ModelWeights::new(dir.path()).unwrap();
        let config = weights.config_json();
        
        assert!(config.contains("bert"));
        assert!(config.contains("hidden_size"));
    }

    #[test]
    fn test_cast_or_copy_aligned() {
        let floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        
        let result: Vec<f32> = cast_or_copy(&bytes);
        assert_eq!(result, floats);
    }

    #[test]
    fn test_cast_or_copy_misaligned() {
        let floats = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut bytes: Vec<u8> = vec![0]; // 1-byte offset
        bytes.extend(floats.iter().flat_map(|f| f.to_le_bytes()));
        
        // Skip first byte to get misaligned slice
        let result: Vec<f32> = cast_or_copy(&bytes[1..]);
        assert_eq!(result, floats);
    }
}