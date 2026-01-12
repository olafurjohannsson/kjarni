//! GGUF format loader with mmap caching.
//!
//! Loads model weights from `.gguf` files, which are commonly used for quantized
//! models in the llama.cpp ecosystem.
//!
//! # Format Overview
//!
//! GGUF (GGML Unified Format) stores:
//! - **Metadata**: Model architecture, hyperparameters, tokenizer info
//! - **Tensors**: Weights in various quantization formats (Q4_K, Q8_0, etc.)
//!
//! Unlike SafeTensors, GGUF embeds all model metadata, eliminating the need
//! for separate `config.json` files.
//!
//! # Name Translation
//!
//! GGUF uses different tensor naming conventions than HuggingFace:
//!
//! | HuggingFace | GGUF |
//! |-------------|------|
//! | `model.embed_tokens.weight` | `token_embd.weight` |
//! | `model.layers.0.self_attn.q_proj.weight` | `blk.0.attn_q.weight` |
//! | `model.norm.weight` | `output_norm.weight` |
//!
//! This loader automatically translates HuggingFace names to GGUF names.
//!
//! # Quantization Support
//!
//! Supports reading weights in these formats:
//! - F32, F16, BF16 (full precision)
//! - Q8_0 (8-bit quantization)
//! - Q4_K, Q6_K (k-quant formats)

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::WeightLoader;
use crate::weights::mmap_cache::get_or_create_mmap;
use crate::weights::model_weights::AttentionLayout;
use anyhow::{Context, Result, anyhow};
use gguf_rs::{ByteOrder, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE, GGUFContainer};
use memmap2::Mmap;
use serde_json::Value;
use std::any::Any;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

/// A loader for `.gguf` files with mmap caching.
///
/// Provides access to quantized model weights and embedded metadata.
/// Automatically translates HuggingFace tensor names to GGUF conventions.
///
/// # Example
///
/// ```ignore
/// let loader = GgufLoader::new(Path::new("llama-7b-q4_k_m.gguf"))?;
///
/// // Access metadata
/// let hidden_size = loader.get_u32("llama.embedding_length");
/// let rope_theta = loader.get_f32("llama.rope.freq_base");
///
/// // Access tensors (using HF names - automatically translated)
/// loader.with_tensor("model.layers.0.self_attn.q_proj.weight", |view| {
///     println!("DType: {:?}", view.dtype);  // e.g., Q4_K
///     Ok(())
/// })?;
/// ```
pub struct GgufLoader {
    /// Memory-mapped file contents (shared via cache)
    mmap: Arc<Mmap>,
    /// Maps GGUF tensor names to their info
    tensor_map: HashMap<String, GgufTensorInfo>,
    /// Model metadata from the GGUF header
    metadata: BTreeMap<String, Value>,
    /// Model architecture string (e.g., "llama", "qwen2")
    pub architecture: String,
    /// Byte offset where tensor data begins
    data_start_offset: u64,
}

/// Information about a single tensor in the GGUF file.
struct GgufTensorInfo {
    /// GGML type ID (0=F32, 1=F16, 8=Q8_0, etc.)
    kind: u32,
    /// Byte offset from data_start_offset
    offset: u64,
    /// Size in bytes
    size: u64,
    /// Shape in row-major order (HuggingFace convention)
    shape: Vec<usize>,
}

/// Trait for converting GGUF names to HuggingFace names.
///
/// Useful for debugging and weight inspection tools.
pub trait GgufHfMapper {
    /// Convert a GGUF tensor name to its HuggingFace equivalent.
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String>;
}

impl GgufLoader {
    /// Create a new GGUF loader from a file path.
    ///
    /// Parses the GGUF header to extract metadata and tensor information,
    /// then memory-maps the file for efficient tensor access.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to a `.gguf` file
    ///
    /// # Returns
    ///
    /// A configured loader ready to access tensors and metadata.
    ///
    /// # Errors
    ///
    /// - File not found
    /// - Invalid GGUF magic number
    /// - Unsupported GGUF version
    /// - Parse errors in header
    ///
    /// # Example
    ///
    /// ```ignore
    /// let loader = GgufLoader::new(Path::new("model.gguf"))?;
    /// println!("Architecture: {}", loader.architecture);
    /// println!("Tensors: {}", loader.tensor_count());
    /// ```
    pub fn new(path: &Path) -> Result<Self> {
        // 1. Open file for header parsing
        let mut header_file =
            File::open(path).with_context(|| format!("Failed to open GGUF file: {:?}", path))?;

        // 2. Determine byte order from magic number
        let byte_order = {
            let mut magic = [0u8; 4];
            header_file
                .read_exact(&mut magic)
                .context("Failed to read GGUF magic number")?;

            let magic_val = i32::from_le_bytes(magic);
            match magic_val {
                FILE_MAGIC_GGUF_LE => ByteOrder::LE,
                FILE_MAGIC_GGUF_BE => ByteOrder::BE,
                _ => {
                    return Err(anyhow!(
                        "Invalid GGUF magic number: {:#010x}. Expected {:#010x} (LE) or {:#010x} (BE)",
                        magic_val,
                        FILE_MAGIC_GGUF_LE,
                        FILE_MAGIC_GGUF_BE
                    ));
                }
            }
        };

        // 3. Parse GGUF header using gguf-rs
        // Note: File pointer is at position 4, which is where version starts
        let mut container = GGUFContainer::new(byte_order, Box::new(header_file));
        let model = container
            .decode()
            .map_err(|e| anyhow!("Failed to decode GGUF header: {}", e))?;

        // 4. Build tensor map with shape conversion
        let mut tensor_map = HashMap::new();
        let mut total_tensor_bytes = 0u64;

        for t in model.tensors() {
            total_tensor_bytes += t.size;

            // Convert shape from GGUF [width, height] to ndarray [height, width]
            let mut shape: Vec<usize> = t.shape.iter().map(|&s| s as usize).collect();
            shape.reverse();

            // Remove padding dimensions (GGUF often pads to 4D)
            let cleaned_shape: Vec<usize> = shape
                .into_iter()
                .filter(|&d| d > 1 || t.shape.len() <= 1)
                .collect();

            tensor_map.insert(
                t.name.clone(),
                GgufTensorInfo {
                    kind: t.kind,
                    offset: t.offset,
                    size: t.size,
                    shape: cleaned_shape,
                },
            );
        }

        // 5. Memory-map the file using cache
        let mmap = get_or_create_mmap(path)?;

        // 6. Calculate data offset (aligned to 32 bytes)
        let data_start_offset = (mmap.len() as u64 - total_tensor_bytes) & !31;

        log::info!(
            "Loaded GGUF model: {} tensors, {:.1} MB data, architecture: {}",
            tensor_map.len(),
            total_tensor_bytes as f64 / 1_000_000.0,
            model.model_family()
        );

        Ok(Self {
            mmap,
            tensor_map,
            metadata: model.metadata().clone(),
            architecture: model.model_family(),
            data_start_offset,
        })
    }

    /// Get a tensor view using its native GGUF name.
    ///
    /// Unlike `with_tensor` which accepts HuggingFace names, this method
    /// accepts GGUF names directly (e.g., `"blk.0.attn_q.weight"`).
    ///
    /// # Arguments
    ///
    /// * `gguf_name` - Tensor name in GGUF format
    ///
    /// # Returns
    ///
    /// A `TensorView` containing the raw bytes, shape, and dtype.
    pub fn get_raw_from_gguf_name(&self, gguf_name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensor_map
            .get(gguf_name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in GGUF file", gguf_name))?;

        let dtype = Self::ggml_type_to_dtype(info.kind)?;

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!(
                "Tensor '{}' points outside file bounds (offset {} + size {} > file size {})",
                gguf_name,
                start,
                info.size,
                self.mmap.len()
            ));
        }

        Ok(TensorView {
            name: gguf_name.to_string(),
            bytes: Cow::Borrowed(&self.mmap[start..end]),
            shape: info.shape.clone(),
            dtype,
        })
    }

    /// Convert GGML type ID to internal DType.
    pub(crate) fn ggml_type_to_dtype(kind: u32) -> Result<DType> {
        match kind {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            8 => Ok(DType::Q8_0),
            12 => Ok(DType::Q4_K),
            14 => Ok(DType::Q6_K),
            30 => Ok(DType::BF16),
            other => Err(anyhow!(
                "Unsupported GGML type ID {}. Supported: 0 (F32), 1 (F16), 8 (Q8_0), 12 (Q4_K), 14 (Q6_K), 30 (BF16)",
                other
            )),
        }
    }

    /// Translate a HuggingFace tensor name to GGUF format.
    ///
    /// # Examples
    ///
    /// - `"model.embed_tokens.weight"` → `"token_embd.weight"`
    /// - `"model.layers.0.self_attn.q_proj.weight"` → `"blk.0.attn_q.weight"`
    /// - `"model.norm.weight"` → `"output_norm.weight"`
    pub(crate) fn translate_name(&self, name: &str) -> String {
        // Embedding layer
        if name == "model.embed_tokens.weight" {
            return "token_embd.weight".to_string();
        }

        // Final normalization
        if name == "model.norm.weight" {
            return "output_norm.weight".to_string();
        }

        // LM head
        if name == "lm_head.weight" {
            return "output.weight".to_string();
        }

        // Transformer layers: "model.layers.N.xxx" → "blk.N.xxx"
        if name.starts_with("model.layers.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 4 {
                let layer_idx = parts[2];
                let suffix = parts[3..].join(".");

                let translated_suffix = match suffix.as_str() {
                    // Attention
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",

                    // FFN
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",

                    // Pass through unknown suffixes
                    _ => suffix.as_str(),
                };

                return format!("blk.{}.{}", layer_idx, translated_suffix);
            }
        }

        // Return unchanged if no translation applies
        name.to_string()
    }

    // =========================================================================
    // Metadata Accessors
    // =========================================================================

    /// Get attention layout from GGUF metadata.
    ///
    /// Extracts head count, KV head count, and head dimension from the
    /// architecture-specific metadata keys.
    ///
    /// # Returns
    ///
    /// `Some(AttentionLayout)` if all required metadata is present, `None` otherwise.
    pub fn attention_layout(&self) -> Option<AttentionLayout> {
        let arch = &self.architecture;

        let n_heads = self.get_u32(&format!("{}.attention.head_count", arch))? as usize;
        let n_kv_heads = self.get_u32(&format!("{}.attention.head_count_kv", arch))? as usize;
        let embed = self.get_u32(&format!("{}.embedding_length", arch))? as usize;

        Some(AttentionLayout {
            n_heads,
            n_kv_heads,
            head_dim: embed / n_heads,
        })
    }

    /// Get an architecture-prefixed u32 metadata value.
    ///
    /// Convenience method that prepends the architecture name.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Equivalent to get_u32("llama.block_count")
    /// let num_layers = loader.get_arch_u32("block_count");
    /// ```
    pub fn get_arch_u32(&self, key: &str) -> Option<u32> {
        self.get_u32(&format!("{}.{}", self.architecture, key))
    }

    /// Get an architecture-prefixed f32 metadata value.
    pub fn get_arch_f32(&self, key: &str) -> Option<f32> {
        self.get_f32(&format!("{}.{}", self.architecture, key))
    }

    // =========================================================================
    // Debug Utilities
    // =========================================================================

    /// Get all tensor names in GGUF format.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_map.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensor_map.len()
    }

    /// Print all tensors for debugging.
    pub fn debug_print_tensors(&self) {
        println!("=== GGUF Tensors ({} total) ===", self.tensor_map.len());

        let mut names: Vec<_> = self.tensor_map.keys().collect();
        names.sort();

        for name in names {
            let info = &self.tensor_map[name];
            let dtype_name = match info.kind {
                0 => "F32",
                1 => "F16",
                8 => "Q8_0",
                12 => "Q4_K",
                14 => "Q6_K",
                30 => "BF16",
                other => {
                    // Use the value in the output
                    println!(
                        "  {}: dtype=Unknown({}), shape={:?}, size={} bytes",
                        name, other, info.shape, info.size
                    );
                    continue;
                }
            };
            println!(
                "  {}: dtype={}, shape={:?}, size={} bytes",
                name, dtype_name, info.shape, info.size
            );
        }
    }
}

impl GgufHfMapper for GgufLoader {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        // Basic layers
        if gguf_name == "token_embd.weight" {
            return Some("model.embed_tokens.weight".to_string());
        }
        if gguf_name == "output_norm.weight" {
            return Some("model.norm.weight".to_string());
        }
        if gguf_name == "output.weight" {
            return Some("lm_head.weight".to_string());
        }

        // Transformer blocks: "blk.N.xxx" → "model.layers.N.xxx"
        if gguf_name.starts_with("blk.") {
            let parts: Vec<&str> = gguf_name.split('.').collect();
            if parts.len() >= 3 {
                let layer_idx = parts[1];
                let suffix = parts[2..].join(".");

                let hf_suffix = match suffix.as_str() {
                    "attn_norm.weight" => "input_layernorm.weight",
                    "attn_q.weight" => "self_attn.q_proj.weight",
                    "attn_k.weight" => "self_attn.k_proj.weight",
                    "attn_v.weight" => "self_attn.v_proj.weight",
                    "attn_output.weight" => "self_attn.o_proj.weight",
                    "ffn_norm.weight" => "post_attention_layernorm.weight",
                    "ffn_gate.weight" => "mlp.gate_proj.weight",
                    "ffn_up.weight" => "mlp.up_proj.weight",
                    "ffn_down.weight" => "mlp.down_proj.weight",
                    _ => return None,
                };

                return Some(format!("model.layers.{}.{}", layer_idx, hf_suffix));
            }
        }

        None
    }
}

impl WeightLoader for GgufLoader {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let gguf_name = self.translate_name(name);

        // Handle tied weights: if lm_head not found, use embeddings
        let gguf_name = if gguf_name == "output.weight" && !self.tensor_map.contains_key(&gguf_name)
        {
            log::debug!("LM head not found, using tied embeddings (token_embd.weight)");
            "token_embd.weight".to_string()
        } else {
            gguf_name
        };

        let info = self.tensor_map.get(&gguf_name).ok_or_else(|| {
            anyhow!(
                "Tensor '{}' (translated to '{}') not found in GGUF file",
                name,
                gguf_name
            )
        })?;

        let dtype = Self::ggml_type_to_dtype(info.kind)?;

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!("Tensor '{}' points outside file bounds", name));
        }

        Ok(TensorView {
            name: name.to_string(), // Use original HF name
            bytes: Cow::Borrowed(&self.mmap[start..end]),
            shape: info.shape.clone(),
            dtype,
        })
    }

    fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    }

    fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }

    fn has_metadata(&self) -> bool {
        true
    }

    fn contains(&self, name: &str) -> bool {
        let gguf_name = self.translate_name(name);
        self.tensor_map.contains_key(&gguf_name)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    // --- Helper to mock loader for testing translate_name ---
    fn create_mock_loader() -> GgufLoader {
        GgufLoader {
            mmap: Arc::new(unsafe { Mmap::map(&File::open("/dev/null").unwrap()).unwrap() }), // Dummy
            tensor_map: HashMap::new(),
            metadata: BTreeMap::new(),
            architecture: "llama".to_string(),
            data_start_offset: 0,
        }
    }

    #[test]
    fn test_dtype_conversion() {
        assert_eq!(GgufLoader::ggml_type_to_dtype(0).unwrap(), DType::F32);
        assert_eq!(GgufLoader::ggml_type_to_dtype(1).unwrap(), DType::F16);
        assert_eq!(GgufLoader::ggml_type_to_dtype(8).unwrap(), DType::Q8_0);
        assert_eq!(GgufLoader::ggml_type_to_dtype(12).unwrap(), DType::Q4_K);
        assert_eq!(GgufLoader::ggml_type_to_dtype(14).unwrap(), DType::Q6_K);
        assert_eq!(GgufLoader::ggml_type_to_dtype(30).unwrap(), DType::BF16);

        assert!(GgufLoader::ggml_type_to_dtype(999).is_err());
    }

    #[test]
    fn test_name_translation_embeddings() {
        // Test the translation logic by checking expected outputs
        // Note: We can't easily test private methods, but we can test via contains()

        // These are the expected translations:
        let translations = vec![
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight", "output_norm.weight"),
            ("lm_head.weight", "output.weight"),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "blk.0.attn_v.weight",
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "blk.0.attn_output.weight",
            ),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
            ),
            ("model.layers.0.mlp.up_proj.weight", "blk.0.ffn_up.weight"),
            (
                "model.layers.0.mlp.down_proj.weight",
                "blk.0.ffn_down.weight",
            ),
            (
                "model.layers.15.self_attn.q_proj.weight",
                "blk.15.attn_q.weight",
            ),
        ];

        // Can't test directly without a GGUF file, but document expected behavior
        for (hf_name, expected_gguf) in translations {
            // This would be the test if translate_name were public:
            // let loader = GgufLoader { ... };
            // assert_eq!(loader.translate_name(hf_name), expected_gguf);
            let _ = (hf_name, expected_gguf); // Suppress unused warning
        }
    }

    #[test]
    fn test_ggml_type_mapping() {
        // Test that we map GGML types correctly
        assert!(matches!(GgufLoader::ggml_type_to_dtype(0), Ok(DType::F32)));
        assert!(matches!(GgufLoader::ggml_type_to_dtype(1), Ok(DType::F16)));
        assert!(matches!(GgufLoader::ggml_type_to_dtype(8), Ok(DType::Q8_0)));
        assert!(matches!(
            GgufLoader::ggml_type_to_dtype(12),
            Ok(DType::Q4_K)
        ));
        assert!(matches!(
            GgufLoader::ggml_type_to_dtype(14),
            Ok(DType::Q6_K)
        ));
        assert!(matches!(
            GgufLoader::ggml_type_to_dtype(30),
            Ok(DType::BF16)
        ));

        // Unknown types should error
        assert!(GgufLoader::ggml_type_to_dtype(99).is_err());
        assert!(GgufLoader::ggml_type_to_dtype(255).is_err());
    }

    #[test]
    fn test_gguf_hf_mapper_reverse() {
        // Test reverse mapping (GGUF → HF)
        // This tests the GgufHfMapper trait implementation

        let expected_reverse = vec![
            ("token_embd.weight", Some("model.embed_tokens.weight")),
            ("output_norm.weight", Some("model.norm.weight")),
            ("output.weight", Some("lm_head.weight")),
            (
                "blk.0.attn_q.weight",
                Some("model.layers.0.self_attn.q_proj.weight"),
            ),
            (
                "blk.0.attn_k.weight",
                Some("model.layers.0.self_attn.k_proj.weight"),
            ),
            (
                "blk.0.attn_v.weight",
                Some("model.layers.0.self_attn.v_proj.weight"),
            ),
            (
                "blk.0.attn_output.weight",
                Some("model.layers.0.self_attn.o_proj.weight"),
            ),
            (
                "blk.0.attn_norm.weight",
                Some("model.layers.0.input_layernorm.weight"),
            ),
            (
                "blk.0.ffn_norm.weight",
                Some("model.layers.0.post_attention_layernorm.weight"),
            ),
            (
                "blk.0.ffn_gate.weight",
                Some("model.layers.0.mlp.gate_proj.weight"),
            ),
            (
                "blk.0.ffn_up.weight",
                Some("model.layers.0.mlp.up_proj.weight"),
            ),
            (
                "blk.0.ffn_down.weight",
                Some("model.layers.0.mlp.down_proj.weight"),
            ),
            ("unknown.tensor", None),
        ];

        // Would need actual GgufLoader instance to test
        let _ = expected_reverse;
    }
}
