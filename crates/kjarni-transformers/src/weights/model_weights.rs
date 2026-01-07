use super::{gguf_loader::GgufLoader, safetensors_loader::SafeTensorsLoader, WeightLoader};
use crate::cpu::kernels::q_common::{BlockQ4_K, BlockQ6_K, BlockQ8_0};
use crate::tensor::{
    dtype::DType,
    raw_tensor::TensorView,
    {CpuTensor, QuantizedMatrix},
};
use crate::weights::raw_to_typed_gguf;
use anyhow::{anyhow, Context, Result};
use half::{bf16, f16};
use ndarray::{Array1, Array2, Array3, ArrayD, IxDyn};
use serde_json::json;
use std::path::Path;

#[derive(Debug, Clone, Copy)]
pub struct AttentionLayout {
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

/// High-level interface for loading model weights from a directory.
///
/// This struct detects the weight format (`.safetensors`, `.gguf`, etc.) and provides
/// a consistent API for accessing tensors in various typed formats.
pub struct ModelWeights {
    pub loader: Box<dyn WeightLoader>,
    pub config_json: String,
    pub is_gguf: bool,
}

impl ModelWeights {
    pub fn new(path: &Path) -> Result<Self> {
        // 1. Check if the path is a direct GGUF file
        if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("gguf") {
            return Self::from_gguf_file(path);
        }

        // 2. If it's a directory, look for contents
        if path.is_dir() {
            // Check for Safetensors
            if path.join("model.safetensors").exists()
                || path.join("model.safetensors.index.json").exists()
            {
                let loader = Box::new(SafeTensorsLoader::new(path)?);
                let config_json = std::fs::read_to_string(path.join("config.json"))
                    .context("Safetensors model directory missing config.json")?;
                return Ok(Self {
                    loader,
                    config_json,
                    is_gguf: false,
                });
            }

            // Check for a GGUF file inside the directory (common for llama.cpp users)
            let gguf_in_dir = std::fs::read_dir(path)?
                .filter_map(|e| e.ok())
                .find(|e| e.path().extension().and_then(|s| s.to_str()) == Some("gguf"));

            if let Some(entry) = gguf_in_dir {
                return Self::from_gguf_file(&entry.path());
            }
        }
        Err(anyhow!("No supported weight format found at {:?}", path))
    }

    /// Determines the DType to use for a specific tensor.
    /// Priority: 1. User Override, 2. File's inherent DType
    pub fn resolve_dtype(&self, tensor_name: &str, override_dtype: Option<DType>) -> Result<DType> {
        if let Some(dt) = override_dtype {
            return Ok(dt);
        }
        // Probe the loader for the actual type in the file
        let raw = self.get_raw(tensor_name)?;
        Ok(raw.dtype)
    }

    /// Internal helper to load a GGUF and synthesize its configuration metadata.
    fn from_gguf_file(path: &Path) -> Result<Self> {
        let loader = GgufLoader::new(path)?;
        loader.debug_print_tensors();

        // GGUF stores configuration in its metadata.
        // We synthesize a JSON string that matches the LlamaConfig expected format.
        // This allows the rest of the pipeline to remain unchanged.
        let config_json = Self::synthesize_config_from_gguf(&loader)?;

        Ok(Self {
            loader: Box::new(loader),
            config_json,
            is_gguf: true,
        })
    }

    /// Maps GGUF metadata keys to standard HuggingFace config keys.
    fn synthesize_config_from_gguf(loader: &GgufLoader) -> Result<String> {
        // We use the model we decoded in the loader.
        // Note: You may need to expose a method on GgufLoader to get metadata values.

        // These keys are standard in Llama GGUF files.
        let arch = loader.get_string("general.architecture").unwrap_or("llama");
        let rope_scaling = if loader
            .get_string(&format!("{}.rope.scaling.type", arch))
            .is_some()
        {
            Some(json!({
                "rope_type": loader.get_string(&format!("{}.rope.scaling.type", arch)).unwrap_or("llama3"),
                "factor": loader.get_f32(&format!("{}.rope.scaling.factor", arch)).unwrap_or(32.0),
                "low_freq_factor": loader.get_f32(&format!("{}.rope.scaling.low_freq_factor", arch)).unwrap_or(1.0),
                "high_freq_factor": loader.get_f32(&format!("{}.rope.scaling.high_freq_factor", arch)).unwrap_or(4.0),
                "original_max_position_embeddings": loader.get_u32(&format!("{}.rope.scaling.orig_ctx_len", arch)).unwrap_or(8192),
            }))
        } else {
            None
        };
        // Map GGUF keys to HF JSON keys
        let config = json!({
            "architecture": arch,
            "hidden_size": loader.get_u32(&format!("{}.embedding_length", arch)),
            "intermediate_size": loader.get_u32(&format!("{}.feed_forward_length", arch)),
            "num_attention_heads": loader.get_u32(&format!("{}.attention.head_count", arch)),
            "num_hidden_layers": loader.get_u32(&format!("{}.block_count", arch)),
            "num_key_value_heads": loader.get_u32(&format!("{}.attention.head_count_kv", arch)),
            "max_position_embeddings": loader.get_u32(&format!("{}.context_length", arch)),
            "rope_theta": loader.get_f32(&format!("{}.rope.freq_base", arch)).unwrap_or(1e-5), // Some map to eps
            "rope_scaling": rope_scaling,
            "rms_norm_eps": loader.get_f32(&format!("{}.attention.layer_norm_rms_epsilon", arch)).unwrap_or(1e-5),
            "vocab_size": loader.get_u32(&format!("{}.vocab_size", arch)).unwrap_or(128256),

            "bos_token_id": loader.get_u32(&format!("{}.bos_token_id", arch)).unwrap_or(128000),
            "eos_token_id": loader.get_u32(&format!("{}.eos_token_id", arch)).unwrap_or(128001),
        });
        let json_str = config.to_string();
        println!("Synthesized GGUF config: {}", json_str);
        Ok(json_str)
    }

    fn parsed_vocab_size(&self) -> usize {
        serde_json::from_str::<serde_json::Value>(&self.config_json)
            .ok()
            .and_then(|v| v["vocab_size"].as_u64())
            .map(|v| v as usize)
            .unwrap_or(128256) // Fallback if parsing fails
    }

    /// Checks if a tensor with the given name exists in the model files.
    pub fn contains(&self, name: &str) -> bool {
        self.loader.contains(name)
    }

    /// Gets a raw, untyped view of a tensor's bytes.
    ///
    /// This is a low-level function intended for advanced use cases like direct
    /// GPU DMA transfers where you need a pointer to the raw data in the mmap'd file.
    pub fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        self.loader.get_raw(name)
    }

    /// Gets a typed tensor, preserving its original, memory-efficient dtype.
    ///
    /// This is the primary and most efficient method for loading tensors for CPU computation,
    /// as it avoids unnecessary type conversions and allocations.
    pub fn get_typed_tensor(&self, name: &str) -> Result<CpuTensor> {
        let raw = self.loader.get_raw(name)?;
        if self.is_gguf {
            let attn = self
                .loader
                .as_any()
                .downcast_ref::<GgufLoader>()
                .and_then(|g| g.attention_layout());

            return raw_to_typed_gguf(raw, attn);
        }

        raw_to_typed(raw)
    }

    // --- High-level accessors (use with care, as they may convert and allocate) ---

    /// Loads a tensor and converts it to `Array1<f32>`.
    ///
    /// This is a convenience method for small 1D tensors like biases or norms.
    /// It will fail if the tensor is not 1D or is a quantized matrix type.
    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed
            .to_array1_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array1<f32>: {}", name, e))
    }

    /// Loads a tensor and converts it to `Array2<f32>`.
    ///
    /// **Warning:** This will perform a slow, full dequantization for quantized types
    /// and should only be used for debugging or for models that are entirely F32/BF16/F16.
    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed
            .to_array2_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array2<f32>: {}", name, e))
    }

    pub fn get_array3(&self, name: &str) -> Result<Array3<f32>> {
        let typed = self.get_typed_tensor(name)?;
        typed
            .to_array3_f32()
            .map_err(|e| anyhow!("Failed to load '{}' as Array3<f32>: {}", name, e))
    }
}

/// Converts a `TensorView` into a `CpuTensor`, performing the necessary parsing.
/// Safely cast bytes to a typed slice, handling unaligned data
pub fn cast_or_copy<T: bytemuck::Pod + bytemuck::Zeroable>(bytes: &[u8]) -> Vec<T> {
    if let Ok(slice) = bytemuck::try_cast_slice(bytes) {
        slice.to_vec()
    } else {
        // Unaligned: copy to aligned buffer first
        let mut aligned = vec![0u8; bytes.len()];
        aligned.copy_from_slice(bytes);
        bytemuck::cast_slice(&aligned).to_vec()
    }
}

pub fn raw_to_typed(raw: TensorView<'_>) -> Result<CpuTensor> {
    match raw.dtype {
        DType::F32 => {
            let data: Vec<f32> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F32(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::F16 => {
            let data: Vec<f16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::F16(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::BF16 => {
            let data: Vec<bf16> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::BF16(ArrayD::from_shape_vec(
                IxDyn(&raw.shape),
                data,
            )?))
        }
        DType::Q8_0 => {
            let blocks: Vec<BlockQ8_0> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::Q8_0(QuantizedMatrix {
                blocks,
                shape: [raw.shape[0], raw.shape[1]],
            }))
        }
        DType::Q4_K => {
            let blocks: Vec<BlockQ4_K> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::Q4_K(QuantizedMatrix {
                blocks,
                shape: [raw.shape[0], raw.shape[1]],
            }))
        }
        DType::Q6_K => {
            let blocks: Vec<BlockQ6_K> = cast_or_copy(&raw.bytes);
            Ok(CpuTensor::Q6_K(QuantizedMatrix {
                blocks,
                shape: [raw.shape[0], raw.shape[1]],
            }))
        }
        _ => Err(anyhow!("Unsupported dtype: {:?}", raw.dtype)),
    }
}
