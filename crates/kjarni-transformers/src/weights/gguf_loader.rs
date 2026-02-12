//! GGUF format loader with mmap caching

use std::any::Any;
use std::borrow::Cow;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use gguf_rs::{ByteOrder, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE, GGUFContainer};
use memmap2::Mmap;
use serde_json::Value;

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::WeightLoader;
use crate::weights::mmap_cache::get_or_create_mmap;
use crate::weights::model_weights::AttentionLayout;

/// A loader for `.gguf` files with mmap caching
pub struct GgufLoader {
    mmap: Arc<Mmap>,
    tensor_map: HashMap<String, GgufTensorInfo>,
    metadata: BTreeMap<String, Value>,
    pub architecture: String,
    data_start_offset: u64,
}

struct GgufTensorInfo {
    kind: u32,
    offset: u64,
    size: u64,
    shape: Vec<usize>,
}

/// Converts GGUF names to HuggingFace names.
pub trait GgufHfMapper {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String>;
}

impl GgufLoader {
    /// Creates a new GGUF loader from a file path.
    pub fn new(path: &Path) -> Result<Self> {
        let mut header_file =
            File::open(path).with_context(|| format!("failed to open GGUF file: {:?}", path))?;

        let byte_order = {
            let mut magic = [0u8; 4];
            header_file
                .read_exact(&mut magic)
                .context("failed to read GGUF magic number")?;

            let magic_val = i32::from_le_bytes(magic);
            match magic_val {
                FILE_MAGIC_GGUF_LE => ByteOrder::LE,
                FILE_MAGIC_GGUF_BE => ByteOrder::BE,
                _ => {
                    return Err(anyhow!("invalid GGUF magic number: {:#010x}", magic_val));
                }
            }
        };

        let max_array_size = usize::MAX as u64;
        let mut container = GGUFContainer::new(byte_order, Box::new(header_file), max_array_size);
        let model = container
            .decode()
            .map_err(|e| anyhow!("failed to decode GGUF header: {}", e))?;

        let mut tensor_map = HashMap::new();
        let mut total_tensor_bytes = 0u64;

        for t in model.tensors() {
            total_tensor_bytes += t.size;

            let mut shape: Vec<usize> = t.shape.iter().map(|&s| s as usize).collect();
            shape.reverse();

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

        let mmap = get_or_create_mmap(path)?;
        let data_start_offset = (mmap.len() as u64 - total_tensor_bytes) & !31;

        log::info!(
            "loaded GGUF model: {} tensors, {:.1} MB data, architecture: {}",
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

    /// Returns a tensor view using its native GGUF name.
    pub fn get_raw_from_gguf_name(&self, gguf_name: &str) -> Result<TensorView<'_>> {
        let info = self
            .tensor_map
            .get(gguf_name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in GGUF file", gguf_name))?;

        let dtype = Self::ggml_type_to_dtype(info.kind)?;

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!("tensor '{}' points outside file bounds", gguf_name));
        }

        Ok(TensorView {
            name: gguf_name.to_string(),
            bytes: Cow::Borrowed(&self.mmap[start..end]),
            shape: info.shape.clone(),
            dtype,
        })
    }

    pub(crate) fn ggml_type_to_dtype(kind: u32) -> Result<DType> {
        match kind {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            8 => Ok(DType::Q8_0),
            12 => Ok(DType::Q4_K),
            14 => Ok(DType::Q6_K),
            30 => Ok(DType::BF16),
            other => Err(anyhow!("unsupported GGML type ID {}", other)),
        }
    }

    pub(crate) fn translate_name(&self, name: &str) -> String {
        if name == "model.embed_tokens.weight" {
            return "token_embd.weight".to_string();
        }
        if name == "model.norm.weight" {
            return "output_norm.weight".to_string();
        }
        if name == "lm_head.weight" {
            return "output.weight".to_string();
        }

        if name.starts_with("model.layers.") {
            let parts: Vec<&str> = name.split('.').collect();
            if parts.len() >= 4 {
                let layer_idx = parts[2];
                let suffix = parts[3..].join(".");

                let translated_suffix = match suffix.as_str() {
                    "input_layernorm.weight" => "attn_norm.weight",
                    "self_attn.q_proj.weight" => "attn_q.weight",
                    "self_attn.k_proj.weight" => "attn_k.weight",
                    "self_attn.v_proj.weight" => "attn_v.weight",
                    "self_attn.o_proj.weight" => "attn_output.weight",
                    "post_attention_layernorm.weight" => "ffn_norm.weight",
                    "mlp.gate_proj.weight" => "ffn_gate.weight",
                    "mlp.up_proj.weight" => "ffn_up.weight",
                    "mlp.down_proj.weight" => "ffn_down.weight",
                    _ => suffix.as_str(),
                };

                return format!("blk.{}.{}", layer_idx, translated_suffix);
            }
        }

        name.to_string()
    }

    /// Returns attention layout from GGUF metadata.
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

    /// Returns an architecture-prefixed u32 metadata value.
    pub fn get_arch_u32(&self, key: &str) -> Option<u32> {
        self.get_u32(&format!("{}.{}", self.architecture, key))
    }

    /// Returns an architecture-prefixed f32 metadata value.
    pub fn get_arch_f32(&self, key: &str) -> Option<f32> {
        self.get_f32(&format!("{}.{}", self.architecture, key))
    }

    /// Returns all tensor names in GGUF format.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_map.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensor_map.len()
    }

    /// Prints all tensors for debugging.
    pub fn debug_print_tensors(&self) {
        println!("=== GGUF tensors ({} total) ===", self.tensor_map.len());

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
        if gguf_name == "token_embd.weight" {
            return Some("model.embed_tokens.weight".to_string());
        }
        if gguf_name == "output_norm.weight" {
            return Some("model.norm.weight".to_string());
        }
        if gguf_name == "output.weight" {
            return Some("lm_head.weight".to_string());
        }

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

        let gguf_name = if gguf_name == "output.weight" && !self.tensor_map.contains_key(&gguf_name)
        {
            log::debug!("lm_head not found, using tied embeddings");
            "token_embd.weight".to_string()
        } else {
            gguf_name
        };

        let info = self.tensor_map.get(&gguf_name).ok_or_else(|| {
            anyhow!(
                "tensor '{}' (translated to '{}') not found",
                name,
                gguf_name
            )
        })?;

        let dtype = Self::ggml_type_to_dtype(info.kind)?;

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!("tensor '{}' points outside file bounds", name));
        }

        Ok(TensorView {
            name: name.to_string(),
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
    fn test_gguf_to_hf_reverse_mapping() {
        // Create a minimal valid mmap from temp file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &[0u8; 64]).unwrap();
        let file = File::open(temp_file.path()).unwrap();

        let loader = GgufLoader {
            mmap: Arc::new(unsafe { Mmap::map(&file).unwrap() }),
            tensor_map: HashMap::new(),
            metadata: BTreeMap::new(),
            architecture: "llama".to_string(),
            data_start_offset: 0,
        };

        assert_eq!(
            loader.gguf_to_hf_name("token_embd.weight"),
            Some("model.embed_tokens.weight".to_string())
        );
        assert_eq!(
            loader.gguf_to_hf_name("blk.0.attn_q.weight"),
            Some("model.layers.0.self_attn.q_proj.weight".to_string())
        );
        assert_eq!(loader.gguf_to_hf_name("unknown.tensor"), None);
    }

    #[test]
    fn test_name_translation() {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), &[0u8; 64]).unwrap();
        let file = File::open(temp_file.path()).unwrap();

        let loader = GgufLoader {
            mmap: Arc::new(unsafe { Mmap::map(&file).unwrap() }),
            tensor_map: HashMap::new(),
            metadata: BTreeMap::new(),
            architecture: "llama".to_string(),
            data_start_offset: 0,
        };

        assert_eq!(
            loader.translate_name("model.embed_tokens.weight"),
            "token_embd.weight"
        );
        assert_eq!(
            loader.translate_name("model.norm.weight"),
            "output_norm.weight"
        );
        assert_eq!(loader.translate_name("lm_head.weight"), "output.weight");
        assert_eq!(
            loader.translate_name("model.layers.0.self_attn.q_proj.weight"),
            "blk.0.attn_q.weight"
        );
        assert_eq!(
            loader.translate_name("model.layers.15.mlp.down_proj.weight"),
            "blk.15.ffn_down.weight"
        );
    }
}
