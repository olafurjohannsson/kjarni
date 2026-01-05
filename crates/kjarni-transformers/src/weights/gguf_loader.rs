use crate::tensor::{DType, TensorView};
use crate::weights::WeightLoader;
use crate::weights::model_weights::AttentionLayout;
use anyhow::{Context, Result, anyhow};
use gguf_rs::{
    ByteOrder, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE, GGUFContainer
};
use memmap2::Mmap;
use serde_json::Value;
use std::any::Any;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read};
use std::path::Path;

pub struct GgufLoader {
    mmap: Mmap,
    tensor_map: HashMap<String, GgufTensorInfo>,
    metadata: BTreeMap<String, Value>,
    pub architecture: String,
    data_start_offset: u64,
}
pub trait GgufHfMapper {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String>;
}

struct GgufTensorInfo {
    kind: u32,
    offset: u64,
    size: u64,
    shape: Vec<usize>,
}

impl GgufLoader {
    pub fn get_raw_from_gguf_name(&self, gguf_name: &str) -> Result<TensorView<'_>> {
        let info = self.tensor_map.get(gguf_name).ok_or_else(|| {
            anyhow!(
                "Tensor with GGUF name '{}' not found in tensor map",
                gguf_name
            )
        })?;

        // Mapping GGML Type IDs to your internal DType
        let dtype = match info.kind {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            12 => DType::Q4_K,
            14 => DType::Q6_K,
            30 => DType::BF16,
            other => return Err(anyhow!("GGUF Loader: Unsupported GGML type ID {}", other)),
        };

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!(
                "Tensor '{}' points outside of file bounds",
                gguf_name
            ));
        }

        Ok(TensorView {
            name: gguf_name.to_string(),
            bytes: Cow::Borrowed(&self.mmap[start..end]),
            shape: info.shape.clone(),
            dtype,
        })
    }
}

impl GgufHfMapper for GgufLoader {
    fn gguf_to_hf_name(&self, gguf_name: &str) -> Option<String> {
        // This is the inverse of your `translate_name` function.
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
impl GgufLoader {
    pub fn new(path: &Path) -> Result<Self> {
        // 1. Open the file for header parsing
        let mut header_file = File::open(path).context("Failed to open GGUF file for header")?;

        // 2. Determine ByteOrder by reading first 4 bytes
        let bo = {
            let mut magic = [0u8; 4];
            header_file
                .read_exact(&mut magic)
                .context("Failed to read GGUF magic")?;
            let magic_val = i32::from_le_bytes(magic);
            match magic_val {
                FILE_MAGIC_GGUF_LE => ByteOrder::LE,
                FILE_MAGIC_GGUF_BE => ByteOrder::BE,
                _ => return Err(anyhow!("Invalid GGUF magic")),
            }
        };

        // 3. Hand the file to gguf-rs.
        // The file pointer is already at position 4, which is exactly where
        // the library expects to start reading the version number.
        let mut container = GGUFContainer::new(bo, Box::new(header_file));
        let model = container
            .decode()
            .map_err(|e| anyhow!("GGUF Decode Error: {}", e))?;

        // 4. Map tensors and metadata
        let mut tensor_map = HashMap::new();
        let mut total_tensor_bytes = 0u64;

        for t in model.tensors() {
            total_tensor_bytes += t.size;

            // REVERSE the shape from GGUF [width, height] to ndarray [height, width]
            let mut shape: Vec<usize> = t.shape.iter().map(|&s| s as usize).collect();
            shape.reverse();

            // Filter out 1s (GGUF often pads shapes to 4D, e.g., [2048, 128256, 1, 1])
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
        let metadata = model.metadata().clone();

        // 5. Create a SEPARATE file handle for the memory mapping
        // This avoids any lifetime or pointer conflicts.
        let mmap_file = File::open(path).context("Failed to open GGUF file for mmap")?;
        let mmap = unsafe { Mmap::map(&mmap_file)? };

        // Calculate data offset
        let data_start_offset = (mmap.len() as u64 - total_tensor_bytes) & !31;

        log::info!(
            "Successfully loaded GGUF model: {} tensors",
            tensor_map.len()
        );

        Ok(Self {
            mmap,
            tensor_map,
            metadata,
            architecture: model.model_family(),
            data_start_offset,
        })
    }
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
    /// Generic getter that prepends the architecture prefix automatically
    pub fn get_arch_u32(&self, key: &str) -> Option<u32> {
        self.get_u32(&format!("{}.{}", self.architecture, key))
    }

    pub fn get_arch_f32(&self, key: &str) -> Option<f32> {
        self.get_f32(&format!("{}.{}", self.architecture, key))
    }
    fn translate_name(&self, name: &str) -> String {
        // 1. Handle the basics
        if name == "model.embed_tokens.weight" {
            return "token_embd.weight".to_string();
        }
        if name == "model.norm.weight" {
            return "output_norm.weight".to_string();
        }
        if name == "lm_head.weight" {
            return "output.weight".to_string();
        }

        // 2. Handle the layers: "model.layers.N.xxx" -> "blk.N.xxx"
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
                    _ => suffix.as_str(), // Fallback
                };
                return format!("blk.{}.{}", layer_idx, translated_suffix);
            }
        }

        name.to_string()
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_map.keys().map(|s| s.as_str()).collect()
    }

    /// Debug: print all tensors with their info
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
                k => &format!("Unknown({})", k),
            };
            println!(
                "  {}: dtype={}, shape={:?}, size={} bytes",
                name, dtype_name, info.shape, info.size
            );
        }
    }
}

impl WeightLoader for GgufLoader {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let gguf_name = self.translate_name(name);
        let gguf_name = if gguf_name == "output.weight" && !self.tensor_map.contains_key(&gguf_name)
        {
            log::debug!("LM head not found, using tied embeddings (token_embd.weight)");
            "token_embd.weight".to_string()
        } else {
            gguf_name
        };
        let info = self.tensor_map.get(&gguf_name).ok_or_else(|| {
            anyhow!(
                "Tensor '{}' (translated to '{}') not found in GGUF",
                name,
                gguf_name
            )
        })?;

        // Mapping GGML Type IDs to your internal DType
        let dtype = match info.kind {
            0 => DType::F32,
            1 => DType::F16,
            8 => DType::Q8_0,
            12 => DType::Q4_K,
            14 => DType::Q6_K,
            30 => DType::BF16,
            other => return Err(anyhow!("GGUF Loader: Unsupported GGML type ID {}", other)),
        };

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        // Safety check to prevent out-of-bounds
        if end > self.mmap.len() {
            return Err(anyhow!("Tensor '{}' points outside of file bounds", name));
        }

        Ok(TensorView {
            name: name.to_string(),
            bytes: Cow::Borrowed(&self.mmap[start..end]),
            shape: info.shape.clone(),
            dtype,
        })
    }

    /// Gets a string value from metadata
    fn get_string(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).and_then(|v| v.as_str())
    }

    /// Gets a u32 value from metadata (casting from JSON u64)
    fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_u64())
            .map(|v| v as u32)
    }

    /// Gets a f32 value from metadata (casting from JSON f64)
    fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata
            .get(key)
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
    }
    fn contains(&self, name: &str) -> bool {
        let gguf_name = self.translate_name(name);
        self.tensor_map.contains_key(gguf_name.as_str())
    }
}
