use crate::tensor::{DType, RawTensor};
use crate::weights::WeightLoader;
use anyhow::{anyhow, Context, Result};
// Import your gguf-rs components
use byteorder::{LittleEndian, ReadBytesExt};
use gguf_rs::{
    get_gguf_container, ByteOrder, GGUFContainer, GGUFModel, FILE_MAGIC_GGUF_BE, FILE_MAGIC_GGUF_LE,
};
use memmap2::Mmap;
use serde_json::Value;
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read, Seek, SeekFrom};
use std::path::Path;

pub struct GgufLoader {
    mmap: Mmap,
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
}

impl WeightLoader for GgufLoader {
    fn get_raw(&self, name: &str) -> Result<RawTensor<'_>> {
        let gguf_name = self.translate_name(name);

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

        Ok(RawTensor {
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
        self.tensor_map.contains_key(name)
    }
}
