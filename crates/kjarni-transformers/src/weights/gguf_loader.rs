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

        self.view(gguf_name, gguf_name, info)
    }

    /// The bytes of one tensor, as the dtype [`Self::ggml_type_to_dtype`] reports.
    fn view<'a>(
        &'a self,
        name: &str,
        gguf_name: &str,
        info: &GgufTensorInfo,
    ) -> Result<TensorView<'a>> {
        let dtype = Self::ggml_type_to_dtype(info.kind)?;

        let start = (self.data_start_offset + info.offset) as usize;
        let end = start + info.size as usize;

        if end > self.mmap.len() {
            return Err(anyhow!("tensor '{}' points outside file bounds", gguf_name));
        }
        let raw = &self.mmap[start..end];

        let bytes = match info.kind {
            GGML_Q4_0 => Cow::Owned(transcode_to_q8_0(raw, Q4_0_BLOCK, decode_q4_0, gguf_name)?),
            GGML_Q5_0 => Cow::Owned(transcode_to_q8_0(raw, Q5_0_BLOCK, decode_q5_0, gguf_name)?),
            _ => Cow::Borrowed(raw),
        };

        Ok(TensorView {
            name: name.to_string(),
            bytes,
            shape: info.shape.clone(),
            dtype,
        })
    }

    pub(crate) fn ggml_type_to_dtype(kind: u32) -> Result<DType> {
        match kind {
            0 => Ok(DType::F32),
            1 => Ok(DType::F16),
            // Converted to Q8_0 as they are read; see `transcode_to_q8_0`.
            GGML_Q4_0 | GGML_Q5_0 => Ok(DType::Q8_0),
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
                    // Qwen2 has Q/K/V biases. Without these four the lookup
                    // missed, the loader took the biases to be absent, and
                    // Qwen GGUFs ran without them and generated noise.
                    "self_attn.q_proj.bias" => "attn_q.bias",
                    "self_attn.k_proj.bias" => "attn_k.bias",
                    "self_attn.v_proj.bias" => "attn_v.bias",
                    "self_attn.o_proj.bias" => "attn_output.bias",
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
    /// The attention layout to undo llama.cpp's Q/K row permutation with, or
    /// `None` when the file has none to undo.
    ///
    /// llama.cpp's converter permutes the query and key rows (to suit its RoPE
    /// layout) only for the `llama` architecture, which Mistral files also use.
    /// Qwen2, Phi-3 and the rest are stored as Hugging Face has them. Un-permuting
    /// those anyway scrambled every attention head: Qwen2.5 GGUFs loaded and
    /// generated noise.
    pub fn attention_layout(&self) -> Option<AttentionLayout> {
        let arch = &self.architecture;
        if arch != "llama" {
            return None;
        }

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
    /// Every metadata key/value in the file, in key order.
    ///
    /// GGUF carries the whole config as metadata, which is what makes it possible to
    /// write a `Config` for a new architecture without guessing.
    pub fn metadata(&self) -> &BTreeMap<String, Value> {
        &self.metadata
    }

    /// dtype and shape of a tensor, without reading its data.
    pub fn tensor_info(&self, name: &str) -> Option<(DType, Vec<usize>)> {
        let info = self.tensor_map.get(name)?;
        let dtype = Self::ggml_type_to_dtype(info.kind).ok()?;
        Some((dtype, info.shape.clone()))
    }

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
                    "attn_q.bias" => "self_attn.q_proj.bias",
                    "attn_k.bias" => "self_attn.k_proj.bias",
                    "attn_v.bias" => "self_attn.v_proj.bias",
                    "attn_output.bias" => "self_attn.o_proj.bias",
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
    fn release(&self, bytes: &[u8]) {
        crate::weights::mmap_cache::release_pages(&self.mmap, bytes);
    }

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

        self.view(name, &gguf_name, info)
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

const GGML_Q4_0: u32 = 2;
const GGML_Q5_0: u32 = 6;
/// Bytes per 32-weight block: an f16 scale, then the packed weights.
const Q4_0_BLOCK: usize = 2 + 16;
const Q5_0_BLOCK: usize = 2 + 4 + 16;
const Q8_0_BLOCK: usize = 2 + 32;

/// Rewrites Q4_0 or Q5_0 blocks as Q8_0, without loss.
///
/// Both store 32 small integers per block and one f16 scale: Q4_0 in [-8, 7],
/// Q5_0 in [-16, 15]. Q8_0 stores 32 int8s and one f16 scale, so the same scale
/// and the same integers are an exact Q8_0 block, and every Q8_0 kernel then
/// applies unchanged. Q4_K_M files use Q5_0 for tensors whose rows are not a
/// multiple of 256, which on Qwen2.5 0.5B (hidden size 896) is most of them;
/// without this, no such file could be loaded at all.
fn transcode_to_q8_0(
    raw: &[u8],
    block: usize,
    decode: fn(&[u8], &mut [i8; 32]),
    name: &str,
) -> Result<Vec<u8>> {
    if !raw.len().is_multiple_of(block) {
        return Err(anyhow!(
            "tensor '{name}' is {} bytes, not a whole number of {block}-byte blocks",
            raw.len()
        ));
    }
    let mut out = Vec::with_capacity(raw.len() / block * Q8_0_BLOCK);
    let mut qs = [0i8; 32];
    for b in raw.chunks_exact(block) {
        decode(b, &mut qs);
        out.extend_from_slice(&b[..2]); // the f16 scale, unchanged
        out.extend(qs.iter().map(|&q| q as u8));
    }
    Ok(out)
}

/// ggml's Q4_0: 16 bytes of nibbles, low nibbles first, offset by 8.
fn decode_q4_0(b: &[u8], out: &mut [i8; 32]) {
    let qs = &b[2..18];
    for j in 0..16 {
        out[j] = (qs[j] & 0x0F) as i8 - 8;
        out[j + 16] = (qs[j] >> 4) as i8 - 8;
    }
}

/// ggml's Q5_0: as Q4_0 plus a fifth bit per weight in a little-endian u32,
/// offset by 16.
fn decode_q5_0(b: &[u8], out: &mut [i8; 32]) {
    let qh = u32::from_le_bytes([b[2], b[3], b[4], b[5]]);
    let qs = &b[6..22];
    for j in 0..16 {
        let hi0 = (((qh >> j) << 4) & 0x10) as u8;
        let hi1 = ((qh >> (j + 12)) & 0x10) as u8;
        out[j] = ((qs[j] & 0x0F) | hi0) as i8 - 16;
        out[j + 16] = ((qs[j] >> 4) | hi1) as i8 - 16;
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
        assert_eq!(
            GgufLoader::ggml_type_to_dtype(GGML_Q4_0).unwrap(),
            DType::Q8_0
        );
        assert_eq!(
            GgufLoader::ggml_type_to_dtype(GGML_Q5_0).unwrap(),
            DType::Q8_0
        );
        assert_eq!(GgufLoader::ggml_type_to_dtype(12).unwrap(), DType::Q4_K);
        assert_eq!(GgufLoader::ggml_type_to_dtype(14).unwrap(), DType::Q6_K);
        assert_eq!(GgufLoader::ggml_type_to_dtype(30).unwrap(), DType::BF16);
        assert!(GgufLoader::ggml_type_to_dtype(999).is_err());
    }

    #[test]
    fn test_gguf_to_hf_reverse_mapping() {
        // Create a minimal valid mmap from temp file
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(temp_file.path(), [0u8; 64]).unwrap();
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
        std::fs::write(temp_file.path(), [0u8; 64]).unwrap();
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

        // Qwen2's attention biases, both ways.
        for (hf, gguf) in [
            ("model.layers.3.self_attn.q_proj.bias", "blk.3.attn_q.bias"),
            ("model.layers.3.self_attn.k_proj.bias", "blk.3.attn_k.bias"),
            ("model.layers.3.self_attn.v_proj.bias", "blk.3.attn_v.bias"),
        ] {
            assert_eq!(loader.translate_name(hf), gguf);
            assert_eq!(loader.gguf_to_hf_name(gguf).as_deref(), Some(hf));
        }
    }

    /// Packs 32 integers the way ggml's quantize_row_q5_0 does.
    fn pack_q5_0(scale: half::f16, v: &[i8; 32]) -> Vec<u8> {
        let mut qh = 0u32;
        let mut qs = [0u8; 16];
        for j in 0..16 {
            let (x0, x1) = ((v[j] + 16) as u8, (v[j + 16] + 16) as u8);
            qs[j] = (x0 & 0x0F) | ((x1 & 0x0F) << 4);
            qh |= u32::from((x0 & 0x10) >> 4) << j;
            qh |= u32::from((x1 & 0x10) >> 4) << (j + 16);
        }
        let mut b = scale.to_le_bytes().to_vec();
        b.extend_from_slice(&qh.to_le_bytes());
        b.extend_from_slice(&qs);
        b
    }

    #[test]
    fn q5_0_transcodes_to_the_same_q8_0_values() {
        let v: [i8; 32] = std::array::from_fn(|i| ((i as i32 * 7) % 32 - 16) as i8);
        let scale = half::f16::from_f32(0.0123);
        let mut raw = pack_q5_0(scale, &v);
        raw.extend(pack_q5_0(half::f16::from_f32(-2.5), &[15; 32]));
        let q8 = transcode_to_q8_0(&raw, Q5_0_BLOCK, decode_q5_0, "t").unwrap();
        assert_eq!(q8.len(), 2 * Q8_0_BLOCK);
        assert_eq!(
            &q8[..2],
            &scale.to_le_bytes(),
            "the scale is carried over unchanged"
        );
        let got: Vec<i8> = q8[2..34].iter().map(|&b| b as i8).collect();
        assert_eq!(got, v.to_vec());
        assert!(q8[36..68].iter().all(|&b| b as i8 == 15));
    }

    #[test]
    fn q4_0_transcodes_to_the_same_q8_0_values() {
        let v: [i8; 32] = std::array::from_fn(|i| (i as i32 % 16 - 8) as i8);
        let mut raw = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        raw.extend((0..16).map(|j| ((v[j] + 8) as u8) | (((v[j + 16] + 8) as u8) << 4)));
        let q8 = transcode_to_q8_0(&raw, Q4_0_BLOCK, decode_q4_0, "t").unwrap();
        let got: Vec<i8> = q8[2..34].iter().map(|&b| b as i8).collect();
        assert_eq!(got, v.to_vec());
    }

    #[test]
    fn a_partial_block_is_an_error() {
        assert!(transcode_to_q8_0(&[0u8; 23], Q5_0_BLOCK, decode_q5_0, "t").is_err());
    }
}
