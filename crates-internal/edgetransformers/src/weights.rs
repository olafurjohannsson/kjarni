use anyhow::{Context, Result, anyhow};
use half::{bf16, f16};
use memmap2::Mmap;
use ndarray::{Array1, Array2, ArrayD, IxDyn};
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

// =========================================================================
//  1. DType & RawTensor
// =========================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    U32, // Restored
    I32, // Added just in case
    I8,  // Restored
    U8,
    // Quantization formats
    Q8_0,
    Q4_0,
    Q4_1,
}

impl DType {
    pub fn size_of(&self) -> usize {
        match self {
            DType::F32 | DType::U32 | DType::I32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I8 | DType::U8 | DType::Q8_0 => 1,
            // For sub-byte types like Q4, usually represented as block size,
            // but for raw buffer calc we often treat them as u8 blocks.
            // Adjust this logic when you implement actual GGUF block reading.
            DType::Q4_0 | DType::Q4_1 => 1,
        }
    }
}

/// A view into the raw bytes of a tensor.
pub struct RawTensor<'a> {
    pub bytes: Cow<'a, [u8]>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl<'a> RawTensor<'a> {
    /// **CPU Compatibility Layer**
    /// Converts raw bytes into the standard ndarray<f32>.
    /// Handles F16/BF16 -> F32 conversion on the fly.
    pub fn to_ndarray_f32(&self) -> Result<ArrayD<f32>> {
        let data: Vec<f32> = match self.dtype {
            DType::F32 => {
                // Check alignment before casting.
                // If aligned, zero-copy cast. If not, copy then cast.
                let slice: &[f32] = bytemuck::try_cast_slice(&self.bytes)
                    .unwrap_or_else(|_| bytemuck::pod_collect_to_vec(&self.bytes).leak());
                // ^ Note: strict safety might prefer explicit copy if unaligned,
                // usually mmap is aligned enough. simpler version:

                if let Ok(s) = bytemuck::try_cast_slice(&self.bytes) {
                    s.to_vec()
                } else {
                    // Fallback for unaligned bytes
                    self.bytes
                        .chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                        .collect()
                }
            }
            DType::F16 => {
                // Requires 'half' crate with "bytemuck" feature
                let f16_data: &[f16] = bytemuck::try_cast_slice(&self.bytes)
                    .map_err(|_| anyhow!("Unaligned F16 bytes"))?; // Handle alignment in real app
                f16_data.iter().map(|x| x.to_f32()).collect()
            }
            DType::BF16 => {
                // Requires 'half' crate with "bytemuck" feature
                let bf16_data: &[bf16] = bytemuck::try_cast_slice(&self.bytes)
                    .map_err(|_| anyhow!("Unaligned BF16 bytes"))?;
                bf16_data.iter().map(|x| x.to_f32()).collect()
            }
            _ => {
                return Err(anyhow!(
                    "Automatic conversion to f32 not implemented for {:?}",
                    self.dtype
                ));
            }
        };

        Ok(ArrayD::from_shape_vec(IxDyn(&self.shape), data)?)
    }
}

// =========================================================================
//  2. ModelWeights
// =========================================================================

enum BackingData {
    Mmap(Mmap),
    Memory(Vec<u8>),
}

enum ShardBacking {
    Mmap(Mmap),
    Memory(&'static [u8]), // Leaked for 'static lifetime
}

struct ShardInfo {
    _backing: ShardBacking,
    safetensors: SafeTensors<'static>,
}

pub struct ModelWeights {
    shards: Vec<ShardInfo>,
    tensor_to_shard: HashMap<String, usize>,
    pub config_json: String,
}

impl ModelWeights {
    pub fn new(path: &Path) -> Result<Self> {
        let config_file = path.join("config.json");
        let config_json = fs::read_to_string(&config_file)?;

        // Check for sharded model
        let index_file = path.join("model.safetensors.index.json");
        
        if index_file.exists() {
            Self::load_sharded(path, config_json)
        } else {
            Self::load_single(path, config_json)
        }
    }

    fn load_single(path: &Path, config_json: String) -> Result<Self> {
        let weights_file = path.join("model.safetensors");
        let (shard, tensor_names) = Self::load_shard(&weights_file)?;
        
        let tensor_to_shard: HashMap<String, usize> = tensor_names
            .into_iter()
            .map(|name| (name, 0))
            .collect();

        Ok(Self {
            shards: vec![shard],
            tensor_to_shard,
            config_json,
        })
    }

    fn load_sharded(path: &Path, config_json: String) -> Result<Self> {
        let index_file = path.join("model.safetensors.index.json");
        let index_content = fs::read_to_string(&index_file)?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;

        // Parse weight_map: { "layer.0.weight": "model-00001-of-00004.safetensors", ... }
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid index.json: missing weight_map"))?;

        // Collect unique shard files
        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect();
        shard_files.sort();
        shard_files.dedup();

        // Load each shard
        let mut shards = Vec::with_capacity(shard_files.len());
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();

        for (idx, filename) in shard_files.iter().enumerate() {
            let shard_path = path.join(filename);
            let (shard, _) = Self::load_shard(&shard_path)
                .with_context(|| format!("Failed to load shard: {}", filename))?;
            shards.push(shard);
            file_to_idx.insert(filename.clone(), idx);
        }

        // Build tensor -> shard index map
        let tensor_to_shard: HashMap<String, usize> = weight_map
            .iter()
            .filter_map(|(tensor_name, file_val)| {
                let filename = file_val.as_str()?;
                let idx = file_to_idx.get(filename)?;
                Some((tensor_name.clone(), *idx))
            })
            .collect();

        log::info!(
            "Loaded {} shards with {} tensors",
            shards.len(),
            tensor_to_shard.len()
        );

        Ok(Self {
            shards,
            tensor_to_shard,
            config_json,
        })
    }

fn load_shard(path: &std::path::PathBuf) -> Result<(ShardInfo, Vec<String>)> {
    let file = fs::File::open(path)
        .with_context(|| format!("Failed to open {:?}", path))?;

    let mmap = unsafe { Mmap::map(&file)? };
    let static_slice: &'static [u8] =
        unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };

    let safetensors = SafeTensors::deserialize(static_slice)?;
    let tensor_names: Vec<String> = safetensors.names().iter().map(|s| s.to_string()).collect();

    Ok((
        ShardInfo {
            _backing: ShardBacking::Mmap(mmap),
            safetensors,
        },
        tensor_names,
    ))
}

    pub fn get_raw(&self, name: &str) -> Result<RawTensor<'_>> {
        let shard_idx = self.tensor_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in any shard", name))?;

        let shard = &self.shards[*shard_idx];
        let view = shard.safetensors.tensor(name)?;

        let dtype = match view.dtype() {
            safetensors::Dtype::F32 => DType::F32,
            safetensors::Dtype::F16 => DType::F16,
            safetensors::Dtype::BF16 => DType::BF16,
            safetensors::Dtype::U32 => DType::U32,
            safetensors::Dtype::I8 => DType::I8,
            safetensors::Dtype::U8 => DType::U8,
            _ => return Err(anyhow!("Unsupported dtype: {:?}", view.dtype())),
        };

        Ok(RawTensor {
            bytes: Cow::Borrowed(view.data()),
            shape: view.shape().to_vec(),
            dtype,
        })
    }

    /// List all tensor names across all shards
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_shard.keys().map(|s| s.as_str()).collect()
    }

    /// Check if tensor exists
    pub fn contains(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }


pub fn from_bytes(data: Vec<u8>, config_json: &str) -> Result<Self> {
        // Store the bytes
        let data = Box::leak(data.into_boxed_slice()); // Leak to get 'static lifetime
        
        let safetensors = SafeTensors::deserialize(data)?;
        
        // Build tensor map (all tensors in shard 0)
        let tensor_names: Vec<String> = safetensors
            .names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let tensor_to_shard: HashMap<String, usize> = tensor_names
            .into_iter()
            .map(|name| (name, 0))
            .collect();

        let shard = ShardInfo {
            _backing: ShardBacking::Memory(data),
            safetensors,
        };

        Ok(Self {
            shards: vec![shard],
            tensor_to_shard,
            config_json: config_json.to_string(),
        })
    }

    /// Load from multiple in-memory byte arrays (for sharded WASM)
    pub fn from_bytes_sharded(
        shard_data: Vec<(String, Vec<u8>)>, // (filename, bytes)
        index_json: &str,
        config_json: &str,
    ) -> Result<Self> {
        let index: serde_json::Value = serde_json::from_str(index_json)?;
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid index.json"))?;

        // Map filename -> shard index
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        let mut shards = Vec::with_capacity(shard_data.len());

        for (idx, (filename, data)) in shard_data.into_iter().enumerate() {
            let data = Box::leak(data.into_boxed_slice());
            let safetensors = SafeTensors::deserialize(data)?;
            
            shards.push(ShardInfo {
                _backing: ShardBacking::Memory(data),
                safetensors,
            });
            file_to_idx.insert(filename, idx);
        }

        let tensor_to_shard: HashMap<String, usize> = weight_map
            .iter()
            .filter_map(|(tensor_name, file_val)| {
                let filename = file_val.as_str()?;
                let idx = file_to_idx.get(filename)?;
                Some((tensor_name.clone(), *idx))
            })
            .collect();

        Ok(Self {
            shards,
            tensor_to_shard,
            config_json: config_json.to_string(),
        })
    }
    pub fn get_linear_weight_bf16(&self, name: &str) -> Result<Array2<u16>> {
        let raw = self.get_raw(name)?;

        // Ensure it's actually BF16 in the file
        anyhow::ensure!(raw.dtype == DType::BF16, "Tensor {} is not BF16", name);
        anyhow::ensure!(raw.shape.len() == 2, "Tensor {} is not 2D", name);

        // Cast bytes to u16
        // Note: Safe because BF16 is 2 bytes, just like u16.
        let u16_data: &[u16] = bytemuck::cast_slice(&raw.bytes);

        // Create ndarray view/copy
        // We transpose here just like the F32 version: [Out, In] -> [In, Out]
        // This makes the rows contiguous for the [In] dimension.
        let arr = Array2::from_shape_vec((raw.shape[0], raw.shape[1]), u16_data.to_vec())?;

        // Return Transposed [In, Out] (stored as [Out, In] in memory effectively)
        // Actually, for our Optimized Matmul, we want the WEIGHTS to be [Out, In].
        // Let's stick to the convention: Weights in file are [Out, In].
        // We want to store them as [Out, In] (Transposed relative to MatMul math).

        // Just return the array as is (Row Major: Out, In).
        // If we need to transpose memory layout, we do .t().to_owned() like before.
        // Let's do the standard transpose we decided on: [Hidden, Out] layout in memory?
        // No, for mixed precision, we iterate over Output rows.
        // So keeping it [Out, In] (Standard Safetensors) is actually perfect for Dot Product.

        Ok(arr)
    }


    // --- Legacy / CPU Helpers ---

    pub fn get_array1(&self, name: &str) -> Result<Array1<f32>> {
        let raw = self.get_raw(name)?;
        anyhow::ensure!(raw.shape.len() == 1, "Expected 1D");
        let dyn_arr = raw.to_ndarray_f32()?;
        Ok(dyn_arr.into_dimensionality::<ndarray::Ix1>()?)
    }

    pub fn get_array2(&self, name: &str) -> Result<Array2<f32>> {
        let raw = self.get_raw(name)?;
        anyhow::ensure!(raw.shape.len() == 2, "Expected 2D");
        let dyn_arr = raw.to_ndarray_f32()?;
        Ok(dyn_arr.into_dimensionality::<ndarray::Ix2>()?)
    }

    pub fn get_linear_weight(&self, name: &str) -> Result<Array2<f32>> {
        let weight = self.get_array2(name)?;
        Ok(weight.t().to_owned())
    }

    pub fn get_embedding_weight(&self, name: &str) -> Result<Array2<f32>> {
        self.get_array2(name)
    }
}
