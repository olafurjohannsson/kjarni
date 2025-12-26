use crate::tensor::{DType, TensorView};
// To satisfy the trait bound
use crate::weights::WeightLoader;
use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// A loader for the .safetensors format, supporting single and sharded files.
pub struct SafeTensorsLoader<'a> {
    shards: Vec<ShardInfo<'a>>,
    tensor_to_shard: HashMap<String, usize>,
}

struct ShardInfo<'a> {
    _backing: Mmap,
    tensors: SafeTensors<'a>,
}

impl<'a> SafeTensorsLoader<'a> {
    pub fn new(path: &Path) -> Result<Self> {
        let index_file = path.join("model.safetensors.index.json");
        let (shards, tensor_to_shard) = if index_file.exists() {
            Self::load_sharded(path)?
        } else {
            Self::load_single(path)?
        };
        Ok(Self { shards, tensor_to_shard })
    }

    fn load_single(path: &Path) -> Result<(Vec<ShardInfo<'a>>, HashMap<String, usize>)> {
        let weights_file = path.join("model.safetensors");
        let shard = Self::load_shard_mmap(&weights_file)?;
        let tensor_to_shard = shard.tensors.names().into_iter().map(|name| (name.clone(), 0)).collect();
        Ok((vec![shard], tensor_to_shard))
    }

    fn load_sharded(path: &Path) -> Result<(Vec<ShardInfo<'a>>, HashMap<String, usize>)> {
        let index_content = fs::read_to_string(path.join("model.safetensors.index.json"))?;
        let index: serde_json::Value = serde_json::from_str(&index_content)?;
        let weight_map = index["weight_map"].as_object().ok_or_else(|| anyhow!("Invalid index.json"))?;

        let mut unique_files: Vec<String> = weight_map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        unique_files.sort();
        unique_files.dedup();

        let mut shards = Vec::with_capacity(unique_files.len());
        let mut file_to_shard_idx = HashMap::new();

        for (idx, filename) in unique_files.iter().enumerate() {
            shards.push(Self::load_shard_mmap(&path.join(filename))?);
            file_to_shard_idx.insert(filename.clone(), idx);
        }

        let tensor_to_shard = weight_map.iter().filter_map(|(name, file_val)| {
            let filename = file_val.as_str()?;
            let shard_idx = file_to_shard_idx.get(filename)?;
            Some((name.clone(), *shard_idx))
        }).collect();

        Ok((shards, tensor_to_shard))
    }

    fn load_shard_mmap(path: &Path) -> Result<ShardInfo<'static>> {
        let file = fs::File::open(path).with_context(|| format!("Failed to open weight file: {:?}", path))?;
        let mmap = unsafe { Mmap::map(&file)? };
        let static_slice: &'static [u8] = unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap) };
        let tensors = SafeTensors::deserialize(static_slice)?;
        Ok(ShardInfo { _backing: mmap, tensors })
    }
}

impl WeightLoader for SafeTensorsLoader<'_> {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let shard_idx = self.tensor_to_shard.get(name).ok_or_else(|| anyhow!("Tensor '{}' not found", name))?;
        let view = self.shards[*shard_idx].tensors.tensor(name)?;
        Ok(TensorView {
            name: name.to_string(),
            bytes: Cow::Borrowed(view.data()),
            shape: view.shape().to_vec(),
            dtype: DType::from_safetensors(view.dtype())?,
        })
    }

    fn contains(&self, name: &str) -> bool {
        self.tensor_to_shard.contains_key(name)
    }
}