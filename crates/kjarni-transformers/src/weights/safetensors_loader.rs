//! SafeTensors format loader with mmap caching.
//!
//! Loads model weights from `.safetensors` files, supporting both single-file
//! and sharded (multi-file) models.
//!
//! # Format Overview
//!
//! SafeTensors is a simple, safe format for storing tensors:
//! - Header: JSON metadata with tensor names, shapes, and byte offsets
//! - Data: Raw tensor bytes, contiguously stored
//!
//! # Sharded Models
//!
//! Large models (>10GB) are often split into multiple files:
//! ```text
//! model/
//! ├── model.safetensors.index.json  # Maps tensor names to files
//! ├── model-00001-of-00003.safetensors
//! ├── model-00002-of-00003.safetensors
//! └── model-00003-of-00003.safetensors
//! ```
//!
//! This loader handles sharding transparently.
//!
//! # Memory Efficiency
//!
//! Uses memory-mapped I/O with global caching:
//! - Tensors are read directly from disk on access
//! - Multiple models sharing the same file share the same mmap
//! - No upfront memory allocation for the full model

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::mmap_cache::get_or_create_mmap;
use crate::weights::WeightLoader;
use anyhow::{anyhow, Context, Result};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

/// A loader for `.safetensors` files with mmap caching.
///
/// Supports both single-file and sharded models. Uses the global mmap cache
/// to deduplicate mappings across multiple model loads.
///
/// # Example
///
/// ```ignore
/// // Single file
/// let loader = SafeTensorsLoader::new(Path::new("/models/model.safetensors"))?;
///
/// // Directory with sharded model
/// let loader = SafeTensorsLoader::new(Path::new("/models/llama-7b/"))?;
///
/// // Access tensors via callback
/// loader.with_tensor("model.embed_tokens.weight", |view| {
///     println!("Shape: {:?}, DType: {:?}", view.shape, view.dtype);
///     Ok(())
/// })?;
/// ```
pub struct SafeTensorsLoader {
    /// Loaded shards, each containing an mmap and parsed SafeTensors
    shards: Vec<ShardInfo>,
    /// Maps tensor names to their shard index
    tensor_to_shard: HashMap<String, usize>,
}

/// Information about a single safetensors shard.
struct ShardInfo {
    /// Memory-mapped file contents (shared via cache)
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    /// Parsed SafeTensors structure
    /// 
    /// SAFETY: The 'static lifetime is safe because:
    /// 1. The mmap is owned by Arc and outlives this struct
    /// 2. ShardInfo holds both the mmap and tensors, ensuring the mmap
    ///    is not dropped while tensors are in use
    tensors: SafeTensors<'static>,
}

impl SafeTensorsLoader {
    /// Create a new SafeTensors loader.
    ///
    /// Accepts either a direct file path or a directory containing:
    /// - `model.safetensors` (single file)
    /// - `model.safetensors.index.json` + shards (sharded model)
    ///
    /// # Arguments
    ///
    /// * `path` - Path to a `.safetensors` file or model directory
    ///
    /// # Returns
    ///
    /// A configured loader ready to access tensors.
    ///
    /// # Errors
    ///
    /// - File not found
    /// - Invalid SafeTensors format
    /// - Missing index.json for sharded models
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Load from file
    /// let loader = SafeTensorsLoader::new(Path::new("model.safetensors"))?;
    ///
    /// // Load from directory
    /// let loader = SafeTensorsLoader::new(Path::new("./llama-7b/"))?;
    /// ```
    pub fn new(path: &Path) -> Result<Self> {
        // Case 1: Direct file path
        if path.is_file() {
            return Self::load_single(path).map(|(shards, map)| Self {
                shards,
                tensor_to_shard: map,
            });
        }

        // Case 2: Directory
        if !path.is_dir() {
            return Err(anyhow!(
                "Path {:?} is neither a file nor a directory",
                path
            ));
        }

        let index_file = path.join("model.safetensors.index.json");
        let (shards, tensor_to_shard) = if index_file.exists() {
            Self::load_sharded(path)?
        } else {
            Self::load_single(&path.join("model.safetensors"))?
        };

        Ok(Self {
            shards,
            tensor_to_shard,
        })
    }

    /// Load a single (non-sharded) safetensors file.
    fn load_single(path: &Path) -> Result<(Vec<ShardInfo>, HashMap<String, usize>)> {
        let shard = Self::load_shard(path)?;
        let tensor_to_shard = shard
            .tensors
            .names()
            .into_iter()
            .map(|name| (name.to_string(), 0))
            .collect();
        
        log::info!(
            "Loaded single safetensors file: {} tensors from {:?}",
            shard.tensors.names().len(),
            path.file_name().unwrap_or_default()
        );
        
        Ok((vec![shard], tensor_to_shard))
    }

    /// Load a sharded model from a directory.
    fn load_sharded(path: &Path) -> Result<(Vec<ShardInfo>, HashMap<String, usize>)> {
        let index_path = path.join("model.safetensors.index.json");
        let index_content = fs::read_to_string(&index_path)
            .with_context(|| format!("Failed to read index file: {:?}", index_path))?;
        
        let index: serde_json::Value = serde_json::from_str(&index_content)
            .context("Failed to parse index.json")?;
        
        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("Invalid index.json: missing 'weight_map' object"))?;

        // Collect unique shard files
        let mut unique_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        unique_files.sort();
        unique_files.dedup();

        log::info!(
            "Loading sharded model: {} shards, {} tensors",
            unique_files.len(),
            weight_map.len()
        );

        // Load each shard
        let mut shards = Vec::with_capacity(unique_files.len());
        let mut file_to_shard_idx = HashMap::new();

        for (idx, filename) in unique_files.iter().enumerate() {
            let shard_path = path.join(filename);
            shards.push(Self::load_shard(&shard_path)?);
            file_to_shard_idx.insert(filename.clone(), idx);
            
            log::debug!("Loaded shard {}/{}: {}", idx + 1, unique_files.len(), filename);
        }

        // Build tensor -> shard index mapping
        let tensor_to_shard = weight_map
            .iter()
            .filter_map(|(name, file_val)| {
                let filename = file_val.as_str()?;
                let shard_idx = file_to_shard_idx.get(filename)?;
                Some((name.clone(), *shard_idx))
            })
            .collect();

        Ok((shards, tensor_to_shard))
    }

    /// Load a single shard file with mmap caching.
    fn load_shard(path: &Path) -> Result<ShardInfo> {
        // Use global mmap cache
        let mmap = get_or_create_mmap(path)?;

        // SAFETY: We transmute to 'static because:
        // 1. The Arc<Mmap> is stored in ShardInfo alongside the SafeTensors
        // 2. The mmap will not be dropped while SafeTensors holds references
        // 3. The global cache keeps the mmap alive even longer
        let static_slice: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };

        let tensors = SafeTensors::deserialize(static_slice)
            .with_context(|| format!("Failed to parse safetensors: {:?}", path))?;

        Ok(ShardInfo { mmap, tensors })
    }

    /// Get the list of all tensor names in this model.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_shard.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of tensors in this model.
    pub fn tensor_count(&self) -> usize {
        self.tensor_to_shard.len()
    }

    /// Get the number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}


impl WeightLoader for SafeTensorsLoader {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let shard_idx = self
            .tensor_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("Tensor '{}' not found in model", name))?;

        let view = self.shards[*shard_idx]
            .tensors
            .tensor(name)
            .with_context(|| format!("Failed to read tensor '{}'", name))?;

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

    fn as_any(&self) -> &dyn Any {
        self
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_contains() {
        // This test would need a real safetensors file
        // For now, just verify the API compiles
    }
}