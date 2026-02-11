//! SafeTensors format loader with mmap caching

use std::any::Any;
use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use memmap2::Mmap;
use safetensors::SafeTensors;

use crate::tensor::DType;
use crate::tensor::raw_tensor::TensorView;
use crate::weights::mmap_cache::get_or_create_mmap;
use crate::weights::WeightLoader;

/// A loader for `.safetensors` files with mmap caching
#[derive(Debug)]
pub struct SafeTensorsLoader {
    shards: Vec<ShardInfo>,
    tensor_to_shard: HashMap<String, usize>,
}

#[derive(Debug)]
struct ShardInfo {
    #[allow(dead_code)]
    mmap: Arc<Mmap>,
    // The 'static lifetime is ecause the mmap is owned by Arc
    // and stored alongside tensors, ensuring it outlives the SafeTensors
    tensors: SafeTensors<'static>,
}

impl SafeTensorsLoader {
    /// Creates a new SafeTensors loader.
    ///
    /// Accepts either a direct file path or a directory containing
    /// `model.safetensors` or `model.safetensors.index.json` + shards.
    pub fn new(path: &Path) -> Result<Self> {
        if path.is_file() {
            return Self::load_single(path).map(|(shards, map)| Self {
                shards,
                tensor_to_shard: map,
            });
        }

        if !path.is_dir() {
            return Err(anyhow!("path {:?} is neither a file nor a directory", path));
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

    fn load_single(path: &Path) -> Result<(Vec<ShardInfo>, HashMap<String, usize>)> {
        let shard = Self::load_shard(path)?;
        let tensor_to_shard = shard
            .tensors
            .names()
            .into_iter()
            .map(|name| (name.to_string(), 0))
            .collect();

        log::info!(
            "loaded single safetensors file: {} tensors from {:?}",
            shard.tensors.names().len(),
            path.file_name().unwrap_or_default()
        );

        Ok((vec![shard], tensor_to_shard))
    }

    fn load_sharded(path: &Path) -> Result<(Vec<ShardInfo>, HashMap<String, usize>)> {
        let index_path = path.join("model.safetensors.index.json");
        let index_content = fs::read_to_string(&index_path)
            .with_context(|| format!("failed to read index file: {:?}", index_path))?;

        let index: serde_json::Value =
            serde_json::from_str(&index_content).context("failed to parse index.json")?;

        let weight_map = index["weight_map"]
            .as_object()
            .ok_or_else(|| anyhow!("invalid index.json: missing 'weight_map' object"))?;

        let mut unique_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();
        unique_files.sort();
        unique_files.dedup();

        log::info!(
            "loading sharded model: {} shards, {} tensors",
            unique_files.len(),
            weight_map.len()
        );

        let mut shards = Vec::with_capacity(unique_files.len());
        let mut file_to_shard_idx = HashMap::new();

        for (idx, filename) in unique_files.iter().enumerate() {
            let shard_path = path.join(filename);
            shards.push(Self::load_shard(&shard_path)?);
            file_to_shard_idx.insert(filename.clone(), idx);

            log::debug!("loaded shard {}/{}: {}", idx + 1, unique_files.len(), filename);
        }

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

    fn load_shard(path: &Path) -> Result<ShardInfo> {
        let mmap = get_or_create_mmap(path)?;
        let static_slice: &'static [u8] =
            unsafe { std::mem::transmute::<&[u8], &'static [u8]>(&mmap[..]) };

        let tensors = SafeTensors::deserialize(static_slice)
            .with_context(|| format!("failed to parse safetensors: {:?}", path))?;

        Ok(ShardInfo { mmap, tensors })
    }

    /// Returns all tensor names in this model.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensor_to_shard.keys().map(|s| s.as_str()).collect()
    }

    /// Returns the number of tensors.
    pub fn tensor_count(&self) -> usize {
        self.tensor_to_shard.len()
    }

    /// Returns the number of shards.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }
}

impl WeightLoader for SafeTensorsLoader {
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
        let shard_idx = self
            .tensor_to_shard
            .get(name)
            .ok_or_else(|| anyhow!("tensor '{}' not found in model", name))?;

        let view = self.shards[*shard_idx]
            .tensors
            .tensor(name)
            .with_context(|| format!("failed to read tensor '{}'", name))?;

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
    use safetensors::tensor::{Dtype, TensorView as StTensorView};
    use tempfile::TempDir;

    fn create_safetensors_file(
        dir: &TempDir,
        filename: &str,
        tensors: &[(&str, Vec<f32>, Vec<usize>)],
    ) -> Result<()> {
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

        let file_path = dir.path().join(filename);
        safetensors::serialize_to_file(&tensor_map, &None, &file_path)?;
        Ok(())
    }

    #[test]
    fn test_load_nonexistent_path() {
        let result = SafeTensorsLoader::new(Path::new("nonexistent.safetensors"));
        assert!(result.is_err());
    }

    #[test]
    fn test_load_empty_directory() {
        let dir = tempfile::tempdir().unwrap();
        let result = SafeTensorsLoader::new(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_load_invalid_file() {
        let dir = tempfile::tempdir().unwrap();
        let model_path = dir.path().join("model.safetensors");
        std::fs::write(&model_path, b"invalid content").unwrap();

        let result = SafeTensorsLoader::new(dir.path());
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("failed to parse"));
    }

    #[test]
    fn test_single_file_loading() {
        let dir = tempfile::tempdir().unwrap();
        create_safetensors_file(
            &dir,
            "model.safetensors",
            &[
                ("layer1.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ("layer1.bias", vec![0.1, 0.2], vec![2]),
            ],
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(dir.path()).unwrap();

        assert_eq!(loader.tensor_count(), 2);
        assert_eq!(loader.shard_count(), 1);
        assert!(loader.contains("layer1.weight"));
        assert!(loader.contains("layer1.bias"));
        assert!(!loader.contains("nonexistent"));
    }

    #[test]
    fn test_tensor_names() {
        let dir = tempfile::tempdir().unwrap();
        create_safetensors_file(
            &dir,
            "model.safetensors",
            &[
                ("a.weight", vec![1.0], vec![1]),
                ("b.weight", vec![2.0], vec![1]),
                ("c.weight", vec![3.0], vec![1]),
            ],
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(dir.path()).unwrap();
        let names = loader.tensor_names();

        assert_eq!(names.len(), 3);
        assert!(names.contains(&"a.weight"));
        assert!(names.contains(&"b.weight"));
        assert!(names.contains(&"c.weight"));
    }

    #[test]
    fn test_get_raw_tensor() {
        let dir = tempfile::tempdir().unwrap();
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        create_safetensors_file(
            &dir,
            "model.safetensors",
            &[("test.weight", values.clone(), vec![2, 3])],
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(dir.path()).unwrap();
        let view = loader.get_raw("test.weight").unwrap();

        assert_eq!(view.name, "test.weight");
        assert_eq!(view.shape, vec![2, 3]);
        assert_eq!(view.dtype, DType::F32);
        assert_eq!(view.bytes.len(), 24);
    }

    #[test]
    fn test_missing_tensor_error() {
        let dir = tempfile::tempdir().unwrap();
        create_safetensors_file(
            &dir,
            "model.safetensors",
            &[("exists.weight", vec![1.0], vec![1])],
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(dir.path()).unwrap();
        let result = loader.get_raw("does_not_exist");

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_direct_file_path() {
        let dir = tempfile::tempdir().unwrap();
        create_safetensors_file(
            &dir,
            "model.safetensors",
            &[("test.weight", vec![1.0], vec![1])],
        )
        .unwrap();

        let file_path = dir.path().join("model.safetensors");
        let loader = SafeTensorsLoader::new(&file_path).unwrap();

        assert!(loader.contains("test.weight"));
    }

    #[test]
    fn test_sharded_model_loading() {
        let dir = tempfile::tempdir().unwrap();

        create_safetensors_file(
            &dir,
            "model-00001-of-00002.safetensors",
            &[("layer0.weight", vec![1.0, 2.0], vec![2])],
        )
        .unwrap();

        create_safetensors_file(
            &dir,
            "model-00002-of-00002.safetensors",
            &[("layer1.weight", vec![3.0, 4.0], vec![2])],
        )
        .unwrap();

        let index = serde_json::json!({
            "weight_map": {
                "layer0.weight": "model-00001-of-00002.safetensors",
                "layer1.weight": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            index.to_string(),
        )
        .unwrap();

        let loader = SafeTensorsLoader::new(dir.path()).unwrap();

        assert_eq!(loader.tensor_count(), 2);
        assert_eq!(loader.shard_count(), 2);
        assert!(loader.contains("layer0.weight"));
        assert!(loader.contains("layer1.weight"));

        let view0 = loader.get_raw("layer0.weight").unwrap();
        assert_eq!(view0.shape, vec![2]);

        let view1 = loader.get_raw("layer1.weight").unwrap();
        assert_eq!(view1.shape, vec![2]);
    }
}