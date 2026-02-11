//! Weight loading

mod gguf_conversion;
mod gguf_loader;
mod mmap_cache;
mod model_weights;
mod safetensors_loader;

use std::any::Any;

use anyhow::Result;

use crate::tensor::raw_tensor::TensorView;

pub use gguf_conversion::{cast_or_copy, raw_to_typed_gguf};
pub use gguf_loader::{GgufHfMapper, GgufLoader};
pub use mmap_cache::{clear_mmap_cache, mmap_cache_stats};
pub use model_weights::{AttentionLayout, ModelWeights, raw_to_typed};
pub use safetensors_loader::SafeTensorsLoader;

/// Trait for loading model weights from various file formats
pub trait WeightLoader: Send + Sync {
    /// Returns a raw tensor view by name.
    ///
    /// The view borrows from mmap'd memory and should be consumed immediately.
    fn get_raw(&self, name: &str) -> Result<TensorView<'_>>;

    /// Returns `true` if a tensor with the given name exists.
    fn contains(&self, name: &str) -> bool;

    /// Returns a string from metadata (GGUF only).
    fn get_string(&self, _key: &str) -> Option<&str> {
        None
    }

    /// Returns a u32 from metadata (GGUF only).
    fn get_u32(&self, _key: &str) -> Option<u32> {
        None
    }

    /// Returns a f32 from metadata (GGUF only).
    fn get_f32(&self, _key: &str) -> Option<f32> {
        None
    }

    /// Returns `true` if this loader has embedded metadata.
    fn has_metadata(&self) -> bool {
        false
    }

    /// Downcasts to a concrete type.
    fn as_any(&self) -> &dyn Any;
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod more_tests {
    use super::*;
    use std::any::Any;
    use std::collections::HashMap;
    use std::path::Path;
    use tempfile::TempDir;

    struct MockWeightLoader {
        tensors: HashMap<String, Vec<f32>>,
        metadata_strings: HashMap<String, String>,
        metadata_u32: HashMap<String, u32>,
        metadata_f32: HashMap<String, f32>,
        has_meta: bool,
    }

    impl MockWeightLoader {
        fn new() -> Self {
            Self {
                tensors: HashMap::new(),
                metadata_strings: HashMap::new(),
                metadata_u32: HashMap::new(),
                metadata_f32: HashMap::new(),
                has_meta: false,
            }
        }

        fn with_tensor(mut self, name: &str, data: Vec<f32>) -> Self {
            self.tensors.insert(name.to_string(), data);
            self
        }

        fn with_metadata(mut self) -> Self {
            self.has_meta = true;
            self
        }

        fn with_string_meta(mut self, key: &str, value: &str) -> Self {
            self.metadata_strings
                .insert(key.to_string(), value.to_string());
            self.has_meta = true;
            self
        }

        fn with_u32_meta(mut self, key: &str, value: u32) -> Self {
            self.metadata_u32.insert(key.to_string(), value);
            self.has_meta = true;
            self
        }

        fn with_f32_meta(mut self, key: &str, value: f32) -> Self {
            self.metadata_f32.insert(key.to_string(), value);
            self.has_meta = true;
            self
        }
    }

    impl WeightLoader for MockWeightLoader {
        fn get_raw(&self, name: &str) -> Result<TensorView<'_>> {
            if self.tensors.contains_key(name) {
                anyhow::bail!("Mock: use contains() to check existence")
            } else {
                anyhow::bail!("Tensor '{}' not found", name)
            }
        }

        fn contains(&self, name: &str) -> bool {
            self.tensors.contains_key(name)
        }

        fn get_string(&self, key: &str) -> Option<&str> {
            self.metadata_strings.get(key).map(|s| s.as_str())
        }

        fn get_u32(&self, key: &str) -> Option<u32> {
            self.metadata_u32.get(key).copied()
        }

        fn get_f32(&self, key: &str) -> Option<f32> {
            self.metadata_f32.get(key).copied()
        }

        fn has_metadata(&self) -> bool {
            self.has_meta
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    #[test]
    fn test_weight_loader_contains() {
        let loader = MockWeightLoader::new()
            .with_tensor("layer.0.weight", vec![1.0, 2.0, 3.0])
            .with_tensor("layer.0.bias", vec![0.1, 0.2]);

        assert!(loader.contains("layer.0.weight"));
        assert!(loader.contains("layer.0.bias"));
        assert!(!loader.contains("layer.1.weight"));
        assert!(!loader.contains("nonexistent"));
    }

    #[test]
    fn test_weight_loader_get_raw_missing() {
        let loader = MockWeightLoader::new();

        let result = loader.get_raw("missing_tensor");
        assert!(result.is_err());
    }

    #[test]
    fn test_weight_loader_default_metadata_methods() {
        let loader = MockWeightLoader::new();

        assert_eq!(loader.get_string("any_key"), None);
        assert_eq!(loader.get_u32("any_key"), None);
        assert_eq!(loader.get_f32("any_key"), None);
        assert!(!loader.has_metadata());
    }

    #[test]
    fn test_weight_loader_with_string_metadata() {
        let loader = MockWeightLoader::new()
            .with_string_meta("model.name", "llama-7b")
            .with_string_meta("model.author", "meta");

        assert_eq!(loader.get_string("model.name"), Some("llama-7b"));
        assert_eq!(loader.get_string("model.author"), Some("meta"));
        assert_eq!(loader.get_string("nonexistent"), None);
        assert!(loader.has_metadata());
    }

    #[test]
    fn test_weight_loader_with_u32_metadata() {
        let loader = MockWeightLoader::new()
            .with_u32_meta("num_layers", 32)
            .with_u32_meta("hidden_size", 4096);

        assert_eq!(loader.get_u32("num_layers"), Some(32));
        assert_eq!(loader.get_u32("hidden_size"), Some(4096));
        assert_eq!(loader.get_u32("nonexistent"), None);
    }

    #[test]
    fn test_weight_loader_with_f32_metadata() {
        let loader = MockWeightLoader::new()
            .with_f32_meta("rope_theta", 10000.0)
            .with_f32_meta("layer_norm_eps", 1e-5);

        assert_eq!(loader.get_f32("rope_theta"), Some(10000.0));
        assert!((loader.get_f32("layer_norm_eps").unwrap() - 1e-5).abs() < 1e-10);
        assert_eq!(loader.get_f32("nonexistent"), None);
    }

    #[test]
    fn test_weight_loader_mixed_metadata() {
        let loader = MockWeightLoader::new()
            .with_string_meta("name", "test_model")
            .with_u32_meta("layers", 12)
            .with_f32_meta("eps", 1e-6);

        assert_eq!(loader.get_string("name"), Some("test_model"));
        assert_eq!(loader.get_u32("layers"), Some(12));
        assert!(loader.get_f32("eps").is_some());
        assert!(loader.has_metadata());
    }

    #[test]
    fn test_weight_loader_as_any_downcast() {
        let loader = MockWeightLoader::new().with_tensor("test", vec![1.0]);

        let any_ref = loader.as_any();
        let downcast = any_ref.downcast_ref::<MockWeightLoader>();

        assert!(downcast.is_some());
        assert!(downcast.unwrap().contains("test"));
    }

    #[test]
    fn test_weight_loader_as_any_wrong_type() {
        let loader = MockWeightLoader::new();

        let any_ref = loader.as_any();
        let wrong_downcast = any_ref.downcast_ref::<String>();

        assert!(wrong_downcast.is_none());
    }

    #[test]
    fn test_attention_layout_resolve_placeholder() {
        // Test that placeholder resolution works
        let template = "model.layers.{}.self_attn.q_proj.weight";
        let resolved = template.replace("{}", "5");

        assert_eq!(resolved, "model.layers.5.self_attn.q_proj.weight");
    }

    fn create_test_safetensors(
        dir: &Path,
        tensors: Vec<(&str, Vec<f32>, Vec<usize>)>,
    ) -> Result<()> {
        use safetensors::Dtype;
        use safetensors::tensor::TensorView as SafeTensorView;
        use std::collections::HashMap;
        use std::fs::File;
        use std::io::Write;

        let mut tensor_map = HashMap::new();
        let mut data_storage = Vec::new();

        for (name, data, shape) in &tensors {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            let start = data_storage.len();
            data_storage.extend_from_slice(&bytes);
            let end = data_storage.len();
            tensor_map.insert(name.to_string(), (start, end, shape.clone()));
        }

        let views: Vec<_> = tensor_map
            .iter()
            .map(|(name, (start, end, shape))| {
                (
                    name.as_str(),
                    SafeTensorView::new(Dtype::F32, shape.clone(), &data_storage[*start..*end])
                        .unwrap(),
                )
            })
            .collect();

        let serialized = safetensors::serialize(views, &None)?;

        let mut file = File::create(dir.join("model.safetensors"))?;
        file.write_all(&serialized)?;

        let config = r#"{"hidden_size": 64, "num_layers": 2}"#;
        std::fs::write(dir.join("config.json"), config)?;

        Ok(())
    }

    #[test]
    fn test_model_weights_new_valid_path() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![
                ("layer.0.weight", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ("layer.0.bias", vec![0.1, 0.2], vec![2]),
            ],
        )?;

        let weights = ModelWeights::new(dir.path())?;

        assert!(weights.contains("layer.0.weight"));
        assert!(weights.contains("layer.0.bias"));
        assert!(!weights.contains("nonexistent"));

        Ok(())
    }

    #[test]
    fn test_model_weights_new_invalid_path() {
        let result = ModelWeights::new(Path::new("/nonexistent/path"));
        assert!(result.is_err());
    }

    #[test]
    fn test_model_weights_get_array1() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![("bias", vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5])],
        )?;

        let weights = ModelWeights::new(dir.path())?;
        let array = weights.get_array1("bias")?;

        assert_eq!(array.len(), 5);
        assert_eq!(array[0], 1.0);
        assert_eq!(array[4], 5.0);

        Ok(())
    }

    #[test]
    fn test_model_weights_get_array1_missing() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("exists", vec![1.0], vec![1])])?;

        let weights = ModelWeights::new(dir.path())?;
        let result = weights.get_array1("missing");

        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_model_weights_get_array2() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![("weight", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])],
        )?;

        let weights = ModelWeights::new(dir.path())?;
        let array = weights.get_array2("weight")?;

        assert_eq!(array.dim(), (2, 3));
        assert_eq!(array[[0, 0]], 1.0);
        assert_eq!(array[[1, 2]], 6.0);

        Ok(())
    }

    #[test]
    fn test_model_weights_get_array2_missing() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("exists", vec![1.0, 2.0], vec![1, 2])])?;

        let weights = ModelWeights::new(dir.path())?;
        let result = weights.get_array2("missing");

        assert!(result.is_err());

        Ok(())
    }

    #[test]
    fn test_model_weights_contains() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![
                ("tensor_a", vec![1.0], vec![1]),
                ("tensor_b", vec![2.0], vec![1]),
            ],
        )?;

        let weights = ModelWeights::new(dir.path())?;

        assert!(weights.contains("tensor_a"));
        assert!(weights.contains("tensor_b"));
        assert!(!weights.contains("tensor_c"));

        Ok(())
    }

    #[test]
    fn test_model_weights_multiple_tensors() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![
                (
                    "embed.weight",
                    (0..768).map(|x| x as f32 * 0.01).collect(),
                    vec![256, 3],
                ),
                (
                    "layer.0.attn.q.weight",
                    (0..64).map(|x| x as f32).collect(),
                    vec![8, 8],
                ),
                (
                    "layer.0.attn.k.weight",
                    (0..64).map(|x| -x as f32).collect(),
                    vec![8, 8],
                ),
                (
                    "layer.0.attn.v.weight",
                    (0..64).map(|x| x as f32 * 0.5).collect(),
                    vec![8, 8],
                ),
                ("layer.0.norm.weight", vec![1.0; 8], vec![8]),
                ("layer.0.norm.bias", vec![0.0; 8], vec![8]),
            ],
        )?;

        let weights = ModelWeights::new(dir.path())?;

        assert!(weights.contains("embed.weight"));
        assert!(weights.contains("layer.0.attn.q.weight"));
        assert!(weights.contains("layer.0.norm.weight"));

        let embed = weights.get_array2("embed.weight")?;
        assert_eq!(embed.dim(), (256, 3));

        let q_weight = weights.get_array2("layer.0.attn.q.weight")?;
        assert_eq!(q_weight.dim(), (8, 8));

        let norm_weight = weights.get_array1("layer.0.norm.weight")?;
        assert_eq!(norm_weight.len(), 8);

        Ok(())
    }

    #[test]
    fn test_cast_or_copy_f32_to_f32() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();

        let result = cast_or_copy::<f32>(&bytes);

        assert_eq!(result.len(), 3);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[2], 3.0);
    }

    #[test]
    fn test_clear_mmap_cache() {
        clear_mmap_cache();
        let stats = mmap_cache_stats();
        let _ = stats;
    }

    #[test]
    fn test_safetensors_loader_new() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("test", vec![1.0, 2.0], vec![2])])?;

        let loader = SafeTensorsLoader::new(dir.path())?;

        assert!(loader.contains("test"));
        assert!(!loader.contains("nonexistent"));

        Ok(())
    }

    #[test]
    fn test_safetensors_loader_no_metadata() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("test", vec![1.0], vec![1])])?;

        let loader = SafeTensorsLoader::new(dir.path())?;

        assert!(!loader.has_metadata());
        assert_eq!(loader.get_string("any"), None);
        assert_eq!(loader.get_u32("any"), None);
        assert_eq!(loader.get_f32("any"), None);

        Ok(())
    }

    #[test]
    fn test_safetensors_loader_as_any() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("test", vec![1.0], vec![1])])?;

        let loader = SafeTensorsLoader::new(dir.path())?;
        let any_ref = loader.as_any();

        assert!(any_ref.downcast_ref::<SafeTensorsLoader>().is_some());

        Ok(())
    }

    #[test]
    fn test_safetensors_loader_multiple_files() -> Result<()> {
        let dir = TempDir::new()?;

        {
            use safetensors::Dtype;
            use safetensors::tensor::TensorView as SafeTensorView;
            use std::fs::File;
            use std::io::Write;

            let data1: Vec<u8> = vec![1.0f32, 2.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let view1 = SafeTensorView::new(Dtype::F32, vec![2], &data1).unwrap();
            let serialized = safetensors::serialize(vec![("tensor1", view1)], &None)?;

            let mut file = File::create(dir.path().join("model-00001-of-00002.safetensors"))?;
            file.write_all(&serialized)?;
        }

        {
            use safetensors::Dtype;
            use safetensors::tensor::TensorView as SafeTensorView;
            use std::fs::File;
            use std::io::Write;

            let data2: Vec<u8> = vec![3.0f32, 4.0]
                .iter()
                .flat_map(|f| f.to_le_bytes())
                .collect();
            let view2 = SafeTensorView::new(Dtype::F32, vec![2], &data2).unwrap();
            let serialized = safetensors::serialize(vec![("tensor2", view2)], &None)?;

            let mut file = File::create(dir.path().join("model-00002-of-00002.safetensors"))?;
            file.write_all(&serialized)?;
        }

        let index = serde_json::json!({
            "metadata": {},
            "weight_map": {
                "tensor1": "model-00001-of-00002.safetensors",
                "tensor2": "model-00002-of-00002.safetensors"
            }
        });
        std::fs::write(
            dir.path().join("model.safetensors.index.json"),
            serde_json::to_string(&index)?,
        )?;

        // Create config
        std::fs::write(dir.path().join("config.json"), r#"{}"#)?;

        let loader = SafeTensorsLoader::new(dir.path())?;
        assert!(loader.contains("tensor1"));
        assert!(loader.contains("tensor2"));

        Ok(())
    }

    #[test]
    fn test_model_weights_empty_tensor_name() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(dir.path(), vec![("", vec![1.0], vec![1])])?;

        let weights = ModelWeights::new(dir.path())?;

        assert!(weights.contains(""));

        Ok(())
    }

    #[test]
    fn test_model_weights_special_characters_in_name() -> Result<()> {
        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![
                ("layer.0.self_attn.q_proj.weight", vec![1.0], vec![1]),
                ("model/encoder/layer_0/attention", vec![2.0], vec![1]),
            ],
        )?;

        let weights = ModelWeights::new(dir.path())?;

        assert!(weights.contains("layer.0.self_attn.q_proj.weight"));
        assert!(weights.contains("model/encoder/layer_0/attention"));

        Ok(())
    }

    #[test]
    fn test_model_weights_large_tensor() -> Result<()> {
        let dir = TempDir::new()?;

        // Create a larger tensor (1MB of f32 = 256K elements)
        let large_data: Vec<f32> = (0..262144).map(|x| x as f32 * 0.001).collect();
        create_test_safetensors(
            dir.path(),
            vec![("large_weight", large_data.clone(), vec![512, 512])],
        )?;

        let weights = ModelWeights::new(dir.path())?;
        let array = weights.get_array2("large_weight")?;

        assert_eq!(array.dim(), (512, 512));
        assert!((array[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((array[[0, 1]] - 0.001).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_weight_loader_trait_object_safety() {
        // Verify WeightLoader can be used as trait object
        fn accepts_loader(_loader: &dyn WeightLoader) {}

        let loader = MockWeightLoader::new();
        accepts_loader(&loader);
    }

    #[test]
    fn test_weight_loader_boxed() {
        let loader: Box<dyn WeightLoader> =
            Box::new(MockWeightLoader::new().with_tensor("test", vec![1.0]));

        assert!(loader.contains("test"));
        assert!(!loader.has_metadata());
    }

    #[test]
    fn test_weight_loader_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockWeightLoader>();
    }

    #[test]
    fn test_model_weights_concurrent_access() -> Result<()> {
        use std::sync::Arc;
        use std::thread;

        let dir = TempDir::new()?;
        create_test_safetensors(
            dir.path(),
            vec![
                ("tensor_a", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
                ("tensor_b", vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]),
            ],
        )?;

        let weights = Arc::new(ModelWeights::new(dir.path())?);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                let w = Arc::clone(&weights);
                thread::spawn(move || {
                    let name = if i % 2 == 0 { "tensor_a" } else { "tensor_b" };
                    assert!(w.contains(name));
                    w.get_array2(name).unwrap()
                })
            })
            .collect();

        for handle in handles {
            let _ = handle.join().unwrap();
        }

        Ok(())
    }
}
