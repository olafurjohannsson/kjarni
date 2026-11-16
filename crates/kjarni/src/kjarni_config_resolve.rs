use kjarni_transformers::{models::base::ModelLoadConfig, tensor::DType};

use crate::generation::GenerationOverrides;
use crate::kjarni_config::{GenerationParams, KjarniConfig};

impl KjarniConfig {
    /// Get generation overrides for a specific model and task.
    pub fn get_generation_overrides(&self, model_name: &str, task: Task) -> GenerationOverrides {
        // Start with task defaults
        let task_params = match task {
            Task::Chat => &self.chat.generation,
            Task::Generate => &self.generate.generation,
            _ => return GenerationOverrides::default(),
        };

        // Apply model-specific overrides if present
        let model_params = self
            .model_override(model_name)
            .and_then(|m| m.generation.as_ref());

        // Merge: model overrides > task defaults
        merge_generation_params(task_params, model_params)
    }

    /// Get load config for a specific model.
    pub fn get_load_config(&self, model_name: &str) -> ModelLoadConfig {
        let mut config = ModelLoadConfig {
            offload_embeddings: self.load.offload_embeddings,
            offload_lm_head: self.load.offload_lm_head,
            target_dtype: parse_dtype(&self.load.dtype),
            quantize_lm_head: self
                .load
                .quantize_lm_head
                .as_ref()
                .and_then(|s| parse_quantize(s)),
            quantize_embeddings: self
                .load
                .quantize_embeddings
                .as_ref()
                .and_then(|s| parse_quantize(s)),
            use_gguf: self.load.prefer_gguf,
            max_batch_size: self.load.max_batch_size,
            max_sequence_length: self.load.max_sequence_length,
            // Download progress follows the caller's own quiet setting, which the
            // builders already thread through; this resolver has none of its own.
            quiet: false,
        };

        // Apply model-specific overrides
        if let Some(model_override) = self.model_override(model_name) {
            if let Some(dtype) = &model_override.dtype {
                config.target_dtype = parse_dtype(dtype);
            }
            if let Some(offload) = model_override.offload_embeddings {
                config.offload_embeddings = offload;
            }
            if let Some(offload) = model_override.offload_lm_head {
                config.offload_lm_head = offload;
            }
            if let Some(quantize) = model_override.quantize_lm_head.as_ref() {
                config.quantize_lm_head = parse_quantize(quantize);
            }
            if let Some(quantize) = model_override.quantize_embeddings.as_ref() {
                config.quantize_embeddings = parse_quantize(quantize);
            }
        }

        config
    }

    /// The `[models.<name>]` entry for `model_name`, written either as the
    /// quoted name (`[models."qwen3.5-4b"]`) or with dots as dashes
    /// (`[models.qwen3-5-4b]`), since a bare TOML key can't hold a dot.
    fn model_override(&self, model_name: &str) -> Option<&crate::kjarni_config::ModelOverride> {
        self.models
            .get(model_name)
            .or_else(|| self.models.get(&normalize_model_name(model_name)))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Task {
    Chat,
    Generate,
    Summarize,
    Translate,
    Classify,
    Embed,
    Rerank,
    Search,
    Transcribe,
}

fn merge_generation_params(
    base: &GenerationParams,
    override_: Option<&GenerationParams>,
) -> GenerationOverrides {
    let mut result = GenerationOverrides {
        temperature: Some(base.temperature),
        max_new_tokens: Some(base.max_tokens),
        top_p: base.top_p,
        top_k: base.top_k,
        min_p: base.min_p,
        repetition_penalty: Some(base.repetition_penalty),
        ..Default::default()
    };

    if let Some(o) = override_ {
        result.temperature = Some(o.temperature);
        result.max_new_tokens = Some(o.max_tokens);
        if o.top_p.is_some() {
            result.top_p = o.top_p;
        }
        if o.top_k.is_some() {
            result.top_k = o.top_k;
        }
        if o.min_p.is_some() {
            result.min_p = o.min_p;
        }
        result.repetition_penalty = Some(o.repetition_penalty);
    }

    result
}

fn normalize_model_name(name: &str) -> String {
    // Convert "llama3.2-1b-instruct" to "llama3-2-1b-instruct" for TOML keys
    name.replace('.', "-")
}

fn parse_dtype(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "f32" | "float32" => Some(DType::F32),
        "f16" | "float16" => Some(DType::F16),
        "bf16" | "bfloat16" => Some(DType::BF16),
        _ => None,
    }
}

fn parse_quantize(s: &str) -> Option<DType> {
    match s.to_lowercase().as_str() {
        "q8" | "q8_0" => Some(DType::Q8_0),
        // "q4" | "q4_0" => Some(DType::Q4_0),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(toml: &str) -> KjarniConfig {
        toml::from_str(toml).unwrap()
    }

    #[test]
    fn an_empty_file_is_the_defaults() {
        let c = parse("");
        let o = c.get_generation_overrides("any-model", Task::Chat);
        assert_eq!(o.temperature, Some(0.7));
        assert_eq!(o.max_new_tokens, Some(512));
        assert_eq!(o.top_p, Some(0.9));
        assert_eq!(o.top_k, Some(40));
        assert_eq!(o.min_p, Some(0.05));
        assert_eq!(o.repetition_penalty, Some(1.1));
        let load = c.get_load_config("any-model");
        assert_eq!(load.target_dtype, Some(DType::F32));
        assert!(!load.use_gguf);
        assert!(c.cache.auto_download);
    }

    #[test]
    fn task_settings_apply_to_their_task_only() {
        let c =
            parse("[chat]\ntemperature = 0.2\nmax_tokens = 64\n[generate]\ntemperature = 1.0\n");
        let chat = c.get_generation_overrides("m", Task::Chat);
        assert_eq!(
            (chat.temperature, chat.max_new_tokens),
            (Some(0.2), Some(64))
        );
        let generate = c.get_generation_overrides("m", Task::Generate);
        assert_eq!(
            (generate.temperature, generate.max_new_tokens),
            (Some(1.0), Some(512))
        );
        // Tasks without generation settings get none.
        let embed = c.get_generation_overrides("m", Task::Embed);
        assert_eq!(embed.temperature, None);
    }

    #[test]
    fn a_model_entry_overrides_the_task_whichever_way_its_name_is_written() {
        for key in ["\"qwen3.5-4b\"", "qwen3-5-4b"] {
            let c = parse(&format!(
                "[chat]\ntemperature = 0.2\ntop_k = 10\n[models.{key}.generation]\ntemperature = 0.9\ntop_p = 0.5\n"
            ));
            let o = c.get_generation_overrides("qwen3.5-4b", Task::Chat);
            assert_eq!(o.temperature, Some(0.9), "{key}");
            assert_eq!(o.top_p, Some(0.5), "{key}");
            // Unset in the model entry, so the task's value stays.
            assert_eq!(o.top_k, Some(10), "{key}");
            // Another model is untouched.
            let other = c.get_generation_overrides("llama3.2-1b-instruct", Task::Chat);
            assert_eq!(other.temperature, Some(0.2), "{key}");
        }
    }

    #[test]
    fn load_settings_and_model_overrides() {
        let c = parse(
            "[load]\ndtype = \"bf16\"\nprefer_gguf = true\nquantize_lm_head = \"q8\"\nmax_batch_size = 4\n\
             [models.\"qwen3.5-0.8b\"]\ndtype = \"F16\"\noffload_embeddings = true\nquantize_embeddings = \"q8_0\"\n",
        );
        let base = c.get_load_config("minilm-l6-v2");
        assert_eq!(base.target_dtype, Some(DType::BF16));
        assert!(base.use_gguf);
        assert_eq!(base.quantize_lm_head, Some(DType::Q8_0));
        assert_eq!(base.quantize_embeddings, None);
        assert_eq!(base.max_batch_size, Some(4));
        assert!(!base.offload_embeddings);

        let tuned = c.get_load_config("qwen3.5-0.8b");
        assert_eq!(tuned.target_dtype, Some(DType::F16));
        assert!(tuned.offload_embeddings);
        assert_eq!(tuned.quantize_embeddings, Some(DType::Q8_0));
        assert_eq!(tuned.quantize_lm_head, Some(DType::Q8_0));
    }

    #[test]
    fn unknown_dtypes_and_quantizations_mean_none() {
        assert_eq!(parse_dtype("float16"), Some(DType::F16));
        assert_eq!(parse_dtype("bfloat16"), Some(DType::BF16));
        assert_eq!(parse_dtype("int3"), None);
        assert_eq!(parse_quantize("Q8"), Some(DType::Q8_0));
        assert_eq!(parse_quantize("q4"), None);
    }

    #[test]
    fn config_files_load_and_the_cache_dir_follows_them() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("kjarni.toml");
        std::fs::write(
            &path,
            "[cache]\ndir = \"/data/models\"\nauto_download = false\n",
        )
        .unwrap();
        let c = crate::kjarni_config_loader::load_config_from_path(&path).unwrap();
        assert!(!c.cache.auto_download);
        assert_eq!(
            crate::kjarni_config_loader::get_cache_dir(&c),
            std::path::PathBuf::from("/data/models")
        );
        std::fs::write(&path, "[chat\n").unwrap();
        assert!(crate::kjarni_config_loader::load_config_from_path(&path).is_err());
        assert!(
            crate::kjarni_config_loader::load_config_from_path(&dir.path().join("none.toml"))
                .is_err()
        );
    }
}
