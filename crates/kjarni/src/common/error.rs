//! Common error types for kjarni.

use std::fmt;
use thiserror::Error;

/// Errors that can occur when using kjarni.
#[derive(Debug, Error)]
pub enum KjarniError {
    /// Model not found in registry.
    #[error("{0}")]
    UnknownModel(String),

    /// Model cannot perform the requested task.
    #[error("Model '{model}' cannot be used for {task}: {reason}")]
    IncompatibleModel {
        model: String,
        task: String,
        reason: String,
    },

    /// Model not downloaded and download policy is Never.
    #[error("Model '{0}' not downloaded and download policy is set to Never")]
    ModelNotDownloaded(String),

    /// Failed to download model.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Failed to load model.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Inference failed.
    #[error("Inference failed: {0}")]
    InferenceFailed(#[from] anyhow::Error),

    /// GPU requested but not available.
    #[error("GPU requested but WebGPU context could not be created")]
    GpuUnavailable,

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// No labels available for classification.
    #[error("Model has no label mapping configured")]
    NoLabels,
}

/// Result type for kjarni operations.
pub type KjarniResult<T> = Result<T, KjarniError>;

/// Warning emitted for suboptimal configurations.
#[derive(Debug, Clone)]
pub struct KjarniWarning {
    pub message: String,
    pub suggestion: Option<String>,
}

impl KjarniWarning {
    /// Create a new warning without a suggestion.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            suggestion: None,
        }
    }

    /// Create a new warning with a suggestion.
    pub fn with_suggestion(message: impl Into<String>, suggestion: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            suggestion: Some(suggestion.into()),
        }
    }
}

impl fmt::Display for KjarniWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Warning: {}", self.message)?;
        if let Some(suggestion) = &self.suggestion {
            write!(f, " {}", suggestion)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    // =========================================================================
    // KjarniError::UnknownModel tests
    // =========================================================================

    #[test]
    fn test_unknown_model_display() {
        let err = KjarniError::UnknownModel("my-model".to_string());
        let msg = err.to_string();
        
        assert!(msg.contains("Unknown model"));
        assert!(msg.contains("my-model"));
        assert!(msg.contains("kjarni model list"));
    }

    #[test]
    fn test_unknown_model_debug() {
        let err = KjarniError::UnknownModel("test".to_string());
        let debug = format!("{:?}", err);
        
        assert!(debug.contains("UnknownModel"));
        assert!(debug.contains("test"));
    }

    #[test]
    fn test_unknown_model_no_source() {
        let err = KjarniError::UnknownModel("model".to_string());
        assert!(err.source().is_none());
    }

    // =========================================================================
    // KjarniError::IncompatibleModel tests
    // =========================================================================

    #[test]
    fn test_incompatible_model_display() {
        let err = KjarniError::IncompatibleModel {
            model: "bert-base".to_string(),
            task: "generation".to_string(),
            reason: "BERT is an encoder model".to_string(),
        };
        let msg = err.to_string();
        
        assert!(msg.contains("bert-base"));
        assert!(msg.contains("generation"));
        assert!(msg.contains("BERT is an encoder model"));
    }

    #[test]
    fn test_incompatible_model_all_fields() {
        let err = KjarniError::IncompatibleModel {
            model: "minilm-l6-v2".to_string(),
            task: "text generation".to_string(),
            reason: "encoder models cannot generate text".to_string(),
        };
        let msg = err.to_string();
        
        assert!(msg.contains("minilm-l6-v2"));
        assert!(msg.contains("text generation"));
        assert!(msg.contains("encoder models cannot generate text"));
    }

    #[test]
    fn test_incompatible_model_no_source() {
        let err = KjarniError::IncompatibleModel {
            model: "m".to_string(),
            task: "t".to_string(),
            reason: "r".to_string(),
        };
        assert!(err.source().is_none());
    }

    // =========================================================================
    // KjarniError::ModelNotDownloaded tests
    // =========================================================================

    #[test]
    fn test_model_not_downloaded_display() {
        let err = KjarniError::ModelNotDownloaded("llama-7b".to_string());
        let msg = err.to_string();
        
        assert!(msg.contains("llama-7b"));
        assert!(msg.contains("not downloaded"));
        assert!(msg.contains("Never"));
    }

    #[test]
    fn test_model_not_downloaded_no_source() {
        let err = KjarniError::ModelNotDownloaded("model".to_string());
        assert!(err.source().is_none());
    }

    // =========================================================================
    // KjarniError::DownloadFailed tests
    // =========================================================================

    #[test]
    fn test_download_failed_display() {
        let source = anyhow::anyhow!("Connection timeout");
        let err = KjarniError::DownloadFailed {
            model: "phi3".to_string(),
            source,
        };
        let msg = err.to_string();
        
        assert!(msg.contains("Failed to download"));
        assert!(msg.contains("phi3"));
        assert!(msg.contains("Connection timeout"));
    }

    #[test]
    fn test_download_failed_has_source() {
        let source = anyhow::anyhow!("Network error");
        let err = KjarniError::DownloadFailed {
            model: "model".to_string(),
            source,
        };
        
        assert!(err.source().is_some());
    }

    #[test]
    fn test_download_failed_source_chain() {
        let inner = anyhow::anyhow!("DNS resolution failed");
        let err = KjarniError::DownloadFailed {
            model: "test".to_string(),
            source: inner,
        };
        
        let source = err.source().unwrap();
        assert!(source.to_string().contains("DNS resolution failed"));
    }

    // =========================================================================
    // KjarniError::LoadFailed tests
    // =========================================================================

    #[test]
    fn test_load_failed_display() {
        let source = anyhow::anyhow!("Invalid safetensors format");
        let err = KjarniError::LoadFailed {
            model: "custom-model".to_string(),
            source,
        };
        let msg = err.to_string();
        
        assert!(msg.contains("Failed to load"));
        assert!(msg.contains("custom-model"));
        assert!(msg.contains("Invalid safetensors format"));
    }

    #[test]
    fn test_load_failed_has_source() {
        let source = anyhow::anyhow!("File not found");
        let err = KjarniError::LoadFailed {
            model: "model".to_string(),
            source,
        };
        
        assert!(err.source().is_some());
    }

    // =========================================================================
    // KjarniError::InferenceFailed tests
    // =========================================================================

    #[test]
    fn test_inference_failed_display() {
        let err = KjarniError::InferenceFailed(anyhow::anyhow!("Out of memory"));
        let msg = err.to_string();
        
        assert!(msg.contains("Inference failed"));
        assert!(msg.contains("Out of memory"));
    }

    #[test]
    fn test_inference_failed_from_anyhow() {
        let anyhow_err = anyhow::anyhow!("Tensor shape mismatch");
        let err: KjarniError = anyhow_err.into();
        
        match err {
            KjarniError::InferenceFailed(_) => {}
            _ => panic!("Expected InferenceFailed variant"),
        }
    }

    #[test]
    fn test_inference_failed_has_source() {
        let err = KjarniError::InferenceFailed(anyhow::anyhow!("error"));
        assert!(err.source().is_some());
    }

    // =========================================================================
    // KjarniError::GpuUnavailable tests
    // =========================================================================

    #[test]
    fn test_gpu_unavailable_display() {
        let err = KjarniError::GpuUnavailable;
        let msg = err.to_string();
        
        assert!(msg.contains("GPU"));
        assert!(msg.contains("WebGPU"));
    }

    #[test]
    fn test_gpu_unavailable_no_source() {
        let err = KjarniError::GpuUnavailable;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_gpu_unavailable_debug() {
        let err = KjarniError::GpuUnavailable;
        let debug = format!("{:?}", err);
        assert!(debug.contains("GpuUnavailable"));
    }

    // =========================================================================
    // KjarniError::InvalidConfig tests
    // =========================================================================

    #[test]
    fn test_invalid_config_display() {
        let err = KjarniError::InvalidConfig("temperature must be >= 0".to_string());
        let msg = err.to_string();
        
        assert!(msg.contains("Invalid configuration"));
        assert!(msg.contains("temperature must be >= 0"));
    }

    #[test]
    fn test_invalid_config_various_messages() {
        let messages = [
            "batch_size must be positive",
            "unknown pooling strategy",
            "max_tokens exceeds model limit",
        ];
        
        for message in messages {
            let err = KjarniError::InvalidConfig(message.to_string());
            assert!(err.to_string().contains(message));
        }
    }

    #[test]
    fn test_invalid_config_no_source() {
        let err = KjarniError::InvalidConfig("error".to_string());
        assert!(err.source().is_none());
    }

    // =========================================================================
    // KjarniError::NoLabels tests
    // =========================================================================

    #[test]
    fn test_no_labels_display() {
        let err = KjarniError::NoLabels;
        let msg = err.to_string();
        
        assert!(msg.contains("label"));
        assert!(msg.contains("mapping") || msg.contains("configured"));
    }

    #[test]
    fn test_no_labels_no_source() {
        let err = KjarniError::NoLabels;
        assert!(err.source().is_none());
    }

    #[test]
    fn test_no_labels_debug() {
        let err = KjarniError::NoLabels;
        let debug = format!("{:?}", err);
        assert!(debug.contains("NoLabels"));
    }

    // =========================================================================
    // KjarniResult type alias tests
    // =========================================================================

    #[test]
    fn test_kjarni_result_ok() {
        let result: KjarniResult<i32> = Ok(42);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }

    #[test]
    fn test_kjarni_result_err() {
        let result: KjarniResult<i32> = Err(KjarniError::NoLabels);
        assert!(result.is_err());
    }

    #[test]
    fn test_kjarni_result_with_string() {
        let result: KjarniResult<String> = Ok("success".to_string());
        assert_eq!(result.unwrap(), "success");
    }

    #[test]
    fn test_kjarni_result_question_mark_operator() {
        fn inner() -> KjarniResult<i32> {
            let _: () = Err(KjarniError::GpuUnavailable)?;
            Ok(42)
        }
        
        let result = inner();
        assert!(result.is_err());
    }

    // =========================================================================
    // KjarniWarning::new tests
    // =========================================================================

    #[test]
    fn test_warning_new() {
        let warning = KjarniWarning::new("This is a warning");
        
        assert_eq!(warning.message, "This is a warning");
        assert!(warning.suggestion.is_none());
    }

    #[test]
    fn test_warning_new_from_string() {
        let warning = KjarniWarning::new(String::from("Warning message"));
        assert_eq!(warning.message, "Warning message");
    }

    // =========================================================================
    // KjarniWarning::with_suggestion tests
    // =========================================================================

    #[test]
    fn test_warning_with_suggestion() {
        let warning = KjarniWarning::with_suggestion(
            "Model is large",
            "Consider using a smaller model for faster inference.",
        );
        
        assert_eq!(warning.message, "Model is large");
        assert_eq!(
            warning.suggestion,
            Some("Consider using a smaller model for faster inference.".to_string())
        );
    }

    #[test]
    fn test_warning_with_suggestion_from_strings() {
        let warning = KjarniWarning::with_suggestion(
            String::from("Warning"),
            String::from("Suggestion"),
        );
        
        assert_eq!(warning.message, "Warning");
        assert_eq!(warning.suggestion, Some("Suggestion".to_string()));
    }

    // =========================================================================
    // KjarniWarning Display tests
    // =========================================================================

    #[test]
    fn test_warning_display_without_suggestion() {
        let warning = KjarniWarning::new("CPU fallback activated");
        let display = warning.to_string();
        
        assert!(display.contains("Warning:"));
        assert!(display.contains("CPU fallback activated"));
    }

    #[test]
    fn test_warning_display_with_suggestion() {
        let warning = KjarniWarning::with_suggestion(
            "Slow performance detected",
            "Try enabling GPU acceleration.",
        );
        let display = warning.to_string();
        
        assert!(display.contains("Warning:"));
        assert!(display.contains("Slow performance detected"));
        assert!(display.contains("Try enabling GPU acceleration."));
    }

    #[test]
    fn test_warning_display_format() {
        let warning = KjarniWarning::new("Test");
        let display = warning.to_string();
        
        // Should start with "Warning: "
        assert!(display.starts_with("Warning: "));
    }

    // =========================================================================
    // KjarniWarning Debug tests
    // =========================================================================

    #[test]
    fn test_warning_debug() {
        let warning = KjarniWarning::new("message");
        let debug = format!("{:?}", warning);
        
        assert!(debug.contains("KjarniWarning"));
        assert!(debug.contains("message"));
    }

    #[test]
    fn test_warning_debug_with_suggestion() {
        let warning = KjarniWarning::with_suggestion("msg", "sug");
        let debug = format!("{:?}", warning);
        
        assert!(debug.contains("msg"));
        assert!(debug.contains("sug"));
    }

    // =========================================================================
    // KjarniWarning Clone tests
    // =========================================================================

    #[test]
    fn test_warning_clone() {
        let warning = KjarniWarning::with_suggestion("message", "suggestion");
        let cloned = warning.clone();
        
        assert_eq!(warning.message, cloned.message);
        assert_eq!(warning.suggestion, cloned.suggestion);
    }

    #[test]
    fn test_warning_clone_independence() {
        let warning = KjarniWarning::new("original");
        let mut cloned = warning.clone();
        cloned.message = "modified".to_string();
        
        // Original should be unchanged
        assert_eq!(warning.message, "original");
        assert_eq!(cloned.message, "modified");
    }

    // =========================================================================
    // Error variant exhaustiveness
    // =========================================================================

    #[test]
    fn test_all_error_variants_are_errors() {
        // Ensure all variants implement Error trait
        fn assert_error<E: std::error::Error>(_: &E) {}
        
        assert_error(&KjarniError::UnknownModel("m".into()));
        assert_error(&KjarniError::IncompatibleModel {
            model: "m".into(),
            task: "t".into(),
            reason: "r".into(),
        });
        assert_error(&KjarniError::ModelNotDownloaded("m".into()));
        assert_error(&KjarniError::DownloadFailed {
            model: "m".into(),
            source: anyhow::anyhow!("e"),
        });
        assert_error(&KjarniError::LoadFailed {
            model: "m".into(),
            source: anyhow::anyhow!("e"),
        });
        assert_error(&KjarniError::InferenceFailed(anyhow::anyhow!("e")));
        assert_error(&KjarniError::GpuUnavailable);
        assert_error(&KjarniError::InvalidConfig("c".into()));
        assert_error(&KjarniError::NoLabels);
    }

    #[test]
    fn test_all_error_variants_are_debug() {
        fn assert_debug<D: std::fmt::Debug>(_: &D) {}
        
        assert_debug(&KjarniError::UnknownModel("m".into()));
        assert_debug(&KjarniError::IncompatibleModel {
            model: "m".into(),
            task: "t".into(),
            reason: "r".into(),
        });
        assert_debug(&KjarniError::ModelNotDownloaded("m".into()));
        assert_debug(&KjarniError::GpuUnavailable);
        assert_debug(&KjarniError::InvalidConfig("c".into()));
        assert_debug(&KjarniError::NoLabels);
    }

    // =========================================================================
    // Edge cases
    // =========================================================================

    #[test]
    fn test_empty_model_name() {
        let err = KjarniError::UnknownModel(String::new());
        let msg = err.to_string();
        assert!(msg.contains("''"));
    }

    #[test]
    fn test_unicode_in_error() {
        let err = KjarniError::UnknownModel("模型名称".to_string());
        let msg = err.to_string();
        assert!(msg.contains("模型名称"));
    }

    #[test]
    fn test_special_chars_in_error() {
        let err = KjarniError::InvalidConfig("value must be > 0 && < 100".to_string());
        let msg = err.to_string();
        assert!(msg.contains("value must be > 0 && < 100"));
    }

    #[test]
    fn test_very_long_message() {
        let long_msg = "a".repeat(10000);
        let err = KjarniError::InvalidConfig(long_msg.clone());
        let msg = err.to_string();
        assert!(msg.contains(&long_msg));
    }

    #[test]
    fn test_warning_empty_message() {
        let warning = KjarniWarning::new("");
        assert_eq!(warning.to_string(), "Warning: ");
    }

    #[test]
    fn test_warning_empty_suggestion() {
        let warning = KjarniWarning::with_suggestion("msg", "");
        let display = warning.to_string();
        // Empty suggestion should still be appended with space
        assert!(display.ends_with(" "));
    }

    // =========================================================================
    // Practical usage tests
    // =========================================================================

    #[test]
    fn test_error_in_result_chain() {
        fn load_model(name: &str) -> KjarniResult<String> {
            if name == "unknown" {
                return Err(KjarniError::UnknownModel(name.to_string()));
            }
            Ok(format!("loaded: {}", name))
        }
        
        assert!(load_model("minilm").is_ok());
        assert!(load_model("unknown").is_err());
    }

    #[test]
    fn test_error_matching() {
        let err = KjarniError::GpuUnavailable;
        
        match err {
            KjarniError::GpuUnavailable => {}
            _ => panic!("Expected GpuUnavailable"),
        }
    }

    #[test]
    fn test_incompatible_model_pattern_matching() {
        let err = KjarniError::IncompatibleModel {
            model: "bert".to_string(),
            task: "generation".to_string(),
            reason: "wrong architecture".to_string(),
        };
        
        match err {
            KjarniError::IncompatibleModel { model, task, reason } => {
                assert_eq!(model, "bert");
                assert_eq!(task, "generation");
                assert_eq!(reason, "wrong architecture");
            }
            _ => panic!("Expected IncompatibleModel"),
        }
    }

    #[test]
    fn test_warnings_can_be_collected() {
        let warnings: Vec<KjarniWarning> = vec![
            KjarniWarning::new("Warning 1"),
            KjarniWarning::with_suggestion("Warning 2", "Fix it"),
            KjarniWarning::new("Warning 3"),
        ];
        
        assert_eq!(warnings.len(), 3);
        
        for warning in &warnings {
            assert!(warning.to_string().starts_with("Warning:"));
        }
    }

    // =========================================================================
    // Source chain tests
    // =========================================================================

    #[test]
    fn test_nested_error_sources() {
        let inner = anyhow::anyhow!("Inner error");
        let err = KjarniError::DownloadFailed {
            model: "test".to_string(),
            source: inner,
        };
        
        // Should be able to get the source
        let source = err.source().expect("Should have source");
        assert!(source.to_string().contains("Inner error"));
    }

    #[test]
    fn test_display_includes_source() {
        let inner = anyhow::anyhow!("Root cause");
        let err = KjarniError::LoadFailed {
            model: "mymodel".to_string(),
            source: inner,
        };
        
        let display = err.to_string();
        assert!(display.contains("Root cause"));
        assert!(display.contains("mymodel"));
    }

    // =========================================================================
    // Formatting tests
    // =========================================================================

    #[test]
    fn test_display_vs_debug_different() {
        let err = KjarniError::GpuUnavailable;
        
        let display = format!("{}", err);
        let debug = format!("{:?}", err);
        
        // Display should be human-readable message
        assert!(display.contains("GPU"));
        
        // Debug should contain variant name
        assert!(debug.contains("GpuUnavailable"));
    }

    #[test]
    fn test_error_formatting_in_panic_message() {
        let err = KjarniError::NoLabels;
        let msg = format!("Operation failed: {}", err);
        assert!(msg.contains("Operation failed:"));
        assert!(msg.contains("label"));
    }
}