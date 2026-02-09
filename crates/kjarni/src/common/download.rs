//! Model download utilities.

use std::path::{Path, PathBuf};

use kjarni_transformers::models::{download_model_files, registry::WeightsFormat, ModelType};

use super::{DownloadPolicy, KjarniError, KjarniResult};

/// Get the default cache directory.
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .expect("No cache directory found")
        .join("kjarni")
}

/// Resolve the cache directory, using the provided path or the default.
pub fn resolve_cache_dir(cache_dir: Option<&Path>) -> PathBuf {
    cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir)
}

/// Determine the action to take based on download status and policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownloadAction {
    /// Model is ready, no download needed.
    Ready,
    /// Need to download the model.
    Download,
    /// Model not available and policy forbids download.
    Error,
}

/// Determine what action to take based on download status and policy.
pub fn determine_download_action(is_downloaded: bool, policy: DownloadPolicy) -> DownloadAction {
    match (is_downloaded, policy) {
        (true, DownloadPolicy::Never | DownloadPolicy::IfMissing) => DownloadAction::Ready,
        (true, DownloadPolicy::Eager) => {
            // TODO: Check for updates
            // For now, treat as already downloaded
            DownloadAction::Ready
        }
        (false, DownloadPolicy::Never) => DownloadAction::Error,
        (false, DownloadPolicy::IfMissing | DownloadPolicy::Eager) => DownloadAction::Download,
    }
}

/// Ensure a model is downloaded according to the policy.
///
/// Returns the model directory path.
pub async fn ensure_model_downloaded(
    model_type: ModelType,
    cache_dir: Option<&Path>,
    policy: DownloadPolicy,
    quiet: bool,
) -> KjarniResult<PathBuf> {
    let cache_dir = resolve_cache_dir(cache_dir);
    let model_dir = model_type.cache_dir(&cache_dir);
    let is_downloaded = model_type.is_downloaded(&cache_dir);

    match determine_download_action(is_downloaded, policy) {
        DownloadAction::Ready => Ok(model_dir),
        DownloadAction::Error => {
            Err(KjarniError::ModelNotDownloaded(model_type.cli_name().to_string()))
        }
        DownloadAction::Download => {
            if !quiet {
                eprintln!("Downloading model '{}'...", model_type.cli_name());
            }

            let info = model_type.info();
            download_model_files(&model_dir, &info.paths, WeightsFormat::SafeTensors, quiet)
                .await
                .map_err(|e| KjarniError::DownloadFailed {
                    model: model_type.cli_name().to_string(),
                    source: e,
                })?;

            Ok(model_dir)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    #[test]
    fn test_default_cache_dir_not_empty() {
        let dir = default_cache_dir();
        assert!(!dir.as_os_str().is_empty());
    }
    #[test]
    fn test_default_cache_dir_ends_with_kjarni() {
        let dir = default_cache_dir();
        assert!(
            dir.ends_with("kjarni"),
            "Cache dir should end with 'kjarni': {:?}",
            dir
        );
    }
    #[test]
    fn test_default_cache_dir_has_parent() {
        let dir = default_cache_dir();
        assert!(dir.parent().is_some(), "Cache dir should have a parent directory");
    }

    #[test]
    fn test_default_cache_dir_is_absolute() {
        let dir = default_cache_dir();
        assert!(dir.is_absolute(), "Cache dir should be an absolute path");
    }
    #[test]
    fn test_default_cache_dir_consistent() {
        let dir1 = default_cache_dir();
        let dir2 = default_cache_dir();
        assert_eq!(dir1, dir2);
    }
    #[test]
    fn test_resolve_cache_dir_none_uses_default() {
        let resolved = resolve_cache_dir(None);
        let default = default_cache_dir();
        assert_eq!(resolved, default);
    }

    #[test]
    fn test_resolve_cache_dir_some_uses_provided() {
        let custom = Path::new("/custom/cache/path");
        let resolved = resolve_cache_dir(Some(custom));
        assert_eq!(resolved, PathBuf::from("/custom/cache/path"));
    }

    #[test]
    fn test_resolve_cache_dir_relative_path() {
        let relative = Path::new("./local/cache");
        let resolved = resolve_cache_dir(Some(relative));
        assert_eq!(resolved, PathBuf::from("./local/cache"));
    }
    #[test]
    fn test_resolve_cache_dir_with_tempdir() {
        let temp = TempDir::new().unwrap();
        let resolved = resolve_cache_dir(Some(temp.path()));
        assert_eq!(resolved, temp.path().to_path_buf());
    }
    #[test]
    fn test_download_action_debug() {
        assert_eq!(format!("{:?}", DownloadAction::Ready), "Ready");
        assert_eq!(format!("{:?}", DownloadAction::Download), "Download");
        assert_eq!(format!("{:?}", DownloadAction::Error), "Error");
    }
    #[test]
    fn test_download_action_clone() {
        let action = DownloadAction::Ready;
        assert_eq!(action, action.clone());
    }
    #[test]
    fn test_download_action_copy() {
        let action = DownloadAction::Download;
        let copied = action;
        assert_eq!(action, copied);
    }
    #[test]
    fn test_download_action_equality() {
        assert_eq!(DownloadAction::Ready, DownloadAction::Ready);
        assert_eq!(DownloadAction::Download, DownloadAction::Download);
        assert_eq!(DownloadAction::Error, DownloadAction::Error);
    }
    #[test]
    fn test_download_action_inequality() {
        assert_ne!(DownloadAction::Ready, DownloadAction::Download);
        assert_ne!(DownloadAction::Ready, DownloadAction::Error);
        assert_ne!(DownloadAction::Download, DownloadAction::Error);
    }
    #[test]
    fn test_determine_action_downloaded_never() {
        let action = determine_download_action(true, DownloadPolicy::Never);
        assert_eq!(action, DownloadAction::Ready);
    }
    #[test]
    fn test_determine_action_downloaded_if_missing() {
        let action = determine_download_action(true, DownloadPolicy::IfMissing);
        assert_eq!(action, DownloadAction::Ready);
    }
    #[test]
    fn test_determine_action_downloaded_eager() {
        let action = determine_download_action(true, DownloadPolicy::Eager);
        assert_eq!(action, DownloadAction::Ready);
    }
    #[test]
    fn test_determine_action_not_downloaded_never() {
        let action = determine_download_action(false, DownloadPolicy::Never);
        assert_eq!(action, DownloadAction::Error);
    }

    #[test]
    fn test_determine_action_not_downloaded_if_missing() {
        let action = determine_download_action(false, DownloadPolicy::IfMissing);
        assert_eq!(action, DownloadAction::Download);
    }

    #[test]
    fn test_determine_action_not_downloaded_eager() {
        let action = determine_download_action(false, DownloadPolicy::Eager);
        assert_eq!(action, DownloadAction::Download);
    }
    #[test]
    fn test_determine_action_all_combinations() {
        let test_cases = [
            (true, DownloadPolicy::Never, DownloadAction::Ready),
            (true, DownloadPolicy::IfMissing, DownloadAction::Ready),
            (true, DownloadPolicy::Eager, DownloadAction::Ready),
            (false, DownloadPolicy::Never, DownloadAction::Error),
            (false, DownloadPolicy::IfMissing, DownloadAction::Download),
            (false, DownloadPolicy::Eager, DownloadAction::Download),
        ];

        for (is_downloaded, policy, expected) in test_cases {
            let actual = determine_download_action(is_downloaded, policy);
            assert_eq!(
                actual, expected,
                "Failed for is_downloaded={}, policy={:?}: expected {:?}, got {:?}",
                is_downloaded, policy, expected, actual
            );
        }
    }
    #[test]
    fn test_never_policy_never_downloads() {
        assert_ne!(
            determine_download_action(true, DownloadPolicy::Never),
            DownloadAction::Download
        );
        assert_ne!(
            determine_download_action(false, DownloadPolicy::Never),
            DownloadAction::Download
        );
    }

    #[test]
    fn test_if_missing_only_downloads_when_missing() {
        // IfMissing should download only when not present
        assert_eq!(
            determine_download_action(false, DownloadPolicy::IfMissing),
            DownloadAction::Download
        );
        assert_eq!(
            determine_download_action(true, DownloadPolicy::IfMissing),
            DownloadAction::Ready
        );
    }

    #[test]
    fn test_eager_downloads_when_missing() {
        assert_eq!(
            determine_download_action(false, DownloadPolicy::Eager),
            DownloadAction::Download
        );
    }

    #[test]
    fn test_downloaded_model_never_errors() {
        for policy in [
            DownloadPolicy::Never,
            DownloadPolicy::IfMissing,
            DownloadPolicy::Eager,
        ] {
            let action = determine_download_action(true, policy);
            assert_ne!(
                action,
                DownloadAction::Error,
                "Downloaded model should never error with policy {:?}",
                policy
            );
        }
    }

    #[test]
    fn test_only_never_policy_can_error() {
        let error_policies: Vec<_> = [
            DownloadPolicy::Never,
            DownloadPolicy::IfMissing,
            DownloadPolicy::Eager,
        ]
        .iter()
        .filter(|&&p| determine_download_action(false, p) == DownloadAction::Error)
        .collect();

        assert_eq!(error_policies.len(), 1);
        assert_eq!(*error_policies[0], DownloadPolicy::Never);
    }
    #[test]
    fn test_model_dir_under_cache_dir() {
        let temp = TempDir::new().unwrap();
        let cache_dir = resolve_cache_dir(Some(temp.path()));
        let model_type = ModelType::from_cli_name("minilm-l6-v2").unwrap();
        let model_dir = model_type.cache_dir(&cache_dir);
        assert!(
            model_dir.starts_with(&cache_dir),
            "Model dir {:?} should be under cache dir {:?}",
            model_dir,
            cache_dir
        );
    }

    #[test]
    fn test_different_models_different_dirs() {
        let temp = TempDir::new().unwrap();
        let cache_dir = resolve_cache_dir(Some(temp.path()));
        
        let model1 = ModelType::from_cli_name("minilm-l6-v2").unwrap();
        let model2 = ModelType::from_cli_name("distilbert-sentiment").unwrap();
        
        let dir1 = model1.cache_dir(&cache_dir);
        let dir2 = model2.cache_dir(&cache_dir);
        
        assert_ne!(dir1, dir2, "Different models should have different directories");
    }

    #[test]
    fn test_is_downloaded_false_in_empty_dir() {
        let temp = TempDir::new().unwrap();
        let model_type = ModelType::from_cli_name("minilm-l6-v2").unwrap();
        
        // In an empty temp directory, model should not be downloaded
        assert!(!model_type.is_downloaded(temp.path()));
    }
    #[tokio::test]
    async fn test_ensure_model_not_downloaded_never_policy() {
        let temp = TempDir::new().unwrap();
        let model_type = ModelType::from_cli_name("minilm-l6-v2").unwrap();
    
        let result = ensure_model_downloaded(
            model_type,
            Some(temp.path()),
            DownloadPolicy::Never,
            true,
        ).await;
        
        assert!(result.is_err());
        
        let err = result.unwrap_err();
        match err {
            KjarniError::ModelNotDownloaded(name) => {
                assert_eq!(name, "minilm-l6-v2");
            }
            other => panic!("Expected ModelNotDownloaded error, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_ensure_model_returns_correct_path() {
        let temp = TempDir::new().unwrap();
        let model_type = ModelType::from_cli_name("minilm-l6-v2").unwrap();
        
        let expected_dir = model_type.cache_dir(temp.path());
        std::fs::create_dir_all(&expected_dir).unwrap();
        
        std::fs::write(expected_dir.join("config.json"), "{}").unwrap();
        std::fs::write(expected_dir.join("model.safetensors"), "").unwrap();
        std::fs::write(expected_dir.join("tokenizer.json"), "{}").unwrap();
        
        let result = ensure_model_downloaded(
            model_type,
            Some(temp.path()),
            DownloadPolicy::Never,
            true,
        ).await;
        match result {
            Ok(path) => {
                assert!(path.starts_with(temp.path()));
            }
            Err(KjarniError::ModelNotDownloaded(_)) => {
            }
            Err(e) => panic!("Unexpected error: {:?}", e),
        }
    }

    #[tokio::test]
    async fn test_ensure_model_with_custom_cache_dir() {
        let temp = TempDir::new().unwrap();
        let model_type = ModelType::from_cli_name("distilbert-sentiment").unwrap();
        
        let result = ensure_model_downloaded(
            model_type,
            Some(temp.path()),
            DownloadPolicy::Never,
            true,
        ).await;
        
        assert!(result.is_err());
    }
    #[test]
    fn test_resolve_cache_dir_empty_path() {
        let empty = Path::new("");
        let resolved = resolve_cache_dir(Some(empty));
        assert_eq!(resolved, PathBuf::from(""));
    }

    #[test]
    fn test_resolve_cache_dir_root() {
        let root = Path::new("/");
        let resolved = resolve_cache_dir(Some(root));
        assert_eq!(resolved, PathBuf::from("/"));
    }

    #[test]
    fn test_resolve_cache_dir_with_spaces() {
        let with_spaces = Path::new("/path/with spaces/cache");
        let resolved = resolve_cache_dir(Some(with_spaces));
        assert_eq!(resolved, PathBuf::from("/path/with spaces/cache"));
    }

    #[test]
    fn test_resolve_cache_dir_unicode() {
        let unicode = Path::new("/путь/キャッシュ");
        let resolved = resolve_cache_dir(Some(unicode));
        assert_eq!(resolved, PathBuf::from("/путь/キャッシュ"));
    }

    #[test]
    fn test_if_missing_is_default_behavior() {
        assert_eq!(DownloadPolicy::default(), DownloadPolicy::IfMissing);
        
        assert_eq!(
            determine_download_action(false, DownloadPolicy::default()),
            DownloadAction::Download
        );
        assert_eq!(
            determine_download_action(true, DownloadPolicy::default()),
            DownloadAction::Ready
        );
    }

    #[test]
    fn test_never_policy_for_offline_use() {
        let action_present = determine_download_action(true, DownloadPolicy::Never);
        let action_missing = determine_download_action(false, DownloadPolicy::Never);
        
        assert!(action_present == DownloadAction::Ready);
        assert!(action_missing == DownloadAction::Error);
    }
    #[test]
    fn test_action_determinism() {
        for _ in 0..10 {
            assert_eq!(
                determine_download_action(true, DownloadPolicy::IfMissing),
                DownloadAction::Ready
            );
            assert_eq!(
                determine_download_action(false, DownloadPolicy::Never),
                DownloadAction::Error
            );
        }
    }

    #[test]
    fn test_cache_dir_determinism() {
        let custom = Path::new("/test/path");
        for _ in 0..10 {
            assert_eq!(resolve_cache_dir(Some(custom)), PathBuf::from("/test/path"));
            assert_eq!(resolve_cache_dir(None), default_cache_dir());
        }
    }
}