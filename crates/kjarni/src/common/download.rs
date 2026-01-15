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

/// Ensure a model is downloaded according to the policy.
///
/// Returns the model directory path.
pub async fn ensure_model_downloaded(
    model_type: ModelType,
    cache_dir: Option<&Path>,
    policy: DownloadPolicy,
    quiet: bool,
) -> KjarniResult<PathBuf> {
    let cache_dir = cache_dir
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_cache_dir);

    let model_dir = model_type.cache_dir(&cache_dir);
    let is_downloaded = model_type.is_downloaded(&cache_dir);

    match (is_downloaded, policy) {
        (true, DownloadPolicy::Never | DownloadPolicy::IfMissing) => {
            // Already downloaded, nothing to do
            Ok(model_dir)
        }
        (true, DownloadPolicy::Eager) => {
            // TODO: Check for updates
            // For now, treat as already downloaded
            Ok(model_dir)
        }
        (false, DownloadPolicy::Never) => {
            Err(KjarniError::ModelNotDownloaded(model_type.cli_name().to_string()))
        }
        (false, DownloadPolicy::IfMissing | DownloadPolicy::Eager) => {
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