//! Common types and utilities shared across kjarni modules.

mod device;
mod download;
mod error;
mod load_config;

pub use device::{DownloadPolicy, KjarniDevice};
pub use download::{default_cache_dir, ensure_model_downloaded};
pub use error::{KjarniError, KjarniResult};
pub use load_config::{LoadConfig, LoadConfigBuilder};

/// A folder's name, to call a model loaded from it by.
pub(crate) fn folder_name(path: &std::path::Path) -> String {
    path.file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}
