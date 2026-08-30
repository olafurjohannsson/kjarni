pub mod base;
pub mod registry;

// Re-export commonly used items
pub use base::LanguageModel;

pub use registry::{
    ModelArchitecture, ModelInfo, ModelPaths, ModelTask, ModelType, format_params, format_size,
    get_default_cache_dir,
};

#[cfg(not(target_arch = "wasm32"))]
pub use registry::download_model_files;

#[cfg(test)]
mod tests;
