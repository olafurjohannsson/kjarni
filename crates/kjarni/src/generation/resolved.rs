//! Fully resolved generation configuration.
//!
//! This is what the decoder pipeline actually consumes.
//! Created by merging model defaults with user/runtime overrides.

use kjarni_transformers::common::GenerationConfig;

/// Fully resolved generation configuration.
///
/// This wraps the engine's `GenerationConfig` after all overrides
/// have been applied. It's immutable once created.
#[derive(Debug, Clone)]
pub struct ResolvedGenerationConfig {
    pub(crate) inner: GenerationConfig,
}

impl ResolvedGenerationConfig {
    /// Create from a resolved config.
    pub fn new(config: GenerationConfig) -> Self {
        Self { inner: config }
    }

    /// Consume and return the inner config.
    pub fn into_inner(self) -> GenerationConfig {
        self.inner
    }

    /// Get a reference to the inner config.
    pub fn as_ref(&self) -> &GenerationConfig {
        &self.inner
    }

    /// Get the max new tokens setting.
    pub fn max_new_tokens(&self) -> Option<usize> {
        self.inner.max_new_tokens
    }

    /// Check if using sampling strategy.
    pub fn is_sampling(&self) -> bool {
        matches!(
            self.inner.strategy,
            kjarni_transformers::common::DecodingStrategy::Sample(_)
        )
    }

    /// Check if using beam search.
    pub fn is_beam_search(&self) -> bool {
        matches!(
            self.inner.strategy,
            kjarni_transformers::common::DecodingStrategy::BeamSearch(_)
        )
    }

    /// Check if using greedy decoding.
    pub fn is_greedy(&self) -> bool {
        matches!(
            self.inner.strategy,
            kjarni_transformers::common::DecodingStrategy::Greedy
        )
    }
}

impl AsRef<GenerationConfig> for ResolvedGenerationConfig {
    fn as_ref(&self) -> &GenerationConfig {
        &self.inner
    }
}

impl From<GenerationConfig> for ResolvedGenerationConfig {
    fn from(config: GenerationConfig) -> Self {
        Self::new(config)
    }
}
