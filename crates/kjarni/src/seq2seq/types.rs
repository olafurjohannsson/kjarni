//! Types for seq2seq generation.

use thiserror::Error;

// =============================================================================
// Errors
// =============================================================================

/// Errors that can occur during seq2seq generation.
#[derive(Debug, Error)]
pub enum Seq2SeqError {
    /// Model name not found in registry.
    #[error("Unknown model: '{0}'. Use seq2seq::available_models() to list valid models.")]
    UnknownModel(String),

    /// Model exists but hasn't been downloaded.
    #[error("Model '{0}' not downloaded. Set download_policy(DownloadPolicy::IfMissing) or download manually.")]
    ModelNotDownloaded(String),

    /// Download failed.
    #[error("Failed to download model '{model}': {source}")]
    DownloadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// Model loading failed.
    #[error("Failed to load model '{model}': {source}")]
    LoadFailed {
        model: String,
        #[source]
        source: anyhow::Error,
    },

    /// GPU requested but not available.
    #[error("GPU unavailable. Use .cpu() or ensure GPU drivers are installed.")]
    GpuUnavailable,

    /// Generation failed during inference.
    #[error("Generation failed: {0}")]
    GenerationFailed(#[source] anyhow::Error),

    /// Model architecture incompatible with seq2seq.
    #[error("Model '{model}' incompatible: {reason}")]
    IncompatibleModel { model: String, reason: String },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),
}

/// Result type for seq2seq operations.
pub type Seq2SeqResult<T> = Result<T, Seq2SeqError>;

// =============================================================================
// Generated Token
// =============================================================================

/// A single token from streaming generation.
#[derive(Debug, Clone)]
pub struct Seq2SeqToken {
    /// The decoded text for this token.
    pub text: String,

    /// The token ID.
    pub id: u32,

    /// Whether this is a special token (EOS, PAD, etc.).
    pub is_special: bool,
}

// =============================================================================
// Task Hint
// =============================================================================

/// Hint for what task we're doing (for models that support task-specific config).
///
/// Some models (like T5) have different default generation parameters for
/// different tasks. This enum allows the generator to select appropriate defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Seq2SeqTask {
    /// Translation task (typically: no min_length, no n-gram blocking).
    Translation,

    /// Summarization task (typically: min_length, n-gram blocking, length penalty).
    Summarization,

    /// General text-to-text (use model defaults).
    #[default]
    General,
}

// =============================================================================
// Overrides
// =============================================================================

/// User-facing overrides for seq2seq generation.
///
/// All fields are optional. `None` means "use model/task default".
/// This exposes seq2seq-appropriate parameters (beam search, length control)
/// rather than sampling parameters used by decoder-only models.
///
/// # Example
///
/// ```ignore
/// use kjarni::seq2seq::Seq2SeqOverrides;
///
/// // Use greedy decoding for speed
/// let fast = Seq2SeqOverrides::greedy();
///
/// // High quality with more beams
/// let quality = Seq2SeqOverrides {
///     num_beams: Some(6),
///     length_penalty: Some(1.5),
///     ..Default::default()
/// };
///
/// // Short summary
/// let short = Seq2SeqOverrides::short_summary();
/// ```
#[derive(Debug, Clone, Default)]
pub struct Seq2SeqOverrides {
    // =========================================================================
    // Length Control
    // =========================================================================
    /// Minimum output length in tokens.
    ///
    /// - Summarization: typically 30-100
    /// - Translation: typically 0 (don't force minimum)
    pub min_length: Option<usize>,

    /// Maximum output length in tokens.
    ///
    /// - Summarization: typically 100-300
    /// - Translation: typically 2-3x input length
    pub max_length: Option<usize>,

    // =========================================================================
    // Beam Search Control
    // =========================================================================
    /// Number of beams for beam search.
    ///
    /// - `None` = use model default (typically 4)
    /// - `Some(1)` = greedy decoding (fastest, lower quality)
    /// - `Some(4-6)` = good quality/speed balance
    /// - `Some(8+)` = higher quality, slower
    pub num_beams: Option<usize>,

    /// Length penalty for beam search.
    ///
    /// - `< 1.0` = favor shorter outputs
    /// - `= 1.0` = neutral (T5 default)
    /// - `> 1.0` = favor longer outputs (BART summarization uses 2.0)
    pub length_penalty: Option<f32>,

    /// Stop when `num_beams` complete sequences are found.
    ///
    /// - `true` = stop early (faster, default for most models)
    /// - `false` = continue until max_length
    pub early_stopping: Option<bool>,

    // =========================================================================
    // Repetition Control
    // =========================================================================
    /// Prevent repeating n-grams of this size.
    ///
    /// - `0` = disabled (use for translation)
    /// - `3` = typical for summarization (prevents repeating phrases)
    pub no_repeat_ngram_size: Option<usize>,

    /// Penalty for repeating tokens (1.0 = no penalty).
    pub repetition_penalty: Option<f32>,
}

impl Seq2SeqOverrides {
    /// Create overrides optimized for translation.
    ///
    /// - No minimum length (translations can be short)
    /// - No n-gram blocking (repetition is natural in translation)
    pub fn for_translation() -> Self {
        Self {
            min_length: Some(0),
            no_repeat_ngram_size: Some(0),
            ..Default::default()
        }
    }

    /// Create overrides optimized for summarization.
    ///
    /// - N-gram blocking to prevent phrase repetition
    pub fn for_summarization() -> Self {
        Self {
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        }
    }

    /// Create overrides for short summaries (30-60 tokens).
    pub fn short_summary() -> Self {
        Self {
            min_length: Some(30),
            max_length: Some(60),
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        }
    }

    /// Create overrides for medium summaries (50-150 tokens).
    pub fn medium_summary() -> Self {
        Self {
            min_length: Some(50),
            max_length: Some(150),
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        }
    }

    /// Create overrides for long summaries (100-300 tokens).
    pub fn long_summary() -> Self {
        Self {
            min_length: Some(100),
            max_length: Some(300),
            no_repeat_ngram_size: Some(3),
            ..Default::default()
        }
    }

    /// Use greedy decoding (fastest, single beam).
    pub fn greedy() -> Self {
        Self {
            num_beams: Some(1),
            ..Default::default()
        }
    }

    /// Use high-quality beam search (6 beams).
    pub fn high_quality() -> Self {
        Self {
            num_beams: Some(6),
            ..Default::default()
        }
    }

    /// Check if any overrides are set.
    pub fn is_empty(&self) -> bool {
        self.min_length.is_none()
            && self.max_length.is_none()
            && self.num_beams.is_none()
            && self.length_penalty.is_none()
            && self.early_stopping.is_none()
            && self.no_repeat_ngram_size.is_none()
            && self.repetition_penalty.is_none()
    }

    /// Merge with another set of overrides.
    ///
    /// Values from `other` take precedence where set.
    pub fn merge(&self, other: &Self) -> Self {
        Self {
            min_length: other.min_length.or(self.min_length),
            max_length: other.max_length.or(self.max_length),
            num_beams: other.num_beams.or(self.num_beams),
            length_penalty: other.length_penalty.or(self.length_penalty),
            early_stopping: other.early_stopping.or(self.early_stopping),
            no_repeat_ngram_size: other.no_repeat_ngram_size.or(self.no_repeat_ngram_size),
            repetition_penalty: other.repetition_penalty.or(self.repetition_penalty),
        }
    }
}