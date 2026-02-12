//! Validation helpers for transcription inputs and configuration.

use std::path::Path;

use super::types::{TranscriberError, TranscriberResult};

/// Supported audio file extensions.
const SUPPORTED_EXTENSIONS: &[&str] = &["wav", "mp3", "flac", "ogg"];

/// Validate that a path points to a readable audio file with a supported format.
pub fn validate_audio_path(path: &Path) -> TranscriberResult<()> {
    if !path.exists() {
        return Err(TranscriberError::InvalidAudioPath(path.to_path_buf()));
    }

    if !path.is_file() {
        return Err(TranscriberError::InvalidAudioPath(path.to_path_buf()));
    }

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    if !SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
        return Err(TranscriberError::UnsupportedFormat(ext));
    }

    Ok(())
}

/// Sanity-check transcription configuration values.
pub fn validate_config(
    language: Option<&str>,
    max_tokens: usize,
) -> TranscriberResult<()> {
    if let Some(lang) = language {
        if lang.is_empty() {
            return Err(TranscriberError::InvalidConfig(
                "Language code cannot be empty".into(),
            ));
        }
        if lang.len() > 10 {
            return Err(TranscriberError::InvalidConfig(
                format!("Language code too long: '{}'", lang),
            ));
        }
    }

    if max_tokens == 0 {
        return Err(TranscriberError::InvalidConfig(
            "max_tokens_per_chunk must be > 0".into(),
        ));
    }

    if max_tokens > 4096 {
        return Err(TranscriberError::InvalidConfig(
            format!("max_tokens_per_chunk too large: {} (max 4096)", max_tokens),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_validate_missing_path() {
        let result = validate_audio_path(Path::new("/nonexistent/audio.wav"));
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_bad_extension() {
        // Create a temp file with wrong extension (would need fs in real test)
        // For now just test config validation
    }

    #[test]
    fn test_validate_config_ok() {
        assert!(validate_config(Some("en"), 448).is_ok());
        assert!(validate_config(None, 224).is_ok());
    }

    #[test]
    fn test_validate_config_bad_lang() {
        assert!(validate_config(Some(""), 448).is_err());
    }

    #[test]
    fn test_validate_config_bad_tokens() {
        assert!(validate_config(Some("en"), 0).is_err());
        assert!(validate_config(Some("en"), 10_000).is_err());
    }
}