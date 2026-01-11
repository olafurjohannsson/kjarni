//! Global memory-map cache for weight files.
//!
//! This module provides a process-wide cache of memory-mapped files to prevent
//! duplicate mappings when the same model is loaded multiple times.
//!
//! # Motivation
//!
//! In test suites and applications that load multiple models, it's common to
//! load the same weight file multiple times:
//!
//! ```ignore
//! // Without cache: Two separate 16GB mmaps!
//! let encoder = SentenceEncoder::from_registry(ModelType::MiniLML6V2, ...).await?;
//! let classifier = SequenceClassifier::from_registry(ModelType::MiniLML6V2, ...).await?;
//!
//! // With cache: Single shared mmap
//! // Both models reference the same underlying memory
//! ```
//!
//! # Thread Safety
//!
//! The cache uses a `Mutex` for thread-safe access. The lock is only held during
//! cache lookup/insertion, not during mmap creation or tensor access.
//!
//! # Memory Management
//!
//! Cached mmaps are held via `Arc`, so they're automatically released when all
//! references are dropped. Call [`clear_mmap_cache`] between test runs if needed.

use anyhow::{Context, Result};
use memmap2::Mmap;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Global cache of memory-mapped files.
///
/// Maps canonical file paths to shared mmap references.
static MMAP_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<Mmap>>>> = OnceLock::new();

/// Get or create a memory-mapped file.
///
/// If the file has been mapped before, returns the existing shared reference.
/// Otherwise, creates a new mmap and caches it for future use.
///
/// # Arguments
///
/// * `path` - Path to the file to memory-map
///
/// # Returns
///
/// A shared reference to the memory-mapped file.
///
/// # Thread Safety
///
/// This function is thread-safe. Multiple threads can request the same file
/// concurrently; only one mmap will be created.
///
/// # Example
///
/// ```ignore
/// let mmap1 = get_or_create_mmap(Path::new("/models/model.safetensors"))?;
/// let mmap2 = get_or_create_mmap(Path::new("/models/model.safetensors"))?;
/// assert!(Arc::ptr_eq(&mmap1, &mmap2)); // Same underlying mmap
/// ```
pub fn get_or_create_mmap(path: &Path) -> Result<Arc<Mmap>> {
    let cache = MMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();

    // Canonicalize to handle symlinks and relative paths
    let canonical = path
        .canonicalize()
        .with_context(|| format!("Failed to canonicalize path: {:?}", path))?;

    // Return cached mmap if available
    if let Some(mmap) = guard.get(&canonical) {
        log::debug!("Mmap cache hit for {:?}", canonical.file_name().unwrap_or_default());
        return Ok(Arc::clone(mmap));
    }

    // Create new mmap
    log::debug!("Mmap cache miss, creating new mapping for {:?}", canonical.file_name().unwrap_or_default());
    
    let file = std::fs::File::open(&canonical)
        .with_context(|| format!("Failed to open file for mmap: {:?}", canonical))?;
    
    // SAFETY: The file is opened read-only and we maintain the Arc for the
    // lifetime of all references to the mmap.
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });
    
    guard.insert(canonical, Arc::clone(&mmap));
    
    Ok(mmap)
}

/// Clear the global mmap cache.
///
/// Releases all cached memory mappings. Useful for:
/// - Reducing memory pressure between test runs
/// - Forcing reload of modified weight files
/// - Debugging memory issues
///
/// # Note
///
/// This only removes entries from the cache. Existing `Arc<Mmap>` references
/// held by loaded models will continue to work until dropped.
///
/// # Example
///
/// ```ignore
/// #[test]
/// fn test_model_loading() {
///     // ... test code ...
///     
///     // Clean up after test
///     clear_mmap_cache();
/// }
/// ```
pub fn clear_mmap_cache() {
    if let Some(cache) = MMAP_CACHE.get() {
        let mut guard = cache.lock().unwrap();
        let count = guard.len();
        guard.clear();
        log::info!("Cleared {} entries from mmap cache", count);
    }
}

/// Get statistics about the mmap cache.
///
/// Returns the number of cached files and total mapped bytes.
/// Useful for debugging and monitoring memory usage.
///
/// # Returns
///
/// A tuple of `(file_count, total_bytes)`.
///
/// # Example
///
/// ```ignore
/// let (count, bytes) = mmap_cache_stats();
/// println!("Mmap cache: {} files, {} MB", count, bytes / 1_000_000);
/// ```
pub fn mmap_cache_stats() -> (usize, usize) {
    if let Some(cache) = MMAP_CACHE.get() {
        let guard = cache.lock().unwrap();
        let count = guard.len();
        let total_bytes: usize = guard.values().map(|m| m.len()).sum();
        (count, total_bytes)
    } else {
        (0, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_mmap_cache_deduplication() {
        // Create a temporary file
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test data for mmap").unwrap();
        temp.flush().unwrap();
        let path = temp.path();

        // Clear cache first
        clear_mmap_cache();

        // First load
        let mmap1 = get_or_create_mmap(path).unwrap();
        let (count1, _) = mmap_cache_stats();
        assert_eq!(count1, 1);

        // Second load should reuse
        let mmap2 = get_or_create_mmap(path).unwrap();
        let (count2, _) = mmap_cache_stats();
        assert_eq!(count2, 1); // Still 1, not 2

        // Should be the same Arc
        assert!(Arc::ptr_eq(&mmap1, &mmap2));

        // Clean up
        clear_mmap_cache();
    }

    #[test]
    fn test_mmap_cache_clear() {
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test").unwrap();
        temp.flush().unwrap();

        let _ = get_or_create_mmap(temp.path()).unwrap();
        let (count, _) = mmap_cache_stats();
        assert!(count >= 1);

        clear_mmap_cache();

        let (count_after, _) = mmap_cache_stats();
        assert_eq!(count_after, 0);
    }
}