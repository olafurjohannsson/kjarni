//! Global memory-map cache for weight files
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result};
use memmap2::Mmap;

static MMAP_CACHE: OnceLock<Mutex<HashMap<PathBuf, Arc<Mmap>>>> = OnceLock::new();

/// Returns a shared memory-mapped file
pub fn get_or_create_mmap(path: &Path) -> Result<Arc<Mmap>> {
    let cache = MMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();

    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize path: {:?}", path))?;

    if let Some(mmap) = guard.get(&canonical) {
        log::debug!("mmap cache hit for {:?}", canonical.file_name().unwrap_or_default());
        return Ok(Arc::clone(mmap));
    }

    log::debug!("mmap cache miss, creating new mapping for {:?}", canonical.file_name().unwrap_or_default());

    let file = std::fs::File::open(&canonical)
        .with_context(|| format!("failed to open file for mmap: {:?}", canonical))?;

    let mmap = Arc::new(unsafe { Mmap::map(&file)? });

    guard.insert(canonical, Arc::clone(&mmap));

    Ok(mmap)
}

/// Clears all cached memory mappings.
pub fn clear_mmap_cache() {
    if let Some(cache) = MMAP_CACHE.get() {
        let mut guard = cache.lock().unwrap();
        let count = guard.len();
        guard.clear();
        log::info!("cleared {} entries from mmap cache", count);
    }
}

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
        // MMAP_CACHE is a process-wide global, and cargo runs tests in
        // parallel, so other tests elsewhere in the crate (anything that
        // constructs a ModelWeights/SafeTensorsLoader from a path) can
        // insert or clear entries concurrently. Assert on our own
        // request's behavior (same Arc returned => cache hit) rather
        // than on the cache's total size, which isn't ours to own.
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test data for mmap").unwrap();
        temp.flush().unwrap();
        let path = temp.path();

        let mmap1 = get_or_create_mmap(path).unwrap();
        let mmap2 = get_or_create_mmap(path).unwrap();

        assert!(Arc::ptr_eq(&mmap1, &mmap2));
    }

    #[test]
    fn test_mmap_cache_clear() {
        // Same reasoning as test_mmap_cache_deduplication: don't assert
        // on the shared global's total size. Instead check that our own
        // path gets a fresh mapping (a new Arc) after a clear, which is
        // the directly-observable effect of "clear" on the entry we own.
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test").unwrap();
        temp.flush().unwrap();
        let path = temp.path();

        let mmap1 = get_or_create_mmap(path).unwrap();

        clear_mmap_cache();

        let mmap2 = get_or_create_mmap(path).unwrap();

        assert!(!Arc::ptr_eq(&mmap1, &mmap2));
    }
}