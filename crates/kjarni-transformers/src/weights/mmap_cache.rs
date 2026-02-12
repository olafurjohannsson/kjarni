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
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"test data for mmap").unwrap();
        temp.flush().unwrap();
        let path = temp.path();

        clear_mmap_cache();

        let mmap1 = get_or_create_mmap(path).unwrap();
        let (count1, _) = mmap_cache_stats();
        assert_eq!(count1, 1);

        let mmap2 = get_or_create_mmap(path).unwrap();
        let (count2, _) = mmap_cache_stats();
        assert_eq!(count2, 1);

        assert!(Arc::ptr_eq(&mmap1, &mmap2));

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