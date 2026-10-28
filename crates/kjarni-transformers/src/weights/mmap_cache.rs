//! Global memory-map cache for weight files
//!
//! Holds `Weak` references: a file mapped twice at once shares one mapping, but
//! a mapping nothing uses any more is released, rather than staying resident for
//! the life of the process next to the model's own copy of the weights.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, Weak};

use anyhow::{Context, Result};
use memmap2::Mmap;

static MMAP_CACHE: OnceLock<Mutex<HashMap<PathBuf, Weak<Mmap>>>> = OnceLock::new();

/// Returns a shared memory-mapped file
pub fn get_or_create_mmap(path: &Path) -> Result<Arc<Mmap>> {
    let cache = MMAP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();

    let canonical = path
        .canonicalize()
        .with_context(|| format!("failed to canonicalize path: {:?}", path))?;

    if let Some(mmap) = guard.get(&canonical).and_then(Weak::upgrade) {
        log::debug!(
            "mmap cache hit for {:?}",
            canonical.file_name().unwrap_or_default()
        );
        return Ok(mmap);
    }
    // Entries whose mapping has been released are only paths by now.
    guard.retain(|_, m| m.strong_count() > 0);

    log::debug!(
        "mmap cache miss, creating new mapping for {:?}",
        canonical.file_name().unwrap_or_default()
    );

    let file = std::fs::File::open(&canonical)
        .with_context(|| format!("failed to open file for mmap: {:?}", canonical))?;

    let mmap = Arc::new(unsafe { Mmap::map(&file)? });

    guard.insert(canonical, Arc::downgrade(&mmap));

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
        let live: Vec<Arc<Mmap>> = guard.values().filter_map(Weak::upgrade).collect();
        let total_bytes: usize = live.iter().map(|m| m.len()).sum();
        (live.len(), total_bytes)
    } else {
        (0, 0)
    }
}

/// Tells the kernel the pages under `bytes`, a range of `mmap` the caller has
/// copied out, are not needed now.
///
/// Weights are copied out of the mapping as they load, so without this a model
/// briefly holds its file and its copy at once: Whisper-small peaked at 1.9GB
/// for a 1GB model. Dropping each tensor's pages as it is copied keeps the peak
/// near one copy.
///
/// Always safe for these mappings: they are read-only views of a file, so a
/// dropped page is read back from disk if anything touches it again. Only whole
/// 64KiB-aligned blocks inside the range are dropped, a multiple of every
/// common page size, so neighbouring tensors are never affected.
pub fn release_pages(mmap: &Mmap, bytes: &[u8]) {
    #[cfg(unix)]
    {
        const ALIGN: usize = 64 * 1024;
        let base = mmap.as_ptr() as usize;
        let start = bytes.as_ptr() as usize;
        let end = start + bytes.len();
        if start < base || end > base + mmap.len() {
            return; // not ours: owned or converted bytes
        }
        let first = (start - base).next_multiple_of(ALIGN);
        let last = (end - base) / ALIGN * ALIGN;
        if last > first {
            // SAFETY: a read-only file mapping, so MADV_DONTNEED drops clean pages
            // that are re-read on next access; nothing can be lost.
            let _ = unsafe {
                mmap.unchecked_advise_range(memmap2::UncheckedAdvice::DontNeed, first, last - first)
            };
        }
    }
    #[cfg(not(unix))]
    let _ = (mmap, bytes);
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

    /// A mapping nothing holds any more is released, not kept for the process.
    #[test]
    fn a_dropped_mapping_is_released() {
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(b"weights").unwrap();
        temp.flush().unwrap();

        let mmap = get_or_create_mmap(temp.path()).unwrap();
        let weak = Arc::downgrade(&mmap);
        drop(mmap);
        assert!(weak.upgrade().is_none(), "the cache kept the mapping alive");

        // Asking again maps the file afresh.
        assert_eq!(&get_or_create_mmap(temp.path()).unwrap()[..], b"weights");
    }

    /// Released pages read back unchanged: the mapping is read-only, so
    /// dropping them only means reading them from the file again.
    #[test]
    fn released_pages_still_read_back() {
        let data: Vec<u8> = (0..1_000_000u32).map(|i| (i % 251) as u8).collect();
        let mut temp = NamedTempFile::new().unwrap();
        temp.write_all(&data).unwrap();
        temp.flush().unwrap();

        let mmap = get_or_create_mmap(temp.path()).unwrap();
        assert_eq!(&mmap[..], &data[..]);
        // An unaligned middle range and the whole file; neither may corrupt.
        release_pages(&mmap, &mmap[12_345..800_001]);
        release_pages(&mmap, &mmap[..]);
        assert_eq!(
            &mmap[..],
            &data[..],
            "data changed after releasing its pages"
        );
        // Bytes that are not part of the mapping are ignored.
        release_pages(&mmap, &data[..100]);
    }
}
