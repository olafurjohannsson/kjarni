//! Searcher FFI bindings.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use kjarni::searcher::{SearchOptions, Searcher, SearcherError};
use kjarni::{SearchMode, SearchResult};

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};

#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniSearchMode {
    Keyword = 0,
    Semantic = 1,
    Hybrid = 2,
}

impl From<KjarniSearchMode> for SearchMode {
    fn from(mode: KjarniSearchMode) -> Self {
        match mode {
            KjarniSearchMode::Keyword => SearchMode::Keyword,
            KjarniSearchMode::Semantic => SearchMode::Semantic,
            KjarniSearchMode::Hybrid => SearchMode::Hybrid,
        }
    }
}

impl From<SearchMode> for KjarniSearchMode {
    fn from(mode: SearchMode) -> Self {
        match mode {
            SearchMode::Keyword => KjarniSearchMode::Keyword,
            SearchMode::Semantic => KjarniSearchMode::Semantic,
            SearchMode::Hybrid => KjarniSearchMode::Hybrid,
        }
    }
}

#[repr(C)]
pub struct KjarniSearchResult {
    pub score: f32,
    pub document_id: usize,
    pub text: *mut c_char,
    pub metadata_json: *mut c_char,
}

#[repr(C)]
pub struct KjarniSearchResults {
    pub results: *mut KjarniSearchResult,
    pub len: usize,
}

impl KjarniSearchResults {
    pub fn empty() -> Self {
        Self {
            results: ptr::null_mut(),
            len: 0,
        }
    }

    pub fn from_results(results: Vec<SearchResult>) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let len = results.len();
        let mut c_results: Vec<KjarniSearchResult> = results
            .into_iter()
            .map(|r| {
                let text = CString::new(r.text).unwrap_or_default().into_raw();
                let metadata_json = serde_json::to_string(&r.metadata)
                    .ok()
                    .and_then(|s| CString::new(s).ok())
                    .map(|c| c.into_raw())
                    .unwrap_or(ptr::null_mut());

                KjarniSearchResult {
                    score: r.score,
                    document_id: r.document_id,
                    text,
                    metadata_json,
                }
            })
            .collect();

        c_results.shrink_to_fit();

        let ptr = c_results.as_mut_ptr();
        std::mem::forget(c_results);

        Self { results: ptr, len }
    }
}

/// # Safety
/// Must only be called once per `KjarniSearchResults` returned from search functions.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_search_results_free(results: KjarniSearchResults) {
    if !results.results.is_null() && results.len > 0 {
        // SAFETY: results.results was allocated by from_results() using Vec::into_raw_parts
        // pattern (as_mut_ptr + forget). We have exclusive ownership since this is the free
        // function and caller guarantees single call. Length is tracked in results.len.
        unsafe {
            let slice = std::slice::from_raw_parts_mut(results.results, results.len);
            for result in slice.iter_mut() {
                if !result.text.is_null() {
                    let _ = CString::from_raw(result.text);
                }
                if !result.metadata_json.is_null() {
                    let _ = CString::from_raw(result.metadata_json);
                }
            }
            let _ = Vec::from_raw_parts(results.results, results.len, results.len);
        }
    }
}

#[repr(C)]
pub struct KjarniSearchOptions {
    pub mode: i32,
    pub top_k: usize,
    pub use_reranker: i32,
    pub threshold: f32,
    pub source_pattern: *const c_char,
    pub filter_key: *const c_char,
    pub filter_value: *const c_char,
}

#[unsafe(no_mangle)]
pub extern "C" fn kjarni_search_options_default() -> KjarniSearchOptions {
    KjarniSearchOptions {
        mode: -1,
        top_k: 0,
        use_reranker: -1,
        threshold: 0.0,
        source_pattern: ptr::null(),
        filter_key: ptr::null(),
        filter_value: ptr::null(),
    }
}

#[repr(C)]
pub struct KjarniSearcherConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    pub model_name: *const c_char,
    pub rerank_model: *const c_char,
    pub default_mode: KjarniSearchMode,
    pub default_top_k: usize,
    pub quiet: i32,
}

#[unsafe(no_mangle)]
pub extern "C" fn kjarni_searcher_config_default() -> KjarniSearcherConfig {
    KjarniSearcherConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        rerank_model: ptr::null(),
        default_mode: KjarniSearchMode::Hybrid,
        default_top_k: 10,
        quiet: 0,
    }
}

pub struct KjarniSearcher {
    inner: Searcher,
}

/// - The returned handle must be freed with `kjarni_searcher_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_new(
    config: *const KjarniSearcherConfig,
    out: *mut *mut KjarniSearcher,
) -> KjarniErrorCode {
    if out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let default_config = kjarni_searcher_config_default();
    let config = unsafe {
        if config.is_null() {
            &default_config
        } else {
            &*config
        }
    };

    let result = get_runtime().block_on(async {
        let model_name = if !config.model_name.is_null() {
            match unsafe { CStr::from_ptr(config.model_name) }.to_str() {
                Ok(s) => s,
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        } else {
            "minilm-l6-v2"
        };

        let mut builder = Searcher::builder(model_name);

        match config.device {
            KjarniDevice::Gpu => builder = builder.gpu(),
            KjarniDevice::Cpu => builder = builder.cpu(),
        }

        if !config.cache_dir.is_null() {
            match unsafe { CStr::from_ptr(config.cache_dir) }.to_str() {
                Ok(s) => builder = builder.cache_dir(s),
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        }

        if !config.rerank_model.is_null() {
            match unsafe { CStr::from_ptr(config.rerank_model) }.to_str() {
                Ok(s) => builder = builder.reranker(s),
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        }

        builder = builder.default_mode(config.default_mode.into());

        if config.default_top_k > 0 {
            builder = builder.default_top_k(config.default_top_k);
        }

        builder = builder.quiet(config.quiet != 0);

        builder.build().await.map_err(|e| {
            set_last_error(e.to_string());
            KjarniErrorCode::LoadFailed
        })
    });

    match result {
        Ok(searcher) => {
            let handle = Box::new(KjarniSearcher { inner: searcher });
            unsafe { *out = Box::into_raw(handle) };
            KjarniErrorCode::Ok
        }
        Err(e) => e,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_free(searcher: *mut KjarniSearcher) {
    if !searcher.is_null() {
        unsafe {
            let _ = Box::from_raw(searcher);
        }
    }
}

/// All pointers must be valid. Results must be freed with `kjarni_search_results_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_search(
    searcher: *mut KjarniSearcher,
    index_path: *const c_char,
    query: *const c_char,
    out: *mut KjarniSearchResults,
) -> KjarniErrorCode {
    let options = kjarni_search_options_default();
    unsafe { kjarni_searcher_search_with_options(searcher, index_path, query, &options, out) }
}

/// All pointers must be valid. Results must be freed with `kjarni_search_results_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_search_with_options(
    searcher: *mut KjarniSearcher,
    index_path: *const c_char,
    query: *const c_char,
    options: *const KjarniSearchOptions,
    out: *mut KjarniSearchResults,
) -> KjarniErrorCode {
    if searcher.is_null() || index_path.is_null() || query.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let searcher_ref = unsafe { &(*searcher).inner };
    let index_path = match unsafe { CStr::from_ptr(index_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let query = match unsafe { CStr::from_ptr(query) }.to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let mut search_opts = SearchOptions::default();

    if !options.is_null() {
        let opts = unsafe { &*options };

        if opts.mode >= 0 {
            search_opts.mode = Some(match opts.mode {
                0 => SearchMode::Keyword,
                1 => SearchMode::Semantic,
                _ => SearchMode::Hybrid,
            });
        }

        if opts.top_k > 0 {
            search_opts.top_k = Some(opts.top_k);
        }

        if opts.use_reranker >= 0 {
            search_opts.rerank = Some(opts.use_reranker != 0);
        }

        if opts.threshold > 0.0 {
            search_opts.threshold = Some(opts.threshold);
        }

        let mut filter = kjarni::MetadataFilter::default();
        let mut has_filter = false;

        if !opts.source_pattern.is_null() {
            if let Ok(s) = unsafe { CStr::from_ptr(opts.source_pattern) }.to_str() {
                filter = filter.source(s);
                has_filter = true;
            }
        }

        if !opts.filter_key.is_null() && !opts.filter_value.is_null() {
            if let (Ok(k), Ok(v)) = (
                unsafe { CStr::from_ptr(opts.filter_key) }.to_str(),
                unsafe { CStr::from_ptr(opts.filter_value) }.to_str(),
            ) {
                filter = filter.must(k, v);
                has_filter = true;
            }
        }

        if has_filter {
            search_opts.filter = Some(filter);
        }
    }

    let result = get_runtime().block_on(async {
        searcher_ref
            .search_with_options(index_path, query, &search_opts)
            .await
    });

    match result {
        Ok(results) => {
            unsafe { *out = KjarniSearchResults::from_results(results) };
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            unsafe { *out = KjarniSearchResults::empty() };
            match e {
                SearcherError::IndexNotFound(_) => KjarniErrorCode::ModelNotFound,
                SearcherError::DimensionMismatch { .. } => KjarniErrorCode::InvalidConfig,
                _ => KjarniErrorCode::InferenceFailed,
            }
        }
    }
}

/// All pointers must be valid. Results must be freed with `kjarni_search_results_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_search_keywords(
    index_path: *const c_char,
    query: *const c_char,
    top_k: usize,
    out: *mut KjarniSearchResults,
) -> KjarniErrorCode {
    if index_path.is_null() || query.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let index_path = match unsafe { CStr::from_ptr(index_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let query = match unsafe { CStr::from_ptr(query) }.to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    match Searcher::search_keywords(index_path, query, top_k) {
        Ok(results) => {
            unsafe { *out = KjarniSearchResults::from_results(results) };
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            unsafe { *out = KjarniSearchResults::empty() };
            KjarniErrorCode::InferenceFailed
        }
    }
}

/// `searcher` must be a valid handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_has_reranker(searcher: *const KjarniSearcher) -> bool {
    if searcher.is_null() {
        return false;
    }
    unsafe { (*searcher).inner.has_reranker() }
}

/// `searcher` must be a valid handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_mode(
    searcher: *const KjarniSearcher,
) -> KjarniSearchMode {
    if searcher.is_null() {
        return KjarniSearchMode::Hybrid;
    }
    unsafe { (*searcher).inner.default_mode().into() }
}

/// `searcher` must be a valid handle or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_top_k(searcher: *const KjarniSearcher) -> usize {
    if searcher.is_null() {
        return 10;
    }
    unsafe { (*searcher).inner.default_top_k() }
}

/// Get searcher model name into caller-provided buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_model_name(
    searcher: *const KjarniSearcher,
    buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if searcher.is_null() {
        return 0;
    }

    let name = (*searcher).inner.model_name();
    let name_bytes = name.as_bytes();
    let required = name_bytes.len() + 1; // +1 for null terminator

    // If no buffer provided, just return required size
    if buf.is_null() || buf_len == 0 {
        return required;
    }

    // Copy as much as fits
    let copy_len = name_bytes.len().min(buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);

        // Null terminate
        *buf.add(copy_len) = 0;
    }

    required
}

/// Get reranker model name into caller-provided buffer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_reranker_model(
    searcher: *const KjarniSearcher,
    buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if searcher.is_null() {
        return 0;
    }

    let name = match (*searcher).inner.reranker_model() {
        Some(n) => n,
        None => return 0,
    };

    let name_bytes = name.as_bytes();
    let required = name_bytes.len() + 1; // +1 for null terminator

    // If no buffer provided, just return required size
    if buf.is_null() || buf_len == 0 {
        return required;
    }

    // Copy as much as fits
    let copy_len = name_bytes.len().min(buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);

        // Null terminate
        *buf.add(copy_len) = 0;
    }
    required
}
