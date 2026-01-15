//! Searcher FFI bindings

use crate::callback::{
    is_cancelled, FfiProgressCallback, KjarniCancelToken, KjarniProgressCallbackFn,
};
use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::{SearchMode, SearchResult};
use kjarni::searcher::{SearchOptions, Searcher, SearcherError};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

/// Search mode enum for FFI
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

/// Single search result
#[repr(C)]
pub struct KjarniSearchResult {
    pub score: f32,
    pub document_id: usize,
    pub text: *mut c_char,
    pub metadata_json: *mut c_char, // JSON string for simplicity
}

/// Array of search results
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

        let ptr = c_results.as_mut_ptr();
        std::mem::forget(c_results);

        Self { results: ptr, len }
    }
}

/// Free search results
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_search_results_free(results: KjarniSearchResults) {
    if !results.results.is_null() && results.len > 0 {
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

/// Search options
#[repr(C)]
pub struct KjarniSearchOptions {
    /// Search mode (-1 = use default)
    pub mode: i32,
    /// Number of results (0 = use default)
    pub top_k: usize,
    /// Use reranker (-1 = auto, 0 = no, 1 = yes)
    pub use_reranker: i32,
    /// Minimum score threshold (0 = no threshold)
    pub threshold: f32,
    /// Source pattern filter (NULL = no filter)
    pub source_pattern: *const c_char,
    /// Required metadata key (NULL = no filter)
    pub filter_key: *const c_char,
    /// Required metadata value (NULL = no filter)
    pub filter_value: *const c_char,
}

/// Get default search options
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_search_options_default() -> KjarniSearchOptions {
    KjarniSearchOptions {
        mode: -1,       // Use default
        top_k: 0,       // Use default
        use_reranker: -1, // Auto
        threshold: 0.0,
        source_pattern: ptr::null(),
        filter_key: ptr::null(),
        filter_value: ptr::null(),
    }
}

/// Searcher configuration
#[repr(C)]
pub struct KjarniSearcherConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    pub model_name: *const c_char,
    pub rerank_model: *const c_char, // NULL = no reranking
    pub default_mode: KjarniSearchMode,
    pub default_top_k: usize,
    pub quiet: i32,
}

/// Get default searcher configuration
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

/// Opaque handle to a Searcher
pub struct KjarniSearcher {
    inner: Searcher,
}

/// Create a new Searcher
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_new(
    config: *const KjarniSearcherConfig,
    out: *mut *mut KjarniSearcher,
) -> KjarniErrorCode {
    if out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let default_config = kjarni_searcher_config_default();
    let config = if config.is_null() {
        &default_config
    } else {
        &*config
    };

    let result = get_runtime().block_on(async {
        let model_name = if !config.model_name.is_null() {
            match CStr::from_ptr(config.model_name).to_str() {
                Ok(s) => s,
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        } else {
            "minilm-l6-v2"
        };

        let mut builder = Searcher::builder(model_name);

        // Device
        match config.device {
            KjarniDevice::Gpu => builder = builder.gpu(),
            KjarniDevice::Cpu => builder = builder.cpu(),
        }

        // Cache dir
        if !config.cache_dir.is_null() {
            match CStr::from_ptr(config.cache_dir).to_str() {
                Ok(s) => builder = builder.cache_dir(s),
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        }

        // Reranker
        if !config.rerank_model.is_null() {
            match CStr::from_ptr(config.rerank_model).to_str() {
                Ok(s) => builder = builder.reranker(s),
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        }

        // Defaults
        // builder = builder.default_mode(config.default_mode.into());
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
            *out = Box::into_raw(handle);
            KjarniErrorCode::Ok
        }
        Err(e) => e,
    }
}

/// Free a Searcher
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_free(searcher: *mut KjarniSearcher) {
    if !searcher.is_null() {
        let _ = Box::from_raw(searcher);
    }
}

/// Search with default options
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_search(
    searcher: *mut KjarniSearcher,
    index_path: *const c_char,
    query: *const c_char,
    out: *mut KjarniSearchResults,
) -> KjarniErrorCode {
    let options = kjarni_search_options_default();
    kjarni_searcher_search_with_options(searcher, index_path, query, &options, out)
}

/// Search with custom options
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

    let searcher_ref = &(*searcher).inner;

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let query = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    // Build SearchOptions
    let mut search_opts = SearchOptions::default();

    if !options.is_null() {
        let opts = &*options;

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

        // Build metadata filter
        // let mut filter = MetadataFilter::default();
        // let mut has_filter = false;

        // if !opts.source_pattern.is_null() {
        //     if let Ok(s) = CStr::from_ptr(opts.source_pattern).to_str() {
        //         filter = filter.source(s);
        //         has_filter = true;
        //     }
        // }

        // if !opts.filter_key.is_null() && !opts.filter_value.is_null() {
        //     if let (Ok(k), Ok(v)) = (
        //         CStr::from_ptr(opts.filter_key).to_str(),
        //         CStr::from_ptr(opts.filter_value).to_str(),
        //     ) {
        //         filter = filter.must(k, v);
        //         has_filter = true;
        //     }
        // }

        // if has_filter {
        //     search_opts.filter = Some(filter);
        // }
    }

    let result = get_runtime().block_on(async {
        searcher_ref
            .search_with_options(index_path, query, &search_opts)
            .await
    });

    match result {
        Ok(results) => {
            *out = KjarniSearchResults::from_results(results);
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniSearchResults::empty();
            match e {
                SearcherError::IndexNotFound(_) => KjarniErrorCode::ModelNotFound,
                SearcherError::DimensionMismatch { .. } => KjarniErrorCode::InvalidConfig,
                _ => KjarniErrorCode::InferenceFailed,
            }
        }
    }
}

/// Static keyword search (no embedder needed)
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

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let query = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    match Searcher::search_keywords(index_path, query, top_k) {
        Ok(results) => {
            *out = KjarniSearchResults::from_results(results);
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniSearchResults::empty();
            KjarniErrorCode::InferenceFailed
        }
    }
}

// Accessors
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_has_reranker(searcher: *const KjarniSearcher) -> bool {
    if searcher.is_null() {
        return false;
    }
    (*searcher).inner.has_reranker()
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_mode(
    searcher: *const KjarniSearcher,
) -> KjarniSearchMode {
    if searcher.is_null() {
        return KjarniSearchMode::Hybrid;
    }
    match (*searcher).inner.default_mode() {
        SearchMode::Keyword => KjarniSearchMode::Keyword,
        SearchMode::Semantic => KjarniSearchMode::Semantic,
        SearchMode::Hybrid => KjarniSearchMode::Hybrid,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_top_k(searcher: *const KjarniSearcher) -> usize {
    if searcher.is_null() {
        return 10;
    }
    (*searcher).inner.default_top_k()
}