//! Searcher FFI bindings
//!
//! This module provides C-compatible bindings for the Kjarni searcher functionality.
//! The searcher allows performing semantic, keyword, and hybrid search over vector indexes.
//!
//! # Search Modes
//!
//! - **Keyword**: BM25-based text matching
//! - **Semantic**: Embedding-based similarity search
//! - **Hybrid**: Combined keyword + semantic with reciprocal rank fusion
//!
//! # Reranking
//!
//! Optionally, a cross-encoder reranker can be configured to improve result quality
//! by re-scoring the top candidates.

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::searcher::{SearchOptions, Searcher, SearcherError};
use kjarni::{SearchMode, SearchResult};
use std::ffi::{c_char, CStr, CString};
use std::ptr;

// =============================================================================
// FFI Enums
// =============================================================================

/// Search mode enum for FFI.
///
/// Determines how the search query is processed:
/// - Keyword (0): BM25 text matching only
/// - Semantic (1): Vector similarity only
/// - Hybrid (2): Combined approach (recommended)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniSearchMode {
    /// BM25 keyword search
    Keyword = 0,
    /// Embedding-based semantic search
    Semantic = 1,
    /// Combined keyword + semantic (default, recommended)
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

// =============================================================================
// Search Result Structures
// =============================================================================

/// Single search result returned from a query.
///
/// Contains the matched text, relevance score, and associated metadata.
#[repr(C)]
pub struct KjarniSearchResult {
    /// Relevance score (higher is better, scale depends on search mode)
    pub score: f32,
    /// Document ID within the index
    pub document_id: usize,
    /// The matched text chunk (must be freed)
    pub text: *mut c_char,
    /// JSON-encoded metadata (must be freed, may be NULL)
    pub metadata_json: *mut c_char,
}

/// Array of search results.
///
/// Must be freed with `kjarni_search_results_free` after use.
#[repr(C)]
pub struct KjarniSearchResults {
    /// Pointer to array of results
    pub results: *mut KjarniSearchResult,
    /// Number of results
    pub len: usize,
}

impl KjarniSearchResults {
    /// Create empty results
    pub fn empty() -> Self {
        Self {
            results: ptr::null_mut(),
            len: 0,
        }
    }

    /// Convert from Rust results to FFI results
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

/// Free search results and all contained strings.
///
/// # Safety
///
/// Must only be called once per `KjarniSearchResults` returned from search functions.
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

// =============================================================================
// Search Options
// =============================================================================

/// Search options for customizing query behavior.
///
/// Use `kjarni_search_options_default()` to get defaults, then modify as needed.
#[repr(C)]
pub struct KjarniSearchOptions {
    /// Search mode (-1 = use searcher default)
    pub mode: i32,
    /// Number of results to return (0 = use searcher default)
    pub top_k: usize,
    /// Use reranker: -1 = auto (use if available), 0 = no, 1 = yes
    pub use_reranker: i32,
    /// Minimum score threshold (0.0 = no threshold)
    pub threshold: f32,
    /// Filter by source file pattern, glob syntax (NULL = no filter)
    pub source_pattern: *const c_char,
    /// Metadata key to filter on (NULL = no filter)
    pub filter_key: *const c_char,
    /// Required value for filter_key (NULL = no filter)
    pub filter_value: *const c_char,
}

/// Get default search options.
///
/// Default values:
/// - mode: -1 (use searcher default, typically Hybrid)
/// - top_k: 0 (use searcher default, typically 10)
/// - use_reranker: -1 (auto - use if configured)
/// - threshold: 0.0 (no minimum score)
/// - All filters: NULL (no filtering)
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

// =============================================================================
// Searcher Configuration
// =============================================================================

/// Configuration for creating a Searcher.
///
/// Use `kjarni_searcher_config_default()` to get sensible defaults,
/// then modify fields as needed before passing to `kjarni_searcher_new()`.
#[repr(C)]
pub struct KjarniSearcherConfig {
    /// Device to run models on (CPU or GPU)
    pub device: KjarniDevice,
    /// Directory to cache downloaded models (NULL = system default)
    pub cache_dir: *const c_char,
    /// Embedding model name (NULL = "minilm-l6-v2")
    pub model_name: *const c_char,
    /// Cross-encoder reranker model (NULL = no reranking)
    pub rerank_model: *const c_char,
    /// Default search mode
    pub default_mode: KjarniSearchMode,
    /// Default number of results
    pub default_top_k: usize,
    /// Suppress progress output (1 = quiet, 0 = verbose)
    pub quiet: i32,
}

/// Get default searcher configuration.
///
/// Returns a configuration with sensible defaults:
/// - CPU device
/// - minilm-l6-v2 embedding model
/// - No reranker
/// - Hybrid search mode
/// - Top 10 results
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

// =============================================================================
// Searcher Handle
// =============================================================================

/// Opaque handle to a Searcher instance.
///
/// Created via `kjarni_searcher_new`, must be freed via `kjarni_searcher_free`.
pub struct KjarniSearcher {
    inner: Searcher,
}

/// Create a new Searcher.
///
/// # Arguments
///
/// * `config` - Configuration options (NULL for defaults)
/// * `out` - Pointer to receive the created searcher handle
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
/// On error, call `kjarni_last_error_message()` for details.
///
/// # Safety
///
/// - `out` must be a valid pointer
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

        // Default mode
        builder = builder.default_mode(config.default_mode.into());

        // Default top_k
        if config.default_top_k > 0 {
            builder = builder.default_top_k(config.default_top_k);
        }

        // Quiet
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

/// Free a Searcher handle.
///
/// # Safety
///
/// - `searcher` must be a handle returned by `kjarni_searcher_new`
/// - Must not be called more than once per handle
/// - Handle must not be used after freeing
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_free(searcher: *mut KjarniSearcher) {
    if !searcher.is_null() {
        let _ = Box::from_raw(searcher);
    }
}

// =============================================================================
// Search Functions
// =============================================================================

/// Search with default options.
///
/// Equivalent to calling `kjarni_searcher_search_with_options` with default options.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
/// * `index_path` - Path to the index directory
/// * `query` - Search query string
/// * `out` - Pointer to receive search results
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - Results must be freed with `kjarni_search_results_free`
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

/// Search with custom options.
///
/// Performs a search query against the specified index with custom options
/// for search mode, result count, reranking, filtering, etc.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
/// * `index_path` - Path to the index directory
/// * `query` - Search query string
/// * `options` - Search options (use `kjarni_search_options_default()` as base)
/// * `out` - Pointer to receive search results
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - Results must be freed with `kjarni_search_results_free`
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

    // Build SearchOptions from FFI options
    let mut search_opts = SearchOptions::default();

    if !options.is_null() {
        let opts = &*options;

        // Mode
        if opts.mode >= 0 {
            search_opts.mode = Some(match opts.mode {
                0 => SearchMode::Keyword,
                1 => SearchMode::Semantic,
                _ => SearchMode::Hybrid,
            });
        }

        // Top-k
        if opts.top_k > 0 {
            search_opts.top_k = Some(opts.top_k);
        }

        // Reranker
        if opts.use_reranker >= 0 {
            search_opts.rerank = Some(opts.use_reranker != 0);
        }

        // Threshold
        if opts.threshold > 0.0 {
            search_opts.threshold = Some(opts.threshold);
        }

        // Build metadata filter
        let mut filter = kjarni::MetadataFilter::default();
        let mut has_filter = false;

        if !opts.source_pattern.is_null() {
            if let Ok(s) = CStr::from_ptr(opts.source_pattern).to_str() {
                filter = filter.source(s);
                has_filter = true;
            }
        }

        if !opts.filter_key.is_null() && !opts.filter_value.is_null() {
            if let (Ok(k), Ok(v)) = (
                CStr::from_ptr(opts.filter_key).to_str(),
                CStr::from_ptr(opts.filter_value).to_str(),
            ) {
                filter = filter.must(k, v);
                has_filter = true;
            }
        }

        if has_filter {
            search_opts.filter = Some(filter);
        }
    }

    // Execute search
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

/// Static keyword search (BM25) - no embedder needed.
///
/// This is a convenience function for pure keyword search that doesn't
/// require loading an embedding model. Useful for quick text matching.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `query` - Search query string
/// * `top_k` - Maximum number of results to return
/// * `out` - Pointer to receive search results
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - Results must be freed with `kjarni_search_results_free`
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

// =============================================================================
// Accessor Functions
// =============================================================================

/// Check if the searcher has a reranker configured.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
///
/// # Returns
///
/// `true` if a reranker is configured, `false` otherwise.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_has_reranker(searcher: *const KjarniSearcher) -> bool {
    if searcher.is_null() {
        return false;
    }
    (*searcher).inner.has_reranker()
}

/// Get the default search mode.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
///
/// # Returns
///
/// The default search mode, or Hybrid if searcher is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_mode(
    searcher: *const KjarniSearcher,
) -> KjarniSearchMode {
    if searcher.is_null() {
        return KjarniSearchMode::Hybrid;
    }
    (*searcher).inner.default_mode().into()
}

/// Get the default number of results.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
///
/// # Returns
///
/// The default top_k value, or 10 if searcher is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_default_top_k(searcher: *const KjarniSearcher) -> usize {
    if searcher.is_null() {
        return 10;
    }
    (*searcher).inner.default_top_k()
}

/// Get the embedding model name.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
///
/// # Returns
///
/// Pointer to model name string, or NULL if searcher is NULL.
/// The returned pointer is valid until the next call to this function.
///
/// # Safety
///
/// - `searcher` must be a valid handle or NULL
/// - Returned string must not be modified or freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_model_name(
    searcher: *const KjarniSearcher,
) -> *const c_char {
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static MODEL_NAME_BUF: OnceLock<Mutex<CString>> = OnceLock::new();

    if searcher.is_null() {
        return ptr::null();
    }

    let name = (*searcher).inner.model_name();
    let mutex = MODEL_NAME_BUF.get_or_init(|| Mutex::new(CString::default()));

    if let Ok(mut guard) = mutex.lock() {
        if let Ok(cstr) = CString::new(name) {
            *guard = cstr;
            return guard.as_ptr();
        }
    }
    ptr::null()
}

/// Get the reranker model name, if configured.
///
/// # Arguments
///
/// * `searcher` - Searcher handle
///
/// # Returns
///
/// Pointer to reranker model name string, or NULL if no reranker is configured.
/// The returned pointer is valid until the next call to this function.
///
/// # Safety
///
/// - `searcher` must be a valid handle or NULL
/// - Returned string must not be modified or freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_searcher_reranker_model(
    searcher: *const KjarniSearcher,
) -> *const c_char {
    use std::sync::Mutex;
    use std::sync::OnceLock;

    static RERANKER_MODEL_BUF: OnceLock<Mutex<CString>> = OnceLock::new();

    if searcher.is_null() {
        return ptr::null();
    }

    match (*searcher).inner.reranker_model() {
        Some(name) => {
            let mutex = RERANKER_MODEL_BUF.get_or_init(|| Mutex::new(CString::default()));
            if let Ok(mut guard) = mutex.lock() {
                if let Ok(cstr) = CString::new(name) {
                    *guard = cstr;
                    return guard.as_ptr();
                }
            }
            ptr::null()
        }
        None => ptr::null(),
    }
}