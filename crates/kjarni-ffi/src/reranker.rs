//! Reranker FFI bindings

use crate::{get_runtime, KjarniError, KjarniDevice};
use crate::error::set_last_error;
use kjarni::Reranker;
use std::ffi::{c_char, c_float, CStr, CString};
use std::ptr;

/// Single rerank result
#[repr(C)]
pub struct KjarniRerankResult {
    pub index: usize,
    pub score: c_float,
}

/// Array of rerank results
#[repr(C)]
pub struct KjarniRerankResults {
    pub results: *mut KjarniRerankResult,
    pub len: usize,
}

impl KjarniRerankResults {
    pub fn empty() -> Self {
        Self { results: ptr::null_mut(), len: 0 }
    }

    pub fn from_results(results: Vec<(usize, f32)>) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let len = results.len();
        let mut c_results: Vec<KjarniRerankResult> = results
            .into_iter()
            .map(|(index, score)| KjarniRerankResult { index, score })
            .collect();

        let ptr = c_results.as_mut_ptr();
        std::mem::forget(c_results);

        Self { results: ptr, len }
    }
}

/// Free rerank results
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_rerank_results_free(results: KjarniRerankResults) {
    if !results.results.is_null() && results.len > 0 {
        let _ = Vec::from_raw_parts(results.results, results.len, results.len);
    }
}

/// Configuration for Reranker
#[repr(C)]
pub struct KjarniRerankerConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    pub model_name: *const c_char,
    pub model_path: *const c_char,
    pub quiet: i32,
}

/// Get default reranker configuration
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_reranker_config_default() -> KjarniRerankerConfig {
    KjarniRerankerConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        quiet: 0,
    }
}

/// Opaque handle to a Reranker
pub struct KjarniReranker {
    inner: Reranker,
}

/// Create a new Reranker
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_new(
    config: *const KjarniRerankerConfig,
    out: *mut *mut KjarniReranker,
) -> KjarniError {
    if out.is_null() {
        return KjarniError::NullPointer;
    }

    let default_config = kjarni_reranker_config_default();
    let config = if config.is_null() { &default_config } else { &*config };

    let result = get_runtime().block_on(async {
        // Default model name
        let model_name = if !config.model_name.is_null() {
            match CStr::from_ptr(config.model_name).to_str() {
                Ok(s) => s,
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        } else {
            "cross-encoder-minilm"  // Default cross-encoder
        };

        let mut builder = Reranker::builder(model_name);

        // Device
        match config.device {
            KjarniDevice::Gpu => builder = builder.gpu(),
            KjarniDevice::Cpu => builder = builder.cpu(),
        }

        // Cache dir
        if !config.cache_dir.is_null() {
            match CStr::from_ptr(config.cache_dir).to_str() {
                Ok(s) => builder = builder.cache_dir(s),
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        }

        // Model path
        if !config.model_path.is_null() {
            match CStr::from_ptr(config.model_path).to_str() {
                Ok(s) => builder = builder.model_path(s),
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        }

        builder = builder.quiet(config.quiet != 0);

        builder.build().await.map_err(|e| {
            set_last_error(e.to_string());
            KjarniError::LoadFailed
        })
    });

    match result {
        Ok(reranker) => {
            let handle = Box::new(KjarniReranker { inner: reranker });
            *out = Box::into_raw(handle);
            KjarniError::Ok
        }
        Err(e) => e,
    }
}

/// Free a Reranker
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_free(reranker: *mut KjarniReranker) {
    if !reranker.is_null() {
        let _ = Box::from_raw(reranker);
    }
}

/// Score a single query-document pair
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_score(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    document: *const c_char,
    out: *mut c_float,
) -> KjarniError {
    if reranker.is_null() || query.is_null() || document.is_null() || out.is_null() {
        return KjarniError::NullPointer;
    }

    let reranker_ref = &(*reranker).inner;

    let query = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    let document = match CStr::from_ptr(document).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    let result = get_runtime().block_on(async {
        reranker_ref.score(query, document).await
    });

    match result {
        Ok(score) => {
            *out = score;
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = 0.0;
            KjarniError::InferenceFailed
        }
    }
}

/// Rerank documents by relevance to query
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    out: *mut KjarniRerankResults,
) -> KjarniError {
    if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
        return KjarniError::NullPointer;
    }

    if num_docs == 0 {
        *out = KjarniRerankResults::empty();
        return KjarniError::Ok;
    }

    let reranker_ref = &(*reranker).inner;

    let query = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    // Convert documents
    let mut doc_vec = Vec::with_capacity(num_docs);
    for i in 0..num_docs {
        let doc_ptr = *documents.add(i);
        if doc_ptr.is_null() {
            return KjarniError::NullPointer;
        }
        match CStr::from_ptr(doc_ptr).to_str() {
            Ok(s) => doc_vec.push(s.to_string()),
            Err(_) => return KjarniError::InvalidUtf8,
        }
    }
    let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

    let result = get_runtime().block_on(async {
        reranker_ref.rerank(query, &doc_refs).await
    });

    match result {
        Ok(results) => {
            let ranked: Vec<(usize, f32)> = results.into_iter()
                .map(|r| (r.index, r.score))
                .collect();
            *out = KjarniRerankResults::from_results(ranked);
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniRerankResults::empty();
            KjarniError::InferenceFailed
        }
    }
}

/// Rerank and return top-k results
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank_top_k(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    top_k: usize,
    out: *mut KjarniRerankResults,
) -> KjarniError {
    if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
        return KjarniError::NullPointer;
    }

    if num_docs == 0 {
        *out = KjarniRerankResults::empty();
        return KjarniError::Ok;
    }

    let reranker_ref = &(*reranker).inner;

    let query = match CStr::from_ptr(query).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    // Convert documents
    let mut doc_vec = Vec::with_capacity(num_docs);
    for i in 0..num_docs {
        let doc_ptr = *documents.add(i);
        if doc_ptr.is_null() {
            return KjarniError::NullPointer;
        }
        match CStr::from_ptr(doc_ptr).to_str() {
            Ok(s) => doc_vec.push(s.to_string()),
            Err(_) => return KjarniError::InvalidUtf8,
        }
    }
    let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

    let result = get_runtime().block_on(async {
        reranker_ref.rerank_top_k(query, &doc_refs, top_k).await
    });

    match result {
        Ok(results) => {
            let ranked: Vec<(usize, f32)> = results.into_iter()
                .map(|r| (r.index, r.score))
                .collect();
            *out = KjarniRerankResults::from_results(ranked);
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniRerankResults::empty();
            KjarniError::InferenceFailed
        }
    }
}