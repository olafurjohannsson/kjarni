//! Reranker FFI bindings

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::Reranker;
use kjarni::reranker::RerankerError;
use std::ffi::{CStr, c_char, c_float};
use std::ptr;

/// Puts a cross-encoder logit on a 0..1 scale.
///
/// Exported rather than reimplemented per binding. It is one line of arithmetic,
/// but the branch is not decoration: `exp(-x)` overflows to infinity for a large
/// negative x and `inf / inf` is NaN, which sorts unpredictably. A binding author
/// writing the obvious one-liner gets a function that passes every value they try
/// and returns NaN in production, so there is one implementation and it lives
/// here.
///
/// Monotonic, so ranking is identical either way; only the numbers move. Scores
/// stay logits by default because that is what PyTorch returns.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_sigmoid(x: c_float) -> c_float {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

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
        Self {
            results: ptr::null_mut(),
            len: 0,
        }
    }

    pub fn from_results(results: Vec<(usize, f32)>) -> Self {
        if results.is_empty() {
            return Self::empty();
        }

        let c_results: Box<[KjarniRerankResult]> = results
            .into_iter()
            .map(|(index, score)| KjarniRerankResult { index, score })
            .collect();
        let len = c_results.len();

        Self {
            results: Box::into_raw(c_results) as *mut KjarniRerankResult,
            len,
        }
    }
}

/// Free rerank results
///
/// # Safety
///
/// - `results` must be a value returned by a rerank call, passed here exactly once. Its
///   buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_rerank_results_free(results: KjarniRerankResults) {
    crate::panic::guard("kjarni_rerank_results_free", (), || unsafe {
        if !results.results.is_null() && results.len > 0 {
            let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                results.results,
                results.len,
            ));
        }
    })
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
///
/// # Safety
///
/// - `out` must be null, or a live handle returned by `kjarni_reranker_new` that has not been
///   freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_new(
    config: *const KjarniRerankerConfig,
    out: *mut *mut KjarniReranker,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let default_config = kjarni_reranker_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };

                let result = get_runtime().block_on(async {
                    // Default model name
                    let model_name = if !config.model_name.is_null() {
                        match CStr::from_ptr(config.model_name).to_str() {
                            Ok(s) => s,
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    } else {
                        "minilm-l6-v2-cross-encoder" // Default cross-encoder
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
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    }

                    // Model path
                    if !config.model_path.is_null() {
                        match CStr::from_ptr(config.model_path).to_str() {
                            Ok(s) => builder = builder.model_path(s),
                            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                        }
                    }

                    builder = builder.quiet(config.quiet != 0);

                    builder.build().await.map_err(|e| {
                        set_last_error(e.to_string());
                        match &e {
                            RerankerError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
                            RerankerError::ModelNotDownloaded(_) => KjarniErrorCode::ModelNotFound,
                            RerankerError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
                            RerankerError::InvalidConfig(_) => KjarniErrorCode::InvalidConfig,
                            _ => KjarniErrorCode::LoadFailed,
                        }
                    })
                });

                match result {
                    Ok(reranker) => {
                        let handle = Box::new(KjarniReranker { inner: reranker });
                        *out = Box::into_raw(handle);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => e,
                }
            }
        },
    )
}

/// Free a Reranker
///
/// # Safety
///
/// - `reranker` must be null, or a handle returned by `kjarni_reranker_new` that has not
///   already been passed to this function. It is invalid to use afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_free(reranker: *mut KjarniReranker) {
    crate::panic::guard("kjarni_reranker_free", (), || unsafe {
        if !reranker.is_null() {
            let _ = Box::from_raw(reranker);
        }
    })
}

/// Score a single query-document pair
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `document` must be null, or a valid pointer to a nul-terminated C string.
/// - `out` must be a valid, writable pointer to a `c_float`. It is written on both success and
///   failure, and any buffers it receives are owned by the caller.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_score(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    document: *const c_char,
    out: *mut c_float,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_score",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || document.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let document = match CStr::from_ptr(document).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let result =
                    get_runtime().block_on(async { reranker_ref.score(query, document).await });

                match result {
                    Ok(score) => {
                        *out = score;
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = 0.0;
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Rerank documents by relevance to query
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `documents` must point to `num_docs` readable pointers, each null or a valid
///   nul-terminated C string that stays valid for the call.
/// - `out` must be a valid, writable pointer to a `KjarniRerankResults`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    out: *mut KjarniRerankResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_rerank",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if num_docs == 0 {
                    *out = KjarniRerankResults::empty();
                    return KjarniErrorCode::Ok;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Convert documents
                let mut doc_vec = Vec::with_capacity(num_docs);
                for i in 0..num_docs {
                    let doc_ptr = *documents.add(i);
                    if doc_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    match CStr::from_ptr(doc_ptr).to_str() {
                        Ok(s) => doc_vec.push(s.to_string()),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                }
                let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

                let result =
                    get_runtime().block_on(async { reranker_ref.rerank(query, &doc_refs).await });

                match result {
                    Ok(results) => {
                        let ranked: Vec<(usize, f32)> =
                            results.into_iter().map(|r| (r.index, r.score)).collect();
                        *out = KjarniRerankResults::from_results(ranked);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniRerankResults::empty();
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Rerank and return top-k results
///
/// # Safety
///
/// - `reranker` must be null, or a live handle returned by `kjarni_reranker_new` that has not
///   been freed.
/// - `query` must be null, or a valid pointer to a nul-terminated C string.
/// - `documents` must point to `num_docs` readable pointers, each null or a valid
///   nul-terminated C string that stays valid for the call.
/// - `out` must be a valid, writable pointer to a `KjarniRerankResults`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_reranker_rerank_top_k(
    reranker: *mut KjarniReranker,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    top_k: usize,
    out: *mut KjarniRerankResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_reranker_rerank_top_k",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if reranker.is_null() || query.is_null() || documents.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                if num_docs == 0 {
                    *out = KjarniRerankResults::empty();
                    return KjarniErrorCode::Ok;
                }

                let reranker_ref = &(*reranker).inner;

                let query = match CStr::from_ptr(query).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                // Convert documents
                let mut doc_vec = Vec::with_capacity(num_docs);
                for i in 0..num_docs {
                    let doc_ptr = *documents.add(i);
                    if doc_ptr.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    match CStr::from_ptr(doc_ptr).to_str() {
                        Ok(s) => doc_vec.push(s.to_string()),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                }
                let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();

                let result = get_runtime()
                    .block_on(async { reranker_ref.rerank_top_k(query, &doc_refs, top_k).await });

                match result {
                    Ok(results) => {
                        let ranked: Vec<(usize, f32)> =
                            results.into_iter().map(|r| (r.index, r.score)).collect();
                        *out = KjarniRerankResults::from_results(ranked);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniRerankResults::empty();
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn sigmoid_matches_the_formula_and_never_returns_nan() {
        assert_eq!(kjarni_sigmoid(0.0), 0.5);
        // 1 / (1 + e^-2) and its complement.
        assert!((kjarni_sigmoid(2.0) - 0.880_797_1).abs() < 1e-6);
        assert!((kjarni_sigmoid(-2.0) - 0.119_202_92).abs() < 1e-6);
        // exp(1000) overflows; the naive formula gives NaN here.
        assert_eq!(kjarni_sigmoid(-1000.0), 0.0);
        assert_eq!(kjarni_sigmoid(1000.0), 1.0);
    }

    #[test]
    fn empty_results_free_cleanly() {
        let empty = KjarniRerankResults::from_results(Vec::new());
        assert!(empty.results.is_null());
        assert_eq!(empty.len, 0);
        unsafe { kjarni_rerank_results_free(empty) };
    }

    #[test]
    fn null_arguments_are_refused_before_any_model_is_needed() {
        let q = CString::new("q").unwrap();
        let docs = [q.as_ptr()];
        let mut score = 1.0f32;
        let mut out = KjarniRerankResults::empty();
        unsafe {
            assert_eq!(
                kjarni_reranker_new(std::ptr::null(), std::ptr::null_mut()),
                KjarniErrorCode::NullPointer
            );
            let none = std::ptr::null_mut();
            assert_eq!(
                kjarni_reranker_score(none, q.as_ptr(), q.as_ptr(), &mut score),
                KjarniErrorCode::NullPointer
            );
            assert_eq!(
                kjarni_reranker_rerank(none, q.as_ptr(), docs.as_ptr(), 1, &mut out),
                KjarniErrorCode::NullPointer
            );
            assert_eq!(
                kjarni_reranker_rerank_top_k(none, q.as_ptr(), docs.as_ptr(), 1, 1, &mut out),
                KjarniErrorCode::NullPointer
            );
            kjarni_reranker_free(none);
        }
    }

    /// Through the C interface, the order and scores the library gives.
    #[test]
    #[ignore = "downloads minilm-l6-v2-cross-encoder"]
    fn rerank_through_c_matches_the_library() {
        let mut config = kjarni_reranker_config_default();
        let name = CString::new("minilm-l6-v2-cross-encoder").unwrap();
        config.model_name = name.as_ptr();
        config.quiet = 1;
        let mut handle = ptr::null_mut();
        assert_eq!(
            unsafe { kjarni_reranker_new(&config, &mut handle) },
            KjarniErrorCode::Ok
        );

        let query = "How long do refunds take?";
        let docs = [
            "Our office is closed on public holidays.",
            "Refunds are processed within thirty days of the return.",
            "Shipping is free on orders over fifty euros.",
        ];
        let c_query = CString::new(query).unwrap();
        let c_docs: Vec<CString> = docs.iter().map(|d| CString::new(*d).unwrap()).collect();
        let ptrs: Vec<*const c_char> = c_docs.iter().map(|d| d.as_ptr()).collect();

        let want = get_runtime().block_on(async {
            let reranker = Reranker::builder("minilm-l6-v2-cross-encoder")
                .quiet(true)
                .build()
                .await
                .unwrap();
            reranker.rerank(query, &docs).await.unwrap()
        });

        unsafe {
            let mut out = KjarniRerankResults::empty();
            assert_eq!(
                kjarni_reranker_rerank(
                    handle,
                    c_query.as_ptr(),
                    ptrs.as_ptr(),
                    ptrs.len(),
                    &mut out
                ),
                KjarniErrorCode::Ok
            );
            let got = std::slice::from_raw_parts(out.results, out.len);
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.index, w.index);
                assert!(
                    (g.score - w.score).abs() < 1e-5,
                    "{} vs {}",
                    g.score,
                    w.score
                );
            }
            assert_eq!(got[0].index, 1, "the refund passage ranks first");
            kjarni_rerank_results_free(out);

            let mut top = KjarniRerankResults::empty();
            assert_eq!(
                kjarni_reranker_rerank_top_k(
                    handle,
                    c_query.as_ptr(),
                    ptrs.as_ptr(),
                    ptrs.len(),
                    1,
                    &mut top
                ),
                KjarniErrorCode::Ok
            );
            assert_eq!(top.len, 1);
            assert_eq!((*top.results).index, 1);
            kjarni_rerank_results_free(top);

            let mut score = 0.0f32;
            assert_eq!(
                kjarni_reranker_score(handle, c_query.as_ptr(), ptrs[1], &mut score),
                KjarniErrorCode::Ok
            );
            assert!((score - want[0].score).abs() < 1e-5);

            let mut none = KjarniRerankResults::empty();
            assert_eq!(
                kjarni_reranker_rerank(handle, c_query.as_ptr(), ptrs.as_ptr(), 0, &mut none),
                KjarniErrorCode::Ok
            );
            assert_eq!(none.len, 0);
            kjarni_reranker_free(handle);
        }
    }
}
