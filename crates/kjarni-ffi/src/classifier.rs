//! Classifier FFI bindings

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, KjarniStringArray, get_runtime};
use kjarni::Classifier;
use kjarni::classifier::ClassifierError;
use std::ffi::{CStr, CString, c_char, c_float};
use std::ptr;

/// Single classification result (label + score).
#[repr(C)]
pub struct KjarniClassResult {
    pub label: *mut c_char,
    pub score: c_float,
}

#[repr(C)]
pub struct KjarniClassResults {
    pub results: *mut KjarniClassResult,
    pub len: usize,
}

impl KjarniClassResults {
    pub fn empty() -> Self {
        Self {
            results: ptr::null_mut(),
            len: 0,
        }
    }

    pub fn from_scores(scores: Vec<(String, f32)>) -> Self {
        if scores.is_empty() {
            return Self::empty();
        }

        // A boxed slice has exactly `len` elements, which is the layout
        // `kjarni_class_results_free` frees. A collected Vec need not: collecting
        // in place from the larger (String, f32) items can leave spare capacity.
        let results: Box<[KjarniClassResult]> = scores
            .into_iter()
            .map(|(label, score)| {
                let label_cstr = CString::new(label).unwrap_or_default();
                KjarniClassResult {
                    label: label_cstr.into_raw(),
                    score,
                }
            })
            .collect();
        let len = results.len();

        Self {
            results: Box::into_raw(results) as *mut KjarniClassResult,
            len,
        }
    }
}

/// # Safety
///
/// - `results` must be a value returned by `kjarni_classifier_classify`, passed here exactly
///   once. Its buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_class_results_free(results: KjarniClassResults) {
    crate::panic::guard("kjarni_class_results_free", (), || unsafe {
        if !results.results.is_null() && results.len > 0 {
            let slice = std::ptr::slice_from_raw_parts_mut(results.results, results.len);
            for result in (*slice).iter_mut() {
                if !result.label.is_null() {
                    let _ = CString::from_raw(result.label);
                }
            }
            // The whole slice: freeing `as_mut_ptr()` as a Box<KjarniClassResult>
            // deallocated it with the size of one element.
            let _ = Box::from_raw(slice);
        }
    })
}

#[repr(C)]
pub struct KjarniClassifierConfig {
    /// Device to use
    pub device: KjarniDevice,
    /// Cache directory (NULL = default)
    pub cache_dir: *const c_char,
    /// Model name (NULL = default)
    pub model_name: *const c_char,
    /// Model path (NULL = use registry)
    pub model_path: *const c_char,
    /// Custom labels (NULL = use model labels)
    pub labels: *const *const c_char,
    /// Number of custom labels
    pub num_labels: usize,
    /// Multi-label mode (1 = multi-label, 0 = single-label)
    pub multi_label: i32,
    /// Suppress output
    pub quiet: i32,
}

/// Get default classifier configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_classifier_config_default() -> KjarniClassifierConfig {
    KjarniClassifierConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        labels: ptr::null(),
        num_labels: 0,
        multi_label: 0,
        quiet: 0,
    }
}

/// Opaque handle to a Classifier.
pub struct KjarniClassifier {
    inner: Classifier,
}

/// Create a new Classifier.
///
/// # Safety
/// - `config` must be valid or NULL
/// - `out` must be a valid pointer
/// - The returned handle must be freed with `kjarni_classifier_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_new(
    config: *const KjarniClassifierConfig,
    out: *mut *mut KjarniClassifier,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_classifier_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let default_config = kjarni_classifier_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };

                let result = get_runtime().block_on(async {
                    let mut builder = Classifier::builder("sentiment"); // default

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

                    // Model name
                    if !config.model_name.is_null() {
                        match CStr::from_ptr(config.model_name).to_str() {
                            Ok(s) => builder = Classifier::builder(s),
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

                    // Custom labels
                    if !config.labels.is_null() && config.num_labels > 0 {
                        let mut labels_vec = Vec::with_capacity(config.num_labels);
                        for i in 0..config.num_labels {
                            let label_ptr = *config.labels.add(i);
                            if label_ptr.is_null() {
                                return Err(KjarniErrorCode::NullPointer);
                            }
                            match CStr::from_ptr(label_ptr).to_str() {
                                Ok(s) => labels_vec.push(s.to_string()),
                                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
                            }
                        }
                        builder = builder.labels(labels_vec);
                    }

                    // Multi-label mode
                    if config.multi_label != 0 {
                        builder = builder.multi_label();
                    }

                    builder = builder.quiet(config.quiet != 0);

                    builder.build().await.map_err(|e| {
                        set_last_error(e.to_string());
                        match &e {
                            ClassifierError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
                            ClassifierError::ModelNotDownloaded(_) => {
                                KjarniErrorCode::ModelNotFound
                            }
                            ClassifierError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
                            ClassifierError::InvalidConfig(_) => KjarniErrorCode::InvalidConfig,
                            _ => KjarniErrorCode::LoadFailed,
                        }
                    })
                });

                match result {
                    Ok(classifier) => {
                        let handle = Box::new(KjarniClassifier { inner: classifier });
                        *out = Box::into_raw(handle);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => e,
                }
            }
        },
    )
}

/// Free a Classifier.
///
/// # Safety
///
/// - `classifier` must be null, or a handle returned by `kjarni_classifier_new` that has not
///   already been passed to this function. It is invalid to use afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_free(classifier: *mut KjarniClassifier) {
    crate::panic::guard("kjarni_classifier_free", (), || unsafe {
        if !classifier.is_null() {
            let _ = Box::from_raw(classifier);
        }
    })
}

/// Classify a single text.
///
/// # Safety
///
/// - `classifier` must be null, or a live handle returned by `kjarni_classifier_new` that has
///   not been freed.
/// - `text` must be null, or a valid pointer to a nul-terminated C string.
/// - `out` must be a valid, writable pointer to a `KjarniClassResults`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_classify(
    classifier: *mut KjarniClassifier,
    text: *const c_char,
    out: *mut KjarniClassResults,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_classifier_classify",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if classifier.is_null() || text.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let classifier_ref = &(*classifier).inner;

                let text = match CStr::from_ptr(text).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };

                let result = get_runtime().block_on(async { classifier_ref.classify(text).await });

                match result {
                    Ok(classification) => {
                        *out = KjarniClassResults::from_scores(classification.all_scores);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniClassResults::empty();
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// One [`KjarniClassResults`] per input text, in input order.
#[repr(C)]
pub struct KjarniClassBatch {
    pub items: *mut KjarniClassResults,
    pub len: usize,
}

impl KjarniClassBatch {
    fn empty() -> Self {
        Self {
            items: ptr::null_mut(),
            len: 0,
        }
    }
}

/// Classify several texts in one call.
///
/// The texts go through the model together, grouped by length, which on a CPU
/// is several times faster than classifying them one at a time. Each result is
/// the same as `kjarni_classifier_classify` gives for that text alone.
///
/// # Safety
///
/// - `classifier` must be null, or a live handle returned by `kjarni_classifier_new`.
/// - `texts` must point to `num_texts` valid, nul-terminated C strings, or be null
///   when `num_texts` is 0.
/// - `out` must be a valid, writable pointer. On `KJARNI_OK` it receives a value the
///   caller owns and must release with `kjarni_class_batch_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_classify_batch(
    classifier: *mut KjarniClassifier,
    texts: *const *const c_char,
    num_texts: usize,
    out: *mut KjarniClassBatch,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_classifier_classify_batch",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if classifier.is_null() || out.is_null() || (texts.is_null() && num_texts > 0) {
                    return KjarniErrorCode::NullPointer;
                }
                *out = KjarniClassBatch::empty();
                if num_texts == 0 {
                    return KjarniErrorCode::Ok;
                }

                let mut strs = Vec::with_capacity(num_texts);
                for &p in std::slice::from_raw_parts(texts, num_texts) {
                    if p.is_null() {
                        return KjarniErrorCode::NullPointer;
                    }
                    match CStr::from_ptr(p).to_str() {
                        Ok(s) => strs.push(s),
                        Err(_) => return KjarniErrorCode::InvalidUtf8,
                    }
                }

                let classifier_ref = &(*classifier).inner;
                match get_runtime().block_on(classifier_ref.classify_batch(&strs)) {
                    Ok(results) => {
                        let items: Box<[KjarniClassResults]> = results
                            .into_iter()
                            .map(|r| KjarniClassResults::from_scores(r.all_scores))
                            .collect();
                        let len = items.len();
                        *out = KjarniClassBatch {
                            items: Box::into_raw(items) as *mut KjarniClassResults,
                            len,
                        };
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        KjarniErrorCode::InferenceFailed
                    }
                }
            }
        },
    )
}

/// Frees a batch from `kjarni_classifier_classify_batch` and every result in it.
///
/// # Safety
///
/// `batch` must be a value returned by `kjarni_classifier_classify_batch`, passed
/// here exactly once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_class_batch_free(batch: KjarniClassBatch) {
    crate::panic::guard("kjarni_class_batch_free", (), || unsafe {
        if batch.items.is_null() {
            return;
        }
        let items = Box::from_raw(std::ptr::slice_from_raw_parts_mut(batch.items, batch.len));
        for r in items.into_vec() {
            kjarni_class_results_free(r);
        }
    })
}

/// Get the classifier's labels.
///
/// # Safety
///
/// - `classifier` must be null, or a live handle returned by `kjarni_classifier_new` that has
///   not been freed.
/// - `out` must be a valid, writable pointer to a `KjarniStringArray`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with the matching `*_free`; on
///   failure it is either cleared or left unmodified, and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_labels(
    classifier: *const KjarniClassifier,
    out: *mut KjarniStringArray,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_classifier_labels",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if classifier.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }

                let classifier_ref = &(*classifier).inner;

                match classifier_ref.labels() {
                    Some(labels) => {
                        // `len` is counted after the filter: a label containing a
                        // nul is skipped, and counting before it made the free
                        // read past the end of the array.
                        let c_strings: Box<[*mut c_char]> = labels
                            .into_iter()
                            .filter_map(|s| CString::new(s).ok())
                            .map(|cs| cs.into_raw())
                            .collect();
                        let len = c_strings.len();

                        *out = KjarniStringArray {
                            strings: Box::into_raw(c_strings) as *mut *mut c_char,
                            len,
                        };
                        KjarniErrorCode::Ok
                    }
                    None => {
                        *out = KjarniStringArray {
                            strings: ptr::null_mut(),
                            len: 0,
                        };
                        KjarniErrorCode::Ok
                    }
                }
            }
        },
    )
}

/// Get number of labels.
///
/// # Safety
///
/// - `classifier` must be null, or a live handle returned by `kjarni_classifier_new` that has
///   not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_num_labels(
    classifier: *const KjarniClassifier,
) -> usize {
    crate::panic::guard("kjarni_classifier_num_labels", 0, || -> usize {
        unsafe {
            if classifier.is_null() {
                return 0;
            }
            (*classifier).inner.num_labels()
        }
    })
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use std::ffi::CString;

    #[test]
    fn batch_rejects_null_pointers() {
        let mut out = KjarniClassBatch::empty();
        unsafe {
            assert_eq!(
                kjarni_classifier_classify_batch(ptr::null_mut(), ptr::null(), 0, &mut out),
                KjarniErrorCode::NullPointer
            );
        }
    }

    #[test]
    fn results_round_trip_through_free() {
        // Several labels, so a free that sized the allocation as one element
        // would be caught by Miri or a sanitizer.
        let r = KjarniClassResults::from_scores(vec![
            ("a".into(), 0.5),
            ("b".into(), 0.3),
            ("c".into(), 0.2),
        ]);
        assert_eq!(r.len, 3);
        unsafe { kjarni_class_results_free(r) };

        let items: Box<[KjarniClassResults]> = (0..3)
            .map(|i| KjarniClassResults::from_scores(vec![(format!("l{i}"), 1.0)]))
            .collect();
        let len = items.len();
        unsafe {
            kjarni_class_batch_free(KjarniClassBatch {
                items: Box::into_raw(items) as *mut KjarniClassResults,
                len,
            });
            kjarni_class_batch_free(KjarniClassBatch::empty());
        }
    }

    /// Real model: the batch must equal classifying each text alone.
    #[test]
    fn batch_matches_single_calls() {
        let mut config = kjarni_classifier_config_default();
        let name = CString::new("distilbert-sentiment").unwrap();
        config.model_name = name.as_ptr();
        config.quiet = 1;
        let mut handle = ptr::null_mut();
        if unsafe { kjarni_classifier_new(&config, &mut handle) } != KjarniErrorCode::Ok {
            eprintln!("skipping: distilbert-sentiment is not available");
            return;
        }
        let texts: Vec<CString> = [
            "Great product!",
            "Awful, broke at once.",
            "It is fine I guess.",
        ]
        .iter()
        .map(|t| CString::new(*t).unwrap())
        .collect();
        let ptrs: Vec<*const c_char> = texts.iter().map(|t| t.as_ptr()).collect();
        let mut batch = KjarniClassBatch::empty();
        unsafe {
            assert_eq!(
                kjarni_classifier_classify_batch(handle, ptrs.as_ptr(), ptrs.len(), &mut batch),
                KjarniErrorCode::Ok
            );
            assert_eq!(batch.len, 3);
            let items = std::slice::from_raw_parts(batch.items, batch.len);
            for (p, b) in ptrs.iter().zip(items) {
                let mut single = KjarniClassResults::empty();
                assert_eq!(
                    kjarni_classifier_classify(handle, *p, &mut single),
                    KjarniErrorCode::Ok
                );
                assert_eq!(single.len, b.len);
                let s = std::slice::from_raw_parts(single.results, single.len);
                let bb = std::slice::from_raw_parts(b.results, b.len);
                for (x, y) in s.iter().zip(bb) {
                    assert_eq!(CStr::from_ptr(x.label), CStr::from_ptr(y.label));
                    assert!((x.score - y.score).abs() < 1e-5);
                }
                kjarni_class_results_free(single);
            }
            kjarni_class_batch_free(batch);

            let mut empty = KjarniClassBatch::empty();
            assert_eq!(
                kjarni_classifier_classify_batch(handle, ptr::null(), 0, &mut empty),
                KjarniErrorCode::Ok
            );
            assert_eq!(empty.len, 0);
            kjarni_classifier_free(handle);
        }
    }
}
