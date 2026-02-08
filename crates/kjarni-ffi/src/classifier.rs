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
    /// Label name (must be freed with kjarni_string_free)
    pub label: *mut c_char,
    /// Confidence score
    pub score: c_float,
}

/// Array of classification results.
#[repr(C)]
pub struct KjarniClassResults {
    /// Array of results
    pub results: *mut KjarniClassResult,
    /// Number of results
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

        let len = scores.len();
        let mut results: Vec<KjarniClassResult> = scores
            .into_iter()
            .map(|(label, score)| {
                let label_cstr = CString::new(label).unwrap_or_default();
                KjarniClassResult {
                    label: label_cstr.into_raw(),
                    score,
                }
            })
            .collect();

        let ptr = results.as_mut_ptr();
        std::mem::forget(results);

        Self { results: ptr, len }
    }
}

/// Free classification results.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_class_results_free(results: KjarniClassResults) {
    if !results.results.is_null() && results.len > 0 {
        let slice = std::slice::from_raw_parts_mut(results.results, results.len);
        for result in slice.iter_mut() {
            if !result.label.is_null() {
                let _ = CString::from_raw(result.label);
            }
        }
        let _ = Box::from_raw(slice.as_mut_ptr());
    }
}

/// Configuration for creating a Classifier.
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
                ClassifierError::ModelNotDownloaded(_) => KjarniErrorCode::ModelNotFound,
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

/// Free a Classifier.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_free(classifier: *mut KjarniClassifier) {
    if !classifier.is_null() {
        let _ = Box::from_raw(classifier);
    }
}

/// Classify a single text.
///
/// # Safety
/// - `classifier` must be valid
/// - `text` must be valid UTF-8
/// - `out` must be valid
/// - Results must be freed with `kjarni_class_results_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_classify(
    classifier: *mut KjarniClassifier,
    text: *const c_char,
    out: *mut KjarniClassResults,
) -> KjarniErrorCode {
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

/// Get the classifier's labels.
///
/// # Safety
/// - `classifier` must be valid
/// - `out` must be valid
/// - Results must be freed with `kjarni_string_array_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_labels(
    classifier: *const KjarniClassifier,
    out: *mut KjarniStringArray,
) -> KjarniErrorCode {
    if classifier.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let classifier_ref = &(*classifier).inner;

    match classifier_ref.labels() {
        Some(labels) => {
            let len = labels.len();
            let mut c_strings: Vec<*mut c_char> = labels
                .into_iter()
                .filter_map(|s| CString::new(s).ok())
                .map(|cs| cs.into_raw())
                .collect();

            let ptr = c_strings.as_mut_ptr();
            std::mem::forget(c_strings);

            *out = KjarniStringArray { strings: ptr, len };
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

/// Get number of labels.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_classifier_num_labels(
    classifier: *const KjarniClassifier,
) -> usize {
    if classifier.is_null() {
        return 0;
    }
    (*classifier).inner.num_labels()
}
