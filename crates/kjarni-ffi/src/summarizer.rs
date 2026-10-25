//! Summarization over the C ABI.

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime, opt_str, owned_c_string, seq2seq_error_code};
use kjarni::summarizer::{Summarizer, SummarizerError};
use std::ffi::{CStr, c_char};
use std::ptr;

/// Configuration for a Summarizer.
#[repr(C)]
pub struct KjarniSummarizerConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    /// Model name, e.g. "distilbart-cnn" (the default) or "bart-large-cnn".
    pub model_name: *const c_char,
    /// Minimum summary length in tokens. 0 uses the model's default.
    pub min_length: usize,
    /// Maximum summary length in tokens. 0 uses the model's default.
    pub max_length: usize,
    /// Non-zero suppresses progress output.
    pub quiet: i32,
}

/// Get default summarizer configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_summarizer_config_default() -> KjarniSummarizerConfig {
    KjarniSummarizerConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        min_length: 0,
        max_length: 0,
        quiet: 0,
    }
}

/// Opaque handle to a Summarizer.
pub struct KjarniSummarizer {
    inner: Summarizer,
}

fn error_code(e: &SummarizerError) -> KjarniErrorCode {
    match e {
        SummarizerError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
        SummarizerError::IncompatibleModel { .. } => KjarniErrorCode::InvalidConfig,
        SummarizerError::SummarizationFailed(_) => KjarniErrorCode::InferenceFailed,
        SummarizerError::Seq2Seq(e) => seq2seq_error_code(e),
    }
}

/// Create a new Summarizer. Downloads the model on first use.
///
/// # Safety
///
/// - `config` must be null (defaults) or point to a valid config whose strings
///   are null or nul-terminated.
/// - `out` must be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_summarizer_new(
    config: *const KjarniSummarizerConfig,
    out: *mut *mut KjarniSummarizer,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_summarizer_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let default_config = kjarni_summarizer_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };
                let model = match opt_str(config.model_name) {
                    Ok(m) => m.unwrap_or("distilbart-cnn"),
                    Err(e) => return e,
                };
                let cache_dir = match opt_str(config.cache_dir) {
                    Ok(c) => c,
                    Err(e) => return e,
                };

                let mut builder = Summarizer::builder(model);
                builder = match config.device {
                    KjarniDevice::Gpu => builder.gpu(),
                    KjarniDevice::Cpu => builder.cpu(),
                };
                if let Some(dir) = cache_dir {
                    builder = builder.cache_dir(dir);
                }
                if config.min_length > 0 {
                    builder = builder.min_length(config.min_length);
                }
                if config.max_length > 0 {
                    builder = builder.max_length(config.max_length);
                }
                if config.quiet != 0 {
                    builder = builder.quiet();
                }

                match get_runtime().block_on(builder.build()) {
                    Ok(inner) => {
                        *out = Box::into_raw(Box::new(KjarniSummarizer { inner }));
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        error_code(&e)
                    }
                }
            }
        },
    )
}

/// Free a Summarizer.
///
/// # Safety
///
/// - `summarizer` must be null, or a handle from `kjarni_summarizer_new` not
///   already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_summarizer_free(summarizer: *mut KjarniSummarizer) {
    crate::panic::guard("kjarni_summarizer_free", (), || unsafe {
        if !summarizer.is_null() {
            let _ = Box::from_raw(summarizer);
        }
    })
}

/// Summarize a text.
///
/// # Safety
///
/// - `summarizer` must be a live handle; `text` a nul-terminated string.
/// - `out` must be a valid, writable pointer. On success it receives a string the
///   caller frees with `kjarni_string_free`; on failure it is set to null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_summarizer_summarize(
    summarizer: *const KjarniSummarizer,
    text: *const c_char,
    out: *mut *mut c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_summarizer_summarize",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if summarizer.is_null() || text.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                *out = ptr::null_mut();
                let text = match CStr::from_ptr(text).to_str() {
                    Ok(t) => t,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                match get_runtime().block_on((*summarizer).inner.summarize(text)) {
                    Ok(summary) => {
                        *out = owned_c_string(&summary);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        error_code(&e)
                    }
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        unsafe {
            assert_eq!(kjarni_summarizer_new(ptr::null(), ptr::null_mut()), KjarniErrorCode::NullPointer);
            let mut out = ptr::null_mut();
            assert_eq!(
                kjarni_summarizer_summarize(ptr::null(), ptr::null(), &mut out),
                KjarniErrorCode::NullPointer
            );
            kjarni_summarizer_free(ptr::null_mut());
        }
    }

    #[test]
    fn an_unknown_model_is_reported_as_such() {
        unsafe {
            let mut config = kjarni_summarizer_config_default();
            config.model_name = c"no-such-model".as_ptr();
            config.quiet = 1;
            let mut s = ptr::null_mut();
            let code = kjarni_summarizer_new(&config, &mut s);
            assert_ne!(code, KjarniErrorCode::Ok);
            assert!(s.is_null());
        }
    }

    #[test]
    fn summarizes_when_the_model_is_present() {
        if !kjarni::registry::is_model_downloaded("distilbart-cnn").unwrap_or(false) {
            eprintln!("skipping: distilbart-cnn is not downloaded");
            return;
        }
        unsafe {
            let mut config = kjarni_summarizer_config_default();
            config.quiet = 1;
            let mut s = ptr::null_mut();
            assert_eq!(kjarni_summarizer_new(&config, &mut s), KjarniErrorCode::Ok);
            // Article length: summarization models have a minimum output of
            // around fifty tokens, so a short input comes back padded, not shorter.
            let text = c"The city council voted on Tuesday to build a new central library downtown, ending a debate that has run for almost a decade. The building, designed by a local architecture firm, will open in 2028 and will hold two hundred thousand books, a children's wing, a public makerspace with 3D printers and a recording studio, and meeting rooms that community groups can book for free. Construction is expected to cost forty million dollars, paid for by a bond that voters approved last year by a wide margin. Supporters said the current library, built in 1962, is too small and cannot be made accessible to wheelchair users without a costly renovation. Opponents argued that the money would be better spent on branch libraries in outlying neighbourhoods, several of which have cut their opening hours. The mayor said the branches would keep their budgets and that two of them would get longer hours next year. The old building will be sold, and the proceeds will go to the city's affordable housing fund. Work on the site begins in the spring, after the demolition of a disused parking garage that currently stands there.";
            let mut out = ptr::null_mut();
            assert_eq!(kjarni_summarizer_summarize(s, text.as_ptr(), &mut out), KjarniErrorCode::Ok);
            let summary = CStr::from_ptr(out).to_str().unwrap().to_string();
            crate::kjarni_string_free(out);
            kjarni_summarizer_free(s);
            assert!(!summary.trim().is_empty());
            assert!(summary.len() < text.to_bytes().len(), "a summary should be shorter: {summary}");
        }
    }
}
