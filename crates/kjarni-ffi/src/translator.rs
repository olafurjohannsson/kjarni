//! Translation over the C ABI.

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime, opt_str, owned_c_string, seq2seq_error_code};
use kjarni::translator::{Translator, TranslatorError};
use std::ffi::{CStr, c_char};
use std::ptr;

/// Configuration for a Translator.
#[repr(C)]
pub struct KjarniTranslatorConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    /// Model name, e.g. "flan-t5-base" (the default) or "flan-t5-large".
    pub model_name: *const c_char,
    /// Default source language ("english", "en", ...), used when a call passes
    /// none. Null for no default.
    pub from: *const c_char,
    /// Default target language, used when a call passes none. Null for no
    /// default.
    pub to: *const c_char,
    /// Non-zero suppresses progress output.
    pub quiet: i32,
}

/// Get default translator configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_translator_config_default() -> KjarniTranslatorConfig {
    KjarniTranslatorConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        from: ptr::null(),
        to: ptr::null(),
        quiet: 0,
    }
}

/// Opaque handle to a Translator.
pub struct KjarniTranslator {
    inner: Translator,
}

fn error_code(e: &TranslatorError) -> KjarniErrorCode {
    match e {
        TranslatorError::UnknownModel(_) => KjarniErrorCode::ModelNotFound,
        TranslatorError::IncompatibleModel { .. }
        | TranslatorError::UnknownLanguage(_)
        | TranslatorError::MissingLanguage => KjarniErrorCode::InvalidConfig,
        TranslatorError::TranslationFailed(_) => KjarniErrorCode::InferenceFailed,
        TranslatorError::Seq2Seq(e) => seq2seq_error_code(e),
    }
}

/// Create a new Translator. Downloads the model on first use.
///
/// # Safety
///
/// - `config` must be null (defaults) or point to a valid config whose strings
///   are null or nul-terminated.
/// - `out` must be a valid, writable pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_translator_new(
    config: *const KjarniTranslatorConfig,
    out: *mut *mut KjarniTranslator,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_translator_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let default_config = kjarni_translator_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };
                let (model, cache_dir, from, to) = match (
                    opt_str(config.model_name),
                    opt_str(config.cache_dir),
                    opt_str(config.from),
                    opt_str(config.to),
                ) {
                    (Ok(m), Ok(c), Ok(f), Ok(t)) => (m.unwrap_or("flan-t5-base"), c, f, t),
                    _ => return KjarniErrorCode::InvalidUtf8,
                };

                let mut builder = Translator::builder(model);
                builder = match config.device {
                    KjarniDevice::Gpu => builder.gpu(),
                    KjarniDevice::Cpu => builder.cpu(),
                };
                if let Some(dir) = cache_dir {
                    builder = builder.cache_dir(dir);
                }
                if let Some(from) = from {
                    builder = builder.from(from);
                }
                if let Some(to) = to {
                    builder = builder.to(to);
                }
                if config.quiet != 0 {
                    builder = builder.quiet();
                }

                match get_runtime().block_on(builder.build()) {
                    Ok(inner) => {
                        *out = Box::into_raw(Box::new(KjarniTranslator { inner }));
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

/// Free a Translator.
///
/// # Safety
///
/// - `translator` must be null, or a handle from `kjarni_translator_new` not
///   already freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_translator_free(translator: *mut KjarniTranslator) {
    crate::panic::guard("kjarni_translator_free", (), || unsafe {
        if !translator.is_null() {
            let _ = Box::from_raw(translator);
        }
    })
}

/// Translate a text. A null `from` or `to` uses the translator's default for it;
/// with neither a language nor a default, this fails with
/// `KJARNI_ERROR_INVALID_CONFIG`.
///
/// # Safety
///
/// - `translator` must be a live handle; `text` a nul-terminated string; `from`
///   and `to` null or nul-terminated.
/// - `out` must be a valid, writable pointer. On success it receives a string the
///   caller frees with `kjarni_string_free`; on failure it is set to null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_translator_translate(
    translator: *const KjarniTranslator,
    text: *const c_char,
    from: *const c_char,
    to: *const c_char,
    out: *mut *mut c_char,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_translator_translate",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if translator.is_null() || text.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                *out = ptr::null_mut();
                let text = match CStr::from_ptr(text).to_str() {
                    Ok(t) => t,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                let inner = &(*translator).inner;
                let (from, to) = match (opt_str(from), opt_str(to)) {
                    (Ok(f), Ok(t)) => (f.or(inner.default_from()), t.or(inner.default_to())),
                    _ => return KjarniErrorCode::InvalidUtf8,
                };
                let (Some(from), Some(to)) = (from, to) else {
                    let e = TranslatorError::MissingLanguage;
                    set_last_error(e.to_string());
                    return error_code(&e);
                };
                match get_runtime().block_on(inner.translate(text, from, to)) {
                    Ok(translation) => {
                        *out = owned_c_string(&translation);
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
            assert_eq!(kjarni_translator_new(ptr::null(), ptr::null_mut()), KjarniErrorCode::NullPointer);
            let mut out = ptr::null_mut();
            assert_eq!(
                kjarni_translator_translate(ptr::null(), ptr::null(), ptr::null(), ptr::null(), &mut out),
                KjarniErrorCode::NullPointer
            );
            kjarni_translator_free(ptr::null_mut());
        }
    }

    #[test]
    fn translates_when_the_model_is_present() {
        if !kjarni::registry::is_model_downloaded("flan-t5-base").unwrap_or(false) {
            eprintln!("skipping: flan-t5-base is not downloaded");
            return;
        }
        unsafe {
            let mut config = kjarni_translator_config_default();
            config.quiet = 1;
            let mut t = ptr::null_mut();
            assert_eq!(kjarni_translator_new(&config, &mut t), KjarniErrorCode::Ok);

            // No languages anywhere: a clear configuration error.
            let mut out = ptr::null_mut();
            let code = kjarni_translator_translate(t, c"Hello".as_ptr(), ptr::null(), ptr::null(), &mut out);
            assert_eq!(code, KjarniErrorCode::InvalidConfig);
            assert!(out.is_null());

            let code = kjarni_translator_translate(
                t,
                c"The house is wonderful.".as_ptr(),
                c"en".as_ptr(),
                c"de".as_ptr(),
                &mut out,
            );
            assert_eq!(code, KjarniErrorCode::Ok);
            let german = CStr::from_ptr(out).to_str().unwrap().to_lowercase();
            crate::kjarni_string_free(out);
            kjarni_translator_free(t);
            assert!(german.contains("haus"), "{german}");
        }
    }
}
