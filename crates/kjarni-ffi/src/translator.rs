//! Translation over the C ABI.

use crate::error::set_last_error;
use crate::{
    KjarniDevice, KjarniErrorCode, get_runtime, opt_str, owned_c_string, seq2seq_error_code,
};
use kjarni::translator::{FALLBACK_MODEL, Translator, TranslatorError, default_model};
use std::ffi::{CStr, c_char};
use std::ptr;

/// Configuration for a Translator.
#[repr(C)]
pub struct KjarniTranslatorConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    /// Model name. Null picks by the `from` and `to` below: Opus-MT for English and
    /// German, otherwise "flan-t5-base", which often returns the input unchanged.
    pub model_name: *const c_char,
    /// Default source language ("english", "en", ...), used when a call passes
    /// none. Null for no default.
    pub from: *const c_char,
    /// Default target language, used when a call passes none. Null for no
    /// default.
    pub to: *const c_char,
    /// Non-zero suppresses progress output.
    pub quiet: i32,
    /// Folder holding a Marian (opus-mt) or T5 model in the Hugging Face layout, used instead of
    /// `model_name`. Null uses the registry.
    pub model_path: *const c_char,
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
        model_path: ptr::null(),
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

/// The model to load when the config names none. Default languages decide it, since
/// a translator holds one model and calls may only name languages it can serve.
fn model_for(from: Option<&str>, to: Option<&str>) -> &'static str {
    match (from, to) {
        (Some(from), Some(to)) => default_model(from, to),
        _ => FALLBACK_MODEL,
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
                let (model, cache_dir, from, to, model_path) = match (
                    opt_str(config.model_name),
                    opt_str(config.cache_dir),
                    opt_str(config.from),
                    opt_str(config.to),
                    opt_str(config.model_path),
                ) {
                    (Ok(m), Ok(c), Ok(f), Ok(t), Ok(p)) => (m, c, f, t, p),
                    _ => return KjarniErrorCode::InvalidUtf8,
                };
                // A folder's config says what the model is.
                let model = model.or(model_path.map(|_| "")).unwrap_or_else(|| {
                    let model = model_for(from, to);
                    if model == FALLBACK_MODEL && config.quiet == 0 {
                        eprintln!(
                            "Warning: {FALLBACK_MODEL} often returns the input unchanged. \
                             Set the default languages to get a dedicated model for \
                             English and German."
                        );
                    }
                    model
                });

                let mut builder = Translator::builder(model);
                if let Some(path) = model_path {
                    builder = builder.model_path(path);
                }
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

    /// A folder whose config.json says it holds a `model_type` model.
    fn folder_of(model_type: &str) -> (tempfile::TempDir, std::ffi::CString) {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            format!(r#"{{"model_type": "{model_type}"}}"#),
        )
        .unwrap();
        let path = std::ffi::CString::new(dir.path().to_str().unwrap()).unwrap();
        (dir, path)
    }

    /// A snapshot of `repo` in the Hugging Face cache that has `file`.
    fn cached(repo: &str, file: &str) -> Option<std::ffi::CString> {
        let home = std::env::var_os("HF_HOME")
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::var_os("HOME")
                    .map(|h| std::path::Path::new(&h).join(".cache/huggingface"))
            })?;
        let snapshots = home.join(format!("hub/models--{}/snapshots", repo.replace('/', "--")));
        std::fs::read_dir(snapshots)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| p.join(file).exists())
            .map(|p| std::ffi::CString::new(p.to_str().unwrap()).unwrap())
    }

    #[test]
    fn a_folder_of_another_family_is_invalid_config() {
        let (_dir, path) = folder_of("bart");
        unsafe {
            let mut config = kjarni_translator_config_default();
            config.model_path = path.as_ptr();
            config.quiet = 1;
            let mut t = ptr::null_mut();
            assert_eq!(
                kjarni_translator_new(&config, &mut t),
                KjarniErrorCode::InvalidConfig
            );
            assert!(t.is_null());
        }
    }

    #[test]
    fn translates_from_a_marian_folder() {
        let Some(path) = cached("Helsinki-NLP/opus-mt-en-ROMANCE", "pytorch_model.bin") else {
            eprintln!("skipping: opus-mt-en-ROMANCE not in the Hugging Face cache");
            return;
        };
        unsafe {
            let mut config = kjarni_translator_config_default();
            config.model_path = path.as_ptr();
            config.from = c"en".as_ptr();
            config.quiet = 1;
            let mut t = ptr::null_mut();
            assert_eq!(kjarni_translator_new(&config, &mut t), KjarniErrorCode::Ok);
            let mut out = ptr::null_mut();
            // transformers 4.57: MarianMTModel.generate on ">>es<< Where is the library?"
            let code = kjarni_translator_translate(
                t,
                c"Where is the library?".as_ptr(),
                ptr::null(),
                c"es".as_ptr(),
                &mut out,
            );
            assert_eq!(code, KjarniErrorCode::Ok);
            let text = CStr::from_ptr(out).to_str().unwrap().to_string();
            crate::kjarni_string_free(out);
            let code = kjarni_translator_translate(
                t,
                c"Hello".as_ptr(),
                ptr::null(),
                c"de".as_ptr(),
                &mut out,
            );
            assert_eq!(code, KjarniErrorCode::InvalidConfig);
            kjarni_translator_free(t);
            assert_eq!(text, "¿Dónde está la biblioteca?");
        }
    }

    #[test]
    fn a_null_model_is_chosen_from_the_default_languages() {
        assert_eq!(model_for(Some("en"), Some("de")), "opus-mt-en-de");
        assert_eq!(model_for(Some("German"), Some("English")), "opus-mt-de-en");
        assert_eq!(model_for(Some("en"), Some("fr")), FALLBACK_MODEL);
        assert_eq!(model_for(None, Some("de")), FALLBACK_MODEL);
        assert_eq!(model_for(None, None), FALLBACK_MODEL);
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        unsafe {
            assert_eq!(
                kjarni_translator_new(ptr::null(), ptr::null_mut()),
                KjarniErrorCode::NullPointer
            );
            let mut out = ptr::null_mut();
            assert_eq!(
                kjarni_translator_translate(
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    ptr::null(),
                    &mut out
                ),
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
            let code = kjarni_translator_translate(
                t,
                c"Hello".as_ptr(),
                ptr::null(),
                ptr::null(),
                &mut out,
            );
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
