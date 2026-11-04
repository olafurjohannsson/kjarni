//! Named-entity recognition FFI bindings

use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::ner::{Aggregation, Entity, EntityRecognizer, NerError};
use std::ffi::{CStr, CString, c_char, c_float};
use std::ptr;

/// One entity found in a text.
#[repr(C)]
pub struct KjarniEntity {
    /// The entity type, such as `PER` or `Location`.
    pub label: *mut c_char,
    /// The entity's text, exactly as it appears in the input.
    pub text: *mut c_char,
    /// Where the entity starts in the input, in Unicode code points (not bytes).
    pub start: usize,
    /// Where it ends, in code points, exclusive.
    pub end: usize,
    /// The mean probability of its tokens' labels.
    pub score: c_float,
}

#[repr(C)]
pub struct KjarniEntities {
    pub entities: *mut KjarniEntity,
    pub len: usize,
}

impl KjarniEntities {
    fn empty() -> Self {
        Self {
            entities: ptr::null_mut(),
            len: 0,
        }
    }

    fn from_entities(entities: Vec<Entity>) -> Self {
        if entities.is_empty() {
            return Self::empty();
        }
        // A boxed slice has exactly `len` elements, the layout
        // `kjarni_entities_free` frees.
        let items: Box<[KjarniEntity]> = entities
            .into_iter()
            .map(|e| KjarniEntity {
                label: CString::new(e.label).unwrap_or_default().into_raw(),
                text: CString::new(e.text).unwrap_or_default().into_raw(),
                start: e.start,
                end: e.end,
                score: e.score,
            })
            .collect();
        let len = items.len();
        Self {
            entities: Box::into_raw(items) as *mut KjarniEntity,
            len,
        }
    }
}

/// Free the entities `kjarni_ner_recognize` returned.
///
/// # Safety
///
/// - `entities` must be a value returned by `kjarni_ner_recognize`, passed here exactly
///   once. Its buffers are invalid to read afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_entities_free(entities: KjarniEntities) {
    crate::panic::guard("kjarni_entities_free", (), || unsafe {
        if !entities.entities.is_null() && entities.len > 0 {
            let slice = std::ptr::slice_from_raw_parts_mut(entities.entities, entities.len);
            for e in (*slice).iter_mut() {
                if !e.label.is_null() {
                    let _ = CString::from_raw(e.label);
                }
                if !e.text.is_null() {
                    let _ = CString::from_raw(e.text);
                }
            }
            let _ = Box::from_raw(slice);
        }
    })
}

/// How tokens become entities, as transformers' `aggregation_strategy`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniAggregation {
    /// A word takes the label of its first token, so it is never split.
    First = 0,
    /// Each token keeps its own label; neighbours with the same tag join.
    Simple = 1,
    /// A word takes the label of its most confident token.
    Max = 2,
    /// A word takes the label with the highest probability averaged over it.
    Average = 3,
}

#[repr(C)]
pub struct KjarniNerConfig {
    /// Device to use
    pub device: KjarniDevice,
    /// Cache directory (NULL = default)
    pub cache_dir: *const c_char,
    /// Model name (NULL = distilbert-ner)
    pub model_name: *const c_char,
    /// Model folder (NULL = use the registry)
    pub model_path: *const c_char,
    /// How tokens become entities
    pub aggregation: KjarniAggregation,
    /// Suppress output
    pub quiet: i32,
}

/// Get the default entity recognizer configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_ner_config_default() -> KjarniNerConfig {
    KjarniNerConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        aggregation: KjarniAggregation::First,
        quiet: 0,
    }
}

/// Opaque handle to an entity recognizer.
pub struct KjarniNer {
    inner: EntityRecognizer,
}

unsafe fn opt_str<'a>(p: *const c_char) -> Result<Option<&'a str>, KjarniErrorCode> {
    if p.is_null() {
        return Ok(None);
    }
    unsafe { CStr::from_ptr(p) }
        .to_str()
        .map(Some)
        .map_err(|_| KjarniErrorCode::InvalidUtf8)
}

/// Create an entity recognizer.
///
/// # Safety
/// - `config` must be valid or NULL
/// - `out` must be a valid pointer
/// - The returned handle must be freed with `kjarni_ner_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_ner_new(
    config: *const KjarniNerConfig,
    out: *mut *mut KjarniNer,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_ner_new",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let default_config = kjarni_ner_config_default();
                let config = if config.is_null() {
                    &default_config
                } else {
                    &*config
                };

                let result = get_runtime().block_on(async {
                    let model = opt_str(config.model_name)?.unwrap_or(kjarni::ner::DEFAULT_MODEL);
                    let mut builder = match opt_str(config.model_path)? {
                        Some(path) => EntityRecognizer::from_path(path),
                        None => EntityRecognizer::builder(model),
                    };
                    if let Some(dir) = opt_str(config.cache_dir)? {
                        builder = builder.cache_dir(dir);
                    }
                    builder = match config.device {
                        KjarniDevice::Gpu => builder.gpu(),
                        KjarniDevice::Cpu => builder.cpu(),
                    };
                    builder = builder
                        .aggregation(match config.aggregation {
                            KjarniAggregation::First => Aggregation::First,
                            KjarniAggregation::Simple => Aggregation::Simple,
                            KjarniAggregation::Max => Aggregation::Max,
                            KjarniAggregation::Average => Aggregation::Average,
                        })
                        .quiet(config.quiet != 0);

                    builder.build().await.map_err(|e| {
                        set_last_error(e.to_string());
                        match &e {
                            NerError::UnknownModel(_) | NerError::ModelNotDownloaded(_) => {
                                KjarniErrorCode::ModelNotFound
                            }
                            NerError::GpuUnavailable => KjarniErrorCode::GpuUnavailable,
                            NerError::IncompatibleModel { .. } | NerError::ModelPathNotFound(_) => {
                                KjarniErrorCode::InvalidConfig
                            }
                            _ => KjarniErrorCode::LoadFailed,
                        }
                    })
                });

                match result {
                    Ok(inner) => {
                        *out = Box::into_raw(Box::new(KjarniNer { inner }));
                        KjarniErrorCode::Ok
                    }
                    Err(e) => e,
                }
            }
        },
    )
}

/// Free an entity recognizer.
///
/// # Safety
///
/// - `ner` must be null, or a handle returned by `kjarni_ner_new` that has not already
///   been passed to this function. It is invalid to use afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_ner_free(ner: *mut KjarniNer) {
    crate::panic::guard("kjarni_ner_free", (), || unsafe {
        if !ner.is_null() {
            let _ = Box::from_raw(ner);
        }
    })
}

/// Find the entities in a text.
///
/// # Safety
///
/// - `ner` must be null, or a live handle returned by `kjarni_ner_new`.
/// - `text` must be null, or a valid pointer to a nul-terminated C string.
/// - `out` must be a valid, writable pointer to a `KjarniEntities`. On `KJARNI_OK` it
///   receives a value the caller owns and must release with `kjarni_entities_free`; on
///   failure it is cleared and must not be read.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_ner_recognize(
    ner: *mut KjarniNer,
    text: *const c_char,
    out: *mut KjarniEntities,
) -> KjarniErrorCode {
    crate::panic::guard(
        "kjarni_ner_recognize",
        KjarniErrorCode::Panic,
        || -> KjarniErrorCode {
            unsafe {
                if ner.is_null() || text.is_null() || out.is_null() {
                    return KjarniErrorCode::NullPointer;
                }
                let text = match CStr::from_ptr(text).to_str() {
                    Ok(s) => s,
                    Err(_) => return KjarniErrorCode::InvalidUtf8,
                };
                let ner = &(*ner).inner;
                match get_runtime().block_on(ner.recognize(text)) {
                    Ok(entities) => {
                        *out = KjarniEntities::from_entities(entities);
                        KjarniErrorCode::Ok
                    }
                    Err(e) => {
                        set_last_error(e.to_string());
                        *out = KjarniEntities::empty();
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

    #[test]
    fn entities_round_trip_through_the_c_layout_and_free_cleanly() {
        let list = KjarniEntities::from_entities(vec![
            Entity {
                label: "PER".into(),
                text: "Þórður".into(),
                start: 0,
                end: 6,
                score: 0.99,
            },
            Entity {
                label: "LOC".into(),
                text: "Reykjavík".into(),
                start: 13,
                end: 22,
                score: 0.98,
            },
        ]);
        assert_eq!(list.len, 2);
        unsafe {
            let second = &*list.entities.add(1);
            assert_eq!(CStr::from_ptr(second.text).to_str().unwrap(), "Reykjavík");
            assert_eq!((second.start, second.end), (13, 22));
            kjarni_entities_free(list);
        }
        unsafe { kjarni_entities_free(KjarniEntities::from_entities(vec![])) };
    }

    #[test]
    fn a_missing_model_folder_is_an_invalid_config() {
        let path = CString::new("/no/such/ner/model").unwrap();
        let mut config = kjarni_ner_config_default();
        config.model_path = path.as_ptr();
        config.quiet = 1;
        let mut handle = ptr::null_mut();
        let code = unsafe { kjarni_ner_new(&config, &mut handle) };
        assert_eq!(code, KjarniErrorCode::InvalidConfig);
        assert!(handle.is_null());
    }

    #[test]
    fn null_arguments_are_refused() {
        let mut out = KjarniEntities::empty();
        let code = unsafe { kjarni_ner_recognize(ptr::null_mut(), ptr::null(), &mut out) };
        assert_eq!(code, KjarniErrorCode::NullPointer);
    }
}
