//! Embedder FFI bindings

use crate::{get_runtime, KjarniErrorCode, KjarniFloatArray, KjarniFloat2DArray};
use crate::error::{set_last_error, map_result};
use kjarni::Embedder;
use std::ffi::{c_char, c_float, CStr};
use std::ptr;

/// Device selection for inference.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniDevice {
    /// Use CPU for inference
    Cpu = 0,
    /// Use GPU for inference
    Gpu = 1,
}

/// Configuration for creating an Embedder.
#[repr(C)]
pub struct KjarniEmbedderConfig {
    /// Device to use (CPU or GPU)
    pub device: KjarniDevice,
    /// Cache directory for models (NULL = default)
    pub cache_dir: *const c_char,
    /// Model name from registry (NULL = "minilm-l6-v2")
    pub model_name: *const c_char,
    /// Path to local model (NULL = use registry)
    pub model_path: *const c_char,
    /// Whether to L2-normalize embeddings (1 = yes, 0 = no)
    pub normalize: i32,
    /// Suppress progress output (1 = quiet, 0 = verbose)
    pub quiet: i32,
}

/// Get default embedder configuration.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_embedder_config_default() -> KjarniEmbedderConfig {
    KjarniEmbedderConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        model_path: ptr::null(),
        normalize: 1,
        quiet: 0,
    }
}

/// Opaque handle to an Embedder instance.
pub struct KjarniEmbedder {
    inner: Embedder,
}

/// Create a new Embedder.
///
/// # Safety
/// - `config` must be valid or NULL (uses defaults)
/// - `out` must be a valid pointer
/// - The returned handle must be freed with `kjarni_embedder_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_new(
    config: *const KjarniEmbedderConfig,
    out: *mut *mut KjarniEmbedder,
) -> KjarniErrorCode {
    if out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    // Use defaults if config is null
    let default_config = kjarni_embedder_config_default();
    let config = if config.is_null() { &default_config } else { &*config };

    let result = get_runtime().block_on(async {
        let mut builder = Embedder::builder("minilm-l6-v2");

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
                Ok(s) => builder = builder.model(s),
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

        // Options
        builder = builder.normalize(config.normalize != 0);
        builder = builder.quiet(config.quiet != 0);

        // Build
        builder.build().await.map_err(|e| {
            set_last_error(e.to_string());
            KjarniErrorCode::LoadFailed
        })
    });

    match result {
        Ok(embedder) => {
            let handle = Box::new(KjarniEmbedder { inner: embedder });
            *out = Box::into_raw(handle);
            KjarniErrorCode::Ok
        }
        Err(e) => e,
    }
}

/// Free an Embedder instance.
///
/// # Safety
/// - `embedder` must be a valid handle or NULL
/// - The handle must not be used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_free(embedder: *mut KjarniEmbedder) {
    if !embedder.is_null() {
        let _ = Box::from_raw(embedder);
    }
}

/// Encode a single text to an embedding vector.
///
/// # Safety
/// - `embedder` must be a valid handle
/// - `text` must be a valid null-terminated UTF-8 string
/// - `out` must be a valid pointer
/// - The returned array must be freed with `kjarni_float_array_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_encode(
    embedder: *mut KjarniEmbedder,
    text: *const c_char,
    out: *mut KjarniFloatArray,
) -> KjarniErrorCode {
    if embedder.is_null() || text.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let embedder = &(*embedder).inner;
    let text = match CStr::from_ptr(text).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let result = get_runtime().block_on(async {
        embedder.embed(text).await
    });

    match result {
        Ok(embedding) => {
            *out = KjarniFloatArray::from_vec(embedding);
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniFloatArray::empty();
            KjarniErrorCode::InferenceFailed
        }
    }
}

/// Encode multiple texts to embedding vectors.
///
/// # Safety
/// - `embedder` must be a valid handle
/// - `texts` must be an array of `num_texts` valid null-terminated UTF-8 strings
/// - `out` must be a valid pointer
/// - The returned array must be freed with `kjarni_float_2d_array_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_encode_batch(
    embedder: *mut KjarniEmbedder,
    texts: *const *const c_char,
    num_texts: usize,
    out: *mut KjarniFloat2DArray,
) -> KjarniErrorCode {
    if embedder.is_null() || texts.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    if num_texts == 0 {
        *out = KjarniFloat2DArray::empty();
        return KjarniErrorCode::Ok;
    }

    let embedder_ref = &(*embedder).inner;

    // Convert C strings to Rust
    let mut text_vec = Vec::with_capacity(num_texts);
    for i in 0..num_texts {
        let text_ptr = *texts.add(i);
        if text_ptr.is_null() {
            return KjarniErrorCode::NullPointer;
        }
        match CStr::from_ptr(text_ptr).to_str() {
            Ok(s) => text_vec.push(s.to_string()),
            Err(_) => return KjarniErrorCode::InvalidUtf8,
        }
    }

    let text_refs: Vec<&str> = text_vec.iter().map(|s| s.as_str()).collect();

    let result = get_runtime().block_on(async {
        embedder_ref.embed_batch_flat(&text_refs).await
        // embedder_ref.embed_batch(&text_refs).await
    });

    match result {
        Ok((embeddings, rows, cols)) => {
            // *out = KjarniFloat2DArray::from_vecs(embeddings);
            *out = KjarniFloat2DArray::from_flat(embeddings, rows, cols);
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = KjarniFloat2DArray::empty();
            KjarniErrorCode::InferenceFailed
        }
    }
}

/// Compute cosine similarity between two texts.
///
/// # Safety
/// - `embedder` must be a valid handle
/// - `text1` and `text2` must be valid null-terminated UTF-8 strings
/// - `out` must be a valid pointer
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_similarity(
    embedder: *mut KjarniEmbedder,
    text1: *const c_char,
    text2: *const c_char,
    out: *mut c_float,
) -> KjarniErrorCode {
    if embedder.is_null() || text1.is_null() || text2.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let embedder_ref = &(*embedder).inner;

    let text1 = match CStr::from_ptr(text1).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let text2 = match CStr::from_ptr(text2).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    let result = get_runtime().block_on(async {
        embedder_ref.similarity(text1, text2).await
    });

    match result {
        Ok(sim) => {
            *out = sim;
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *out = 0.0;
            KjarniErrorCode::InferenceFailed
        }
    }
}

/// Get the embedding dimension.
///
/// # Safety
/// - `embedder` must be a valid handle
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_embedder_dim(embedder: *const KjarniEmbedder) -> usize {
    if embedder.is_null() {
        return 0;
    }

    let embedder_ref = &(*embedder).inner;
    
    get_runtime().block_on(async {
        embedder_ref.dimension()
    })
}