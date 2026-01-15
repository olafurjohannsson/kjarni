//! Indexer FFI bindings

use crate::callback::{
    is_cancelled, FfiProgressCallback, KjarniCancelToken, KjarniProgress,
    KjarniProgressCallbackFn,
};
use crate::error::set_last_error;
use crate::{get_runtime, KjarniDevice, KjarniError};
use kjarni::indexer::{Indexer, IndexerError, IndexInfo, IndexStats};
use std::ffi::{c_char, c_void, CStr, CString};
use std::ptr;

/// Statistics returned after indexing
#[repr(C)]
pub struct KjarniIndexStats {
    pub documents_indexed: usize,
    pub chunks_created: usize,
    pub dimension: usize,
    pub size_bytes: u64,
    pub files_processed: usize,
    pub files_skipped: usize,
    pub elapsed_ms: u64,
}

impl From<IndexStats> for KjarniIndexStats {
    fn from(s: IndexStats) -> Self {
        Self {
            documents_indexed: s.documents_indexed,
            chunks_created: s.chunks_created,
            dimension: s.dimension,
            size_bytes: s.size_bytes,
            files_processed: s.files_processed,
            files_skipped: s.files_skipped,
            elapsed_ms: s.elapsed_ms,
        }
    }
}

/// Information about an existing index
#[repr(C)]
pub struct KjarniIndexInfo {
    pub path: *mut c_char,
    pub document_count: usize,
    pub segment_count: usize,
    pub dimension: usize,
    pub size_bytes: u64,
    pub embedding_model: *mut c_char, // May be NULL
}

impl KjarniIndexInfo {
    fn from_info(info: IndexInfo) -> Self {
        let path = CString::new(info.path).unwrap_or_default().into_raw();
        let embedding_model = info
            .embedding_model
            .and_then(|s| CString::new(s).ok())
            .map(|c| c.into_raw())
            .unwrap_or(ptr::null_mut());

        Self {
            path,
            document_count: info.document_count,
            segment_count: info.segment_count,
            dimension: info.dimension,
            size_bytes: info.size_bytes,
            embedding_model,
        }
    }
}

/// Free index info strings
#[no_mangle]
pub unsafe extern "C" fn kjarni_index_info_free(info: KjarniIndexInfo) {
    if !info.path.is_null() {
        let _ = CString::from_raw(info.path);
    }
    if !info.embedding_model.is_null() {
        let _ = CString::from_raw(info.embedding_model);
    }
}

/// Configuration for creating an Indexer
#[repr(C)]
pub struct KjarniIndexerConfig {
    pub device: KjarniDevice,
    pub cache_dir: *const c_char,
    pub model_name: *const c_char,
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub batch_size: usize,
    /// Comma-separated list of extensions (NULL = use defaults)
    pub extensions: *const c_char,
    /// Comma-separated list of exclude patterns
    pub exclude_patterns: *const c_char,
    pub recursive: i32,
    pub include_hidden: i32,
    pub max_file_size: usize, // 0 = no limit
    pub quiet: i32,
}

/// Get default indexer configuration
#[no_mangle]
pub extern "C" fn kjarni_indexer_config_default() -> KjarniIndexerConfig {
    KjarniIndexerConfig {
        device: KjarniDevice::Cpu,
        cache_dir: ptr::null(),
        model_name: ptr::null(),
        chunk_size: 512,
        chunk_overlap: 50,
        batch_size: 32,
        extensions: ptr::null(),
        exclude_patterns: ptr::null(),
        recursive: 1,
        include_hidden: 0,
        max_file_size: 10 * 1024 * 1024, // 10MB
        quiet: 0,
    }
}

/// Opaque handle to an Indexer
pub struct KjarniIndexer {
    inner: Indexer,
}

/// Create a new Indexer
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_new(
    config: *const KjarniIndexerConfig,
    out: *mut *mut KjarniIndexer,
) -> KjarniError {
    if out.is_null() {
        return KjarniError::NullPointer;
    }

    let default_config = kjarni_indexer_config_default();
    let config = if config.is_null() {
        &default_config
    } else {
        &*config
    };

    let result = get_runtime().block_on(async {
        // Get model name
        let model_name = if !config.model_name.is_null() {
            match CStr::from_ptr(config.model_name).to_str() {
                Ok(s) => s,
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        } else {
            "minilm-l6-v2" // Default
        };

        let mut builder = Indexer::builder(model_name);

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

        // Chunking
        builder = builder.chunk_size(config.chunk_size);
        builder = builder.chunk_overlap(config.chunk_overlap);
        builder = builder.batch_size(config.batch_size);

        // Extensions
        if !config.extensions.is_null() {
            match CStr::from_ptr(config.extensions).to_str() {
                Ok(s) => {
                    let exts: Vec<&str> = s.split(',').map(|e| e.trim()).collect();
                    builder = builder.extensions(&exts);
                }
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        }

        // Exclude patterns
        if !config.exclude_patterns.is_null() {
            match CStr::from_ptr(config.exclude_patterns).to_str() {
                Ok(s) => {
                    for pattern in s.split(',').map(|p| p.trim()) {
                        builder = builder.exclude(pattern);
                    }
                }
                Err(_) => return Err(KjarniError::InvalidUtf8),
            }
        }

        // Flags
        builder = builder.recursive(config.recursive != 0);
        builder = builder.include_hidden(config.include_hidden != 0);
        if config.max_file_size > 0 {
            builder = builder.max_file_size(config.max_file_size);
        }
        builder = builder.quiet(config.quiet != 0);

        builder.build().await.map_err(|e| {
            set_last_error(e.to_string());
            KjarniError::LoadFailed
        })
    });

    match result {
        Ok(indexer) => {
            let handle = Box::new(KjarniIndexer { inner: indexer });
            *out = Box::into_raw(handle);
            KjarniError::Ok
        }
        Err(e) => e,
    }
}

/// Free an Indexer
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_free(indexer: *mut KjarniIndexer) {
    if !indexer.is_null() {
        let _ = Box::from_raw(indexer);
    }
}

/// Create a new index (simple version, outputs to stderr)
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_create(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    force: i32,
    out: *mut KjarniIndexStats,
) -> KjarniError {
    kjarni_indexer_create_with_callback(
        indexer, index_path, inputs, num_inputs, force, None, ptr::null_mut(), ptr::null(), out,
    )
}

/// Create a new index with progress callback
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_create_with_callback(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    force: i32,
    progress_callback: KjarniProgressCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
    out: *mut KjarniIndexStats,
) -> KjarniError {
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || out.is_null() {
        return KjarniError::NullPointer;
    }

    if num_inputs == 0 {
        set_last_error("No input paths specified");
        return KjarniError::InvalidConfig;
    }

    let indexer_ref = &(*indexer).inner;

    // Convert index path
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    // Convert input paths
    let mut input_vec = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        let input_ptr = *inputs.add(i);
        if input_ptr.is_null() {
            return KjarniError::NullPointer;
        }
        match CStr::from_ptr(input_ptr).to_str() {
            Ok(s) => input_vec.push(s),
            Err(_) => return KjarniError::InvalidUtf8,
        }
    }

    // Setup progress callback wrapper
    let ffi_callback = FfiProgressCallback::new(progress_callback, user_data);

    // Create cancel check function
    let check_cancel = move || is_cancelled(cancel_token);

    let result = get_runtime().block_on(async {
        indexer_ref
            .create_with_options(
                index_path,
                &input_vec,
                force != 0,
                ffi_callback.as_ref(),
                &check_cancel,
            )
            .await
    });

    match result {
        Ok(stats) => {
            *out = stats.into();
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            match e {
                IndexerError::Cancelled => KjarniError::Cancelled,
                IndexerError::IndexExists(_) => KjarniError::InvalidConfig,
                IndexerError::IndexNotFound(_) => KjarniError::ModelNotFound,
                _ => KjarniError::InferenceFailed,
            }
        }
    }
}

/// Add documents to existing index
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_add(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    documents_added: *mut usize,
) -> KjarniError {
    kjarni_indexer_add_with_callback(
        indexer,
        index_path,
        inputs,
        num_inputs,
        None,
        ptr::null_mut(),
        ptr::null(),
        documents_added,
    )
}

/// Add documents with progress callback
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_add_with_callback(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    progress_callback: KjarniProgressCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
    documents_added: *mut usize,
) -> KjarniError {
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || documents_added.is_null() {
        return KjarniError::NullPointer;
    }

    if num_inputs == 0 {
        *documents_added = 0;
        return KjarniError::Ok;
    }

    let indexer_ref = &(*indexer).inner;

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    let mut input_vec = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        let input_ptr = *inputs.add(i);
        if input_ptr.is_null() {
            return KjarniError::NullPointer;
        }
        match CStr::from_ptr(input_ptr).to_str() {
            Ok(s) => input_vec.push(s),
            Err(_) => return KjarniError::InvalidUtf8,
        }
    }

    let ffi_callback = FfiProgressCallback::new(progress_callback, user_data);
    let check_cancel = move || is_cancelled(cancel_token);

    let result = get_runtime().block_on(async {
        indexer_ref
            .add_with_callback(index_path, &input_vec, ffi_callback.as_ref(), &check_cancel)
            .await
    });

    match result {
        Ok(count) => {
            *documents_added = count;
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *documents_added = 0;
            match e {
                IndexerError::Cancelled => KjarniError::Cancelled,
                IndexerError::DimensionMismatch { .. } => KjarniError::InvalidConfig,
                _ => KjarniError::InferenceFailed,
            }
        }
    }
}

/// Get index information (static - no indexer needed)
#[no_mangle]
pub unsafe extern "C" fn kjarni_index_info(
    index_path: *const c_char,
    out: *mut KjarniIndexInfo,
) -> KjarniError {
    if index_path.is_null() || out.is_null() {
        return KjarniError::NullPointer;
    }

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    match Indexer::info(index_path) {
        Ok(info) => {
            *out = KjarniIndexInfo::from_info(info);
            KjarniError::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            KjarniError::ModelNotFound
        }
    }
}

/// Delete an index
#[no_mangle]
pub unsafe extern "C" fn kjarni_index_delete(index_path: *const c_char) -> KjarniError {
    if index_path.is_null() {
        return KjarniError::NullPointer;
    }

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniError::InvalidUtf8,
    };

    match Indexer::delete(index_path) {
        Ok(()) => KjarniError::Ok,
        Err(e) => {
            set_last_error(e.to_string());
            KjarniError::InferenceFailed
        }
    }
}

// Accessors
#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_model_name(indexer: *const KjarniIndexer) -> *const c_char {
    static mut MODEL_NAME_BUF: Option<CString> = None;

    if indexer.is_null() {
        return ptr::null();
    }

    let name = (*indexer).inner.model_name();
    MODEL_NAME_BUF = CString::new(name).ok();
    MODEL_NAME_BUF
        .as_ref()
        .map(|c| c.as_ptr())
        .unwrap_or(ptr::null())
}

#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_dimension(indexer: *const KjarniIndexer) -> usize {
    if indexer.is_null() {
        return 0;
    }
    (*indexer).inner.dimension()
}

#[no_mangle]
pub unsafe extern "C" fn kjarni_indexer_chunk_size(indexer: *const KjarniIndexer) -> usize {
    if indexer.is_null() {
        return 0;
    }
    (*indexer).inner.chunk_size()
}