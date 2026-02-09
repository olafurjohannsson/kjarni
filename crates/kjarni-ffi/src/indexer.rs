//! Indexer FFI bindings
//!
//! This module provides C-compatible bindings for the Kjarni indexer functionality.
//! The indexer allows creating and managing vector indexes for RAG (Retrieval Augmented Generation)
//! applications.
//!
//! # Architecture
//!
//! The FFI layer is intentionally thin:
//! - Simple functions (`kjarni_indexer_create`, `kjarni_indexer_add`) call the Rust API directly
//! - Callback functions (`kjarni_indexer_create_with_callback`, etc.) wrap FFI callbacks and pass them through
//!
//! # Thread Safety
//!
//! All functions are thread-safe. The underlying Indexer uses async operations
//! which are executed on a shared Tokio runtime.

use crate::callback::{
    KjarniCancelToken, KjarniProgress, KjarniProgressCallbackFn, KjarniProgressStage, is_cancelled,
};
use crate::error::set_last_error;
use crate::{KjarniDevice, KjarniErrorCode, get_runtime};
use kjarni::ProgressStage;
use kjarni::indexer::{IndexInfo, IndexStats, Indexer, IndexerError};
use std::ffi::{CStr, CString, c_char, c_void};
use std::ptr;


// FFI Structures


/// Statistics returned after indexing operations.
///
/// This struct is returned by `kjarni_indexer_create` and contains
/// information about what was indexed and how long it took.
#[repr(C)]
pub struct KjarniIndexStats {
    /// Number of document chunks indexed (after splitting)
    pub documents_indexed: usize,
    /// Number of chunks created from source files
    pub chunks_created: usize,
    /// Embedding dimension used
    pub dimension: usize,
    /// Total index size on disk in bytes
    pub size_bytes: u64,
    /// Number of source files successfully processed
    pub files_processed: usize,
    /// Number of source files skipped (errors, unsupported format, too large)
    pub files_skipped: usize,
    /// Total time taken in milliseconds
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

/// Information about an existing index.
///
/// Retrieved via `kjarni_index_info`. The caller must free this struct
/// using `kjarni_index_info_free` to avoid memory leaks.
#[repr(C)]
pub struct KjarniIndexInfo {
    /// Path to the index directory (must be freed)
    pub path: *mut c_char,
    /// Total number of documents in the index
    pub document_count: usize,
    /// Number of index segments
    pub segment_count: usize,
    /// Embedding dimension
    pub dimension: usize,
    /// Total size on disk in bytes
    pub size_bytes: u64,
    /// Embedding model name used to create the index (may be NULL, must be freed if not)
    pub embedding_model: *mut c_char,
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

/// Free memory allocated for index info strings.
///
/// # Safety
///
/// Must only be called once per `KjarniIndexInfo` returned from `kjarni_index_info`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_index_info_free(info: KjarniIndexInfo) {
    if !info.path.is_null() {
        let _ = CString::from_raw(info.path);
    }
    if !info.embedding_model.is_null() {
        let _ = CString::from_raw(info.embedding_model);
    }
}

/// Configuration for creating an Indexer.
///
/// Use `kjarni_indexer_config_default()` to get sensible defaults,
/// then modify fields as needed before passing to `kjarni_indexer_new()`.
#[repr(C)]
pub struct KjarniIndexerConfig {
    /// Device to run embeddings on (CPU or GPU)
    pub device: KjarniDevice,
    /// Directory to cache downloaded models (NULL = system default)
    pub cache_dir: *const c_char,
    /// Embedding model name (NULL = "minilm-l6-v2")
    pub model_name: *const c_char,
    /// Maximum chunk size in characters
    pub chunk_size: usize,
    /// Overlap between adjacent chunks in characters
    pub chunk_overlap: usize,
    /// Batch size for embedding operations
    pub batch_size: usize,
    /// Comma-separated list of file extensions to include (NULL = use defaults)
    pub extensions: *const c_char,
    /// Comma-separated list of glob patterns to exclude
    pub exclude_patterns: *const c_char,
    /// Whether to recurse into subdirectories (1 = true, 0 = false)
    pub recursive: i32,
    /// Whether to include hidden files (1 = true, 0 = false)
    pub include_hidden: i32,
    /// Maximum file size in bytes (0 = no limit)
    pub max_file_size: usize,
    /// Suppress progress output to stderr (1 = quiet, 0 = verbose)
    pub quiet: i32,
}

/// Get default indexer configuration.
///
/// Returns a configuration with sensible defaults:
/// - CPU device
/// - minilm-l6-v2 model
/// - 512 character chunks with 50 character overlap
/// - 32 batch size
/// - 10MB max file size
/// - Recursive directory traversal
/// - Hidden files excluded
#[unsafe(no_mangle)]
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


// Indexer Handle


/// Opaque handle to an Indexer instance.
///
/// Created via `kjarni_indexer_new`, must be freed via `kjarni_indexer_free`.
pub struct KjarniIndexer {
    inner: Indexer,
}

/// Create a new Indexer.
///
/// # Arguments
///
/// * `config` - Configuration options (NULL for defaults)
/// * `out` - Pointer to receive the created indexer handle
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
/// On error, call `kjarni_last_error_message()` for details.
///
/// # Safety
///
/// - `out` must be a valid pointer
/// - The returned handle must be freed with `kjarni_indexer_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_new(
    config: *const KjarniIndexerConfig,
    out: *mut *mut KjarniIndexer,
) -> KjarniErrorCode {
    if out.is_null() {
        return KjarniErrorCode::NullPointer;
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
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
            }
        } else {
            "minilm-l6-v2"
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
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
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
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
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
                Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
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
            KjarniErrorCode::LoadFailed
        })
    });

    match result {
        Ok(indexer) => {
            let handle = Box::new(KjarniIndexer { inner: indexer });
            *out = Box::into_raw(handle);
            KjarniErrorCode::Ok
        }
        Err(e) => e,
    }
}

/// Free an Indexer handle.
///
/// # Safety
///
/// - `indexer` must be a handle returned by `kjarni_indexer_new`
/// - Must not be called more than once per handle
/// - Handle must not be used after freeing
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_free(indexer: *mut KjarniIndexer) {
    if !indexer.is_null() {
        let _ = Box::from_raw(indexer);
    }
}


// Helper Functions


/// Convert Rust ProgressStage to FFI KjarniProgressStage
fn convert_stage(stage: ProgressStage) -> KjarniProgressStage {
    match stage {
        ProgressStage::Scanning => KjarniProgressStage::Scanning,
        ProgressStage::Loading => KjarniProgressStage::Loading,
        ProgressStage::Embedding => KjarniProgressStage::Embedding,
        ProgressStage::Writing => KjarniProgressStage::Writing,
        ProgressStage::Committing => KjarniProgressStage::Committing,
        ProgressStage::Searching => KjarniProgressStage::Searching,
        ProgressStage::Reranking => KjarniProgressStage::Reranking,
    }
}

/// Convert IndexerError to appropriate error code
fn indexer_error_to_code(e: &IndexerError) -> KjarniErrorCode {
    match e {
        IndexerError::Cancelled => KjarniErrorCode::Cancelled,
        IndexerError::IndexExists(_) => KjarniErrorCode::InvalidConfig,
        IndexerError::IndexNotFound(_) => KjarniErrorCode::ModelNotFound,
        IndexerError::DimensionMismatch { .. } => KjarniErrorCode::InvalidConfig,
        IndexerError::NoInputs => KjarniErrorCode::InvalidConfig,
        IndexerError::PathNotFound(_) => KjarniErrorCode::ModelNotFound,
        _ => KjarniErrorCode::InferenceFailed,
    }
}

/// Parse input paths from FFI array
unsafe fn parse_inputs<'a>(
    inputs: *const *const c_char,
    num_inputs: usize,
) -> Result<Vec<&'a str>, KjarniErrorCode> {
    let mut input_vec = Vec::with_capacity(num_inputs);
    for i in 0..num_inputs {
        let input_ptr = *inputs.add(i);
        if input_ptr.is_null() {
            return Err(KjarniErrorCode::NullPointer);
        }
        match CStr::from_ptr(input_ptr).to_str() {
            Ok(s) => input_vec.push(s),
            Err(_) => return Err(KjarniErrorCode::InvalidUtf8),
        }
    }
    Ok(input_vec)
}


// Create Index Functions


/// Create a new index from files/directories (simple version).
///
/// This is the simple API without progress callbacks. For progress reporting
/// and cancellation support, use `kjarni_indexer_create_with_callback`.
///
/// # Arguments
///
/// * `indexer` - Indexer handle from `kjarni_indexer_new`
/// * `index_path` - Path where the index will be created
/// * `inputs` - Array of file/directory paths to index
/// * `num_inputs` - Number of elements in `inputs` array
/// * `force` - If non-zero, overwrite existing index at `index_path`
/// * `out` - Pointer to receive indexing statistics
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - `inputs` must contain at least `num_inputs` valid C strings
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_create(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    force: i32,
    out: *mut KjarniIndexStats,
) -> KjarniErrorCode {
    // Validate pointers
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let indexer_ref = &(*indexer).inner;

    // Parse index path
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    // Parse input paths
    let input_vec = match parse_inputs(inputs, num_inputs) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // Call the simple Rust API directly (no callbacks)
    let result = get_runtime().block_on(async {
        indexer_ref
            .create_with_options(index_path, &input_vec, force != 0)
            .await
    });

    match result {
        Ok(stats) => {
            *out = stats.into();
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            indexer_error_to_code(&e)
        }
    }
}

/// Create a new index with progress callback and cancellation support.
///
/// This is the full-featured API that supports:
/// - Progress reporting via callback
/// - Cancellation via cancel token
///
/// # Arguments
///
/// * `indexer` - Indexer handle from `kjarni_indexer_new`
/// * `index_path` - Path where the index will be created
/// * `inputs` - Array of file/directory paths to index
/// * `num_inputs` - Number of elements in `inputs` array
/// * `force` - If non-zero, overwrite existing index at `index_path`
/// * `progress_callback` - Optional callback for progress updates (may be NULL)
/// * `user_data` - Opaque pointer passed to callback (may be NULL)
/// * `cancel_token` - Optional cancellation token (may be NULL)
/// * `out` - Pointer to receive indexing statistics
///
/// # Callback
///
/// The progress callback receives a `KjarniProgress` struct with:
/// - `stage`: Current operation (scanning, loading, embedding, etc.)
/// - `current`: Current item number
/// - `total`: Total items (may be 0 if unknown)
/// - `message`: Optional status message (may be NULL)
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, `KjarniErrorCode::Cancelled` if cancelled,
/// or other error code on failure.
///
/// # Safety
///
/// - All non-optional pointers must be valid
/// - `inputs` must contain at least `num_inputs` valid C strings
/// - Callback must be thread-safe if indexer uses multiple threads
#[unsafe(no_mangle)]
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
) -> KjarniErrorCode {
    // Validate required pointers
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let indexer_ref = &(*indexer).inner;

    // Parse index path
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    // Parse input paths
    let input_vec = match parse_inputs(inputs, num_inputs) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // Create progress callback closure
    // We always create a closure to avoid generic type inference issues
    // The closure checks internally if the FFI callback is present
    let on_progress =
        move |stage: ProgressStage, current: usize, total: usize, msg: Option<&str>| {
            if let Some(callback) = progress_callback {
                let ffi_stage = convert_stage(stage);

                // Convert message to C string (temporary, valid for callback duration)
                let msg_cstring = msg.and_then(|s| CString::new(s).ok());
                let msg_ptr = msg_cstring
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(ptr::null());

                let progress = KjarniProgress {
                    stage: ffi_stage,
                    current,
                    total,
                    message: msg_ptr,
                };

                callback(progress, user_data);
            }
        };

    // Create cancellation check closure
    let is_cancelled_fn = move || -> bool {
        if cancel_token.is_null() {
            false
        } else {
            is_cancelled(cancel_token)
        }
    };

    // Call the Rust API with callbacks
    let result = get_runtime().block_on(async {
        indexer_ref
            .create_with_callback(
                index_path,
                &input_vec,
                force != 0,
                Some(on_progress),
                Some(is_cancelled_fn),
            )
            .await
    });

    match result {
        Ok(stats) => {
            *out = stats.into();
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            indexer_error_to_code(&e)
        }
    }
}


// Add to Index Functions


/// Add documents to an existing index (simple version).
///
/// This is the simple API without progress callbacks. For progress reporting
/// and cancellation support, use `kjarni_indexer_add_with_callback`.
///
/// # Arguments
///
/// * `indexer` - Indexer handle from `kjarni_indexer_new`
/// * `index_path` - Path to existing index
/// * `inputs` - Array of file/directory paths to add
/// * `num_inputs` - Number of elements in `inputs` array
/// * `documents_added` - Pointer to receive count of documents added
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - `inputs` must contain at least `num_inputs` valid C strings
/// - Index at `index_path` must exist and be compatible (same embedding dimension)
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_add(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    documents_added: *mut usize,
) -> KjarniErrorCode {
    // Validate pointers
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || documents_added.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    // Handle empty input
    if num_inputs == 0 {
        *documents_added = 0;
        return KjarniErrorCode::Ok;
    }

    let indexer_ref = &(*indexer).inner;

    // Parse index path
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    // Parse input paths
    let input_vec = match parse_inputs(inputs, num_inputs) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // Call the simple Rust API directly (no callbacks)
    let result = get_runtime().block_on(async { indexer_ref.add(index_path, &input_vec).await });

    match result {
        Ok(count) => {
            *documents_added = count;
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *documents_added = 0;
            indexer_error_to_code(&e)
        }
    }
}

/// Add documents to an existing index with progress callback and cancellation support.
///
/// This is the full-featured API that supports:
/// - Progress reporting via callback
/// - Cancellation via cancel token
///
/// # Arguments
///
/// * `indexer` - Indexer handle from `kjarni_indexer_new`
/// * `index_path` - Path to existing index
/// * `inputs` - Array of file/directory paths to add
/// * `num_inputs` - Number of elements in `inputs` array
/// * `progress_callback` - Optional callback for progress updates (may be NULL)
/// * `user_data` - Opaque pointer passed to callback (may be NULL)
/// * `cancel_token` - Optional cancellation token (may be NULL)
/// * `documents_added` - Pointer to receive count of documents added
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, `KjarniErrorCode::Cancelled` if cancelled,
/// or other error code on failure.
///
/// # Safety
///
/// - All non-optional pointers must be valid
/// - `inputs` must contain at least `num_inputs` valid C strings
/// - Index at `index_path` must exist and be compatible
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_add_with_callback(
    indexer: *mut KjarniIndexer,
    index_path: *const c_char,
    inputs: *const *const c_char,
    num_inputs: usize,
    progress_callback: KjarniProgressCallbackFn,
    user_data: *mut c_void,
    cancel_token: *const KjarniCancelToken,
    documents_added: *mut usize,
) -> KjarniErrorCode {
    // Validate required pointers
    if indexer.is_null() || index_path.is_null() || inputs.is_null() || documents_added.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    // Handle empty input
    if num_inputs == 0 {
        *documents_added = 0;
        return KjarniErrorCode::Ok;
    }

    let indexer_ref = &(*indexer).inner;

    // Parse index path
    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    // Parse input paths
    let input_vec = match parse_inputs(inputs, num_inputs) {
        Ok(v) => v,
        Err(e) => return e,
    };

    // Create progress callback closure
    let on_progress =
        move |stage: ProgressStage, current: usize, total: usize, msg: Option<&str>| {
            if let Some(callback) = progress_callback {
                let ffi_stage = convert_stage(stage);

                let msg_cstring = msg.and_then(|s| CString::new(s).ok());
                let msg_ptr = msg_cstring
                    .as_ref()
                    .map(|c| c.as_ptr())
                    .unwrap_or(ptr::null());

                let progress = KjarniProgress {
                    stage: ffi_stage,
                    current,
                    total,
                    message: msg_ptr,
                };

                callback(progress, user_data);
            }
        };

    // Create cancellation check closure
    let is_cancelled_fn = move || -> bool {
        if cancel_token.is_null() {
            false
        } else {
            is_cancelled(cancel_token)
        }
    };

    // Call the Rust API with callbacks
    let result = get_runtime().block_on(async {
        indexer_ref
            .add_with_callback(
                index_path,
                &input_vec,
                Some(on_progress),
                Some(is_cancelled_fn),
            )
            .await
    });

    match result {
        Ok(count) => {
            *documents_added = count;
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            *documents_added = 0;
            indexer_error_to_code(&e)
        }
    }
}


// Index Management Functions


/// Get information about an existing index.
///
/// This is a static function that doesn't require an Indexer handle.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory
/// * `out` - Pointer to receive index information
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - All pointers must be valid
/// - Caller must free the returned `KjarniIndexInfo` with `kjarni_index_info_free`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_index_info(
    index_path: *const c_char,
    out: *mut KjarniIndexInfo,
) -> KjarniErrorCode {
    if index_path.is_null() || out.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    match Indexer::info(index_path) {
        Ok(info) => {
            *out = KjarniIndexInfo::from_info(info);
            KjarniErrorCode::Ok
        }
        Err(e) => {
            set_last_error(e.to_string());
            KjarniErrorCode::ModelNotFound
        }
    }
}

/// Delete an index.
///
/// This permanently removes the index directory and all its contents.
/// This is a static function that doesn't require an Indexer handle.
///
/// # Arguments
///
/// * `index_path` - Path to the index directory to delete
///
/// # Returns
///
/// `KjarniErrorCode::Ok` on success, error code otherwise.
///
/// # Safety
///
/// - `index_path` must be a valid C string
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_index_delete(index_path: *const c_char) -> KjarniErrorCode {
    if index_path.is_null() {
        return KjarniErrorCode::NullPointer;
    }

    let index_path = match CStr::from_ptr(index_path).to_str() {
        Ok(s) => s,
        Err(_) => return KjarniErrorCode::InvalidUtf8,
    };

    match Indexer::delete(index_path) {
        Ok(()) => KjarniErrorCode::Ok,
        Err(e) => {
            set_last_error(e.to_string());
            KjarniErrorCode::InferenceFailed
        }
    }
}


// Accessor Functions


/// Get the embedding model name used by the indexer.
///
/// # Arguments
///
/// * `indexer` - Indexer handle
///
/// # Returns
///
/// Pointer to model name string, or NULL if indexer is NULL.
/// The returned pointer is valid until the next call to this function.
///
/// # Safety
///
/// - `indexer` must be a valid handle or NULL
/// - Returned string must not be modified or freed
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_model_name(
    indexer: *const KjarniIndexer,
    buf: *mut c_char,
    buf_len: usize,
) -> usize {
    if indexer.is_null() {
        return 0;
    }

    let name: &str = unsafe { (*indexer).inner.model_name() };
    let name_bytes = name.as_bytes();
    let required = name_bytes.len() + 1; // +1 for null terminator

    // If no buffer provided, just return required size
    if buf.is_null() || buf_len == 0 {
        return required;
    }

    // Copy as much as fits
    let copy_len = name_bytes.len().min(buf_len.saturating_sub(1));
    unsafe {
        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), buf as *mut u8, copy_len);

        // Null terminate
        *buf.add(copy_len) = 0;
    }
    required
}

/// Get the embedding dimension used by the indexer.
///
/// # Arguments
///
/// * `indexer` - Indexer handle
///
/// # Returns
///
/// Embedding dimension, or 0 if indexer is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_dimension(indexer: *const KjarniIndexer) -> usize {
    if indexer.is_null() {
        return 0;
    }
    (*indexer).inner.dimension()
}

/// Get the chunk size configured for the indexer.
///
/// # Arguments
///
/// * `indexer` - Indexer handle
///
/// # Returns
///
/// Chunk size in characters, or 0 if indexer is NULL.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_indexer_chunk_size(indexer: *const KjarniIndexer) -> usize {
    if indexer.is_null() {
        return 0;
    }
    (*indexer).inner.chunk_size()
}
