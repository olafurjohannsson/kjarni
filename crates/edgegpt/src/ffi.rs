// Use the EdgeGPT struct from your library's main file.
use super::EdgeGPT; 
use edgetransformers::prelude::{Device, WgpuContext};
use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::ptr;
use std::slice;
use std::sync::Arc;
use tokio::runtime::Runtime;

// --- Global Tokio Runtime ---
static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

// --- Opaque Handle ---
// The C code will only see this as a void*, but Rust knows it's a pointer to our EdgeGPT engine.
pub type EdgeGptHandle = *mut EdgeGPT;

// --- C-style Enum for Device Selection ---
#[repr(C)]
pub enum EdgeGptDevice {
    Cpu = 0,
    Gpu = 1,
}

// --- C-compatible struct to return embeddings result ---
#[repr(C)]
pub struct CEmbeddingResult {
    pub embeddings_ptr: *mut f32,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
    pub error_message: *mut c_char,
}


/// Creates a new instance of the EdgeGPT engine.
///
/// Returns a handle to the engine. This handle must be freed later
/// by calling `edgegpt_destroy`. Returns `null` if creation fails.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgegpt_create(device_type: EdgeGptDevice) -> EdgeGptHandle {
    let engine = match device_type {
        EdgeGptDevice::Cpu => {
            println!("Creating EdgeGPT engine for CPU...");
            EdgeGPT::new(Device::Cpu, None)
        }
        EdgeGptDevice::Gpu => {
            println!("Creating EdgeGPT engine for GPU...");
            // We must block on the async context creation.
            let context_result = RUNTIME.block_on(WgpuContext::new());
            EdgeGPT::new(Device::Wgpu, Some(Arc::new(context_result)))
        }
    };
    Box::into_raw(Box::new(engine))
}

/// Destroys an instance of the EdgeGPT engine and frees its memory.
/// # Safety
/// - `handle` must be a valid pointer returned by `edgegpt_create`.
#[unsafe(no_mangle)]
pub extern "C" fn edgegpt_destroy(handle: EdgeGptHandle) {
    if handle.is_null() {
        return;
    }
    unsafe {
        let _ = Box::from_raw(handle);
    }
}

/// Encodes a batch of sentences using the specified engine instance.
/// # Safety
/// - `handle` must be a valid pointer returned by `edgegpt_create`.
/// - `sentences_ptr` must be a valid pointer to an array of null-terminated UTF-8 strings.
/// - The returned pointer must be freed by the caller using `edgegpt_free_embedding_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgegpt_encode_batch(
    handle: EdgeGptHandle,
    sentences_ptr: *const *const c_char,
    num_sentences: c_int,
) -> *mut CEmbeddingResult {
    if handle.is_null() || sentences_ptr.is_null() || num_sentences <= 0 {
        return ptr::null_mut();
    }

    let engine = unsafe { &*handle };
    let sentences_slice = unsafe { slice::from_raw_parts(sentences_ptr, num_sentences as usize) };

    let rust_sentences: Vec<&str> = sentences_slice
        .iter()
        .map(|&p| unsafe { CStr::from_ptr(p).to_str().unwrap() })
        .collect();

    // Call the high-level encode_batch method on our EdgeGPT instance
    let result = RUNTIME.block_on(engine.encode_batch(&rust_sentences));

    let result_struct = match result {
        Ok(embeddings) => {
            let num_embeddings = embeddings.len();
            let embedding_dim = if num_embeddings > 0 { embeddings[0].len() } else { 0 };
            
            // Flatten the Vec<Vec<f32>> into a single Vec<f32> for C
            let mut flat_embeddings = embeddings.into_iter().flatten().collect::<Vec<f32>>();
            
            flat_embeddings.shrink_to_fit();
            let embeddings_ptr = flat_embeddings.as_mut_ptr();
            std::mem::forget(flat_embeddings); // Give ownership to the C caller

            CEmbeddingResult {
                embeddings_ptr,
                num_embeddings,
                embedding_dim,
                error_message: ptr::null_mut(),
            }
        }
        Err(e) => {
            let error_message = CString::new(format!("ERROR: {}", e)).unwrap().into_raw();
            CEmbeddingResult {
                embeddings_ptr: ptr::null_mut(),
                num_embeddings: 0,
                embedding_dim: 0,
                error_message,
            }
        }
    };

    Box::into_raw(Box::new(result_struct))
}

/// Frees the memory allocated for a CEmbeddingResult.
/// # Safety
/// - `result_ptr` must be a pointer returned by `edgegpt_encode_batch`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edgegpt_free_embedding_result(result_ptr: *mut CEmbeddingResult) {
    if result_ptr.is_null() {
        return;
    }
    unsafe {
        let result = Box::from_raw(result_ptr);
        if !result.error_message.is_null() {
            let _ = CString::from_raw(result.error_message);
        }
        if !result.embeddings_ptr.is_null() {
            // Reconstruct the flat Vec<f32> to let Rust handle freeing its memory
            let total_elements = result.num_embeddings * result.embedding_dim;
            let _ = Vec::from_raw_parts(result.embeddings_ptr, total_elements, total_elements);
        }
    }
}