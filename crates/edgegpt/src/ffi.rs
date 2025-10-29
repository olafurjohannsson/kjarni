use super::EdgeGPT;
use edgetransformers::prelude::{Device, WgpuContext};
use once_cell::sync::Lazy;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::sync::Arc;
use tokio::runtime::Runtime;

// --- Global Tokio Runtime ---
// A single, global runtime is needed to execute our async Rust code from the sync C world.
static RUNTIME: Lazy<Runtime> = Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

// --- Opaque Handle ---
// The C code will only see this as a `void*`. Rust knows it's a pointer to our EdgeGPT engine.
pub type EdgeGptHandle = *mut EdgeGPT;

// --- C-style Enum for Device Selection ---
#[repr(C)]
pub enum EdgeGptDevice {
    Cpu = 0,
    Gpu = 1,
}

/// Creates a new instance of the EdgeGPT engine for a specified device.
///
/// Returns a handle to the engine instance. This handle must be freed later
/// by calling `edgegpt_destroy`.
///
/// Returns `null` if creation fails (e.g., no compatible GPU found).
#[no_mangle]
pub extern "C" fn edgegpt_create(device_type: EdgeGptDevice) -> EdgeGptHandle {
    let result = RUNTIME.block_on(async {
        match device_type {
            EdgeGptDevice::Cpu => {
                println!("Creating EdgeGPT engine for CPU...");
                Ok(EdgeGPT::new(Device::Cpu, None))
            }
            EdgeGptDevice::Gpu => {
                println!("Creating EdgeGPT engine for GPU...");
                let context = WgpuContext::new().await;
                Ok(EdgeGPT::new(Device::Wgpu, Some(Arc::new(context))))
            }
        }
    });

    match result {
        Ok(engine) => {
            // Put the engine on the heap and return a raw pointer to it.
            // The caller is now responsible for this memory.
            Box::into_raw(Box::new(engine))
        }
        Err(e) => {
            eprintln!("Failed to create EdgeGPT engine: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Destroys an instance of the EdgeGPT engine and frees its memory.
///
/// # Safety
/// - `handle` must be a valid pointer returned by `edgegpt_create`.
/// - This function must only be called once for each handle.
#[no_mangle]
pub extern "C" fn edgegpt_destroy(handle: EdgeGptHandle) {
    if handle.is_null() {
        return;
    }
    // Reconstruct the Box from the raw pointer and let Rust's drop semantics
    // handle the cleanup of the EdgeGPT instance and all its loaded models.
    unsafe {
        let _ = Box::from_raw(handle);
    }
}

/// Summarizes a UTF-8 string using the specified engine instance.
///
/// # Safety
/// - `handle` must be a valid pointer returned by `edgegpt_create`.
/// - `text_ptr` must be a valid, null-terminated UTF-8 string.
/// - The returned pointer must be freed by the caller using `edgegpt_free_string`.
#[no_mangle]
pub extern "C" fn edgegpt_summarize(handle: EdgeGptHandle, text_ptr: *const c_char, max_length: i32) -> *mut c_char {
    if handle.is_null() || text_ptr.is_null() {
        return std::ptr::null_mut();
    }
    
    let engine = unsafe { &*handle };
    let text = unsafe { CStr::from_ptr(text_ptr) }.to_str().unwrap_or("");

    let result = RUNTIME.block_on(engine.summarize(text, max_length as usize));

    let output_str = match result {
        Ok(summary) => summary,
        Err(e) => format!("ERROR: {}", e),
    };

    CString::new(output_str).unwrap().into_raw()
}

/// Generates text from a UTF-8 prompt using the specified engine instance.
///
/// # Safety
/// - `handle` must be a valid pointer returned by `edgegpt_create`.
/// - `prompt_ptr` must be a valid, null-terminated UTF-8 string.
/// - The returned pointer must be freed by the caller using `edgegpt_free_string`.
#[no_mangle]
pub extern "C" fn edgegpt_generate(handle: EdgeGptHandle, prompt_ptr: *const c_char, max_new_tokens: i32) -> *mut c_char {
    if handle.is_null() || prompt_ptr.is_null() {
        return std::ptr::null_mut();
    }

    let engine = unsafe { &*handle };
    let prompt = unsafe { CStr::from_ptr(prompt_ptr) }.to_str().unwrap_or("");

    let result = RUNTIME.block_on(engine.generate(prompt, max_new_tokens as usize));

    let output_str = match result {
        Ok(text) => text,
        Err(e) => format!("ERROR: {}", e),
    };

    CString::new(output_str).unwrap().into_raw()
}

/// Frees a string that was allocated by this library.
///
/// # Safety
/// - `ptr` must be a pointer returned by a function from this library.
#[no_mangle]
pub extern "C" fn edgegpt_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    unsafe {
        let _ = CString::from_raw(ptr);
    }
}