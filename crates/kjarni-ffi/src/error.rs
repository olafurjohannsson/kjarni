//! Error handling for Kjarni FFI

use std::cell::RefCell;
use std::ffi::CString;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Error codes returned by Kjarni FFI functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniError {
    /// Operation completed successfully
    Ok = 0,
    /// Null pointer passed to function
    NullPointer = 1,
    /// Invalid UTF-8 string
    InvalidUtf8 = 2,
    /// Model not found in registry
    ModelNotFound = 3,
    /// Failed to load model
    LoadFailed = 4,
    /// Inference operation failed
    InferenceFailed = 5,
    /// GPU not available
    GpuUnavailable = 6,
    /// Invalid configuration
    InvalidConfig = 7,
    /// Operation was cancelled
    Cancelled = 8,
    /// Operation timed out
    Timeout = 9,
    /// Stream has ended
    StreamEnded = 10,
    /// Unknown error
    Unknown = 255,
}

/// Set the last error message (thread-local).
pub fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = CString::new(msg).ok();
    });
}

/// Map a Result to KjarniError, setting the error message if Err.
pub fn map_result<T, E: std::fmt::Display>(result: Result<T, E>, err_code: KjarniError) -> Result<T, KjarniError> {
    result.map_err(|e| {
        set_last_error(e.to_string());
        err_code
    })
}

/// Get the name of an error code as a C string.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_error_name(err: KjarniError) -> *const std::ffi::c_char {
    let name: &'static [u8] = match err {
        KjarniError::Ok => b"KJARNI_OK\0",
        KjarniError::NullPointer => b"KJARNI_ERROR_NULL_POINTER\0",
        KjarniError::InvalidUtf8 => b"KJARNI_ERROR_INVALID_UTF8\0",
        KjarniError::ModelNotFound => b"KJARNI_ERROR_MODEL_NOT_FOUND\0",
        KjarniError::LoadFailed => b"KJARNI_ERROR_LOAD_FAILED\0",
        KjarniError::InferenceFailed => b"KJARNI_ERROR_INFERENCE_FAILED\0",
        KjarniError::GpuUnavailable => b"KJARNI_ERROR_GPU_UNAVAILABLE\0",
        KjarniError::InvalidConfig => b"KJARNI_ERROR_INVALID_CONFIG\0",
        KjarniError::Cancelled => b"KJARNI_ERROR_CANCELLED\0",
        KjarniError::Timeout => b"KJARNI_ERROR_TIMEOUT\0",
        KjarniError::StreamEnded => b"KJARNI_ERROR_STREAM_ENDED\0",
        KjarniError::Unknown => b"KJARNI_ERROR_UNKNOWN\0",
    };
    name.as_ptr() as *const std::ffi::c_char
}

/// Get the last error message. Returns NULL if no error.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_last_error_message() -> *const std::ffi::c_char {
    LAST_ERROR.with(|e| {
        e.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(std::ptr::null())
    })
}

/// Clear the last error.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_clear_error() {
    LAST_ERROR.with(|e| {
        *e.borrow_mut() = None;
    });
}