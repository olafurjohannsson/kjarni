//! Error handling for Kjarni FFI

use std::cell::RefCell;
use std::ffi::CString;
use std::ffi::c_char;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Error codes returned by Kjarni FFI functions.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniErrorCode {
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
pub fn map_result<T, E: std::fmt::Display>(result: Result<T, E>, err_code: KjarniErrorCode) -> Result<T, KjarniErrorCode> {
    result.map_err(|e| {
        set_last_error(e.to_string());
        err_code
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn kjarni_error_code_to_string(
    err: KjarniErrorCode
) -> *const c_char {
    kjarni_error_name(err)
}

/// Get the name of an error code as a C string.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_error_name(err: KjarniErrorCode) -> *const std::ffi::c_char {
    let name: &'static [u8] = match err {
        KjarniErrorCode::Ok => b"KJARNI_OK\0",
        KjarniErrorCode::NullPointer => b"KJARNI_ERROR_NULL_POINTER\0",
        KjarniErrorCode::InvalidUtf8 => b"KJARNI_ERROR_INVALID_UTF8\0",
        KjarniErrorCode::ModelNotFound => b"KJARNI_ERROR_MODEL_NOT_FOUND\0",
        KjarniErrorCode::LoadFailed => b"KJARNI_ERROR_LOAD_FAILED\0",
        KjarniErrorCode::InferenceFailed => b"KJARNI_ERROR_INFERENCE_FAILED\0",
        KjarniErrorCode::GpuUnavailable => b"KJARNI_ERROR_GPU_UNAVAILABLE\0",
        KjarniErrorCode::InvalidConfig => b"KJARNI_ERROR_INVALID_CONFIG\0",
        KjarniErrorCode::Cancelled => b"KJARNI_ERROR_CANCELLED\0",
        KjarniErrorCode::Timeout => b"KJARNI_ERROR_TIMEOUT\0",
        KjarniErrorCode::StreamEnded => b"KJARNI_ERROR_STREAM_ENDED\0",
        KjarniErrorCode::Unknown => b"KJARNI_ERROR_UNKNOWN\0",
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

#[cfg(test)]
mod ff_error_tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn test_error_code_values() {
        assert_eq!(KjarniErrorCode::Ok as i32, 0);
        assert_eq!(KjarniErrorCode::NullPointer as i32, 1);
        assert_eq!(KjarniErrorCode::InvalidUtf8 as i32, 2);
        assert_eq!(KjarniErrorCode::ModelNotFound as i32, 3);
        assert_eq!(KjarniErrorCode::LoadFailed as i32, 4);
        assert_eq!(KjarniErrorCode::InferenceFailed as i32, 5);
        assert_eq!(KjarniErrorCode::GpuUnavailable as i32, 6);
        assert_eq!(KjarniErrorCode::InvalidConfig as i32, 7);
        assert_eq!(KjarniErrorCode::Cancelled as i32, 8);
        assert_eq!(KjarniErrorCode::Timeout as i32, 9);
        assert_eq!(KjarniErrorCode::StreamEnded as i32, 10);
        assert_eq!(KjarniErrorCode::Unknown as i32, 255);
    }

    #[test]
    fn test_error_code_repr_c_size() {
        assert!(std::mem::size_of::<KjarniErrorCode>() >= std::mem::size_of::<i32>());
    }

    #[test]
    fn test_error_code_copy_clone() {
        let err = KjarniErrorCode::InferenceFailed;
        let copied = err;
        let cloned = err.clone();
        
        assert_eq!(err, copied);
        assert_eq!(err, cloned);
    }

    #[test]
    fn test_error_code_debug() {
        let debug_str = format!("{:?}", KjarniErrorCode::ModelNotFound);
        assert!(debug_str.contains("ModelNotFound"));
    }

    #[test]
    fn test_error_code_equality() {
        assert_eq!(KjarniErrorCode::Ok, KjarniErrorCode::Ok);
        assert_ne!(KjarniErrorCode::Ok, KjarniErrorCode::Unknown);
    }

    #[test]
    fn test_error_name_all_variants() {
        let cases = [
            (KjarniErrorCode::Ok, "KJARNI_OK"),
            (KjarniErrorCode::NullPointer, "KJARNI_ERROR_NULL_POINTER"),
            (KjarniErrorCode::InvalidUtf8, "KJARNI_ERROR_INVALID_UTF8"),
            (KjarniErrorCode::ModelNotFound, "KJARNI_ERROR_MODEL_NOT_FOUND"),
            (KjarniErrorCode::LoadFailed, "KJARNI_ERROR_LOAD_FAILED"),
            (KjarniErrorCode::InferenceFailed, "KJARNI_ERROR_INFERENCE_FAILED"),
            (KjarniErrorCode::GpuUnavailable, "KJARNI_ERROR_GPU_UNAVAILABLE"),
            (KjarniErrorCode::InvalidConfig, "KJARNI_ERROR_INVALID_CONFIG"),
            (KjarniErrorCode::Cancelled, "KJARNI_ERROR_CANCELLED"),
            (KjarniErrorCode::Timeout, "KJARNI_ERROR_TIMEOUT"),
            (KjarniErrorCode::StreamEnded, "KJARNI_ERROR_STREAM_ENDED"),
            (KjarniErrorCode::Unknown, "KJARNI_ERROR_UNKNOWN"),
        ];

        for (code, expected_name) in cases {
            let ptr = kjarni_error_name(code);
            assert!(!ptr.is_null(), "Error name for {:?} is null", code);
            
            let name = unsafe { CStr::from_ptr(ptr) };
            assert_eq!(
                name.to_str().unwrap(),
                expected_name,
                "Mismatch for {:?}",
                code
            );
        }
    }

    #[test]
    fn test_error_name_returns_static_pointer() {
        let ptr1 = kjarni_error_name(KjarniErrorCode::Ok);
        let ptr2 = kjarni_error_name(KjarniErrorCode::Ok);
        let ptr3 = kjarni_error_name(KjarniErrorCode::Ok);
        
        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr2, ptr3);
    }

    #[test]
    fn test_error_code_to_string_same_as_error_name() {
        for code in [
            KjarniErrorCode::Ok,
            KjarniErrorCode::NullPointer,
            KjarniErrorCode::InferenceFailed,
            KjarniErrorCode::Unknown,
        ] {
            let name_ptr = kjarni_error_name(code);
            let str_ptr = kjarni_error_code_to_string(code);
            assert_eq!(name_ptr, str_ptr);
        }
    }

    #[test]
    fn test_set_last_error_basic() {
        kjarni_clear_error();
        
        set_last_error("Something went wrong");
        
        let msg_ptr = kjarni_last_error_message();
        assert!(!msg_ptr.is_null());
        
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Something went wrong");
    }

    #[test]
    fn test_set_last_error_overwrites_previous() {
        kjarni_clear_error();
        
        set_last_error("First error");
        set_last_error("Second error");
        set_last_error("Third error");
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Third error");
    }

    #[test]
    fn test_set_last_error_from_string() {
        kjarni_clear_error();
        
        let owned = String::from("Owned string error");
        set_last_error(owned);
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Owned string error");
    }

    #[test]
    fn test_set_last_error_from_str() {
        kjarni_clear_error();
        
        set_last_error("Static str error");
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Static str error");
    }

    #[test]
    fn test_set_last_error_empty_string() {
        kjarni_clear_error();
        
        set_last_error("");
        
        let msg_ptr = kjarni_last_error_message();
        assert!(!msg_ptr.is_null());
        
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "");
    }

    #[test]
    fn test_set_last_error_unicode() {
        kjarni_clear_error();
        
        set_last_error("Unicode: こんにちは 🦀 émoji");
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Unicode: こんにちは 🦀 émoji");
    }

    #[test]
    fn test_set_last_error_with_null_byte() {
        kjarni_clear_error();
        
        set_last_error("Before\0After");
        let _ = kjarni_last_error_message();
    }

    #[test]
    fn test_set_last_error_long_message() {
        kjarni_clear_error();
        
        let long_msg = "A".repeat(10_000);
        set_last_error(long_msg.clone());
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), long_msg);
    }

    #[test]
    fn test_last_error_message_null_when_no_error() {
        kjarni_clear_error();
        
        let msg_ptr = kjarni_last_error_message();
        assert!(msg_ptr.is_null());
    }

    #[test]
    fn test_last_error_message_stable_pointer() {
        kjarni_clear_error();
        set_last_error("Test error");
        let ptr1 = kjarni_last_error_message();
        let ptr2 = kjarni_last_error_message();
        let ptr3 = kjarni_last_error_message();
        
        assert_eq!(ptr1, ptr2);
        assert_eq!(ptr2, ptr3);
    }
    #[test]
    fn test_clear_error_clears_message() {
        set_last_error("Some error");
        assert!(!kjarni_last_error_message().is_null());
        
        kjarni_clear_error();
        assert!(kjarni_last_error_message().is_null());
    }

    #[test]
    fn test_clear_error_idempotent() {
        kjarni_clear_error();
        kjarni_clear_error();
        kjarni_clear_error();
        
        assert!(kjarni_last_error_message().is_null());
    }

    #[test]
    fn test_clear_error_when_no_error() {
        LAST_ERROR.with(|e| {
            *e.borrow_mut() = None;
        });
        
        kjarni_clear_error();
        assert!(kjarni_last_error_message().is_null());
    }
    #[test]
    fn test_map_result_ok_passes_through() {
        kjarni_clear_error();
        
        let input: Result<i32, &str> = Ok(42);
        let result = map_result(input, KjarniErrorCode::Unknown);
        
        assert_eq!(result, Ok(42));
        assert!(kjarni_last_error_message().is_null()); 
    }

    #[test]
    fn test_map_result_err_sets_message() {
        kjarni_clear_error();
        
        let input: Result<i32, &str> = Err("Something failed");
        let result = map_result(input, KjarniErrorCode::InferenceFailed);
        
        assert_eq!(result, Err(KjarniErrorCode::InferenceFailed));
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Something failed");
    }

    #[test]
    fn test_map_result_with_custom_error_type() {
        kjarni_clear_error();
        
        #[derive(Debug)]
        struct CustomError(String);
        
        impl std::fmt::Display for CustomError {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "Custom: {}", self.0)
            }
        }
        
        let input: Result<(), CustomError> = Err(CustomError("details".into()));
        let result = map_result(input, KjarniErrorCode::LoadFailed);
        
        assert_eq!(result, Err(KjarniErrorCode::LoadFailed));
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Custom: details");
    }

    #[test]
    fn test_map_result_with_io_error() {
        kjarni_clear_error();
        
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let input: Result<(), std::io::Error> = Err(io_err);
        let result = map_result(input, KjarniErrorCode::ModelNotFound);
        
        assert_eq!(result, Err(KjarniErrorCode::ModelNotFound));
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert!(msg.to_str().unwrap().contains("file not found"));
    }

    #[test]
    fn test_map_result_preserves_value_type() {
        kjarni_clear_error();
        
        let input: Result<Vec<f32>, &str> = Ok(vec![1.0, 2.0, 3.0]);
        let result = map_result(input, KjarniErrorCode::Unknown);
        
        assert_eq!(result.unwrap(), vec![1.0, 2.0, 3.0]);
    }
    #[test]
    fn test_error_thread_local_isolation() {
        use std::thread;
        
        kjarni_clear_error();
        set_last_error("Main thread error");
        
        let handle = thread::spawn(|| {
            // New thread should have no error
            assert!(kjarni_last_error_message().is_null());
            
            set_last_error("Child thread error");
            
            let msg_ptr = kjarni_last_error_message();
            let msg = unsafe { CStr::from_ptr(msg_ptr) };
            assert_eq!(msg.to_str().unwrap(), "Child thread error");
        });
        
        handle.join().unwrap();
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Main thread error");
    }

    #[test]
    fn test_error_multiple_threads_independent() {
        use std::thread;
        use std::sync::Barrier;
        use std::sync::Arc;
        
        let barrier = Arc::new(Barrier::new(4));
        
        let handles: Vec<_> = (0..4)
            .map(|i| {
                let b = Arc::clone(&barrier);
                thread::spawn(move || {
                    kjarni_clear_error();
                    
                    // All threads set error at roughly the same time
                    b.wait();
                    set_last_error(format!("Error from thread {}", i));
                    b.wait();
                    
                    // Each thread should see only its own error
                    let msg_ptr = kjarni_last_error_message();
                    let msg = unsafe { CStr::from_ptr(msg_ptr) };
                    let expected = format!("Error from thread {}", i);
                    assert_eq!(msg.to_str().unwrap(), expected);
                })
            })
            .collect();
        
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_clear_error_thread_local() {
        use std::thread;
        
        set_last_error("Main error");
        
        let handle = thread::spawn(|| {
            set_last_error("Child error");
            kjarni_clear_error();
            assert!(kjarni_last_error_message().is_null());
        });
        
        handle.join().unwrap();
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Main error");
    }

    #[test]
    fn test_ffi_error_handling_pattern() {
        kjarni_clear_error();
        
        let result: Result<(), &str> = Err("Model file corrupt");
        let code = match map_result(result, KjarniErrorCode::LoadFailed) {
            Ok(_) => KjarniErrorCode::Ok,
            Err(e) => e,
        };
        
        assert_ne!(code, KjarniErrorCode::Ok);
        let name_ptr = kjarni_error_name(code);
        let name = unsafe { CStr::from_ptr(name_ptr) };
        assert_eq!(name.to_str().unwrap(), "KJARNI_ERROR_LOAD_FAILED");
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Model file corrupt");
        
        kjarni_clear_error();
        assert!(kjarni_last_error_message().is_null());
    }

    #[test]
    fn test_ffi_success_pattern() {
        kjarni_clear_error();
        
        let result: Result<i32, &str> = Ok(100);
        let code = match map_result(result, KjarniErrorCode::Unknown) {
            Ok(_) => KjarniErrorCode::Ok,
            Err(e) => e,
        };
        
        assert_eq!(code, KjarniErrorCode::Ok);
        assert!(kjarni_last_error_message().is_null());
    }

    #[test]
    fn test_ffi_sequential_errors() {
        kjarni_clear_error();
        
        // First error
        let _ = map_result::<(), _>(Err("First"), KjarniErrorCode::NullPointer);
        let msg1 = unsafe { CStr::from_ptr(kjarni_last_error_message()) };
        assert_eq!(msg1.to_str().unwrap(), "First");
        
        // Second error overwrites without explicit clear
        let _ = map_result::<(), _>(Err("Second"), KjarniErrorCode::InvalidUtf8);
        let msg2 = unsafe { CStr::from_ptr(kjarni_last_error_message()) };
        assert_eq!(msg2.to_str().unwrap(), "Second");
    }
    #[test]
    fn test_error_name_all_codes_exhaustive() {
        let all_codes = [
            KjarniErrorCode::Ok,
            KjarniErrorCode::NullPointer,
            KjarniErrorCode::InvalidUtf8,
            KjarniErrorCode::ModelNotFound,
            KjarniErrorCode::LoadFailed,
            KjarniErrorCode::InferenceFailed,
            KjarniErrorCode::GpuUnavailable,
            KjarniErrorCode::InvalidConfig,
            KjarniErrorCode::Cancelled,
            KjarniErrorCode::Timeout,
            KjarniErrorCode::StreamEnded,
            KjarniErrorCode::Unknown,
        ];
        
        for code in all_codes {
            let ptr = kjarni_error_name(code);
            assert!(!ptr.is_null());
            
            let s = unsafe { CStr::from_ptr(ptr) };
            assert!(s.to_str().is_ok());
            assert!(!s.to_str().unwrap().is_empty());
        }
    }

    #[test]
    fn test_error_names_are_valid_c_identifiers() {
        let all_codes = [
            KjarniErrorCode::Ok,
            KjarniErrorCode::NullPointer,
            KjarniErrorCode::InvalidUtf8,
            KjarniErrorCode::ModelNotFound,
            KjarniErrorCode::LoadFailed,
            KjarniErrorCode::InferenceFailed,
            KjarniErrorCode::GpuUnavailable,
            KjarniErrorCode::InvalidConfig,
            KjarniErrorCode::Cancelled,
            KjarniErrorCode::Timeout,
            KjarniErrorCode::StreamEnded,
            KjarniErrorCode::Unknown,
        ];
        
        for code in all_codes {
            let ptr = kjarni_error_name(code);
            let name = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
            
            assert!(name.starts_with("KJARNI_"), "Name '{}' doesn't start with KJARNI_", name);
            
            assert!(
                name.chars().all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'),
                "Name '{}' contains invalid characters",
                name
            );
        }
    }

    #[test]
    fn test_set_error_after_clear_works() {
        set_last_error("Error 1");
        kjarni_clear_error();
        set_last_error("Error 2");
        
        let msg_ptr = kjarni_last_error_message();
        let msg = unsafe { CStr::from_ptr(msg_ptr) };
        assert_eq!(msg.to_str().unwrap(), "Error 2");
    }

    #[test]
    fn test_rapid_set_clear_cycles() {
        for i in 0..1000 {
            set_last_error(format!("Error {}", i));
            
            let msg_ptr = kjarni_last_error_message();
            let msg = unsafe { CStr::from_ptr(msg_ptr) };
            assert_eq!(msg.to_str().unwrap(), format!("Error {}", i));
            
            kjarni_clear_error();
            assert!(kjarni_last_error_message().is_null());
        }
    }

    #[test]
    fn test_error_message_pointer_validity_after_new_error() {
        set_last_error("First error message");
        let ptr1 = kjarni_last_error_message();
        
        let msg1 = unsafe { CStr::from_ptr(ptr1) }.to_str().unwrap().to_owned();
        assert_eq!(msg1, "First error message");
        
        set_last_error("Second error message");
        let ptr2 = kjarni_last_error_message();
        
        let msg2 = unsafe { CStr::from_ptr(ptr2) }.to_str().unwrap();
        assert_eq!(msg2, "Second error message");
    }

    #[test]
    fn test_error_message_pointer_validity_after_clear() {
        set_last_error("Error message");
        let ptr = kjarni_last_error_message();
        
        // Read before clear
        let msg = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap().to_owned();
        assert_eq!(msg, "Error message");
        kjarni_clear_error();
        assert!(kjarni_last_error_message().is_null());
    }
}