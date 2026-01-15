//! FFI callback infrastructure for progress reporting and streaming

use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Progress stage enum for FFI
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KjarniProgressStage {
    Scanning = 0,
    Loading = 1,
    Embedding = 2,
    Writing = 3,
    Committing = 4,
    Searching = 5,
    Reranking = 6,
}

impl From<kjarni_rag::ProgressStage> for KjarniProgressStage {
    fn from(stage: kjarni_rag::ProgressStage) -> Self {
        match stage {
            kjarni_rag::ProgressStage::Scanning => KjarniProgressStage::Scanning,
            kjarni_rag::ProgressStage::Loading => KjarniProgressStage::Loading,
            kjarni_rag::ProgressStage::Embedding => KjarniProgressStage::Embedding,
            kjarni_rag::ProgressStage::Writing => KjarniProgressStage::Writing,
            kjarni_rag::ProgressStage::Committing => KjarniProgressStage::Committing,
            kjarni_rag::ProgressStage::Searching => KjarniProgressStage::Searching,
            kjarni_rag::ProgressStage::Reranking => KjarniProgressStage::Reranking,
        }
    }
}

/// Progress data passed to callbacks
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KjarniProgress {
    pub stage: KjarniProgressStage,
    pub current: usize,
    pub total: usize,
    /// Message string (may be NULL)
    pub message: *const c_char,
}

/// Progress callback function pointer type
/// 
/// Arguments:
/// - progress: Progress data
/// - user_data: User-provided context pointer
pub type KjarniProgressCallbackFn = Option<extern "C" fn(progress: KjarniProgress, user_data: *mut c_void)>;

/// Cancellation token handle
pub struct KjarniCancelToken {
    pub(crate) inner: Arc<AtomicBool>,
}

/// Create a new cancellation token
#[no_mangle]
pub extern "C" fn kjarni_cancel_token_new() -> *mut KjarniCancelToken {
    let token = KjarniCancelToken {
        inner: Arc::new(AtomicBool::new(false)),
    };
    Box::into_raw(Box::new(token))
}

/// Cancel the operation associated with this token
#[no_mangle]
pub unsafe extern "C" fn kjarni_cancel_token_cancel(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        (*token).inner.store(true, Ordering::SeqCst);
    }
}

/// Check if token is cancelled
#[no_mangle]
pub unsafe extern "C" fn kjarni_cancel_token_is_cancelled(token: *const KjarniCancelToken) -> bool {
    if token.is_null() {
        return false;
    }
    (*token).inner.load(Ordering::SeqCst)
}

/// Reset the cancellation token
#[no_mangle]
pub unsafe extern "C" fn kjarni_cancel_token_reset(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        (*token).inner.store(false, Ordering::SeqCst);
    }
}

/// Free a cancellation token
#[no_mangle]
pub unsafe extern "C" fn kjarni_cancel_token_free(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        let _ = Box::from_raw(token);
    }
}

/// Helper to invoke FFI callback from Rust
pub(crate) struct FfiProgressCallback {
    callback: KjarniProgressCallbackFn,
    user_data: *mut c_void,
    message_buffer: std::cell::RefCell<std::ffi::CString>,
}

// SAFETY: We're careful about the raw pointer usage
unsafe impl Send for FfiProgressCallback {}
unsafe impl Sync for FfiProgressCallback {}

impl FfiProgressCallback {
    pub fn new(callback: KjarniProgressCallbackFn, user_data: *mut c_void) -> Option<Self> {
        callback.map(|_| Self {
            callback,
            user_data,
            message_buffer: std::cell::RefCell::new(std::ffi::CString::default()),
        })
    }
    
    pub fn report(&self, progress: &kjarni_rag::Progress, message: Option<&str>) {
        if let Some(cb) = self.callback {
            let message_ptr = if let Some(msg) = message {
                if let Ok(cstr) = std::ffi::CString::new(msg) {
                    *self.message_buffer.borrow_mut() = cstr;
                    self.message_buffer.borrow().as_ptr()
                } else {
                    std::ptr::null()
                }
            } else {
                std::ptr::null()
            };
            
            let ffi_progress = KjarniProgress {
                stage: progress.stage.into(),
                current: progress.current,
                total: progress.total,
                message: message_ptr,
            };
            
            cb(ffi_progress, self.user_data);
        }
    }
}

/// Check if a cancel token is cancelled (helper for internal use)
pub(crate) fn is_cancelled(token: *const KjarniCancelToken) -> bool {
    if token.is_null() {
        return false;
    }
    unsafe { (*token).inner.load(Ordering::SeqCst) }
}