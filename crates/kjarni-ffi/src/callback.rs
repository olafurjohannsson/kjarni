//! FFI callback infrastructure for progress reporting and streaming

use std::ffi::{CString, c_char, c_void};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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

/// Progress data passed to callbacks
#[repr(C)]
#[derive(Debug, Clone)]
pub struct KjarniProgress {
    pub stage: KjarniProgressStage,
    pub current: usize,
    pub total: usize,
    pub message: *const c_char,
}

/// Generic callback function pointer type for progress
pub type KjarniProgressCallbackFn =
    Option<extern "C" fn(progress: KjarniProgress, user_data: *mut c_void)>;

/// Token callback for streaming (future use)
#[repr(C)]
pub struct KjarniToken {
    pub text: *const c_char,
    pub token_id: u32,
    pub is_special: bool,
}

/// Generic callback for token streaming
pub type KjarniTokenCallbackFn =
    Option<extern "C" fn(token: KjarniToken, user_data: *mut c_void) -> bool>;

/// Cancellation token
pub struct KjarniCancelToken {
    pub(crate) inner: Arc<AtomicBool>,
}

#[unsafe(no_mangle)]
pub extern "C" fn kjarni_cancel_token_new() -> *mut KjarniCancelToken {
    Box::into_raw(Box::new(KjarniCancelToken {
        inner: Arc::new(AtomicBool::new(false)),
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_cancel_token_cancel(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        unsafe {
            (*token).inner.store(true, Ordering::SeqCst);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_cancel_token_is_cancelled(token: *const KjarniCancelToken) -> bool {
    if token.is_null() {
        false
    } else {
        unsafe { (*token).inner.load(Ordering::SeqCst) }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_cancel_token_reset(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        unsafe {
            (*token).inner.store(false, Ordering::SeqCst);
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_cancel_token_free(token: *mut KjarniCancelToken) {
    if !token.is_null() {
        unsafe {
            let _ = Box::from_raw(token);
        }
    }
}

/// Check if cancelled (internal helper)
pub fn is_cancelled(token: *const KjarniCancelToken) -> bool {
    if token.is_null() {
        false
    } else {
        unsafe { (*token).inner.load(Ordering::SeqCst) }
    }
}

pub struct FfiCallback<T> {
    callback: Option<extern "C" fn(T, *mut c_void)>,
    user_data: *mut c_void,
}

unsafe impl<T> Send for FfiCallback<T> {}
unsafe impl<T> Sync for FfiCallback<T> {}

impl<T> FfiCallback<T> {
    pub fn new(callback: Option<extern "C" fn(T, *mut c_void)>, user_data: *mut c_void) -> Self {
        Self {
            callback,
            user_data,
        }
    }

    pub fn is_some(&self) -> bool {
        self.callback.is_some()
    }

    pub fn call(&self, value: T) {
        if let Some(cb) = self.callback {
            cb(value, self.user_data);
        }
    }
}

/// Progress callback wrapper with message buffer
pub struct ProgressCallbackWrapper {
    inner: FfiCallback<KjarniProgress>,
    message_buf: std::cell::UnsafeCell<Option<CString>>,
}

unsafe impl Send for ProgressCallbackWrapper {}
unsafe impl Sync for ProgressCallbackWrapper {}

impl ProgressCallbackWrapper {
    pub fn new(callback: KjarniProgressCallbackFn, user_data: *mut c_void) -> Option<Self> {
        if callback.is_some() {
            Some(Self {
                inner: FfiCallback::new(callback, user_data),
                message_buf: std::cell::UnsafeCell::new(None),
            })
        } else {
            None
        }
    }

    pub fn report(
        &self,
        stage: KjarniProgressStage,
        current: usize,
        total: usize,
        message: Option<&str>,
    ) {
        let message_ptr = if let Some(msg) = message {
            if let Ok(cstr) = CString::new(msg) {
                unsafe {
                    *self.message_buf.get() = Some(cstr);
                    (*self.message_buf.get())
                        .as_ref()
                        .map(|c| c.as_ptr())
                        .unwrap_or(std::ptr::null())
                }
            } else {
                std::ptr::null()
            }
        } else {
            std::ptr::null()
        };

        self.inner.call(KjarniProgress {
            stage,
            current,
            total,
            message: message_ptr,
        });
    }
}
