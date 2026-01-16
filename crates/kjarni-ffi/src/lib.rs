//! Kjarni FFI - C bindings for Kjarni ML library
//!
//! This crate provides C-compatible bindings for use from C, C++, C#, Go, Python, etc.
mod callback;
mod error;
mod embedder;
mod classifier;
mod reranker;
mod indexer;
mod searcher;


pub use callback::*;
pub use error::*;
pub use embedder::*;
pub use classifier::*;
pub use reranker::*;
pub use indexer::*;
pub use searcher::*;

use std::sync::OnceLock;
use tokio::runtime::Runtime;

/// Global tokio runtime for async operations
static RUNTIME: OnceLock<Runtime> = OnceLock::new();

/// Get or initialize the global runtime
pub(crate) fn get_runtime() -> &'static Runtime {
    RUNTIME.get_or_init(|| {
        Runtime::new().expect("Failed to create tokio runtime")
    })
}

// =============================================================================
// Global Functions
// =============================================================================

/// Initialize the Kjarni runtime. Optional - auto-initialized on first use.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_init() -> KjarniErrorCode {
    // 1. Initialize Rayon Global Thread Pool (Compute)
    // We explicitly set this to physical cores to maximize AVX/SIMD throughput.
    // This fixes the issue where Rayon defaults to 1 thread when loaded inside Python.
    let physical_cores = num_cpus::get_physical();
    
    // build_global() returns an Err if the pool is already initialized.
    // We intentionally ignore the error because it means we are good to go.
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(physical_cores)
        .build_global();

    // 2. Initialize Tokio Runtime (Async/IO)
    let _ = get_runtime();

    KjarniErrorCode::Ok
}
/// Shutdown and cleanup. Call before process exit.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_shutdown() {
    // Runtime will be dropped when process exits
}

/// Get the library version string.
#[unsafe(no_mangle)]
pub extern "C" fn kjarni_version() -> *const std::ffi::c_char {
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as *const std::ffi::c_char
}

// =============================================================================
// Memory Management - Common Types
// =============================================================================

/// Float array returned by FFI functions. Caller must free with kjarni_float_array_free.
#[repr(C)]
pub struct KjarniFloatArray {
    pub data: *mut f32,
    pub len: usize,
}

impl KjarniFloatArray {
    pub fn from_vec(v: Vec<f32>) -> Self {
        let len = v.len();
        let boxed = v.into_boxed_slice();
        let data = Box::into_raw(boxed) as *mut f32;
        Self { data, len }
    }
    
    pub fn empty() -> Self {
        Self { data: std::ptr::null_mut(), len: 0 }
    }
}

/// 2D float array for batch results. Caller must free with kjarni_float_2d_array_free.
#[repr(C)]
pub struct KjarniFloat2DArray {
    pub data: *mut f32,
    pub rows: usize,
    pub cols: usize,
}

impl KjarniFloat2DArray {
    pub fn from_vecs(vecs: Vec<Vec<f32>>) -> Self {
        if vecs.is_empty() {
            return Self { data: std::ptr::null_mut(), rows: 0, cols: 0 };
        }
        let rows = vecs.len();
        let cols = vecs[0].len();
        let flat: Vec<f32> = vecs.into_iter().flatten().collect();
        let boxed = flat.into_boxed_slice();
        let data = Box::into_raw(boxed) as *mut f32;
        Self { data, rows, cols }
    }

    pub fn from_flat(v: Vec<f32>, rows: usize, cols: usize) -> Self {
        // Ensure the vector length matches dimensions
        assert_eq!(v.len(), rows * cols, "Vector length must match rows*cols");

        // Convert Vec to Boxed Slice to prevent deallocation
        let mut boxed_slice = v.into_boxed_slice();
        
        // Get the raw pointer
        let data = boxed_slice.as_mut_ptr();
        
        // Forget the box so Rust doesn't free the memory at end of scope
        // Ownership is now passed to C (and eventually Python)
        std::mem::forget(boxed_slice);

        Self { data, rows, cols }
    }
    
    pub fn empty() -> Self {
        Self { data: std::ptr::null_mut(), rows: 0, cols: 0 }
    }
}

/// String array. Caller must free with kjarni_string_array_free.
#[repr(C)]
pub struct KjarniStringArray {
    pub strings: *mut *mut std::ffi::c_char,
    pub len: usize,
}

/// Free a float array allocated by Kjarni.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_float_array_free(arr: KjarniFloatArray) {
    if !arr.data.is_null() && arr.len > 0 {
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(arr.data, arr.len));
    }
}

/// Free a 2D float array allocated by Kjarni.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_float_2d_array_free(arr: KjarniFloat2DArray) {
    if !arr.data.is_null() && arr.rows > 0 && arr.cols > 0 {
        let total = arr.rows * arr.cols;
        let _ = Box::from_raw(std::slice::from_raw_parts_mut(arr.data, total));
    }
}

/// Free a string allocated by Kjarni.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_string_free(s: *mut std::ffi::c_char) {
    if !s.is_null() {
        let _ = std::ffi::CString::from_raw(s);
    }
}

/// Free a string array allocated by Kjarni.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn kjarni_string_array_free(arr: KjarniStringArray) {
    if !arr.strings.is_null() && arr.len > 0 {
        let strings = std::slice::from_raw_parts_mut(arr.strings, arr.len);
        for s in strings.iter() {
            if !s.is_null() {
                let _ = std::ffi::CString::from_raw(*s);
            }
        }
        let _ = Box::from_raw(strings.as_mut_ptr());
    }
}