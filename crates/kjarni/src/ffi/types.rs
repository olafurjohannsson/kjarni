//! FFI-safe type conversions

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

/// Convert a Rust String to a C string (caller must free)
pub fn string_to_c(s: String) -> *mut c_char {
    CString::new(s).unwrap().into_raw()
}

/// Convert a C string to a Rust String (does not take ownership)
pub unsafe fn c_to_string(s: *const c_char) -> String {
    CStr::from_ptr(s).to_string_lossy().into_owned()
}

/// Free a C string created by string_to_c
pub unsafe fn free_c_string(s: *mut c_char) {
    if !s.is_null() {
        let _ = CString::from_raw(s);
    }
}

/// FFI-safe float array
#[repr(C)]
pub struct FloatArray {
    pub data: *mut f32,
    pub len: usize,
}

impl FloatArray {
    pub fn from_vec(v: Vec<f32>) -> Self {
        let len = v.len();
        let data = Box::into_raw(v.into_boxed_slice()) as *mut f32;
        Self { data, len }
    }

    pub unsafe fn to_vec(&self) -> Vec<f32> {
        std::slice::from_raw_parts(self.data, self.len).to_vec()
    }

    pub unsafe fn free(self) {
        if !self.data.is_null() {
            let _ = Box::from_raw(std::slice::from_raw_parts_mut(self.data, self.len));
        }
    }
}