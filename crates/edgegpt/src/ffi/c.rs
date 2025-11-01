//! C FFI for EdgeGPT
//!
//! This module provides C-compatible bindings that can be used from C, C++, and other languages.

use crate::edge_gpt::EdgeGPT;
use crate::ffi::types::{FloatArray, string_to_c, c_to_string, free_c_string};
use edgetransformers::prelude::Device;
use std::ffi::c_char;
use std::ptr;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Opaque handle to an EdgeGPT instance
#[repr(C)]
pub struct EdgeGPTHandle {
    _private: [u8; 0],
}

/// Error codes
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum EdgeGPTError {
    Success = 0,
    NullPointer = 1,
    InvalidUtf8 = 2,
    ModelLoadError = 3,
    InferenceError = 4,
    RuntimeError = 5,
}

struct EdgeGPTContext {
    edge_gpt: EdgeGPT,
    runtime: Runtime,
}

/// Create a new EdgeGPT instance for CPU
///
/// # Safety
/// The returned handle must be freed with `edge_gpt_free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_new_cpu() -> *mut EdgeGPTHandle {
    let runtime = match Runtime::new() {
        Ok(rt) => rt,
        Err(_) => return ptr::null_mut(),
    };
    
    let edge_gpt = EdgeGPT::new(Device::Cpu, None);
    
    let ctx = Box::new(EdgeGPTContext { edge_gpt, runtime });
    Box::into_raw(ctx) as *mut EdgeGPTHandle
}

/// Free an EdgeGPT instance
///
/// # Safety
/// The handle must be valid and not used after this call
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_free(handle: *mut EdgeGPTHandle) {
    if !handle.is_null() {
        let _ = Box::from_raw(handle as *mut EdgeGPTContext);
    }
}

/// Encode a single text
///
/// # Safety
/// - `handle` must be valid
/// - `text` must be a valid null-terminated UTF-8 string
/// - `out_embedding` will be allocated and must be freed with `edge_gpt_free_float_array()`
/// - `out_len` will be set to the embedding dimension
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_encode(
    handle: *mut EdgeGPTHandle,
    text: *const c_char,
    out_embedding: *mut *mut f32,
    out_len: *mut usize,
) -> EdgeGPTError {
    if handle.is_null() || text.is_null() || out_embedding.is_null() || out_len.is_null() {
        return EdgeGPTError::NullPointer;
    }
    
    let ctx = &*(handle as *const EdgeGPTContext);
    let text_str = match std::panic::catch_unwind(|| c_to_string(text)) {
        Ok(s) => s,
        Err(_) => return EdgeGPTError::InvalidUtf8,
    };
    
    let result = ctx.runtime.block_on(async {
        ctx.edge_gpt.encode(&text_str).await
    });
    
    match result {
        Ok(embedding) => {
            let len = embedding.len();
            let array = FloatArray::from_vec(embedding);
            *out_embedding = array.data;
            *out_len = len;
            EdgeGPTError::Success
        }
        Err(_) => EdgeGPTError::InferenceError,
    }
}

/// Encode a batch of texts
///
/// # Safety
/// - `handle` must be valid
/// - `texts` must be an array of `num_texts` valid null-terminated UTF-8 strings
/// - `out_embeddings` will be allocated and must be freed with `edge_gpt_free_batch_embeddings()`
/// - `out_lens` will be allocated and must be freed with `free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_encode_batch(
    handle: *mut EdgeGPTHandle,
    texts: *const *const c_char,
    num_texts: usize,
    out_embeddings: *mut *mut *mut f32,
    out_lens: *mut *mut usize,
    embedding_dim: *mut usize,
) -> EdgeGPTError {
    if handle.is_null() || texts.is_null() || out_embeddings.is_null() || out_lens.is_null() {
        return EdgeGPTError::NullPointer;
    }
    
    let ctx = &*(handle as *const EdgeGPTContext);
    
    // Convert C strings to Rust strings
    let mut text_vec = Vec::with_capacity(num_texts);
    for i in 0..num_texts {
        let text_ptr = *texts.add(i);
        if text_ptr.is_null() {
            return EdgeGPTError::NullPointer;
        }
        match std::panic::catch_unwind(|| c_to_string(text_ptr)) {
            Ok(s) => text_vec.push(s),
            Err(_) => return EdgeGPTError::InvalidUtf8,
        }
    }
    
    let text_refs: Vec<&str> = text_vec.iter().map(|s| s.as_str()).collect();
    
    let result = ctx.runtime.block_on(async {
        ctx.edge_gpt.encode_batch(&text_refs).await
    });
    
    match result {
        Ok(embeddings) => {
            let dim = embeddings[0].len();
            
            // Allocate array of pointers
            let mut emb_ptrs = Vec::with_capacity(num_texts);
            let mut lens = Vec::with_capacity(num_texts);
            
            for embedding in embeddings {
                let len = embedding.len();
                let array = FloatArray::from_vec(embedding);
                emb_ptrs.push(array.data);
                lens.push(len);
            }
            
            *out_embeddings = Box::into_raw(emb_ptrs.into_boxed_slice()) as *mut *mut f32;
            *out_lens = Box::into_raw(lens.into_boxed_slice()) as *mut usize;
            *embedding_dim = dim;
            
            EdgeGPTError::Success
        }
        Err(_) => EdgeGPTError::InferenceError,
    }
}

/// Compute similarity between two texts
///
/// # Safety
/// - `handle` must be valid
/// - `text1` and `text2` must be valid null-terminated UTF-8 strings
/// - `out_similarity` will be set to the similarity score
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_similarity(
    handle: *mut EdgeGPTHandle,
    text1: *const c_char,
    text2: *const c_char,
    out_similarity: *mut f32,
) -> EdgeGPTError {
    if handle.is_null() || text1.is_null() || text2.is_null() || out_similarity.is_null() {
        return EdgeGPTError::NullPointer;
    }
    
    let ctx = &*(handle as *const EdgeGPTContext);
    let text1_str = match std::panic::catch_unwind(|| c_to_string(text1)) {
        Ok(s) => s,
        Err(_) => return EdgeGPTError::InvalidUtf8,
    };
    let text2_str = match std::panic::catch_unwind(|| c_to_string(text2)) {
        Ok(s) => s,
        Err(_) => return EdgeGPTError::InvalidUtf8,
    };
    
    let result = ctx.runtime.block_on(async {
        ctx.edge_gpt.similarity(&text1_str, &text2_str).await
    });
    
    match result {
        Ok(sim) => {
            *out_similarity = sim;
            EdgeGPTError::Success
        }
        Err(_) => EdgeGPTError::InferenceError,
    }
}

/// Rerank documents by relevance to a query
///
/// # Safety
/// - `handle` must be valid
/// - `query` must be a valid null-terminated UTF-8 string
/// - `documents` must be an array of `num_docs` valid null-terminated UTF-8 strings
/// - `out_indices` and `out_scores` will be allocated and must be freed with `free()`
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_rerank(
    handle: *mut EdgeGPTHandle,
    query: *const c_char,
    documents: *const *const c_char,
    num_docs: usize,
    out_indices: *mut *mut usize,
    out_scores: *mut *mut f32,
) -> EdgeGPTError {
    if handle.is_null() || query.is_null() || documents.is_null() || out_indices.is_null() || out_scores.is_null() {
        return EdgeGPTError::NullPointer;
    }
    
    let ctx = &*(handle as *const EdgeGPTContext);
    let query_str = match std::panic::catch_unwind(|| c_to_string(query)) {
        Ok(s) => s,
        Err(_) => return EdgeGPTError::InvalidUtf8,
    };
    
    // Convert documents
    let mut doc_vec = Vec::with_capacity(num_docs);
    for i in 0..num_docs {
        let doc_ptr = *documents.add(i);
        if doc_ptr.is_null() {
            return EdgeGPTError::NullPointer;
        }
        match std::panic::catch_unwind(|| c_to_string(doc_ptr)) {
            Ok(s) => doc_vec.push(s),
            Err(_) => return EdgeGPTError::InvalidUtf8,
        }
    }
    
    let doc_refs: Vec<&str> = doc_vec.iter().map(|s| s.as_str()).collect();
    
    let result = ctx.runtime.block_on(async {
        ctx.edge_gpt.rerank(&query_str, &doc_refs).await
    });
    
    match result {
        Ok(ranked) => {
            let mut indices = Vec::with_capacity(num_docs);
            let mut scores = Vec::with_capacity(num_docs);
            
            for (idx, score) in ranked {
                indices.push(idx);
                scores.push(score);
            }
            
            *out_indices = Box::into_raw(indices.into_boxed_slice()) as *mut usize;
            *out_scores = Box::into_raw(scores.into_boxed_slice()) as *mut f32;
            
            EdgeGPTError::Success
        }
        Err(_) => EdgeGPTError::InferenceError,
    }
}

/// Free a float array allocated by edge_gpt_encode
///
/// # Safety
/// - `data` must have been allocated by an EdgeGPT function
/// - `len` must match the original length
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_free_float_array(data: *mut f32, len: usize) {
    if !data.is_null() {
        let _ = Vec::from_raw_parts(data, len, len);
    }
}

/// Free batch embeddings
///
/// # Safety
/// - Arrays must have been allocated by edge_gpt_encode_batch
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_free_batch_embeddings(
    embeddings: *mut *mut f32,
    lens: *mut usize,
    num_texts: usize,
    embedding_dim: usize,
) {
    if !embeddings.is_null() {
        let emb_vec = Vec::from_raw_parts(embeddings, num_texts, num_texts);
        for emb_ptr in emb_vec {
            if !emb_ptr.is_null() {
                let _ = Vec::from_raw_parts(emb_ptr, embedding_dim, embedding_dim);
            }
        }
    }
    
    if !lens.is_null() {
        let _ = Vec::from_raw_parts(lens, num_texts, num_texts);
    }
}

/// Free indices/scores arrays
#[unsafe(no_mangle)]
pub unsafe extern "C" fn edge_gpt_free_usize_array(data: *mut usize, len: usize) {
    if !data.is_null() {
        let _ = Vec::from_raw_parts(data, len, len);
    }
}