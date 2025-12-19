//! edgegpt.h - C API for EdgeGPT
#ifndef EDGEGPT_H
#define EDGEGPT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque handle to EdgeGPT instance
typedef struct EdgeGPTHandle EdgeGPTHandle;

/// Opaque handle to Search Index
typedef struct EdgeGPTIndex EdgeGPTIndex;

/// Error codes
typedef enum {
    EDGE_GPT_SUCCESS = 0,
    EDGE_GPT_NULL_POINTER = 1,
    EDGE_GPT_INVALID_UTF8 = 2,
    EDGE_GPT_MODEL_LOAD_ERROR = 3,
    EDGE_GPT_INFERENCE_ERROR = 4,
    EDGE_GPT_RUNTIME_ERROR = 5,
} EdgeGPTError;

/// Create EdgeGPT instance for CPU
EdgeGPTHandle* edge_gpt_new_cpu(void);

/// Create EdgeGPT instance for GPU (WGPU)
/// Returns NULL if GPU initialization fails
EdgeGPTHandle* edge_gpt_new_gpu(void);

/// Free EdgeGPT instance
void edge_gpt_free(EdgeGPTHandle* handle);

// ============================================================================
// Embeddings & Similarity
// ============================================================================

/// Encode single text
EdgeGPTError edge_gpt_encode(
    EdgeGPTHandle* handle,
    const char* text,
    float** out_embedding,
    size_t* out_len
);

/// Encode batch of texts
EdgeGPTError edge_gpt_encode_batch(
    EdgeGPTHandle* handle,
    const char* const* texts,
    size_t num_texts,
    float*** out_embeddings,
    size_t** out_lens,
    size_t* embedding_dim
);

/// Compute similarity
EdgeGPTError edge_gpt_similarity(
    EdgeGPTHandle* handle,
    const char* text1,
    const char* text2,
    float* out_similarity
);

/// Rerank documents
EdgeGPTError edge_gpt_rerank(
    EdgeGPTHandle* handle,
    const char* query,
    const char* const* documents,
    size_t num_docs,
    size_t** out_indices,
    float** out_scores
);

// ============================================================================
// Generation & Summarization
// ============================================================================

/// Generate text continuation
EdgeGPTError edge_gpt_generate(
    EdgeGPTHandle* handle,
    const char* prompt,
    char** out_text
);

/// Summarize text
EdgeGPTError edge_gpt_summarize(
    EdgeGPTHandle* handle,
    const char* text,
    char** out_summary
);

// ============================================================================
// Indexing & Search (RAG)
// ============================================================================

/// Build a search index from documents
EdgeGPTError edge_gpt_build_index(
    EdgeGPTHandle* handle,
    const char* const* documents,
    size_t num_docs,
    EdgeGPTIndex** out_index
);

/// Search the index
/// Returns a JSON string of results
EdgeGPTError edge_gpt_search(
    EdgeGPTHandle* handle,
    EdgeGPTIndex* index_handle,
    const char* query,
    size_t limit,
    char** out_json_results
);

/// Save index to disk
EdgeGPTError edge_gpt_save_index(
    EdgeGPTIndex* index_handle,
    const char* path
);

/// Load index from disk
EdgeGPTError edge_gpt_load_index(
    const char* path,
    EdgeGPTIndex** out_index
);

/// Free the index
void edge_gpt_free_index(EdgeGPTIndex* index_handle);

// ============================================================================
// Memory Management
// ============================================================================

/// Free float array
void edge_gpt_free_float_array(float* data, size_t len);

/// Free batch embeddings
void edge_gpt_free_batch_embeddings(
    float** embeddings,
    size_t* lens,
    size_t num_texts,
    size_t embedding_dim
);

/// Free size_t array
void edge_gpt_free_usize_array(size_t* data, size_t len);

/// Free string returned by generate/summarize/search
void edge_gpt_free_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // EDGEGPT_H