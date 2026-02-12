"""Low-level FFI bindings for Kjarni."""

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_int, c_int32, c_size_t, c_float, c_uint64, c_bool,
    POINTER, Structure, byref, CFUNCTYPE
)
from pathlib import Path
import platform
import json
import numpy as np
from typing import Optional, Callable

def _find_library():
    """Find the native library."""
    system = platform.system()
    if system == "Windows":
        lib_name = "kjarni_ffi.dll"
    elif system == "Darwin":
        lib_name = "libkjarni_ffi.dylib"
    else:
        lib_name = "libkjarni_ffi.so"

    search_paths = [
        Path(__file__).parent / lib_name,
        Path(__file__).parent / "lib" / lib_name,
        Path(__file__).parent.parent.parent.parent.parent / "target" / "release" / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    return lib_name

_lib = ctypes.CDLL(_find_library())

class KjarniError:
    OK = 0
    NULL_POINTER = 1
    INVALID_UTF8 = 2
    MODEL_NOT_FOUND = 3
    LOAD_FAILED = 4
    INFERENCE_FAILED = 5
    GPU_UNAVAILABLE = 6
    INVALID_CONFIG = 7
    CANCELLED = 8
    TIMEOUT = 9
    STREAM_ENDED = 10
    UNKNOWN = 255

class KjarniDevice:
    CPU = 0
    GPU = 1

class KjarniSearchMode:
    KEYWORD = 0
    SEMANTIC = 1
    HYBRID = 2

class KjarniProgressStage:
    SCANNING = 0
    LOADING = 1
    EMBEDDING = 2
    WRITING = 3
    COMMITTING = 4
    SEARCHING = 5
    RERANKING = 6

class KjarniProgress(Structure):
    _fields_ = [
        ("stage", c_int),
        ("current", c_size_t),
        ("total", c_size_t),
        ("message", c_char_p),
    ]

# Callback type: void callback(KjarniProgress progress, void* user_data)
PROGRESS_CALLBACK_TYPE = CFUNCTYPE(None, KjarniProgress, c_void_p)

class KjarniFloatArray(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("len", c_size_t),
    ]

    def to_list(self) -> list:
        if not self.data or self.len == 0:
            return []
        return [self.data[i] for i in range(self.len)]

    def free(self):
        _lib.kjarni_float_array_free(self)

class KjarniFloat2DArray(Structure):
    _fields_ = [
        ("data", POINTER(c_float)),
        ("rows", c_size_t),
        ("cols", c_size_t),
    ]

    def to_numpy(self) -> np.ndarray:
        """Convert to NumPy array efficiently."""
        if not self.data or self.rows == 0 or self.cols == 0:
            return np.empty((0, 0), dtype=np.float32)
        total_size = self.rows * self.cols
        c_array_type = c_float * total_size
        c_array = c_array_type.from_address(ctypes.addressof(self.data.contents))

        raw_view = np.ctypeslib.as_array(c_array)
        return raw_view.reshape((self.rows, self.cols)).copy()

    def to_list(self) -> list:
        if not self.data or self.rows == 0 or self.cols == 0:
            return []
        result = []
        for i in range(self.rows):
            row = [self.data[i * self.cols + j] for j in range(self.cols)]
            result.append(row)
        return result

    def free(self):
        _lib.kjarni_float_2d_array_free(self)

class KjarniEmbedderConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("normalize", c_int32),
        ("quiet", c_int32),
    ]

class KjarniClassResult(Structure):
    _fields_ = [
        ("label", c_char_p),
        ("score", c_float),
    ]

class KjarniClassResults(Structure):
    _fields_ = [
        ("results", POINTER(KjarniClassResult)),
        ("len", c_size_t),
    ]

    def to_list(self) -> list:
        if not self.results or self.len == 0:
            return []
        return [
            (self.results[i].label.decode("utf-8"), self.results[i].score)
            for i in range(self.len)
        ]

    def free(self):
        _lib.kjarni_class_results_free(self)

class KjarniClassifierConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("labels", POINTER(c_char_p)),
        ("num_labels", c_size_t),
        ("multi_label", c_int32),
        ("quiet", c_int32),
    ]

class KjarniRerankResult(Structure):
    _fields_ = [
        ("index", c_size_t),
        ("score", c_float),
    ]

class KjarniRerankResults(Structure):
    _fields_ = [
        ("results", POINTER(KjarniRerankResult)),
        ("len", c_size_t),
    ]

    def to_list(self) -> list:
        if not self.results or self.len == 0:
            return []
        return [
            (self.results[i].index, self.results[i].score)
            for i in range(self.len)
        ]

    def free(self):
        _lib.kjarni_rerank_results_free(self)

class KjarniRerankerConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("quiet", c_int32),
    ]

class KjarniIndexStats(Structure):
    _fields_ = [
        ("documents_indexed", c_size_t),
        ("chunks_created", c_size_t),
        ("dimension", c_size_t),
        ("size_bytes", c_uint64),
        ("files_processed", c_size_t),
        ("files_skipped", c_size_t),
        ("elapsed_ms", c_uint64),
    ]

class KjarniIndexInfo(Structure):
    _fields_ = [
        ("path", c_char_p),
        ("document_count", c_size_t),
        ("segment_count", c_size_t),
        ("dimension", c_size_t),
        ("size_bytes", c_uint64),
        ("embedding_model", c_char_p),
    ]

class KjarniIndexerConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("chunk_size", c_size_t),
        ("chunk_overlap", c_size_t),
        ("batch_size", c_size_t),
        ("extensions", c_char_p),
        ("exclude_patterns", c_char_p),
        ("recursive", c_int32),
        ("include_hidden", c_int32),
        ("max_file_size", c_size_t),
        ("quiet", c_int32),
    ]

class KjarniSearchResult(Structure):
    _fields_ = [
        ("score", c_float),
        ("document_id", c_size_t),
        ("text", c_char_p),
        ("metadata_json", c_char_p),
    ]

class KjarniSearchResults(Structure):
    _fields_ = [
        ("results", POINTER(KjarniSearchResult)),
        ("len", c_size_t),
    ]

    def to_list(self) -> list:
        if not self.results or self.len == 0:
            return []
        result = []
        for i in range(self.len):
            r = self.results[i]
            text = r.text.decode("utf-8") if r.text else ""
            metadata = {}
            if r.metadata_json:
                try:
                    metadata = json.loads(r.metadata_json.decode("utf-8"))
                except:
                    pass
            result.append({
                "score": r.score,
                "document_id": r.document_id,
                "text": text,
                "metadata": metadata,
            })
        return result

    def free(self):
        _lib.kjarni_search_results_free(self)

class KjarniSearchOptions(Structure):
    _fields_ = [
        ("mode", c_int32),
        ("top_k", c_size_t),
        ("use_reranker", c_int32),
        ("threshold", c_float),
        ("source_pattern", c_char_p),
        ("filter_key", c_char_p),
        ("filter_value", c_char_p),
    ]

class KjarniSearcherConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("rerank_model", c_char_p),
        ("default_mode", c_int),
        ("default_top_k", c_size_t),
        ("quiet", c_int32),
    ]

# Error handling
_lib.kjarni_last_error_message.restype = c_char_p
_lib.kjarni_last_error_message.argtypes = []
_lib.kjarni_clear_error.restype = None
_lib.kjarni_clear_error.argtypes = []
_lib.kjarni_error_name.restype = c_char_p
_lib.kjarni_error_name.argtypes = [c_int]

# Global
_lib.kjarni_init.restype = c_int
_lib.kjarni_init.argtypes = []
_lib.kjarni_version.restype = c_char_p
_lib.kjarni_version.argtypes = []

# Memory
_lib.kjarni_float_array_free.restype = None
_lib.kjarni_float_array_free.argtypes = [KjarniFloatArray]
_lib.kjarni_float_2d_array_free.restype = None
_lib.kjarni_float_2d_array_free.argtypes = [KjarniFloat2DArray]
_lib.kjarni_string_free.restype = None
_lib.kjarni_string_free.argtypes = [c_char_p]
_lib.kjarni_class_results_free.restype = None
_lib.kjarni_class_results_free.argtypes = [KjarniClassResults]
_lib.kjarni_rerank_results_free.restype = None
_lib.kjarni_rerank_results_free.argtypes = [KjarniRerankResults]
_lib.kjarni_search_results_free.restype = None
_lib.kjarni_search_results_free.argtypes = [KjarniSearchResults]

# Cancel token
_lib.kjarni_cancel_token_new.restype = c_void_p
_lib.kjarni_cancel_token_new.argtypes = []
_lib.kjarni_cancel_token_cancel.restype = None
_lib.kjarni_cancel_token_cancel.argtypes = [c_void_p]
_lib.kjarni_cancel_token_is_cancelled.restype = c_bool
_lib.kjarni_cancel_token_is_cancelled.argtypes = [c_void_p]
_lib.kjarni_cancel_token_reset.restype = None
_lib.kjarni_cancel_token_reset.argtypes = [c_void_p]
_lib.kjarni_cancel_token_free.restype = None
_lib.kjarni_cancel_token_free.argtypes = [c_void_p]

# Embedder
_lib.kjarni_embedder_config_default.restype = KjarniEmbedderConfig
_lib.kjarni_embedder_config_default.argtypes = []
_lib.kjarni_embedder_new.restype = c_int
_lib.kjarni_embedder_new.argtypes = [POINTER(KjarniEmbedderConfig), POINTER(c_void_p)]
_lib.kjarni_embedder_free.restype = None
_lib.kjarni_embedder_free.argtypes = [c_void_p]
_lib.kjarni_embedder_encode.restype = c_int
_lib.kjarni_embedder_encode.argtypes = [c_void_p, c_char_p, POINTER(KjarniFloatArray)]
_lib.kjarni_embedder_encode_batch.restype = c_int
_lib.kjarni_embedder_encode_batch.argtypes = [c_void_p, POINTER(c_char_p), c_size_t, POINTER(KjarniFloat2DArray)]
_lib.kjarni_embedder_similarity.restype = c_int
_lib.kjarni_embedder_similarity.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(c_float)]
_lib.kjarni_embedder_dim.restype = c_size_t
_lib.kjarni_embedder_dim.argtypes = [c_void_p]

# Classifier
_lib.kjarni_classifier_config_default.restype = KjarniClassifierConfig
_lib.kjarni_classifier_config_default.argtypes = []
_lib.kjarni_classifier_new.restype = c_int
_lib.kjarni_classifier_new.argtypes = [POINTER(KjarniClassifierConfig), POINTER(c_void_p)]
_lib.kjarni_classifier_free.restype = None
_lib.kjarni_classifier_free.argtypes = [c_void_p]
_lib.kjarni_classifier_classify.restype = c_int
_lib.kjarni_classifier_classify.argtypes = [c_void_p, c_char_p, POINTER(KjarniClassResults)]
_lib.kjarni_classifier_num_labels.restype = c_size_t
_lib.kjarni_classifier_num_labels.argtypes = [c_void_p]

# Reranker
_lib.kjarni_reranker_config_default.restype = KjarniRerankerConfig
_lib.kjarni_reranker_config_default.argtypes = []
_lib.kjarni_reranker_new.restype = c_int
_lib.kjarni_reranker_new.argtypes = [POINTER(KjarniRerankerConfig), POINTER(c_void_p)]
_lib.kjarni_reranker_free.restype = None
_lib.kjarni_reranker_free.argtypes = [c_void_p]
_lib.kjarni_reranker_score.restype = c_int
_lib.kjarni_reranker_score.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(c_float)]
_lib.kjarni_reranker_rerank.restype = c_int
_lib.kjarni_reranker_rerank.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, POINTER(KjarniRerankResults)]
_lib.kjarni_reranker_rerank_top_k.restype = c_int
_lib.kjarni_reranker_rerank_top_k.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, c_size_t, POINTER(KjarniRerankResults)]

# Indexer
_lib.kjarni_indexer_config_default.restype = KjarniIndexerConfig
_lib.kjarni_indexer_config_default.argtypes = []
_lib.kjarni_indexer_new.restype = c_int
_lib.kjarni_indexer_new.argtypes = [POINTER(KjarniIndexerConfig), POINTER(c_void_p)]
_lib.kjarni_indexer_free.restype = None
_lib.kjarni_indexer_free.argtypes = [c_void_p]
_lib.kjarni_indexer_create.restype = c_int
_lib.kjarni_indexer_create.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, c_int32, POINTER(KjarniIndexStats)]
_lib.kjarni_indexer_create_with_callback.restype = c_int
_lib.kjarni_indexer_create_with_callback.argtypes = [
    c_void_p, c_char_p, POINTER(c_char_p), c_size_t, c_int32,
    PROGRESS_CALLBACK_TYPE, c_void_p, c_void_p, POINTER(KjarniIndexStats)
]
_lib.kjarni_indexer_add.restype = c_int
_lib.kjarni_indexer_add.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, POINTER(c_size_t)]
_lib.kjarni_indexer_add_with_callback.restype = c_int
_lib.kjarni_indexer_add_with_callback.argtypes = [
    c_void_p, c_char_p, POINTER(c_char_p), c_size_t,
    PROGRESS_CALLBACK_TYPE, c_void_p, c_void_p, POINTER(c_size_t)
]
_lib.kjarni_index_info.restype = c_int
_lib.kjarni_index_info.argtypes = [c_char_p, POINTER(KjarniIndexInfo)]
_lib.kjarni_index_info_free.restype = None
_lib.kjarni_index_info_free.argtypes = [KjarniIndexInfo]
_lib.kjarni_index_delete.restype = c_int
_lib.kjarni_index_delete.argtypes = [c_char_p]
_lib.kjarni_indexer_model_name.restype = c_size_t
_lib.kjarni_indexer_model_name.argtypes = [c_void_p, c_char_p, c_size_t]
_lib.kjarni_indexer_dimension.restype = c_size_t
_lib.kjarni_indexer_dimension.argtypes = [c_void_p]
_lib.kjarni_indexer_chunk_size.restype = c_size_t
_lib.kjarni_indexer_chunk_size.argtypes = [c_void_p]

# Searcher
_lib.kjarni_searcher_config_default.restype = KjarniSearcherConfig
_lib.kjarni_searcher_config_default.argtypes = []
_lib.kjarni_search_options_default.restype = KjarniSearchOptions
_lib.kjarni_search_options_default.argtypes = []
_lib.kjarni_searcher_new.restype = c_int
_lib.kjarni_searcher_new.argtypes = [POINTER(KjarniSearcherConfig), POINTER(c_void_p)]
_lib.kjarni_searcher_free.restype = None
_lib.kjarni_searcher_free.argtypes = [c_void_p]
_lib.kjarni_searcher_search.restype = c_int
_lib.kjarni_searcher_search.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(KjarniSearchResults)]
_lib.kjarni_searcher_search_with_options.restype = c_int
_lib.kjarni_searcher_search_with_options.argtypes = [
    c_void_p, c_char_p, c_char_p, POINTER(KjarniSearchOptions), POINTER(KjarniSearchResults)
]
_lib.kjarni_search_keywords.restype = c_int
_lib.kjarni_search_keywords.argtypes = [c_char_p, c_char_p, c_size_t, POINTER(KjarniSearchResults)]
_lib.kjarni_searcher_has_reranker.restype = c_bool
_lib.kjarni_searcher_has_reranker.argtypes = [c_void_p]
_lib.kjarni_searcher_default_mode.restype = c_int
_lib.kjarni_searcher_default_mode.argtypes = [c_void_p]
_lib.kjarni_searcher_default_top_k.restype = c_size_t
_lib.kjarni_searcher_default_top_k.argtypes = [c_void_p]
_lib.kjarni_searcher_model_name.restype = c_size_t
_lib.kjarni_searcher_model_name.argtypes = [c_void_p, c_char_p, c_size_t]
_lib.kjarni_searcher_reranker_model.restype = c_size_t
_lib.kjarni_searcher_reranker_model.argtypes = [c_void_p, c_char_p, c_size_t]

class KjarniException(Exception):
    """Exception raised by Kjarni operations."""
    def __init__(self, message: str, code: int):
        self.code = code
        super().__init__(message)

def check_error(err: int):
    """Raise exception if error occurred."""
    if err != KjarniError.OK:
        msg = _lib.kjarni_last_error_message()
        if msg:
            msg = msg.decode("utf-8")
        else:
            name = _lib.kjarni_error_name(err)
            msg = name.decode("utf-8") if name else f"Error code {err}"
        raise KjarniException(msg, err)

def version() -> str:
    """Get Kjarni version."""
    return _lib.kjarni_version().decode("utf-8")