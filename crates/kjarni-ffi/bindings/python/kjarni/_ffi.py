"""Low-level FFI bindings for Kjarni."""

import ctypes
from ctypes import (
    c_void_p, c_char_p, c_int, c_int32, c_size_t, c_float,
    POINTER, Structure, byref
)
from pathlib import Path
import platform
import sys

def _find_library():
    """Find the native library."""
    system = platform.system()
    if system == "Windows":
        lib_name = "kjarni_ffi.dll"
    elif system == "Darwin":
        lib_name = "libkjarni_ffi.dylib"
    else:
        lib_name = "libkjarni_ffi.so"

    # Search paths
    search_paths = [
        Path(__file__).parent / lib_name,
        Path(__file__).parent / "lib" / lib_name,
        Path(__file__).parent.parent.parent.parent.parent / "target" / "release" / lib_name,
    ]

    for path in search_paths:
        if path.exists():
            return str(path)

    # Fall back to system search
    return lib_name

_lib = ctypes.CDLL(_find_library())

# =============================================================================
# Error Codes
# =============================================================================

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

# =============================================================================
# Structures
# =============================================================================

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

# =============================================================================
# Function Signatures
# =============================================================================

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

# =============================================================================
# Error Handling
# =============================================================================

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