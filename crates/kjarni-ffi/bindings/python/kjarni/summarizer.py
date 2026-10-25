"""High-level Summarizer API."""

from typing import Optional
from ctypes import c_void_p, c_char_p, c_int, c_int32, c_size_t, byref, POINTER, Structure

from ._ffi import _lib, check_error, take_string, encode_or_none, KjarniDevice


class KjarniSummarizerConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("min_length", c_size_t),
        ("max_length", c_size_t),
        ("quiet", c_int32),
    ]


_lib.kjarni_summarizer_config_default.restype = KjarniSummarizerConfig
_lib.kjarni_summarizer_config_default.argtypes = []
_lib.kjarni_summarizer_new.restype = c_int
_lib.kjarni_summarizer_new.argtypes = [POINTER(KjarniSummarizerConfig), POINTER(c_void_p)]
_lib.kjarni_summarizer_free.restype = None
_lib.kjarni_summarizer_free.argtypes = [c_void_p]
_lib.kjarni_summarizer_summarize.restype = c_int
_lib.kjarni_summarizer_summarize.argtypes = [c_void_p, c_char_p, POINTER(c_void_p)]


class Summarizer:
    """Summarizes article-length text.

    Example:
        >>> summarizer = Summarizer()
        >>> print(summarizer.summarize(article))
    """

    def __init__(
        self,
        model: Optional[str] = None,
        min_length: int = 0,
        max_length: int = 0,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        quiet: bool = False,
    ):
        """Load a summarization model, downloading it on first use.

        Args:
            model: e.g. "distilbart-cnn" (the default) or "bart-large-cnn"
            min_length: Minimum summary length in tokens; 0 uses the model's default
            max_length: Maximum summary length in tokens; 0 uses the model's default
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            quiet: Suppress progress output
        """
        if min_length < 0 or max_length < 0:
            raise ValueError("min_length and max_length must not be negative")
        config = _lib.kjarni_summarizer_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.min_length = min_length
        config.max_length = max_length
        config.quiet = 1 if quiet else 0
        if model:
            config.model_name = model.encode("utf-8")
        config.cache_dir = encode_or_none(cache_dir)

        self._handle = c_void_p()
        check_error(_lib.kjarni_summarizer_new(byref(config), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_summarizer_free(self._handle)
            self._handle = None

    def summarize(self, text: str) -> str:
        """Summarize `text`."""
        out = c_void_p()
        check_error(_lib.kjarni_summarizer_summarize(self._handle, text.encode("utf-8"), byref(out)))
        return take_string(out)
