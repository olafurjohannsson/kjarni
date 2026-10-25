"""High-level Translator API."""

from typing import Optional
from ctypes import c_void_p, c_char_p, c_int, c_int32, byref, POINTER, Structure

from ._ffi import _lib, check_error, take_string, encode_or_none, KjarniDevice


class KjarniTranslatorConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("from", c_char_p),
        ("to", c_char_p),
        ("quiet", c_int32),
    ]


_lib.kjarni_translator_config_default.restype = KjarniTranslatorConfig
_lib.kjarni_translator_config_default.argtypes = []
_lib.kjarni_translator_new.restype = c_int
_lib.kjarni_translator_new.argtypes = [POINTER(KjarniTranslatorConfig), POINTER(c_void_p)]
_lib.kjarni_translator_free.restype = None
_lib.kjarni_translator_free.argtypes = [c_void_p]
_lib.kjarni_translator_translate.restype = c_int
_lib.kjarni_translator_translate.argtypes = [c_void_p, c_char_p, c_char_p, c_char_p, POINTER(c_void_p)]


class Translator:
    """Translates text between languages.

    Example:
        >>> translator = Translator(source="en", target="de")
        >>> translator.translate("The house is small.")
        'Das Haus ist klein.'
    """

    def __init__(
        self,
        model: Optional[str] = None,
        source: Optional[str] = None,
        target: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        quiet: bool = False,
    ):
        """Load a translation model, downloading it on first use.

        `source` and `target` rather than `from` and `to`, which Python reserves.

        Args:
            model: e.g. "flan-t5-base" (the default) or "flan-t5-large"
            source: Default source language ("en", "english", ...)
            target: Default target language
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            quiet: Suppress progress output
        """
        config = _lib.kjarni_translator_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.quiet = 1 if quiet else 0
        if model:
            config.model_name = model.encode("utf-8")
        config.cache_dir = encode_or_none(cache_dir)
        setattr(config, "from", encode_or_none(source))
        config.to = encode_or_none(target)

        self._handle = c_void_p()
        check_error(_lib.kjarni_translator_new(byref(config), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_translator_free(self._handle)
            self._handle = None

    def translate(self, text: str, source: Optional[str] = None, target: Optional[str] = None) -> str:
        """Translate `text`.

        Languages left as None fall back to the defaults given at construction;
        with neither, this raises KjarniException with code INVALID_CONFIG.
        """
        out = c_void_p()
        check_error(_lib.kjarni_translator_translate(
            self._handle, text.encode("utf-8"), encode_or_none(source), encode_or_none(target), byref(out)))
        return take_string(out)
