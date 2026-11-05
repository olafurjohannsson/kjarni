"""Named-entity recognition: people, places, organizations and the like."""

import os
from ctypes import POINTER, Structure, byref, c_char_p, c_float, c_int, c_int32, c_size_t, c_void_p
from typing import List, NamedTuple, Optional

from ._ffi import KjarniDevice, _lib, check_error


class _KjarniEntity(Structure):
    _fields_ = [
        ("label", c_char_p),
        ("text", c_char_p),
        ("start", c_size_t),
        ("end", c_size_t),
        ("score", c_float),
    ]


class _KjarniEntities(Structure):
    _fields_ = [
        ("entities", POINTER(_KjarniEntity)),
        ("len", c_size_t),
    ]


class _KjarniNerConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("aggregation", c_int),
        ("quiet", c_int32),
    ]


_lib.kjarni_ner_config_default.restype = _KjarniNerConfig
_lib.kjarni_ner_config_default.argtypes = []
_lib.kjarni_ner_new.restype = c_int
_lib.kjarni_ner_new.argtypes = [POINTER(_KjarniNerConfig), POINTER(c_void_p)]
_lib.kjarni_ner_free.restype = None
_lib.kjarni_ner_free.argtypes = [c_void_p]
_lib.kjarni_ner_recognize.restype = c_int
_lib.kjarni_ner_recognize.argtypes = [c_void_p, c_char_p, POINTER(_KjarniEntities)]
_lib.kjarni_entities_free.restype = None
_lib.kjarni_entities_free.argtypes = [_KjarniEntities]

_AGGREGATIONS = {"first": 0, "simple": 1, "max": 2, "average": 3}


class Entity(NamedTuple):
    """One entity. ``start`` and ``end`` index the input string, so
    ``text[start:end] == entity.text``."""

    label: str
    text: str
    start: int
    end: int
    score: float


class EntityRecognizer:
    """Finds named entities with a token classification model.

    Example:
        >>> from kjarni import EntityRecognizer
        >>> ner = EntityRecognizer()
        >>> for e in ner.recognize("Sarah Connor met Apple's CEO in San Francisco."):
        ...     print(e.label, e.text)
        PER Sarah Connor
        ORG Apple
        LOC San Francisco

    The entities are those transformers' ``pipeline("ner")`` gives for the
    same ``aggregation``, with the same spans and scores.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        aggregation: str = "first",
        quiet: bool = False,
        model_path: Optional[str] = None,
    ):
        """Create an EntityRecognizer.

        Args:
            model: Model name; the default is "distilbert-ner", for English
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            aggregation: How tokens become entities: "first" (keeps words
                whole), "simple", "max" or "average", as in transformers
            quiet: Suppress progress output
            model_path: A local folder holding any ``*ForTokenClassification``
                model, such as one downloaded from the Hugging Face Hub. Used
                instead of ``model`` when given.
        """
        if aggregation not in _AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(_AGGREGATIONS)}, not {aggregation!r}")
        config = _lib.kjarni_ner_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.aggregation = _AGGREGATIONS[aggregation]
        config.quiet = 1 if quiet else 0
        if model:
            config.model_name = model.encode("utf-8")
        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")
        if model_path:
            config.model_path = os.fspath(model_path).encode("utf-8")

        self._handle = c_void_p()
        check_error(_lib.kjarni_ner_new(byref(config), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_ner_free(self._handle)

    def recognize(self, text: str) -> List[Entity]:
        """The entities in ``text``, in order."""
        out = _KjarniEntities()
        check_error(_lib.kjarni_ner_recognize(self._handle, text.encode("utf-8"), byref(out)))
        try:
            return [
                Entity(
                    label=out.entities[i].label.decode("utf-8"),
                    text=out.entities[i].text.decode("utf-8"),
                    start=out.entities[i].start,
                    end=out.entities[i].end,
                    score=out.entities[i].score,
                )
                for i in range(out.len)
            ]
        finally:
            _lib.kjarni_entities_free(out)
