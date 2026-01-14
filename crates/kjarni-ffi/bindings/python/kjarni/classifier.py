"""High-level Classifier API."""

from typing import Optional, List, Tuple
from ctypes import c_void_p, c_char_p, byref, POINTER

from ._ffi import (
    _lib, check_error, KjarniDevice,
    KjarniClassifierConfig, KjarniClassResults
)

class Classifier:
    """Text classification model.
    
    Example:
        >>> classifier = Classifier("sentiment")
        >>> result = classifier.classify("I love this product!")
        >>> print(result)
        [('positive', 0.95), ('negative', 0.05)]
    """

    def __init__(
        self,
        model: str = "sentiment",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        labels: Optional[List[str]] = None,
        multi_label: bool = False,
        quiet: bool = False,
    ):
        """Create a new Classifier.
        
        Args:
            model: Model name or task (e.g., "sentiment", "emotion")
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            labels: Custom labels (overrides model defaults)
            multi_label: Enable multi-label classification
            quiet: Suppress progress output
        """
        config = _lib.kjarni_classifier_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.model_name = model.encode("utf-8")
        config.multi_label = 1 if multi_label else 0
        config.quiet = 1 if quiet else 0

        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")

        # Handle custom labels
        self._label_refs = None
        if labels:
            c_labels = (c_char_p * len(labels))()
            for i, label in enumerate(labels):
                c_labels[i] = label.encode("utf-8")
            self._label_refs = c_labels  # Keep reference
            config.labels = c_labels
            config.num_labels = len(labels)

        self._handle = c_void_p()
        err = _lib.kjarni_classifier_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_classifier_free(self._handle)

    def classify(self, text: str) -> List[Tuple[str, float]]:
        """Classify a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of (label, score) tuples sorted by score
        """
        result = KjarniClassResults()
        err = _lib.kjarni_classifier_classify(
            self._handle,
            text.encode("utf-8"),
            byref(result)
        )
        check_error(err)

        scores = result.to_list()
        result.free()
        return scores

    @property
    def num_labels(self) -> int:
        """Get number of classification labels."""
        return _lib.kjarni_classifier_num_labels(self._handle)