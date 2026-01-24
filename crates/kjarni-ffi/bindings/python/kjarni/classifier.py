"""High-level Classifier API."""

from typing import Optional, List, Tuple
from ctypes import c_void_p, c_char_p, byref
from dataclasses import dataclass

from ._ffi import (
    _lib, check_error, KjarniDevice,
    KjarniClassifierConfig, KjarniClassResults
)


@dataclass
class ClassificationResult:
    """Result of classifying a single text."""
    
    label: str
    """The predicted label (highest score)."""
    
    score: float
    """Confidence score for the predicted label (0.0 - 1.0)."""
    
    all_scores: List[Tuple[str, float]]
    """All labels with their scores, sorted by score descending."""
    
    @property
    def label_index(self) -> int:
        """Index of the predicted label."""
        return 0  # Already sorted, top is at 0
    
    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Get the top K predictions."""
        return self.all_scores[:k]
    
    def is_confident(self, threshold: float) -> bool:
        """Check if the top prediction exceeds a confidence threshold."""
        return self.score >= threshold
    
    def above_threshold(self, threshold: float) -> List[Tuple[str, float]]:
        """Get predictions above a threshold."""
        return [(label, score) for label, score in self.all_scores if score >= threshold]
    
    def __str__(self) -> str:
        return f"{self.label} ({self.score * 100:.1f}%)"


class Classifier:
    """Text classification model.
    
    Example:
        >>> classifier = Classifier("distilbert-sentiment")
        >>> result = classifier.classify("I love this product!")
        >>> print(result.label, result.score)
        POSITIVE 0.9998
    """

    def __init__(
        self,
        model: str,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        labels: Optional[List[str]] = None,
        multi_label: bool = False,
        quiet: bool = False,
    ):
        """Create a new Classifier.
        
        Args:
            model: Model name (e.g., "distilbert-sentiment", "roberta-sentiment")
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            labels: Custom labels (overrides model defaults)
            multi_label: Enable multi-label classification (sigmoid vs softmax)
            quiet: Suppress progress output
        """
        self._model = model
        self._device = device
        
        config = _lib.kjarni_classifier_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.model_name = model.encode("utf-8")
        config.multi_label = 1 if multi_label else 0
        config.quiet = 1 if quiet else 0

        if cache_dir:
            self._cache_dir = cache_dir
            config.cache_dir = cache_dir.encode("utf-8")

        # Handle custom labels
        self._label_refs = None
        self._custom_labels = labels
        if labels:
            c_labels = (c_char_p * len(labels))()
            for i, label in enumerate(labels):
                c_labels[i] = label.encode("utf-8")
            self._label_refs = c_labels
            config.labels = c_labels
            config.num_labels = len(labels)

        self._handle = c_void_p()
        err = _lib.kjarni_classifier_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_classifier_free(self._handle)

    def classify(self, text: str) -> ClassificationResult:
        """Classify a single text.
        
        Args:
            text: Input text
            
        Returns:
            ClassificationResult with label, score, and all_scores
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
        
        if not scores:
            raise RuntimeError("Classification returned no results")
        
        return ClassificationResult(
            label=scores[0][0],
            score=scores[0][1],
            all_scores=scores
        )

    def classify_batch(self, texts: List[str]) -> List[ClassificationResult]:
        """Classify multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of ClassificationResult objects
        """
        # TODO: Add batch FFI call for better performance
        return [self.classify(text) for text in texts]

    @property
    def model_id(self) -> str:
        """Get the model identifier."""
        return self._model

    @property
    def device(self) -> str:
        """Get the device (cpu or gpu)."""
        return self._device

    @property
    def num_labels(self) -> int:
        """Get number of classification labels."""
        return _lib.kjarni_classifier_num_labels(self._handle)

    @property
    def has_custom_labels(self) -> bool:
        """Check if using custom labels."""
        return self._custom_labels is not None