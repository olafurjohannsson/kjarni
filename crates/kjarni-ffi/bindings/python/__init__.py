"""Kjarni - Fast ML inference library."""

from kjarni.embedder import Embedder
from kjarni.classifier import Classifier
from kjarni._ffi import version, KjarniException

__all__ = ["Embedder", "Classifier", "version", "KjarniException"]
__version__ = version()