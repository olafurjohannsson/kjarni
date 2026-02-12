"""Kjarni - Fast ML inference library."""

from .embedder import Embedder
from .classifier import Classifier
from .reranker import Reranker, RerankResult
from .indexer import Indexer, IndexStats, IndexInfo, Progress, CancelToken
from .searcher import Searcher, SearchResult, SearchMode
from ._ffi import version, KjarniException

__all__ = [
    # Models
    "Embedder",
    "Classifier",
    "Reranker",
    "RerankResult",
    # RAG
    "Indexer",
    "IndexStats",
    "IndexInfo",
    "Searcher",
    "SearchResult",
    "SearchMode",
    # Utilities
    "Progress",
    "CancelToken",
    "version",
    "KjarniException",
]

__version__ = version()