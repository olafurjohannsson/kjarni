"""Kjarni - Fast ML inference library."""

from .embedder import Embedder
from .classifier import Classifier
from .reranker import Reranker, RerankResult
from .indexer import Indexer, IndexStats, IndexInfo, Progress, CancelToken
from .searcher import Searcher, SearchResult, SearchMode
from .chat import Chat, Conversation, Turn
from .transcriber import Transcriber, Transcript, Segment
from .summarizer import Summarizer
from .translator import Translator
from .image_index import ImageIndex, ImageHit, ScanReport
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
    # Generation
    "Chat",
    "Conversation",
    "Turn",
    "Summarizer",
    "Translator",
    # Speech
    "Transcriber",
    "Transcript",
    "Segment",
    # Images
    "ImageIndex",
    "ImageHit",
    "ScanReport",
    # Utilities
    "Progress",
    "CancelToken",
    "version",
    "KjarniException",
]

__version__ = version()