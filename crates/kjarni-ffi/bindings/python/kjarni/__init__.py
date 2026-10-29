"""Kjarni: run transformer models inside your program.

Embeddings, classification, reranking, search, chat, speech-to-text,
summarization, translation and image search, on a native engine whose output
matches PyTorch. Models download on first use and are cached, so everything
after that works offline.

    from kjarni import Embedder

    e = Embedder("minilm-l6-v2")
    print(e.similarity("doctor", "physician"))  # 0.8598
"""

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