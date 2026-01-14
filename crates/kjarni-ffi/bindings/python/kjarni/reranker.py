"""High-level Reranker API."""

from typing import Optional, List, NamedTuple
from ctypes import c_void_p, c_char_p, c_float, c_size_t, byref, POINTER, Structure
from ctypes import c_int, c_int32

from ._ffi import (
    _lib, check_error, KjarniDevice, KjarniError,
    c_float, c_size_t
)

class KjarniRerankResult(Structure):
    _fields_ = [
        ("index", c_size_t),
        ("score", c_float),
    ]

class KjarniRerankResults(Structure):
    _fields_ = [
        ("results", POINTER(KjarniRerankResult)),
        ("len", c_size_t),
    ]

    def to_list(self) -> list:
        if not self.results or self.len == 0:
            return []
        return [
            (self.results[i].index, self.results[i].score)
            for i in range(self.len)
        ]

    def free(self):
        _lib.kjarni_rerank_results_free(self)

class KjarniRerankerConfig(Structure):
    _fields_ = [
        ("device", c_int),
        ("cache_dir", c_char_p),
        ("model_name", c_char_p),
        ("model_path", c_char_p),
        ("quiet", c_int32),
    ]

# Register FFI functions

_lib.kjarni_reranker_config_default.restype = KjarniRerankerConfig
_lib.kjarni_reranker_config_default.argtypes = []
_lib.kjarni_reranker_new.restype = c_int
_lib.kjarni_reranker_new.argtypes = [POINTER(KjarniRerankerConfig), POINTER(c_void_p)]
_lib.kjarni_reranker_free.restype = None
_lib.kjarni_reranker_free.argtypes = [c_void_p]
_lib.kjarni_reranker_score.restype = c_int
_lib.kjarni_reranker_score.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(c_float)]
_lib.kjarni_reranker_rerank.restype = c_int
_lib.kjarni_reranker_rerank.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, POINTER(KjarniRerankResults)]
_lib.kjarni_reranker_rerank_top_k.restype = c_int
_lib.kjarni_reranker_rerank_top_k.argtypes = [c_void_p, c_char_p, POINTER(c_char_p), c_size_t, c_size_t, POINTER(KjarniRerankResults)]
_lib.kjarni_rerank_results_free.restype = None
_lib.kjarni_rerank_results_free.argtypes = [KjarniRerankResults]


class RerankResult(NamedTuple):
    """Single rerank result."""
    index: int
    score: float
    document: str


class Reranker:
    """Text reranking model using cross-encoders.
    
    Example:
        >>> reranker = Reranker()
        >>> results = reranker.rerank("What is Python?", [
        ...     "Python is a programming language",
        ...     "Java is also a language",
        ...     "The weather is nice today"
        ... ])
        >>> for r in results:
        ...     print(f"{r.index}: {r.score:.3f} - {r.document[:30]}...")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        quiet: bool = False,
    ):
        """Create a new Reranker.
        
        Args:
            model: Model name (e.g., "cross-encoder-minilm")
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            quiet: Suppress progress output
        """
        config = _lib.kjarni_reranker_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.quiet = 1 if quiet else 0

        if model:
            config.model_name = model.encode("utf-8")
        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")

        self._handle = c_void_p()
        err = _lib.kjarni_reranker_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_reranker_free(self._handle)

    def score(self, query: str, document: str) -> float:
        """Score a single query-document pair.
        
        Args:
            query: Query text
            document: Document text
            
        Returns:
            Relevance score (higher = more relevant)
        """
        result = c_float()
        err = _lib.kjarni_reranker_score(
            self._handle,
            query.encode("utf-8"),
            document.encode("utf-8"),
            byref(result)
        )
        check_error(err)
        return result.value

    def rerank(self, query: str, documents: List[str]) -> List[RerankResult]:
        """Rerank documents by relevance to query.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            
        Returns:
            List of RerankResult sorted by score (highest first)
        """
        if not documents:
            return []

        c_docs = (c_char_p * len(documents))()
        for i, d in enumerate(documents):
            c_docs[i] = d.encode("utf-8")

        results = KjarniRerankResults()
        err = _lib.kjarni_reranker_rerank(
            self._handle,
            query.encode("utf-8"),
            c_docs,
            len(documents),
            byref(results)
        )
        check_error(err)

        raw = results.to_list()
        results.free()

        return [
            RerankResult(index=idx, score=score, document=documents[idx])
            for idx, score in raw
        ]

    def rerank_top_k(self, query: str, documents: List[str], k: int) -> List[RerankResult]:
        """Rerank and return top-k results.
        
        Args:
            query: Query text
            documents: List of documents to rerank
            k: Number of top results to return
            
        Returns:
            Top-k RerankResults sorted by score
        """
        if not documents:
            return []

        c_docs = (c_char_p * len(documents))()
        for i, d in enumerate(documents):
            c_docs[i] = d.encode("utf-8")

        results = KjarniRerankResults()
        err = _lib.kjarni_reranker_rerank_top_k(
            self._handle,
            query.encode("utf-8"),
            c_docs,
            len(documents),
            k,
            byref(results)
        )
        check_error(err)

        raw = results.to_list()
        results.free()

        return [
            RerankResult(index=idx, score=score, document=documents[idx])
            for idx, score in raw
        ]