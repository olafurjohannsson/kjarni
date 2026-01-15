"""High-level Searcher API."""

from typing import Optional, List, Dict, Any, NamedTuple
from ctypes import c_void_p, c_char_p, c_size_t, c_int32, c_float, c_bool, byref, POINTER
from dataclasses import dataclass
from enum import Enum

from ._ffi import (
    _lib, check_error, KjarniDevice, KjarniSearchMode,
    KjarniSearcherConfig, KjarniSearchOptions, KjarniSearchResults,
)


class SearchMode(Enum):
    """Search mode."""
    KEYWORD = KjarniSearchMode.KEYWORD
    SEMANTIC = KjarniSearchMode.SEMANTIC
    HYBRID = KjarniSearchMode.HYBRID


@dataclass
class SearchResult:
    """Single search result."""
    score: float
    document_id: int
    text: str
    metadata: Dict[str, Any]


class Searcher:
    """Document searcher for RAG applications.
    
    Example:
        >>> searcher = Searcher(model="minilm-l6-v2")
        >>> results = searcher.search("my_index", "What is Python?")
        >>> for r in results:
        ...     print(f"{r.score:.4f}: {r.text[:50]}...")
    """

    def __init__(
        self,
        model: str = "minilm-l6-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        rerank_model: Optional[str] = None,
        default_mode: SearchMode = SearchMode.HYBRID,
        default_top_k: int = 10,
        quiet: bool = False,
    ):
        """Create a new Searcher.
        
        Args:
            model: Embedding model name (must match index)
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            rerank_model: Optional cross-encoder for reranking
            default_mode: Default search mode
            default_top_k: Default number of results
            quiet: Suppress progress output
        """
        config = _lib.kjarni_searcher_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.model_name = model.encode("utf-8")
        config.default_mode = default_mode.value
        config.default_top_k = default_top_k
        config.quiet = 1 if quiet else 0

        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")
        if rerank_model:
            config.rerank_model = rerank_model.encode("utf-8")

        self._handle = c_void_p()
        err = _lib.kjarni_searcher_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_searcher_free(self._handle)

    def search(
        self,
        index_path: str,
        query: str,
        mode: Optional[SearchMode] = None,
        top_k: Optional[int] = None,
        rerank: Optional[bool] = None,
        threshold: Optional[float] = None,
        source_pattern: Optional[str] = None,
        filter_key: Optional[str] = None,
        filter_value: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search an index.
        
        Args:
            index_path: Path to the index
            query: Search query
            mode: Search mode (None = use default)
            top_k: Number of results (None = use default)
            rerank: Use reranker (None = auto)
            threshold: Minimum score threshold
            source_pattern: Filter by source file pattern (glob)
            filter_key: Metadata key to filter on
            filter_value: Required metadata value
            
        Returns:
            List of SearchResult
        """
        options = _lib.kjarni_search_options_default()
        
        if mode is not None:
            options.mode = mode.value
        if top_k is not None:
            options.top_k = top_k
        if rerank is not None:
            options.use_reranker = 1 if rerank else 0
        if threshold is not None:
            options.threshold = threshold
        if source_pattern:
            options.source_pattern = source_pattern.encode("utf-8")
        if filter_key:
            options.filter_key = filter_key.encode("utf-8")
        if filter_value:
            options.filter_value = filter_value.encode("utf-8")

        results = KjarniSearchResults()
        err = _lib.kjarni_searcher_search_with_options(
            self._handle,
            index_path.encode("utf-8"),
            query.encode("utf-8"),
            byref(options),
            byref(results),
        )
        check_error(err)

        output = []
        raw_results = results.to_list()
        results.free()

        for r in raw_results:
            output.append(SearchResult(
                score=r["score"],
                document_id=r["document_id"],
                text=r["text"],
                metadata=r["metadata"],
            ))

        return output

    @staticmethod
    def search_keywords(
        index_path: str,
        query: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """Static keyword search (BM25) - no embedder needed.
        
        Args:
            index_path: Path to the index
            query: Search query
            top_k: Number of results
            
        Returns:
            List of SearchResult
        """
        results = KjarniSearchResults()
        err = _lib.kjarni_search_keywords(
            index_path.encode("utf-8"),
            query.encode("utf-8"),
            top_k,
            byref(results),
        )
        check_error(err)

        output = []
        raw_results = results.to_list()
        results.free()

        for r in raw_results:
            output.append(SearchResult(
                score=r["score"],
                document_id=r["document_id"],
                text=r["text"],
                metadata=r["metadata"],
            ))

        return output

    @property
    def has_reranker(self) -> bool:
        """Check if reranker is configured."""
        return _lib.kjarni_searcher_has_reranker(self._handle)

    @property
    def default_mode(self) -> SearchMode:
        """Get the default search mode."""
        mode = _lib.kjarni_searcher_default_mode(self._handle)
        return SearchMode(mode)

    @property
    def default_top_k(self) -> int:
        """Get the default number of results."""
        return _lib.kjarni_searcher_default_top_k(self._handle)