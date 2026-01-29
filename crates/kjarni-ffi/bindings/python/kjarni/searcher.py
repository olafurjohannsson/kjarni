"""High-level Searcher API for Kjarni.

This module provides a Pythonic interface to the Kjarni searcher, which performs
semantic, keyword, and hybrid search over vector indexes created by the Indexer.

Example:
    >>> from kjarni import Searcher, SearchMode
    >>> 
    >>> # Create a searcher
    >>> searcher = Searcher(model="minilm-l6-v2")
    >>> 
    >>> # Search an index
    >>> results = searcher.search("my_index", "What is machine learning?")
    >>> for r in results:
    ...     print(f"{r.score:.4f}: {r.text[:50]}...")
    >>> 
    >>> # Search with options
    >>> results = searcher.search(
    ...     "my_index",
    ...     "neural networks",
    ...     mode=SearchMode.SEMANTIC,
    ...     top_k=5,
    ... )
"""

from typing import Optional, List, Dict, Any
from ctypes import c_void_p, c_char_p, c_size_t, c_int32, c_float, c_bool, byref, create_string_buffer
from dataclasses import dataclass
from enum import Enum

from ._ffi import (
    _lib, check_error, KjarniDevice, KjarniSearchMode,
    KjarniSearcherConfig, KjarniSearchOptions, KjarniSearchResults,
)


class SearchMode(Enum):
    """Search mode determining how queries are processed.
    
    Attributes:
        KEYWORD: BM25-based text matching only
        SEMANTIC: Embedding-based vector similarity only
        HYBRID: Combined keyword + semantic with reciprocal rank fusion (recommended)
    """
    KEYWORD = KjarniSearchMode.KEYWORD
    SEMANTIC = KjarniSearchMode.SEMANTIC
    HYBRID = KjarniSearchMode.HYBRID


@dataclass
class SearchResult:
    """A single search result.
    
    Attributes:
        score: Relevance score (higher is better, scale varies by search mode)
        document_id: Document ID within the index
        text: The matched text chunk
        metadata: Associated metadata (source file, chunk index, custom fields)
    """
    score: float
    document_id: int
    text: str
    metadata: Dict[str, Any]


class Searcher:
    """Document searcher for RAG applications.
    
    The Searcher performs queries against vector indexes created by the Indexer.
    It supports three search modes:
    
    - **Keyword**: BM25 text matching, good for exact term matching
    - **Semantic**: Vector similarity, good for conceptual matching
    - **Hybrid**: Combined approach using reciprocal rank fusion (recommended)
    
    Optionally, a cross-encoder reranker can be configured to improve result
    quality by re-scoring the top candidates.
    
    Args:
        model: Embedding model name (must match the model used to create the index)
        device: Compute device - "cpu" or "gpu" (default: "cpu")
        cache_dir: Directory to cache downloaded models (default: system cache)
        rerank_model: Optional cross-encoder model for reranking results
        default_mode: Default search mode (default: HYBRID)
        default_top_k: Default number of results (default: 10)
        quiet: Suppress progress output (default: False)
    
    Example:
        >>> # Basic searcher
        >>> searcher = Searcher(model="minilm-l6-v2")
        >>> 
        >>> # Searcher with reranking
        >>> searcher = Searcher(
        ...     model="minilm-l6-v2",
        ...     rerank_model="ms-marco-MiniLM-L-6-v2",
        ... )
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
            index_path: Path to the index directory
            query: Search query string
            mode: Search mode (None = use default)
            top_k: Number of results (None = use default)
            rerank: Use reranker (None = auto, use if configured)
            threshold: Minimum score threshold (results below this are filtered)
            source_pattern: Filter by source file pattern (glob syntax, e.g. "*.md")
            filter_key: Metadata key to filter on
            filter_value: Required value for filter_key
            
        Returns:
            List of SearchResult objects, sorted by relevance (highest first)
            
        Raises:
            KjarniException: If the index doesn't exist or search fails
            
        Example:
            >>> # Basic search
            >>> results = searcher.search("my_index", "machine learning")
            >>> 
            >>> # Search with options
            >>> results = searcher.search(
            ...     "my_index",
            ...     "neural networks",
            ...     mode=SearchMode.SEMANTIC,
            ...     top_k=5,
            ...     threshold=0.5,
            ... )
            >>> 
            >>> # Search with filtering
            >>> results = searcher.search(
            ...     "my_index",
            ...     "API reference",
            ...     source_pattern="*.md",
            ... )
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
        """Static keyword search (BM25) - no embedding model needed.
        
        This is a convenience method for pure keyword search that doesn't
        require loading an embedding model. Useful for quick text matching
        or when you only need BM25 results.
        
        Args:
            index_path: Path to the index directory
            query: Search query string
            top_k: Maximum number of results (default: 10)
            
        Returns:
            List of SearchResult objects, sorted by BM25 score
            
        Raises:
            KjarniException: If the index doesn't exist or search fails
            
        Example:
            >>> # Quick keyword search without loading embedder
            >>> results = Searcher.search_keywords("my_index", "Python tutorial")
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
        """Check if a reranker is configured."""
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

    @property
    def model_name(self) -> str:
        """Get the embedding model name used by this searcher."""
        required = _lib.kjarni_searcher_model_name(self._handle, None, 0)
        if required == 0:
            return ""
        
        buf = create_string_buffer(required)
        _lib.kjarni_searcher_model_name(self._handle, buf, required)
        return buf.value.decode("utf-8")

    @property
    def reranker_model(self) -> Optional[str]:
        """Get the reranker model name, or None if no reranker is configured."""
        required = _lib.kjarni_searcher_reranker_model(self._handle, None, 0)
        if required == 0:
            return None
        
        buf = create_string_buffer(required)
        _lib.kjarni_searcher_reranker_model(self._handle, buf, required)
        return buf.value.decode("utf-8")