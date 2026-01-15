"""High-level Indexer API."""

from typing import Optional, List, Sequence, Callable, NamedTuple
from ctypes import c_void_p, c_char_p, c_size_t, c_int32, byref, POINTER
from dataclasses import dataclass
import threading

from ._ffi import (
    _lib, check_error, KjarniDevice, KjarniProgressStage,
    KjarniIndexerConfig, KjarniIndexStats, KjarniIndexInfo,
    KjarniProgress, PROGRESS_CALLBACK_TYPE,
)


@dataclass
class IndexStats:
    """Statistics returned after indexing."""
    documents_indexed: int
    chunks_created: int
    dimension: int
    size_bytes: int
    files_processed: int
    files_skipped: int
    elapsed_ms: int


@dataclass
class IndexInfo:
    """Information about an existing index."""
    path: str
    document_count: int
    segment_count: int
    dimension: int
    size_bytes: int
    embedding_model: Optional[str]


class Progress(NamedTuple):
    """Progress update during indexing."""
    stage: str
    current: int
    total: int
    message: Optional[str]


class CancelToken:
    """Token to cancel long-running operations."""
    
    def __init__(self):
        self._handle = _lib.kjarni_cancel_token_new()
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_cancel_token_free(self._handle)
    
    def cancel(self):
        """Cancel the operation."""
        _lib.kjarni_cancel_token_cancel(self._handle)
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancelled."""
        return _lib.kjarni_cancel_token_is_cancelled(self._handle)
    
    def reset(self):
        """Reset the token for reuse."""
        _lib.kjarni_cancel_token_reset(self._handle)


def _stage_name(stage: int) -> str:
    """Convert stage enum to string."""
    names = {
        KjarniProgressStage.SCANNING: "scanning",
        KjarniProgressStage.LOADING: "loading",
        KjarniProgressStage.EMBEDDING: "embedding",
        KjarniProgressStage.WRITING: "writing",
        KjarniProgressStage.COMMITTING: "committing",
        KjarniProgressStage.SEARCHING: "searching",
        KjarniProgressStage.RERANKING: "reranking",
    }
    return names.get(stage, "unknown")


class Indexer:
    """Document indexer for RAG applications.
    
    Example:
        >>> indexer = Indexer(model="minilm-l6-v2")
        >>> stats = indexer.create("my_index", ["docs/"])
        >>> print(f"Indexed {stats.documents_indexed} documents")
    """

    def __init__(
        self,
        model: str = "minilm-l6-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        batch_size: int = 32,
        extensions: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        recursive: bool = True,
        include_hidden: bool = False,
        max_file_size: int = 10 * 1024 * 1024,
        quiet: bool = False,
    ):
        """Create a new Indexer.
        
        Args:
            model: Embedding model name
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            chunk_size: Maximum chunk size in characters
            chunk_overlap: Overlap between chunks
            batch_size: Batch size for embedding
            extensions: File extensions to include (None = defaults)
            exclude_patterns: Glob patterns to exclude
            recursive: Recurse into subdirectories
            include_hidden: Include hidden files
            max_file_size: Skip files larger than this (bytes)
            quiet: Suppress progress output
        """
        config = _lib.kjarni_indexer_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.model_name = model.encode("utf-8")
        config.chunk_size = chunk_size
        config.chunk_overlap = chunk_overlap
        config.batch_size = batch_size
        config.recursive = 1 if recursive else 0
        config.include_hidden = 1 if include_hidden else 0
        config.max_file_size = max_file_size
        config.quiet = 1 if quiet else 0

        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")
        if extensions:
            config.extensions = ",".join(extensions).encode("utf-8")
        if exclude_patterns:
            config.exclude_patterns = ",".join(exclude_patterns).encode("utf-8")

        self._handle = c_void_p()
        err = _lib.kjarni_indexer_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_indexer_free(self._handle)

    def create(
        self,
        index_path: str,
        inputs: Sequence[str],
        force: bool = False,
        on_progress: Optional[Callable[[Progress], None]] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> IndexStats:
        """Create a new index from files/directories.
        
        Args:
            index_path: Path for the new index
            inputs: List of files/directories to index
            force: Overwrite existing index
            on_progress: Optional progress callback
            cancel_token: Optional cancellation token
            
        Returns:
            IndexStats with indexing statistics
        """
        if not inputs:
            raise ValueError("No input paths specified")

        c_inputs = (c_char_p * len(inputs))()
        for i, inp in enumerate(inputs):
            c_inputs[i] = inp.encode("utf-8")

        stats = KjarniIndexStats()

        # Create callback wrapper if provided
        callback_ref = None  # Keep reference to prevent GC
        c_callback = None
        c_user_data = None
        
        if on_progress is not None:
            def _callback_wrapper(progress: KjarniProgress, user_data):
                msg = progress.message.decode("utf-8") if progress.message else None
                py_progress = Progress(
                    stage=_stage_name(progress.stage),
                    current=progress.current,
                    total=progress.total,
                    message=msg,
                )
                on_progress(py_progress)
            
            callback_ref = PROGRESS_CALLBACK_TYPE(_callback_wrapper)
            c_callback = callback_ref
            c_user_data = None

        c_cancel = cancel_token._handle if cancel_token else None

        err = _lib.kjarni_indexer_create_with_callback(
            self._handle,
            index_path.encode("utf-8"),
            c_inputs,
            len(inputs),
            1 if force else 0,
            c_callback,
            c_user_data,
            c_cancel,
            byref(stats),
        )
        check_error(err)

        return IndexStats(
            documents_indexed=stats.documents_indexed,
            chunks_created=stats.chunks_created,
            dimension=stats.dimension,
            size_bytes=stats.size_bytes,
            files_processed=stats.files_processed,
            files_skipped=stats.files_skipped,
            elapsed_ms=stats.elapsed_ms,
        )

    def add(
        self,
        index_path: str,
        inputs: Sequence[str],
        on_progress: Optional[Callable[[Progress], None]] = None,
        cancel_token: Optional[CancelToken] = None,
    ) -> int:
        """Add documents to an existing index.
        
        Args:
            index_path: Path to existing index
            inputs: List of files/directories to add
            on_progress: Optional progress callback
            cancel_token: Optional cancellation token
            
        Returns:
            Number of documents added
        """
        if not inputs:
            return 0

        c_inputs = (c_char_p * len(inputs))()
        for i, inp in enumerate(inputs):
            c_inputs[i] = inp.encode("utf-8")

        docs_added = c_size_t()

        callback_ref = None
        c_callback = None
        
        if on_progress is not None:
            def _callback_wrapper(progress: KjarniProgress, user_data):
                msg = progress.message.decode("utf-8") if progress.message else None
                py_progress = Progress(
                    stage=_stage_name(progress.stage),
                    current=progress.current,
                    total=progress.total,
                    message=msg,
                )
                on_progress(py_progress)
            
            callback_ref = PROGRESS_CALLBACK_TYPE(_callback_wrapper)
            c_callback = callback_ref

        c_cancel = cancel_token._handle if cancel_token else None

        err = _lib.kjarni_indexer_add_with_callback(
            self._handle,
            index_path.encode("utf-8"),
            c_inputs,
            len(inputs),
            c_callback,
            None,
            c_cancel,
            byref(docs_added),
        )
        check_error(err)

        return docs_added.value

    @staticmethod
    def info(index_path: str) -> IndexInfo:
        """Get information about an existing index.
        
        Args:
            index_path: Path to the index
            
        Returns:
            IndexInfo with index statistics
        """
        info = KjarniIndexInfo()
        err = _lib.kjarni_index_info(index_path.encode("utf-8"), byref(info))
        check_error(err)

        result = IndexInfo(
            path=info.path.decode("utf-8") if info.path else "",
            document_count=info.document_count,
            segment_count=info.segment_count,
            dimension=info.dimension,
            size_bytes=info.size_bytes,
            embedding_model=info.embedding_model.decode("utf-8") if info.embedding_model else None,
        )

        _lib.kjarni_index_info_free(info)
        return result

    @staticmethod
    def delete(index_path: str):
        """Delete an index.
        
        Args:
            index_path: Path to the index to delete
        """
        err = _lib.kjarni_index_delete(index_path.encode("utf-8"))
        check_error(err)

    @property
    def model_name(self) -> str:
        """Get the embedding model name."""
        name = _lib.kjarni_indexer_model_name(self._handle)
        return name.decode("utf-8") if name else ""

    @property
    def dimension(self) -> int:
        """Get the embedding dimension."""
        return _lib.kjarni_indexer_dimension(self._handle)

    @property
    def chunk_size(self) -> int:
        """Get the chunk size."""
        return _lib.kjarni_indexer_chunk_size(self._handle)