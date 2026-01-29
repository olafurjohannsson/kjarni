"""High-level Indexer API for Kjarni.

This module provides a Pythonic interface to the Kjarni indexer, which creates
and manages vector indexes for RAG (Retrieval Augmented Generation) applications.

Example:
    >>> from kjarni import Indexer
    >>> 
    >>> # Create an indexer
    >>> indexer = Indexer(model="minilm-l6-v2")
    >>> 
    >>> # Create an index from documents
    >>> stats = indexer.create("my_index", ["docs/"])
    >>> print(f"Indexed {stats.documents_indexed} documents")
    >>> 
    >>> # Get index info
    >>> info = Indexer.info("my_index")
    >>> print(f"Index has {info.document_count} documents")
    >>> 
    >>> # Add more documents
    >>> added = indexer.add("my_index", ["more_docs/"])
    >>> print(f"Added {added} documents")
    >>> 
    >>> # Delete index when done
    >>> Indexer.delete("my_index")
"""

from typing import Optional, List, Sequence, Callable, NamedTuple
from ctypes import c_void_p, c_char_p, c_size_t, c_int32, byref, POINTER, create_string_buffer
from dataclasses import dataclass

from ._ffi import (
    _lib, check_error, KjarniDevice, KjarniProgressStage,
    KjarniIndexerConfig, KjarniIndexStats, KjarniIndexInfo,
    KjarniProgress, PROGRESS_CALLBACK_TYPE,
)


@dataclass
class IndexStats:
    """Statistics returned after indexing operations.
    
    Attributes:
        documents_indexed: Number of document chunks indexed (after splitting)
        chunks_created: Number of chunks created from source files
        dimension: Embedding dimension used
        size_bytes: Total index size on disk
        files_processed: Number of source files successfully processed
        files_skipped: Number of files skipped (errors, unsupported, too large)
        elapsed_ms: Total time taken in milliseconds
    """
    documents_indexed: int
    chunks_created: int
    dimension: int
    size_bytes: int
    files_processed: int
    files_skipped: int
    elapsed_ms: int


@dataclass
class IndexInfo:
    """Information about an existing index.
    
    Attributes:
        path: Path to the index directory
        document_count: Total number of documents in the index
        segment_count: Number of index segments
        dimension: Embedding dimension
        size_bytes: Total size on disk
        embedding_model: Name of embedding model used (may be None)
    """
    path: str
    document_count: int
    segment_count: int
    dimension: int
    size_bytes: int
    embedding_model: Optional[str]


class Progress(NamedTuple):
    """Progress update during indexing operations.
    
    Attributes:
        stage: Current operation stage (scanning, loading, embedding, etc.)
        current: Current item number being processed
        total: Total items to process (may be 0 if unknown)
        message: Optional status message
    """
    stage: str
    current: int
    total: int
    message: Optional[str]


class CancelToken:
    """Token to cancel long-running operations.
    
    Create a token and pass it to indexing operations. Call `cancel()` from
    another thread to request cancellation.
    
    Example:
        >>> token = CancelToken()
        >>> # In another thread:
        >>> token.cancel()
        >>> # The indexing operation will stop and raise an exception
    """
    
    def __init__(self):
        """Create a new cancellation token."""
        self._handle = _lib.kjarni_cancel_token_new()
    
    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_cancel_token_free(self._handle)
    
    def cancel(self):
        """Request cancellation of the associated operation."""
        _lib.kjarni_cancel_token_cancel(self._handle)
    
    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return _lib.kjarni_cancel_token_is_cancelled(self._handle)
    
    def reset(self):
        """Reset the token for reuse with another operation."""
        _lib.kjarni_cancel_token_reset(self._handle)


def _stage_name(stage: int) -> str:
    """Convert FFI stage enum to human-readable string."""
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
    
    The Indexer creates and manages vector indexes from text documents.
    Documents are split into chunks, embedded using a sentence transformer
    model, and stored in an efficient vector index for similarity search.
    
    Args:
        model: Embedding model name (default: "minilm-l6-v2")
        device: Compute device - "cpu" or "gpu" (default: "cpu")
        cache_dir: Directory to cache downloaded models (default: system cache)
        chunk_size: Maximum chunk size in characters (default: 512)
        chunk_overlap: Overlap between chunks in characters (default: 50)
        batch_size: Batch size for embedding operations (default: 32)
        extensions: File extensions to include (default: common text formats)
        exclude_patterns: Glob patterns for files to exclude
        recursive: Whether to recurse into subdirectories (default: True)
        include_hidden: Whether to include hidden files (default: False)
        max_file_size: Skip files larger than this in bytes (default: 10MB)
        quiet: Suppress progress output to stderr (default: False)
    
    Example:
        >>> indexer = Indexer(model="minilm-l6-v2", chunk_size=256)
        >>> stats = indexer.create("my_index", ["documents/"])
        >>> print(f"Created index with {stats.documents_indexed} chunks")
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
        
        Scans the input paths for supported text files, splits them into
        chunks, generates embeddings, and stores everything in a vector index.
        
        Args:
            index_path: Path where the index will be created
            inputs: List of file or directory paths to index
            force: If True, overwrite existing index at index_path
            on_progress: Optional callback for progress updates
            cancel_token: Optional token to cancel the operation
            
        Returns:
            IndexStats with information about what was indexed
            
        Raises:
            ValueError: If inputs is empty
            KjarniException: If indexing fails
            
        Example:
            >>> def show_progress(p):
            ...     print(f"{p.stage}: {p.current}/{p.total}")
            >>> stats = indexer.create("idx", ["docs/"], on_progress=show_progress)
        """
        if not inputs:
            raise ValueError("No input paths specified")

        # Prepare input array
        c_inputs = (c_char_p * len(inputs))()
        for i, inp in enumerate(inputs):
            c_inputs[i] = inp.encode("utf-8")

        stats = KjarniIndexStats()

        # Decide which FFI function to call based on whether callbacks are needed
        if on_progress is None and cancel_token is None:
            # Simple path: no callbacks, call the simple FFI function
            err = _lib.kjarni_indexer_create(
                self._handle,
                index_path.encode("utf-8"),
                c_inputs,
                len(inputs),
                1 if force else 0,
                byref(stats),
            )
        else:
            # Callback path: need to set up callbacks
            callback_ref = None  # Keep reference to prevent GC
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
                
                # Create ctypes callback - must keep reference!
                callback_ref = PROGRESS_CALLBACK_TYPE(_callback_wrapper)
                c_callback = callback_ref

            c_cancel = cancel_token._handle if cancel_token else None

            err = _lib.kjarni_indexer_create_with_callback(
                self._handle,
                index_path.encode("utf-8"),
                c_inputs,
                len(inputs),
                1 if force else 0,
                c_callback,
                None,  # user_data not needed, closure captures on_progress
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
        
        Processes additional files and appends them to an existing index.
        The index must have been created with the same embedding model.
        
        Args:
            index_path: Path to existing index
            inputs: List of file or directory paths to add
            on_progress: Optional callback for progress updates
            cancel_token: Optional token to cancel the operation
            
        Returns:
            Number of document chunks added
            
        Raises:
            KjarniException: If the index doesn't exist or has incompatible settings
            
        Example:
            >>> added = indexer.add("my_index", ["new_docs/"])
            >>> print(f"Added {added} document chunks")
        """
        if not inputs:
            return 0

        # Prepare input array
        c_inputs = (c_char_p * len(inputs))()
        for i, inp in enumerate(inputs):
            c_inputs[i] = inp.encode("utf-8")

        docs_added = c_size_t()

        # Decide which FFI function to call
        if on_progress is None and cancel_token is None:
            # Simple path
            err = _lib.kjarni_indexer_add(
                self._handle,
                index_path.encode("utf-8"),
                c_inputs,
                len(inputs),
                byref(docs_added),
            )
        else:
            # Callback path
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
        
        This is a static method that doesn't require an Indexer instance.
        
        Args:
            index_path: Path to the index directory
            
        Returns:
            IndexInfo with index statistics and metadata
            
        Raises:
            KjarniException: If the index doesn't exist or is corrupted
            
        Example:
            >>> info = Indexer.info("my_index")
            >>> print(f"Index has {info.document_count} documents")
            >>> print(f"Using model: {info.embedding_model}")
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

        # Free the C strings
        _lib.kjarni_index_info_free(info)
        return result

    @staticmethod
    def delete(index_path: str):
        """Delete an index.
        
        Permanently removes the index directory and all its contents.
        This is a static method that doesn't require an Indexer instance.
        
        Args:
            index_path: Path to the index to delete
            
        Raises:
            KjarniException: If the index doesn't exist or deletion fails
            
        Example:
            >>> Indexer.delete("my_index")
        """
        err = _lib.kjarni_index_delete(index_path.encode("utf-8"))
        check_error(err)

    @property
    def model_name(self) -> str:
        """Get the embedding model name used by this indexer."""
        # First call to get required size
        required = _lib.kjarni_indexer_model_name(self._handle, None, 0)
        if required == 0:
            return ""
        
        # Allocate buffer and get the string
        buf = create_string_buffer(required)
        _lib.kjarni_indexer_model_name(self._handle, buf, required)
        return buf.value.decode("utf-8")

    @property
    def dimension(self) -> int:
        """Get the embedding dimension produced by the model."""
        return _lib.kjarni_indexer_dimension(self._handle)

    @property
    def chunk_size(self) -> int:
        """Get the configured chunk size in characters."""
        return _lib.kjarni_indexer_chunk_size(self._handle)