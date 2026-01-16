"""High-level Embedder API."""
import numpy as np
from typing import Optional, List, Sequence
from ctypes import c_void_p, c_char_p, c_float, byref, POINTER

from ._ffi import (
    _lib, check_error, KjarniDevice,
    KjarniEmbedderConfig, KjarniFloatArray, KjarniFloat2DArray
)

class Embedder:
    """Text embedding model.
    
    Example:
        >>> embedder = Embedder()
        >>> embedding = embedder.encode("Hello, world!")
        >>> print(len(embedding))
        384
        
        >>> sim = embedder.similarity("cat", "dog")
        >>> print(f"Similarity: {sim:.3f}")
    """

    def __init__(
        self,
        model: Optional[str] = None,
        device: str = "cpu",
        cache_dir: Optional[str] = None,
        normalize: bool = True,
        quiet: bool = False,
    ):
        """Create a new Embedder.
        
        Args:
            model: Model name (e.g., "minilm-l6-v2", "mpnet-base-v2")
            device: "cpu" or "gpu"
            cache_dir: Directory to cache models
            normalize: L2-normalize embeddings
            quiet: Suppress progress output
        """
        config = _lib.kjarni_embedder_config_default()
        config.device = KjarniDevice.GPU if device == "gpu" else KjarniDevice.CPU
        config.normalize = 1 if normalize else 0
        config.quiet = 1 if quiet else 0

        if model:
            config.model_name = model.encode("utf-8")
        if cache_dir:
            config.cache_dir = cache_dir.encode("utf-8")

        self._handle = c_void_p()
        err = _lib.kjarni_embedder_new(byref(config), byref(self._handle))
        check_error(err)

    def __del__(self):
        if hasattr(self, '_handle') and self._handle:
            _lib.kjarni_embedder_free(self._handle)

    def encode(self, text: str) -> List[float]:
        """Encode a single text to an embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector as list of floats
        """
        result = KjarniFloatArray()

        err = _lib.kjarni_embedder_encode(
            self._handle,
            text.encode("utf-8"),
            byref(result)
        )
        check_error(err)

        embedding = result.to_list()
        result.free()
        return embedding

    def encode_batch(self, texts: Sequence[str]) -> np.ndarray:
        """Encode multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Convert to C array
        c_texts = (c_char_p * len(texts))()
        for i, t in enumerate(texts):
            c_texts[i] = t.encode("utf-8")

        result = KjarniFloat2DArray()
        err = _lib.kjarni_embedder_encode_batch(
            self._handle,
            c_texts,
            len(texts),
            byref(result)
        )
        check_error(err)

        embeddings = result.to_numpy()
        result.free()
        return embeddings

    def similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        result = c_float()
        err = _lib.kjarni_embedder_similarity(
            self._handle,
            text1.encode("utf-8"),
            text2.encode("utf-8"),
            byref(result)
        )
        check_error(err)
        return result.value

    @property
    def dim(self) -> int:
        """Get the embedding dimension."""
        return _lib.kjarni_embedder_dim(self._handle)