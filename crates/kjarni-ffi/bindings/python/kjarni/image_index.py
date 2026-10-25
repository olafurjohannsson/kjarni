"""High-level image search API."""

from typing import List, NamedTuple, Optional
from ctypes import c_void_p, c_char_p, c_int, c_float, c_size_t, byref, POINTER, Structure

from ._ffi import _lib, check_error, encode_or_none


class KjarniImageHit(Structure):
    _fields_ = [
        ("path", c_char_p),
        ("score", c_float),
    ]


class KjarniImageResults(Structure):
    _fields_ = [
        ("hits", POINTER(KjarniImageHit)),
        ("len", c_size_t),
    ]


class KjarniScanReport(Structure):
    _fields_ = [
        ("added", c_size_t),
        ("skipped", c_size_t),
    ]


# Absent from a library built with --no-default-features; the import then fails
# with AttributeError, which ImageIndex reports as unavailable.
try:
    _lib.kjarni_image_index_new.restype = c_int
    _lib.kjarni_image_index_new.argtypes = [c_char_p, POINTER(c_void_p)]
    _lib.kjarni_image_index_free.restype = None
    _lib.kjarni_image_index_free.argtypes = [c_void_p]
    _lib.kjarni_image_index_add.restype = c_int
    _lib.kjarni_image_index_add.argtypes = [c_void_p, c_char_p]
    _lib.kjarni_image_index_add_directory.restype = c_int
    _lib.kjarni_image_index_add_directory.argtypes = [c_void_p, c_char_p, POINTER(KjarniScanReport)]
    _lib.kjarni_image_index_search.restype = c_int
    _lib.kjarni_image_index_search.argtypes = [c_void_p, c_char_p, c_size_t, POINTER(KjarniImageResults)]
    _lib.kjarni_image_results_free.restype = None
    _lib.kjarni_image_results_free.argtypes = [KjarniImageResults]
    _lib.kjarni_image_index_save.restype = c_int
    _lib.kjarni_image_index_save.argtypes = [c_void_p, c_char_p]
    _lib.kjarni_image_index_load.restype = c_int
    _lib.kjarni_image_index_load.argtypes = [c_void_p, c_char_p, POINTER(c_size_t)]
    _lib.kjarni_image_index_len.restype = c_size_t
    _lib.kjarni_image_index_len.argtypes = [c_void_p]
    _AVAILABLE = True
except AttributeError:
    _AVAILABLE = False


class ImageHit(NamedTuple):
    """An image matching a search."""
    path: str
    #: Cosine similarity; small even for good matches, so only the order matters.
    score: float


class ScanReport(NamedTuple):
    """What scanning a directory did."""
    added: int
    #: Files that could not be decoded.
    skipped: int


class ImageIndex:
    """Photos searched by description, with CLIP.

    Example:
        >>> index = ImageIndex()
        >>> index.add_directory("~/Pictures")
        >>> for hit in index.search("a dog on a beach", top_k=5):
        ...     print(hit.path)
    """

    def __init__(self, model_dir: Optional[str] = None):
        """Load a CLIP model.

        Args:
            model_dir: A CLIP model directory; None uses clip-vit-base-32,
                downloaded on first use
        """
        if not _AVAILABLE:
            raise RuntimeError("this Kjarni library was built without image support")
        self._handle = c_void_p()
        check_error(_lib.kjarni_image_index_new(encode_or_none(model_dir and str(model_dir)), byref(self._handle)))

    def __del__(self):
        if getattr(self, "_handle", None):
            _lib.kjarni_image_index_free(self._handle)
            self._handle = None

    def __len__(self) -> int:
        return _lib.kjarni_image_index_len(self._handle)

    def add(self, path: str) -> None:
        """Index one JPEG or PNG."""
        check_error(_lib.kjarni_image_index_add(self._handle, str(path).encode("utf-8")))

    def add_directory(self, directory: str) -> ScanReport:
        """Index every JPEG and PNG under `directory`."""
        report = KjarniScanReport()
        check_error(_lib.kjarni_image_index_add_directory(
            self._handle, str(directory).encode("utf-8"), byref(report)))
        return ScanReport(report.added, report.skipped)

    def search(self, query: str, top_k: int = 10) -> List[ImageHit]:
        """The `top_k` images best matching `query`."""
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        results = KjarniImageResults()
        check_error(_lib.kjarni_image_index_search(self._handle, query.encode("utf-8"), top_k, byref(results)))
        try:
            return [
                ImageHit(h.path.decode("utf-8") if h.path else "", h.score)
                for h in (results.hits[i] for i in range(results.len))
            ]
        finally:
            _lib.kjarni_image_results_free(results)

    def save(self, path: str) -> None:
        """Write the vectors (not the images) to `path`."""
        check_error(_lib.kjarni_image_index_save(self._handle, str(path).encode("utf-8")))

    def load(self, path: str) -> int:
        """Restore vectors written by `save`; returns how many."""
        count = c_size_t()
        check_error(_lib.kjarni_image_index_load(self._handle, str(path).encode("utf-8"), byref(count)))
        return count.value
