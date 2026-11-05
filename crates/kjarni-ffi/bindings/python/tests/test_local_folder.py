"""A model loaded from a local folder gives what the same model gives by name.

The folder is the one the named model downloads to, so loading it by path must
change nothing. Models from the Hugging Face Hub load the same way.
"""

import os
import sys
import unittest
from pathlib import Path

from _support import TempDir  # noqa: F401  (puts the package on sys.path)
from kjarni import Classifier, Embedder, Reranker


def cache_dir() -> Path:
    """Where kjarni keeps models, as the Rust side decides it."""
    if os.environ.get("KJARNI_CACHE_DIR"):
        return Path(os.environ["KJARNI_CACHE_DIR"])
    if sys.platform == "win32":
        return Path(os.environ["LOCALAPPDATA"]) / "kjarni"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "kjarni"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "kjarni"


class LocalFolderTest(unittest.TestCase):
    def test_embedder_from_folder_matches_by_name(self):
        by_name = Embedder(model="minilm-l6-v2", quiet=True).encode("hello world")
        folder = cache_dir() / "sentence-transformers_all-MiniLM-L6-v2"
        by_path = Embedder(model_path=str(folder), quiet=True).encode("hello world")
        self.assertEqual(len(by_path), 384)
        for a, b in zip(by_name, by_path):
            self.assertAlmostEqual(a, b, places=6)

    def test_classifier_from_folder_matches_by_name(self):
        text = "The box arrived crushed."
        by_name = Classifier(model="distilbert-sentiment", quiet=True).classify(text)
        folder = cache_dir() / "distilbert_distilbert-base-uncased-finetuned-sst-2-english"
        by_path = Classifier(model_path=folder, quiet=True).classify(text)
        self.assertEqual(by_path.label, by_name.label)
        self.assertAlmostEqual(by_path.score, by_name.score, places=6)

    def test_reranker_from_folder_matches_by_name(self):
        q, d = "how do I reset my password", "Click Forgot password on the sign-in page."
        by_name = Reranker(quiet=True).score(q, d)
        folder = cache_dir() / "cross-encoder_ms-marco-MiniLM-L-6-v2"
        by_path = Reranker(model_path=str(folder), quiet=True).score(q, d)
        self.assertAlmostEqual(by_path, by_name, places=5)

    def test_classifier_needs_a_name_or_a_folder(self):
        with self.assertRaises(ValueError):
            Classifier()

    def test_a_folder_that_is_not_a_model_is_an_error(self):
        with TempDir() as empty:
            with self.assertRaises(Exception) as ctx:
                Embedder(model_path=str(empty), quiet=True)
            self.assertIn("config.json", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
