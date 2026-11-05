"""A model loaded from a local folder gives what the same model gives by name.

The folder is the one the named model downloads to, so loading it by path must
change nothing. Models from the Hugging Face Hub load the same way.
"""

import os
import sys
import unittest
from pathlib import Path

from _support import TempDir, heavy  # noqa: F401  (puts the package on sys.path)
from _support import SPEECH_WAV
from kjarni import Chat, Classifier, Embedder, KjarniException, Reranker, Summarizer, Transcriber, Translator
from kjarni._ffi import KjarniError


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


def hf_snapshot(repo: str):
    """The folder of `repo` in the Hugging Face cache, if it is there."""
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    snapshots = home / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    found = sorted(snapshots.glob("*/config.json")) if snapshots.exists() else []
    return found[0].parent if found else None


@heavy
class ChatFromFolderTest(unittest.TestCase):
    QUESTION = "What is the capital of Iceland?"

    @classmethod
    def setUpClass(cls):
        cls.by_name = Chat("qwen2.5-0.5b-instruct", quiet=True).send(
            cls.QUESTION, do_sample=False, max_new_tokens=24)

    def test_chat_from_the_download_folder_matches_by_name(self):
        folder = cache_dir() / "Qwen_Qwen2.5-0.5B-Instruct"
        chat = Chat(model_path=folder, quiet=True)
        self.assertIn("Reykjav", self.by_name)
        self.assertEqual(chat.send(self.QUESTION, do_sample=False, max_new_tokens=24), self.by_name)

    def test_chat_from_a_hub_folder_uses_its_template_and_matches_by_name(self):
        folder = hf_snapshot("Qwen/Qwen2.5-0.5B-Instruct")
        if folder is None:
            self.skipTest("Qwen/Qwen2.5-0.5B-Instruct is not in the Hugging Face cache")
        chat = Chat(model_path=str(folder), quiet=True)
        self.assertEqual(chat.send(self.QUESTION, do_sample=False, max_new_tokens=24), self.by_name)

    def test_chat_needs_a_name_or_a_folder(self):
        with self.assertRaises(ValueError):
            Chat()

    def test_chat_from_an_encoder_folder_is_an_error(self):
        folder = cache_dir() / "distilbert_distilbert-base-uncased-finetuned-sst-2-english"
        with self.assertRaises(Exception) as ctx:
            Chat(model_path=folder, quiet=True)
        self.assertIn("not suitable for chat", str(ctx.exception))


def folder_of(path: Path, model_type: str) -> Path:
    """A folder whose config.json says it holds a `model_type` model."""
    (path / "config.json").write_text('{"model_type": "%s"}' % model_type)
    return path


class Seq2SeqFolderErrorsTest(unittest.TestCase):
    def test_a_folder_of_the_wrong_family_is_invalid_config(self):
        cases = [
            (lambda p: Summarizer(model_path=p, quiet=True), "marian"),
            (lambda p: Translator(model_path=p, quiet=True), "bart"),
            (lambda p: Transcriber(model_path=p, quiet=True), "bart"),
        ]
        for make, model_type in cases:
            with TempDir() as d:
                with self.assertRaises(KjarniException) as e:
                    make(folder_of(d, model_type))
                self.assertEqual(e.exception.code, KjarniError.INVALID_CONFIG)


@heavy
class Seq2SeqFromFolderTest(unittest.TestCase):
    def test_translator_from_the_download_folder_matches_by_name(self):
        text = "The committee postponed its decision until next spring."
        by_name = Translator(model="opus-mt-en-de", quiet=True).translate(text, "en", "de")
        folder = cache_dir() / "kjarni-ai_opus-mt-en-de"
        by_path = Translator(model_path=folder, quiet=True).translate(text, "en", "de")
        self.assertEqual(by_path, by_name)

    def test_summarizer_from_the_download_folder_matches_by_name(self):
        from test_generation import ARTICLE
        by_name = Summarizer(quiet=True).summarize(ARTICLE)
        folder = cache_dir() / "olafuraron_distilbart-cnn-12-6"
        self.assertEqual(Summarizer(model_path=folder, quiet=True).summarize(ARTICLE), by_name)

    def test_transcriber_from_the_download_folder_matches_by_name(self):
        by_name = Transcriber(language="en", quiet=True).transcribe_file(SPEECH_WAV)
        folder = cache_dir() / "openai_whisper-small"
        by_path = Transcriber(model_path=folder, language="en", quiet=True).transcribe_file(SPEECH_WAV)
        self.assertEqual(by_path.text, by_name.text)

    def test_translator_from_a_hub_marian_folder_picks_its_target(self):
        folder = hf_snapshot("Helsinki-NLP/opus-mt-en-ROMANCE")
        if folder is None:
            self.skipTest("Helsinki-NLP/opus-mt-en-ROMANCE is not in the Hugging Face cache")
        t = Translator(model_path=folder, source="en", quiet=True)
        # transformers 4.57, MarianMTModel.generate with the model's own config
        self.assertEqual(t.translate("Where is the library?", target="es"), "¿Dónde está la biblioteca?")
        self.assertEqual(t.translate("The weather is nice today.", target="French"), "Le temps est beau aujourd'hui.")
        with self.assertRaises(KjarniException) as e:
            t.translate("Hello", target="de")
        self.assertEqual(e.exception.code, KjarniError.INVALID_CONFIG)


if __name__ == "__main__":
    unittest.main()
