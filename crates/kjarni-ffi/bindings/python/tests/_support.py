"""Shared fixtures for the binding tests.

Tests that need a model of several hundred megabytes skip when KJARNI_SKIP_HEAVY
is set, so CI can run the rest without downloading gigabytes on every job.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SPEECH_WAV = ROOT / "crates" / "kjarni" / "tests" / "fixtures" / "speech.wav"
PHOTOS = ROOT / "tests" / "fixtures" / "clip" / "photos"

heavy = unittest.skipIf(os.environ.get("KJARNI_SKIP_HEAVY"), "KJARNI_SKIP_HEAVY is set")


class TempDir:
    """A scratch directory removed on exit."""

    def __enter__(self) -> Path:
        self.path = Path(tempfile.mkdtemp(prefix="kjarni_py_"))
        return self.path

    def __exit__(self, *exc):
        shutil.rmtree(self.path, ignore_errors=True)
