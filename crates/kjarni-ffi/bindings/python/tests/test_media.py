import array
import unittest

from _support import PHOTOS, SPEECH_WAV, TempDir, heavy
from kjarni import ImageIndex, Transcriber, KjarniException


@heavy
class TranscriberTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transcriber = Transcriber(language="en", quiet=True)

    def test_transcribes_a_wav_file(self):
        t = self.transcriber.transcribe_file(SPEECH_WAV)
        self.assertTrue(t.text.strip())
        self.assertEqual(t.language, "en")
        self.assertGreater(t.duration_secs, 0)
        self.assertTrue(t.segments)
        for s in t.segments:
            self.assertGreaterEqual(s.end, s.start)

    def test_pcm_from_a_list_and_a_buffer(self):
        # A second of silence at 8 kHz exercises resampling; both input kinds must
        # reach the engine as the same samples.
        from_list = self.transcriber.transcribe([0.0] * 8000, 8000)
        from_buffer = self.transcriber.transcribe(array.array("f", [0.0] * 8000), 8000)
        self.assertEqual(from_list.text, from_buffer.text)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            self.transcriber.transcribe([0.0] * 100, 0)
        with self.assertRaises(KjarniException):
            self.transcriber.transcribe_file("/no/such/file.wav")


@heavy
class ImageIndexTest(unittest.TestCase):
    CASES = [
        ("a sandy beach under a blue sky", "beach_holiday"),
        ("a dense green forest with trees", "deep_forest"),
        ("a red car", "red_car"),
        ("a snowy mountain peak", "snowy_mountain"),
    ]

    @classmethod
    def setUpClass(cls):
        cls.index = ImageIndex()
        cls.report = cls.index.add_directory(PHOTOS)

    def test_scan(self):
        self.assertGreaterEqual(self.report.added, 4)
        self.assertEqual(self.report.skipped, 0)
        self.assertEqual(len(self.index), self.report.added)

    def test_descriptions_find_the_right_photo(self):
        for query, expected in self.CASES:
            with self.subTest(query=query):
                hits = self.index.search(query, top_k=2)
                self.assertEqual(len(hits), 2)
                self.assertIn(expected, hits[0].path)

    def test_save_and_load(self):
        with TempDir() as d:
            saved = d / "photos.idx"
            self.index.save(saved)
            restored = ImageIndex()
            self.assertEqual(restored.load(saved), len(self.index))
            self.assertIn("red_car", restored.search("a red car", 1)[0].path)

    def test_missing_photo(self):
        with self.assertRaises(KjarniException):
            self.index.add("/no/such/photo.jpg")


if __name__ == "__main__":
    unittest.main()
