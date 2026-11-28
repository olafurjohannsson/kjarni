import array
import unittest

from _support import PHOTOS, SPEECH_WAV, TempDir, heavy
from kjarni import (
    ImageIndex,
    KjarniException,
    Segment,
    Transcriber,
    Transcript,
    segments_to_srt,
    segments_to_vtt,
)


class SubtitleFormattingTest(unittest.TestCase):
    def test_empty_segments(self):
        self.assertEqual(segments_to_srt([]), "")
        self.assertEqual(segments_to_vtt([]), "WEBVTT\n\n")

    def test_short_and_empty_segments(self):
        segs = [
            Segment(0.0, 1.5, "  Hello there. "),
            Segment(1.5, 2.0, "  \n\t  "),
            Segment(61.25, 3725.004, " General\n\tKenobi. "),
        ]
        self.assertEqual(
            segments_to_srt(segs),
            "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n"
            "2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n",
        )
        self.assertEqual(
            segments_to_vtt(segs),
            "WEBVTT\n\n"
            "00:00:00.000 --> 00:00:01.500\nHello there.\n\n"
            "00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n",
        )

    def test_wraps_over_42_chars_into_two_lines(self):
        segs = [
            Segment(
                0.0,
                4.0,
                "This is a moderately long subtitle line that should wrap onto two lines.",
            )
        ]
        self.assertEqual(
            segments_to_srt(segs),
            "1\n00:00:00,000 --> 00:00:04,000\n"
            "This is a moderately long subtitle line\n"
            "that should wrap onto two lines.\n\n",
        )

    def test_splits_very_long_segment_proportionally(self):
        segs = [
            Segment(
                10.0,
                20.0,
                "Alpha bravo charlie delta echo foxtrot golf "
                "hotel india juliet kilo lima mike november "
                "oscar papa quebec.",
            )
        ]
        self.assertEqual(
            segments_to_srt(segs),
            "1\n00:00:10,000 --> 00:00:17,404\n"
            "Alpha bravo charlie delta echo foxtrot\n"
            "golf hotel india juliet kilo lima mike\n\n"
            "2\n00:00:17,404 --> 00:00:20,000\n"
            "november oscar papa quebec.\n\n",
        )

    def test_single_word_over_42_chars_stays_intact(self):
        long_word = "supercalifragilisticexpialidociousandmorewords"
        segs = [Segment(0.0, 6.0, f"Prefix {long_word} suffix")]
        self.assertEqual(
            segments_to_srt(segs),
            f"1\n00:00:00,000 --> 00:00:05,390\nPrefix\n{long_word}\n\n"
            "2\n00:00:05,390 --> 00:00:06,000\nsuffix\n\n",
        )

    def test_unicode_character_count(self):
        line = "Þórður sagði halló við alla í Reykjavík í"
        self.assertEqual(len(line), 41)
        segs = [Segment(0.0, 3.0, f"{line} dag.")]
        self.assertEqual(
            segments_to_srt(segs),
            f"1\n00:00:00,000 --> 00:00:03,000\n{line}\ndag.\n\n",
        )

    def test_timestamp_rounding_and_bounds(self):
        segs = [
            Segment(0.0, 0.9996, "Rounding up."),
            Segment(float("nan"), float("inf"), "Non-finite."),
            Segment(-2.5, -1.0, "Negative."),
            Segment(5.0, 3.0, "Reversed."),
        ]
        self.assertEqual(
            segments_to_srt(segs),
            "1\n00:00:00,000 --> 00:00:01,000\nRounding up.\n\n"
            "2\n00:00:00,000 --> 00:00:00,000\nNon-finite.\n\n"
            "3\n00:00:00,000 --> 00:00:00,000\nNegative.\n\n"
            "4\n00:00:05,000 --> 00:00:05,000\nReversed.\n\n",
        )

    def test_vtt_escapes_arrow(self):
        segs = [Segment(0.0, 2.0, "From A --> B and x-->y")]
        self.assertEqual(
            segments_to_vtt(segs),
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFrom A -- > B and x-- >y\n\n",
        )

    def test_transcript_methods_and_fallback(self):
        transcript = Transcript("  Just full text. ", "en", 4.5, [])
        self.assertEqual(
            transcript.to_srt(),
            "1\n00:00:00,000 --> 00:00:04,500\nJust full text.\n\n",
        )
        self.assertEqual(
            transcript.to_vtt(),
            "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust full text.\n\n",
        )
        empty = Transcript("   ", "en", 1.0, [])
        self.assertEqual(empty.to_srt(), "")
        self.assertEqual(empty.to_vtt(), "WEBVTT\n\n")


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
