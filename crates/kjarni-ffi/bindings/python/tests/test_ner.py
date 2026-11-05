"""Named-entity recognition through the Python binding.

The entities are what transformers' pipeline("ner", aggregation_strategy=...)
gives for dslim/distilbert-NER on the CPU.
"""

import unittest

from _support import TempDir, heavy  # noqa: F401  (puts the package on sys.path)
from kjarni import EntityRecognizer

TEXT = "Sarah Connor met Apple's CEO in San Francisco on Tuesday."


@heavy
class EntityRecognizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ner = EntityRecognizer(quiet=True)

    def test_entities_match_transformers(self):
        got = self.ner.recognize(TEXT)
        want = [
            ("PER", "Sarah Connor", 0, 12, 0.99879),
            ("ORG", "Apple", 17, 22, 0.99710),
            ("LOC", "San Francisco", 32, 45, 0.99761),
        ]
        self.assertEqual([(e.label, e.text, e.start, e.end) for e in got], [w[:4] for w in want])
        for e, w in zip(got, want):
            self.assertAlmostEqual(e.score, w[4], places=4)

    def test_spans_index_the_python_string(self):
        text = "Þórður hitti Angela Merkel í Berlín."
        for e in self.ner.recognize(text):
            self.assertEqual(text[e.start:e.end], e.text)

    def test_no_entities_is_an_empty_list(self):
        self.assertEqual(self.ner.recognize("nothing to see here"), [])


class EntityRecognizerArgumentsTest(unittest.TestCase):
    def test_an_unknown_aggregation_is_refused(self):
        with self.assertRaises(ValueError):
            EntityRecognizer(aggregation="word")

    def test_a_folder_that_is_not_a_model_is_an_error(self):
        with TempDir() as empty:
            with self.assertRaises(Exception):
                EntityRecognizer(model_path=str(empty), quiet=True)


if __name__ == "__main__":
    unittest.main()
