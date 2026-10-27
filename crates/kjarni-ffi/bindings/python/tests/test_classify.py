import unittest

from _support import TempDir  # noqa: F401  (puts the package on sys.path)
from kjarni import Classifier

TEXTS = [
    "I love this product, it's absolutely amazing!",
    "Worst purchase of my life.",
    "It is fine I guess.",
    "Quick delivery, but the colour is not what the photos showed and setup took ages.",
]


class ClassifyBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier = Classifier(model="distilbert-sentiment", quiet=True)

    def test_batch_matches_classifying_each_text_alone(self):
        batch = self.classifier.classify_batch(TEXTS)
        self.assertEqual(len(batch), len(TEXTS))
        for text, b in zip(TEXTS, batch):
            single = self.classifier.classify(text)
            self.assertEqual(b.label, single.label, text)
            self.assertAlmostEqual(b.score, single.score, places=5, msg=text)
            self.assertEqual([l for l, _ in b.all_scores], [l for l, _ in single.all_scores])

    def test_empty_batch(self):
        self.assertEqual(self.classifier.classify_batch([]), [])

    def test_sentiment(self):
        labels = [r.label for r in self.classifier.classify_batch(TEXTS[:2])]
        self.assertEqual(labels, ["POSITIVE", "NEGATIVE"])


if __name__ == "__main__":
    unittest.main()
