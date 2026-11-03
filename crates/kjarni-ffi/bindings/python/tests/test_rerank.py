import re
import unittest

from _support import TempDir  # noqa: F401  (puts the package on sys.path)
from kjarni import Reranker

QUERY = "how do I get my money back?"
DOCS = [
    "Our refund policy allows returns within 30 days of purchase.",
    "The store opens at nine in the morning.",
    "Shipping is free on orders over fifty dollars.",
]


class RerankerTest(unittest.TestCase):
    def test_model_named_in_the_docstring_loads(self):
        documented = re.search(r'"([^"]+)"', Reranker.__init__.__doc__).group(1)
        self.assertEqual(documented, "minilm-l6-v2-cross-encoder")
        reranker = Reranker(model=documented, quiet=True)
        ranked = reranker.rerank(QUERY, DOCS)
        self.assertEqual([r.index for r in ranked], [0, 2, 1])
        # sentence-transformers' CrossEncoder gives the same scores for these pairs.
        for result, expected in zip(ranked, [-6.4270, -10.7962, -11.2367]):
            self.assertAlmostEqual(result.score, expected, places=3)


if __name__ == "__main__":
    unittest.main()
