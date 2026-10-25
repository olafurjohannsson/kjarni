import unittest

from _support import TempDir
from kjarni import Indexer, Searcher, SearchMode, KjarniException


class IndexAndSearchTest(unittest.TestCase):
    def test_index_add_and_search(self):
        with TempDir() as docs:
            (docs / "cats.txt").write_text("Cats are small carnivorous mammals that purr and chase mice.")
            (docs / "rust.md").write_text("Rust is a systems programming language focused on memory safety.")
            (docs / "bread.txt").write_text("Sourdough bread is leavened with a starter of wild yeast.")
            (docs / "skip.log").write_text("This file has an extension that is not indexed.")
            index = str(docs / "index")

            indexer = Indexer(extensions=["txt", "md"], quiet=True)
            stats = indexer.create(index, [str(docs)])
            self.assertEqual(stats.documents_indexed, 3)
            self.assertEqual(stats.dimension, 384)

            with self.assertRaises(KjarniException):
                indexer.create(index, [str(docs)])
            indexer.create(index, [str(docs)], force=True)

            (docs / "tea.txt").write_text("Green tea is brewed from unoxidised leaves of the tea plant.")
            self.assertGreaterEqual(indexer.add(index, [str(docs / "tea.txt")]), 1)

            searcher = Searcher(quiet=True)
            self.assertFalse(searcher.has_reranker)

            hits = searcher.search(index, "feline pets")
            self.assertTrue(hits)
            self.assertIn("cats", hits[0].text.lower())
            self.assertIn("cats.txt", str(hits[0].metadata))

            semantic = searcher.search(index, "a drink made from leaves", mode=SearchMode.SEMANTIC, top_k=1)
            self.assertEqual(len(semantic), 1)
            self.assertIn("tea", semantic[0].text.lower())

            keyword = searcher.search(index, "sourdough", mode=SearchMode.KEYWORD)
            self.assertTrue(keyword)
            self.assertIn("sourdough", keyword[0].text.lower())

            filtered = searcher.search(index, "programming", source_pattern="*.md")
            self.assertTrue(filtered)
            for r in filtered:
                self.assertIn("rust.md", str(r.metadata))

            with self.assertRaises(KjarniException):
                searcher.search(str(docs / "nope"), "x")


if __name__ == "__main__":
    unittest.main()
