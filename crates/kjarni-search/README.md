# kjarni-search

Search primitives for [Kjarni](https://github.com/olafurjohannsson/kjarni), a
transformer inference library: a BM25 keyword index (`Bm25Index`), an exact vector
store (`VectorStore`), and hybrid search that combines the two by reciprocal rank
fusion.

Most programs want the `kjarni` crate instead, whose `Indexer` and `Searcher` build
on these, embed the documents and keep the index on disk. This crate is for using
the pieces directly.

MIT or Apache-2.0.
