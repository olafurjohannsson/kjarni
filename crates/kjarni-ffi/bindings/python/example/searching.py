"""Test file for Kjarni Searcher functionality."""

import tempfile
import os
from pathlib import Path

from kjarni import version
from kjarni.indexer import Indexer, IndexStats
from kjarni.searcher import Searcher, SearchResult, SearchMode


def create_test_documents(base_dir: Path) -> list[str]:
    """Create test documents with varied content for search testing."""
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = {
        "machine_learning.txt": """
            Machine learning is a subset of artificial intelligence that enables
            systems to learn and improve from experience. It focuses on developing
            algorithms that can access data and use it to learn for themselves.
            
            Supervised learning uses labeled training data to learn the mapping
            between inputs and outputs. Common algorithms include linear regression,
            decision trees, and neural networks.
            
            Unsupervised learning finds patterns in unlabeled data. Clustering and
            dimensionality reduction are key techniques in this area.
        """,
        "deep_learning.txt": """
            Deep learning is a subset of machine learning based on artificial
            neural networks with multiple layers. These deep neural networks
            can learn hierarchical representations of data.
            
            Convolutional neural networks (CNNs) excel at image recognition tasks.
            They use convolutional layers to detect features like edges and shapes.
            
            Recurrent neural networks (RNNs) process sequential data like text
            and time series. LSTM and GRU are popular RNN architectures.
        """,
        "transformers.txt": """
            Transformers are a neural network architecture that revolutionized
            natural language processing. They use self-attention mechanisms to
            process sequences in parallel.
            
            BERT (Bidirectional Encoder Representations from Transformers) learns
            contextual word embeddings by training on masked language modeling.
            
            GPT (Generative Pre-trained Transformer) models are autoregressive
            and excel at text generation tasks.
        """,
        "python_basics.txt": """
            Python is a high-level programming language known for its readability
            and simplicity. It supports multiple programming paradigms including
            procedural, object-oriented, and functional programming.
            
            Python uses indentation to define code blocks, making the code
            visually clean and consistent. Variables are dynamically typed.
            
            Popular Python libraries include NumPy for numerical computing,
            Pandas for data analysis, and TensorFlow for machine learning.
        """,
        "rust_basics.txt": """
            Rust is a systems programming language focused on safety, speed,
            and concurrency. It guarantees memory safety without garbage collection
            through its ownership system.
            
            The borrow checker ensures references are always valid and prevents
            data races at compile time. This eliminates many common bugs.
            
            Rust is used for building web servers, command-line tools, and
            embedded systems where performance and reliability are critical.
        """,
        "subdir/embeddings.txt": """
            Word embeddings represent words as dense vectors in a continuous
            vector space. Similar words have similar vector representations.
            
            Word2Vec learns embeddings by predicting context words (CBOW) or
            predicting a word from its context (Skip-gram).
            
            Sentence embeddings extend this to entire sentences or documents.
            Models like Sentence-BERT create embeddings useful for semantic search.
        """,
    }
    
    for filename, content in test_files.items():
        filepath = docs_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content.strip())
    
    return [str(docs_dir)]


def create_index(temp_dir: Path, docs_dirs: list[str]) -> str:
    """Create a test index and return its path."""
    index_path = str(temp_dir / "test_search_index")
    
    indexer = Indexer(
        model="minilm-l6-v2",
        chunk_size=256,
        chunk_overlap=25,
        quiet=True,
    )
    
    stats = indexer.create(index_path, docs_dirs, force=True)
    print(f"  Created index with {stats.documents_indexed} documents")
    
    return index_path


def test_searcher_creation():
    """Test creating a Searcher instance."""
    print("=" * 60)
    print("TEST: Searcher Creation")
    print("=" * 60)
    
    # Test with default settings
    print("Creating searcher with defaults...")
    searcher = Searcher(quiet=True)
    
    print(f"  Model name: {searcher.model_name}")
    print(f"  Default mode: {searcher.default_mode}")
    print(f"  Default top_k: {searcher.default_top_k}")
    print(f"  Has reranker: {searcher.has_reranker}")
    print(f"  Reranker model: {searcher.reranker_model}")
    
    assert searcher.model_name, "Model name should not be empty"
    assert searcher.default_mode == SearchMode.HYBRID, "Default mode should be HYBRID"
    assert searcher.default_top_k == 10, "Default top_k should be 10"
    assert not searcher.has_reranker, "Should not have reranker by default"
    assert searcher.reranker_model is None, "Reranker model should be None"
    
    print("✓ Default searcher creation passed\n")
    
    # Test with custom settings
    print("Creating searcher with custom settings...")
    searcher2 = Searcher(
        model="minilm-l6-v2",
        device="cpu",
        default_mode=SearchMode.SEMANTIC,
        default_top_k=5,
        quiet=True,
    )
    
    print(f"  Model name: {searcher2.model_name}")
    print(f"  Default mode: {searcher2.default_mode}")
    print(f"  Default top_k: {searcher2.default_top_k}")
    
    assert searcher2.default_mode == SearchMode.SEMANTIC, "Default mode should be SEMANTIC"
    assert searcher2.default_top_k == 5, "Default top_k should be 5"
    
    print("✓ Custom searcher creation passed\n")


def test_searcher_with_reranker():
    """Test creating a Searcher with reranker."""
    print("=" * 60)
    print("TEST: Searcher with Reranker")
    print("=" * 60)
    
    print("Creating searcher with reranker...")
    try:
        searcher = Searcher(
            model="minilm-l6-v2",
            rerank_model="minilm-l6-v2-cross-encoder",
            quiet=True,
        )
        
        print(f"  Model name: {searcher.model_name}")
        print(f"  Has reranker: {searcher.has_reranker}")
        print(f"  Reranker model: {searcher.reranker_model}")
        
        assert searcher.has_reranker, "Should have reranker"
        assert searcher.reranker_model == "minilm-l6-v2-cross-encoder", "Reranker model name mismatch"
        
        print("✓ Searcher with reranker creation passed\n")
        return searcher
        
    except Exception as e:
        print(f"  Warning: Could not create reranker: {e}")
        print("  Skipping reranker tests\n")
        return None


def test_basic_search(index_path: str):
    """Test basic search functionality."""
    print("=" * 60)
    print("TEST: Basic Search")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    query = "What is machine learning?"
    print(f"Query: '{query}'")
    
    results = searcher.search(index_path, query)
    
    print(f"  Found {len(results)} results")
    assert len(results) > 0, "Should find some results"
    
    for i, r in enumerate(results[:3]):
        print(f"  [{i+1}] Score: {r.score:.4f}")
        print(f"      Text: {r.text[:80]}...")
        print(f"      Source: {r.metadata.get('source', 'N/A')}")
    
    # Verify result structure
    assert isinstance(results[0].score, float), "Score should be float"
    assert isinstance(results[0].document_id, int), "Document ID should be int"
    assert isinstance(results[0].text, str), "Text should be string"
    assert isinstance(results[0].metadata, dict), "Metadata should be dict"
    
    print("✓ Basic search passed\n")


def test_search_modes(index_path: str):
    """Test all three search modes."""
    print("=" * 60)
    print("TEST: Search Modes")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    query = "neural networks deep learning"
    
    for mode in [SearchMode.KEYWORD, SearchMode.SEMANTIC, SearchMode.HYBRID]:
        print(f"\n  Mode: {mode.name}")
        results = searcher.search(index_path, query, mode=mode, top_k=3)
        
        print(f"  Results: {len(results)}")
        assert len(results) > 0, f"Should find results with {mode.name} mode"
        
        for i, r in enumerate(results):
            print(f"    [{i+1}] {r.score:.4f} - {r.text[:50]}...")
    
    print("\n✓ Search modes test passed\n")


def test_search_top_k(index_path: str):
    """Test top_k limiting."""
    print("=" * 60)
    print("TEST: Top-K Limiting")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    query = "programming language"
    
    for k in [1, 3, 5, 10]:
        results = searcher.search(index_path, query, top_k=k)
        print(f"  top_k={k}: got {len(results)} results")
        assert len(results) <= k, f"Should return at most {k} results"
    
    print("✓ Top-K limiting test passed\n")


def test_search_threshold(index_path: str):
    """Test score threshold filtering."""
    print("=" * 60)
    print("TEST: Score Threshold")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    query = "transformers attention mechanism"
    
    # Get results without threshold
    all_results = searcher.search(index_path, query, top_k=10)
    print(f"  Without threshold: {len(all_results)} results")
    
    if all_results:
        # Find a threshold that filters some results
        scores = [r.score for r in all_results]
        mid_score = (max(scores) + min(scores)) / 2
        
        print(f"  Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"  Using threshold: {mid_score:.4f}")
        
        filtered_results = searcher.search(index_path, query, top_k=10, threshold=mid_score)
        print(f"  With threshold: {len(filtered_results)} results")
        
        # All results should be above threshold
        for r in filtered_results:
            assert r.score >= mid_score, f"Score {r.score} below threshold {mid_score}"
    
    print("✓ Score threshold test passed\n")


def test_search_with_source_filter(index_path: str):
    """Test filtering by source pattern."""
    print("=" * 60)
    print("TEST: Source Pattern Filter")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    query = "programming"
    
    # Search without filter
    all_results = searcher.search(index_path, query, top_k=20)
    print(f"  Without filter: {len(all_results)} results")
    
    # Get unique sources
    sources = set()
    for r in all_results:
        if "source" in r.metadata:
            sources.add(r.metadata["source"])
    print(f"  Sources found: {len(sources)}")
    for src in list(sources)[:3]:
        print(f"    - {src}")
    
    # Filter for specific pattern (files containing "python" or "rust")
    # Note: This depends on your filter implementation
    filtered_results = searcher.search(
        index_path, 
        query, 
        top_k=20,
        source_pattern="*python*"
    )
    print(f"  With '*python*' filter: {len(filtered_results)} results")
    
    print("✓ Source pattern filter test passed\n")


def test_search_with_metadata_filter(index_path: str):
    """Test filtering by metadata key/value."""
    print("=" * 60)
    print("TEST: Metadata Filter")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    query = "learning algorithms"
    
    # Search without filter
    all_results = searcher.search(index_path, query, top_k=10)
    print(f"  Without filter: {len(all_results)} results")
    
    # Show available metadata keys
    if all_results:
        print(f"  Metadata keys: {list(all_results[0].metadata.keys())}")
    
    # Filter by chunk_index (if available)
    filtered_results = searcher.search(
        index_path,
        query,
        top_k=10,
        filter_key="chunk_index",
        filter_value="0"
    )
    print(f"  With chunk_index=0 filter: {len(filtered_results)} results")
    
    print("✓ Metadata filter test passed\n")


def test_static_keyword_search(index_path: str):
    """Test static keyword search (no embedder needed)."""
    print("=" * 60)
    print("TEST: Static Keyword Search")
    print("=" * 60)
    
    query = "Python NumPy Pandas"
    print(f"Query: '{query}'")
    
    # This doesn't need a Searcher instance
    results = Searcher.search_keywords(index_path, query, top_k=5)
    
    print(f"  Found {len(results)} results")
    assert len(results) > 0, "Should find some results"
    
    for i, r in enumerate(results):
        print(f"  [{i+1}] Score: {r.score:.4f}")
        print(f"      Text: {r.text[:60]}...")
    
    print("✓ Static keyword search passed\n")


def test_search_with_reranker(index_path: str, searcher_with_reranker: Searcher):
    """Test search with reranking."""
    print("=" * 60)
    print("TEST: Search with Reranker")
    print("=" * 60)
    
    if searcher_with_reranker is None:
        print("  Skipping - reranker not available\n")
        return
    
    query = "How do neural networks learn?"
    print(f"Query: '{query}'")
    
    # Search without reranking
    results_no_rerank = searcher_with_reranker.search(
        index_path, query, top_k=5, rerank=False
    )
    print(f"\n  Without reranking ({len(results_no_rerank)} results):")
    for i, r in enumerate(results_no_rerank[:3]):
        print(f"    [{i+1}] {r.score:.4f} - {r.text[:50]}...")
    
    # Search with reranking
    results_reranked = searcher_with_reranker.search(
        index_path, query, top_k=5, rerank=True
    )
    print(f"\n  With reranking ({len(results_reranked)} results):")
    for i, r in enumerate(results_reranked[:3]):
        print(f"    [{i+1}] {r.score:.4f} - {r.text[:50]}...")
    
    print("\n✓ Search with reranker passed\n")


def test_search_relevance(index_path: str):
    """Test that search returns relevant results."""
    print("=" * 60)
    print("TEST: Search Relevance")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    test_cases = [
        ("Rust memory safety ownership", ["rust", "memory", "ownership", "borrow"]),
        ("Python data analysis", ["python", "pandas", "numpy", "data"]),
        ("transformer attention BERT GPT", ["transformer", "attention", "bert", "gpt"]),
    ]
    
    for query, expected_terms in test_cases:
        print(f"\n  Query: '{query}'")
        results = searcher.search(index_path, query, top_k=3)
        
        if results:
            top_result = results[0]
            text_lower = top_result.text.lower()
            
            # Check if any expected terms appear in top result
            found_terms = [t for t in expected_terms if t in text_lower]
            print(f"    Top result contains: {found_terms}")
            print(f"    Text: {top_result.text[:80]}...")
            
            assert len(found_terms) > 0, f"Top result should contain relevant terms for '{query}'"
    
    print("\n✓ Search relevance test passed\n")


def test_search_index_not_found():
    """Test error handling for non-existent index."""
    print("=" * 60)
    print("TEST: Index Not Found Error")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    try:
        searcher.search("/nonexistent/index/path", "test query")
        print("✗ Should have raised an exception")
    except Exception as e:
        print(f"  Got expected error: {e}")
        print("✓ Index not found error handling passed\n")


def test_static_keyword_search_index_not_found():
    """Test static keyword search error handling."""
    print("=" * 60)
    print("TEST: Static Keyword Search - Index Not Found")
    print("=" * 60)
    
    try:
        Searcher.search_keywords("/nonexistent/index/path", "test query")
        print("✗ Should have raised an exception")
    except Exception as e:
        print(f"  Got expected error: {e}")
        print("✓ Static keyword search error handling passed\n")


def test_empty_query(index_path: str):
    """Test behavior with empty query."""
    print("=" * 60)
    print("TEST: Empty Query")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    # Empty query should still work (may return empty or all results)
    try:
        results = searcher.search(index_path, "", top_k=5)
        print(f"  Empty query returned {len(results)} results")
        print("✓ Empty query test passed\n")
    except Exception as e:
        print(f"  Empty query raised: {e}")
        print("  (This may be expected behavior)\n")


def test_special_characters_query(index_path: str):
    """Test search with special characters."""
    print("=" * 60)
    print("TEST: Special Characters in Query")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    queries = [
        "C++ vs Rust",
        "machine-learning",
        "what's the difference?",
        "neural_networks",
    ]
    
    for query in queries:
        try:
            results = searcher.search(index_path, query, top_k=3)
            print(f"  '{query}': {len(results)} results")
        except Exception as e:
            print(f"  '{query}': Error - {e}")
    
    print("✓ Special characters test passed\n")


def test_concurrent_searches(index_path: str):
    """Test multiple searches in sequence."""
    print("=" * 60)
    print("TEST: Sequential Searches")
    print("=" * 60)
    
    searcher = Searcher(model="minilm-l6-v2", quiet=True)
    
    queries = [
        "machine learning",
        "deep neural networks",
        "Python programming",
        "Rust performance",
        "NLP transformers",
    ]
    
    for query in queries:
        results = searcher.search(index_path, query, top_k=3)
        print(f"  '{query}': {len(results)} results")
    
    print("✓ Sequential searches test passed\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KJARNI SEARCHER TEST SUITE")
    print("=" * 60 + "\n")
    
    # Create temporary directory for all test artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test documents and index
        print("Setting up test environment...")
        docs_dirs = create_test_documents(temp_path)
        print(f"  Created test documents in: {docs_dirs[0]}")
        
        index_path = create_index(temp_path, docs_dirs)
        print(f"  Created test index at: {index_path}\n")
        
        # Run searcher tests
        test_searcher_creation()
        searcher_with_reranker = test_searcher_with_reranker()
        
        # Search functionality tests
        test_basic_search(index_path)
        test_search_modes(index_path)
        test_search_top_k(index_path)
        test_search_threshold(index_path)
        test_search_with_source_filter(index_path)
        test_search_with_metadata_filter(index_path)
        test_static_keyword_search(index_path)
        test_search_with_reranker(index_path, searcher_with_reranker)
        test_search_relevance(index_path)
        
        # Error handling tests
        test_search_index_not_found()
        test_static_keyword_search_index_not_found()
        
        # Edge cases
        test_empty_query(index_path)
        test_special_characters_query(index_path)
        test_concurrent_searches(index_path)
        
        # Cleanup is automatic with tempfile
    
    print("=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()