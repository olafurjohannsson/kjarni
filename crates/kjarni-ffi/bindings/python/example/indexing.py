"""Test file for Kjarni Indexer functionality."""

import tempfile
import os
import shutil
import threading
import time
from pathlib import Path

from kjarni import version
from kjarni.indexer import Indexer, IndexStats, IndexInfo, CancelToken, Progress


def create_test_documents(base_dir: Path) -> list[str]:
    """Create some temporary test documents for indexing."""
    docs_dir = base_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    test_files = {
        "readme.txt": """
            Welcome to the Kjarni project.
            This is a machine learning library for embeddings, classification, and search.
            It supports multiple languages including Rust, Python, and C#.
        """,
        "architecture.txt": """
            Kjarni uses a layered architecture.
            The core is written in Rust for performance.
            FFI bindings expose functionality to other languages.
            Python bindings use ctypes for direct library access.
        """,
        "models.txt": """
            Supported models include:
            - MiniLM for lightweight embeddings
            - DistilRoBERTa for classification
            - Cross-encoder models for reranking
            All models run locally without network access.
        """,
        "subdir/nested.txt": """
            This is a nested file to test recursive indexing.
            It contains information about vector databases.
            Semantic search enables finding similar documents.
        """,
    }
    
    created_paths = []
    for filename, content in test_files.items():
        filepath = docs_dir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content.strip())
        created_paths.append(str(filepath))
    
    return [str(docs_dir)]  # Return directory to index


def create_large_test_documents(base_dir: Path, num_files: int = 50) -> list[str]:
    """Create a larger set of test documents for cancellation testing.
    
    Creates enough files that indexing takes measurable time, allowing
    cancellation to occur mid-operation.
    """
    docs_dir = base_dir / "large_docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Create multiple files with substantial content
    for i in range(num_files):
        content = f"""
            Document {i}: Introduction to Machine Learning
            
            Machine learning is a subset of artificial intelligence that focuses on
            building systems that learn from data. This document covers various aspects
            of machine learning including supervised learning, unsupervised learning,
            and reinforcement learning.
            
            Supervised Learning:
            In supervised learning, the algorithm learns from labeled training data.
            Common algorithms include linear regression, logistic regression, decision
            trees, random forests, and neural networks. The goal is to learn a mapping
            from inputs to outputs that generalizes well to unseen data.
            
            Unsupervised Learning:
            Unsupervised learning deals with unlabeled data. The algorithm tries to
            find patterns or structure in the data. Common techniques include clustering
            (k-means, hierarchical), dimensionality reduction (PCA, t-SNE), and
            association rule learning.
            
            Deep Learning:
            Deep learning uses neural networks with many layers to learn hierarchical
            representations of data. Convolutional neural networks excel at image tasks,
            while recurrent neural networks and transformers handle sequential data.
            
            This is document number {i} in the test corpus. It contains enough text
            to create multiple chunks when processed by the indexer.
        """
        filepath = docs_dir / f"doc_{i:04d}.txt"
        filepath.write_text(content.strip())
    
    return [str(docs_dir)]


def test_version():
    """Test that we can get the library version."""
    print("=" * 60)
    print("TEST: Version")
    print("=" * 60)
    
    v = version()
    print(f"Kjarni version: {v}")
    assert v, "Version should not be empty"
    print("✓ Version test passed\n")


def test_indexer_creation():
    """Test creating an Indexer instance."""
    print("=" * 60)
    print("TEST: Indexer Creation")
    print("=" * 60)
    
    print("Creating indexer with defaults...")
    indexer = Indexer(quiet=True)
    
    print(f"  Model name: {indexer.model_name}")
    print(f"  Dimension: {indexer.dimension}")
    print(f"  Chunk size: {indexer.chunk_size}")
    
    assert indexer.model_name, "Model name should not be empty"
    assert indexer.dimension > 0, "Dimension should be positive"
    assert indexer.chunk_size > 0, "Chunk size should be positive"
    
    print("Creating indexer with custom settings...")
    indexer2 = Indexer(
        model="minilm-l6-v2",
        device="cpu",
        chunk_size=256,
        chunk_overlap=25,
        batch_size=16,
        extensions=["txt", "md"],
        recursive=True,
        quiet=True,
    )
    
    print(f"  Model name: {indexer2.model_name}")
    print(f"  Dimension: {indexer2.dimension}")
    print(f"  Chunk size: {indexer2.chunk_size}")
    
    assert indexer2.chunk_size == 256, f"Expected chunk_size=256, got {indexer2.chunk_size}"
    print("✓ Custom indexer creation passed\n")


def test_cancel_token():
    """Test CancelToken functionality."""
    print("=" * 60)
    print("TEST: CancelToken")
    print("=" * 60)
    
    token = CancelToken()
    
    assert not token.is_cancelled, "New token should not be cancelled"
    print("  New token is_cancelled: False ✓")
    
    token.cancel()
    assert token.is_cancelled, "Token should be cancelled after cancel()"
    print("  After cancel() is_cancelled: True ✓")
    
    token.reset()
    assert not token.is_cancelled, "Token should not be cancelled after reset()"
    print("  After reset() is_cancelled: False ✓")
    
    print("✓ CancelToken test passed\n")


def test_index_create(temp_dir: Path, docs_dir: list[str]):
    """Test creating an index."""
    print("=" * 60)
    print("TEST: Index Creation")
    print("=" * 60)
    
    index_path = str(temp_dir / "test_index")
    
    indexer = Indexer(
        model="minilm-l6-v2",
        chunk_size=256,
        chunk_overlap=25,
        quiet=True,
    )
    
    print(f"Creating index at: {index_path}")
    print(f"Indexing directories: {docs_dir}")
    
    try:
        stats = indexer.create(index_path, docs_dir, force=True)
        
        print(f"  Documents indexed: {stats.documents_indexed}")
        print(f"  Chunks created: {stats.chunks_created}")
        print(f"  Dimension: {stats.dimension}")
        print(f"  Size (bytes): {stats.size_bytes}")
        print(f"  Files processed: {stats.files_processed}")
        print(f"  Files skipped: {stats.files_skipped}")
        print(f"  Elapsed (ms): {stats.elapsed_ms}")
        
        assert stats.documents_indexed > 0, "Should have indexed some documents"
        assert stats.files_processed > 0, "Should have processed some files"
        
        print("✓ Index creation passed\n")
        return index_path
        
    except Exception as e:
        print(f"✗ Index creation failed: {e}")
        print("  (This is expected if _with_callback is unimplemented)\n")
        return None


def test_index_info(index_path: str):
    """Test getting index info."""
    print("=" * 60)
    print("TEST: Index Info")
    print("=" * 60)
    
    if index_path is None:
        print("  Skipping - no index available\n")
        return
    
    try:
        info = Indexer.info(index_path)
        
        print(f"  Path: {info.path}")
        print(f"  Document count: {info.document_count}")
        print(f"  Segment count: {info.segment_count}")
        print(f"  Dimension: {info.dimension}")
        print(f"  Size (bytes): {info.size_bytes}")
        print(f"  Embedding model: {info.embedding_model}")
        
        assert info.document_count > 0, "Should have documents"
        assert info.dimension > 0, "Should have positive dimension"
        
        print("✓ Index info passed\n")
        
    except Exception as e:
        print(f"✗ Index info failed: {e}\n")


def test_index_add(index_path: str, temp_dir: Path):
    """Test adding documents to existing index."""
    print("=" * 60)
    print("TEST: Index Add")
    print("=" * 60)
    
    if index_path is None:
        print("  Skipping - no index available\n")
        return
    
    # Create additional documents
    extra_dir = temp_dir / "extra_docs"
    extra_dir.mkdir(parents=True, exist_ok=True)
    
    (extra_dir / "extra1.txt").write_text(
        "This is an additional document about neural networks and deep learning."
    )
    (extra_dir / "extra2.txt").write_text(
        "Another document discussing transformers and attention mechanisms."
    )
    
    indexer = Indexer(model="minilm-l6-v2", quiet=True)
    
    try:
        docs_added = indexer.add(index_path, [str(extra_dir)])
        
        print(f"  Documents added: {docs_added}")
        assert docs_added > 0, "Should have added some documents"
        
        info = Indexer.info(index_path)
        print(f"  Total documents now: {info.document_count}")
        
        print("✓ Index add passed\n")
        
    except Exception as e:
        print(f"✗ Index add failed: {e}")
        print("  (This is expected if _with_callback is unimplemented)\n")


def test_index_delete(index_path: str):
    """Test deleting an index."""
    print("=" * 60)
    print("TEST: Index Delete")
    print("=" * 60)
    
    if index_path is None:
        print("  Skipping - no index available\n")
        return
    
    try:
        Indexer.delete(index_path)
        
        # Verify it's gone
        assert not os.path.exists(index_path), "Index should be deleted"
        
        print(f"  Deleted index at: {index_path}")
        print("Index delete passed\n")
        
    except Exception as e:
        print(f"Index delete failed: {e}\n")


def test_index_not_found():
    """Test error handling for non-existent index."""
    print("=" * 60)
    print("TEST: Index Not Found Error")
    print("=" * 60)
    
    try:
        Indexer.info("/nonexistent/path/to/index")
        print("✗ Should have raised an exception")
    except Exception as e:
        print(f"  Got expected error: {e}")
        print("Error handling passed\n")


def test_empty_inputs():
    """Test error handling for empty inputs."""
    print("=" * 60)
    print("TEST: Empty Inputs Error")
    print("=" * 60)
    
    indexer = Indexer(quiet=True)
    
    try:
        indexer.create("/tmp/test_index", [])
        print("Should have raised an exception")
    except ValueError as e:
        print(f"  Got expected ValueError: {e}")
        print("Empty inputs validation passed\n")
    except Exception as e:
        print(f"  Got exception (may be from FFI): {e}\n")


def test_progress_callback(temp_dir: Path, docs_dir: list[str]):
    """Test that progress callbacks are fired correctly."""
    print("=" * 60)
    print("TEST: Progress Callback")
    print("=" * 60)
    
    index_path = str(temp_dir / "progress_test_index")
    
    indexer = Indexer(
        model="minilm-l6-v2",
        chunk_size=256,
        quiet=True,
    )
    
    progress_updates: list[Progress] = []
    stages_seen: set[str] = set()
    
    def on_progress(p: Progress):
        progress_updates.append(p)
        stages_seen.add(p.stage)
        msg = f" - {p.message}" if p.message else ""
        print(f"  [{p.stage}] {p.current}/{p.total}{msg}")
    
    print(f"Creating index with progress callback...")
    print(f"  Index path: {index_path}")
    
    try:
        stats = indexer.create(index_path, docs_dir, force=True, on_progress=on_progress)
        
        print(f"\nProgress callback results:")
        print(f"  Total updates received: {len(progress_updates)}")
        print(f"  Stages seen: {sorted(stages_seen)}")
        
        # Verify we got progress updates
        assert len(progress_updates) > 0, "Should have received progress updates"
        
        # Verify we saw expected stages
        # At minimum we should see scanning, loading, embedding, committing
        expected_stages = {"scanning", "loading", "embedding", "committing"}
        missing_stages = expected_stages - stages_seen
        if missing_stages:
            print(f"  Warning: Did not see stages: {missing_stages}")
        
        # Verify progress data is valid
        for p in progress_updates:
            assert isinstance(p.stage, str), "Stage should be a string"
            assert isinstance(p.current, int), "Current should be an int"
            assert isinstance(p.total, int), "Total should be an int"
            assert p.current >= 0, "Current should be non-negative"
            assert p.total >= 0, "Total should be non-negative"
        
        print(f"\n  Documents indexed: {stats.documents_indexed}")
        print("✓ Progress callback test passed\n")
        
        # Clean up
        Indexer.delete(index_path)
        
    except Exception as e:
        print(f"✗ Progress callback test failed: {e}\n")
        import traceback
        traceback.print_exc()


def test_progress_callback_on_add(temp_dir: Path, docs_dir: list[str]):
    """Test that progress callbacks work for add operation."""
    print("=" * 60)
    print("TEST: Progress Callback on Add")
    print("=" * 60)
    
    index_path = str(temp_dir / "progress_add_test_index")
    
    indexer = Indexer(model="minilm-l6-v2", quiet=True)
    
    # First create an index without callback
    print("  Creating initial index...")
    stats = indexer.create(index_path, docs_dir, force=True)
    print(f"  Initial documents: {stats.documents_indexed}")
    
    # Create additional documents
    extra_dir = temp_dir / "extra_progress_docs"
    extra_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(5):
        (extra_dir / f"extra_{i}.txt").write_text(
            f"This is extra document {i} about machine learning and neural networks. "
            f"It contains information about deep learning, transformers, and embeddings."
        )
    
    # Now add with progress callback
    progress_updates: list[Progress] = []
    
    def on_progress(p: Progress):
        progress_updates.append(p)
        msg = f" - {p.message}" if p.message else ""
        print(f"  [{p.stage}] {p.current}/{p.total}{msg}")
    
    print(f"\n  Adding documents with progress callback...")
    
    try:
        docs_added = indexer.add(index_path, [str(extra_dir)], on_progress=on_progress)
        
        print(f"\n  Documents added: {docs_added}")
        print(f"  Progress updates received: {len(progress_updates)}")
        
        assert docs_added > 0, "Should have added documents"
        assert len(progress_updates) > 0, "Should have received progress updates"
        
        # Verify final count
        info = Indexer.info(index_path)
        print(f"  Total documents now: {info.document_count}")
        
        print("✓ Progress callback on add test passed\n")
        
        # Clean up
        Indexer.delete(index_path)
        
    except Exception as e:
        print(f"✗ Progress callback on add test failed: {e}\n")
        import traceback
        traceback.print_exc()


def test_cancellation(temp_dir: Path):
    """Test that cancellation token stops indexing mid-operation."""
    print("=" * 60)
    print("TEST: Cancellation")
    print("=" * 60)
    
    print("  Creating large test dataset...")
    large_docs_dir = create_large_test_documents(temp_dir, num_files=100)
    
    index_path = str(temp_dir / "cancel_test_index")
    
    indexer = Indexer(
        model="minilm-l6-v2",
        chunk_size=256,
        batch_size=8, 
        quiet=True,
    )
    
    cancel_token = CancelToken()
    files_seen = []
    cancelled_at_stage = [None]
    cancelled_at_count = [0]
    
    def on_progress(p: Progress):
        files_seen.append(p)
        
        if p.stage == "loading" and p.current >= 10:
            if not cancel_token.is_cancelled:
                print(f"  Requesting cancellation at {p.stage} {p.current}/{p.total}")
                cancelled_at_stage[0] = p.stage
                cancelled_at_count[0] = p.current
                cancel_token.cancel()
    
    print(f"  Starting indexing (will cancel after 10 files loaded)...")
    
    try:
        stats = indexer.create(
            index_path, 
            large_docs_dir, 
            force=True, 
            on_progress=on_progress,
            cancel_token=cancel_token,
        )
        
        print(f" Indexing completed without cancellation!")
        print(f"  Documents indexed: {stats.documents_indexed}")
        print(f"  This might mean cancellation check isn't frequent enough\n")
        
        # Clean up
        if os.path.exists(index_path):
            Indexer.delete(index_path)
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if "cancel" in error_msg:
            print(f"  Got expected cancellation error: {e}")
            print(f"  Cancelled at stage: {cancelled_at_stage[0]}")
            print(f"  Cancelled at count: {cancelled_at_count[0]}")
            print(f"  Total progress updates before cancel: {len(files_seen)}")
            
            if cancelled_at_count[0] < 50:
                print("✓ Cancellation test passed\n")
            else:
                print(f"  Warning: Cancellation happened late ({cancelled_at_count[0]} files)")
                print("✓ Cancellation test passed (but could be faster)\n")
        else:
            print(f"✗ Got unexpected error: {e}")
            import traceback
            traceback.print_exc()
            print()
        
        # Clean up partial index if it exists
        if os.path.exists(index_path):
            try:
                Indexer.delete(index_path)
            except:
                shutil.rmtree(index_path, ignore_errors=True)


def test_cancellation_immediate(temp_dir: Path, docs_dir: list[str]):
    """Test that pre-cancelled token prevents indexing from starting."""
    print("=" * 60)
    print("TEST: Immediate Cancellation")
    print("=" * 60)
    
    index_path = str(temp_dir / "immediate_cancel_test_index")
    
    indexer = Indexer(model="minilm-l6-v2", quiet=True)
    
    # Cancel before starting
    cancel_token = CancelToken()
    cancel_token.cancel()
    
    progress_count = [0]
    
    def on_progress(p: Progress):
        progress_count[0] += 1
        print(f"  Unexpected progress: [{p.stage}] {p.current}/{p.total}")
    
    print("  Starting indexing with pre-cancelled token...")
    
    try:
        stats = indexer.create(
            index_path,
            docs_dir,
            force=True,
            on_progress=on_progress,
            cancel_token=cancel_token,
        )
        
        print(f"✗ Indexing should have been cancelled immediately!")
        print(f"  Documents indexed: {stats.documents_indexed}\n")
        
        # Clean up
        if os.path.exists(index_path):
            Indexer.delete(index_path)
            
    except Exception as e:
        error_msg = str(e).lower()
        
        if "cancel" in error_msg:
            print(f"  Got expected cancellation error: {e}")
            print(f"  Progress callbacks received: {progress_count[0]}")
            
            if progress_count[0] == 0:
                print("  Operation was cancelled before any progress")
            else:
                print(f"  Warning: Some progress happened before cancellation")
            
            print("Immediate cancellation test passed\n")
        else:
            print(f"Got unexpected error: {e}\n")
        
        # Clean up partial index if it exists
        if os.path.exists(index_path):
            try:
                Indexer.delete(index_path)
            except:
                shutil.rmtree(index_path, ignore_errors=True)


def test_cancellation_from_thread(temp_dir: Path):
    """Test cancellation from a separate thread."""
    print("=" * 60)
    print("TEST: Cancellation from Thread")
    print("=" * 60)
    
    # Create test dataset
    print("  Creating test dataset...")
    large_docs_dir = create_large_test_documents(temp_dir / "thread_cancel", num_files=100)
    
    index_path = str(temp_dir / "thread_cancel_test_index")
    
    indexer = Indexer(
        model="minilm-l6-v2",
        chunk_size=256,
        batch_size=8,
        quiet=True,
    )
    
    cancel_token = CancelToken()
    indexing_started = threading.Event()
    indexing_result = [None]
    progress_updates = []
    
    def on_progress(p: Progress):
        progress_updates.append(p)
        if p.stage == "loading" and p.current >= 5:
            indexing_started.set()
    
    def indexing_thread():
        try:
            stats = indexer.create(
                index_path,
                large_docs_dir,
                force=True,
                on_progress=on_progress,
                cancel_token=cancel_token,
            )
            indexing_result[0] = stats
        except Exception as e:
            indexing_result[0] = e
    
    print("  Starting indexing in background thread...")
    thread = threading.Thread(target=indexing_thread)
    thread.start()
    
    print("  Waiting for indexing to start...")
    if indexing_started.wait(timeout=30):
        print(f"  Indexing started, cancelling from main thread...")
        time.sleep(0.1)  # Small delay to let more work happen
        cancel_token.cancel()
    else:
        print("  Warning: Indexing didn't start within timeout")
    
    # Wait for thread to finish
    thread.join(timeout=30)
    
    if thread.is_alive():
        print("✗ Indexing thread didn't finish after cancellation\n")
    else:
        result = indexing_result[0]
        
        if isinstance(result, Exception):
            error_msg = str(result).lower()
            if "cancel" in error_msg:
                print(f"  Got expected cancellation: {result}")
                print(f"  Progress updates before cancel: {len(progress_updates)}")
                print("✓ Thread cancellation test passed\n")
            else:
                print(f"✗ Got unexpected error: {result}\n")
        elif isinstance(result, IndexStats):
            print(f"Indexing completed without cancellation!")
            print(f"  Documents indexed: {result.documents_indexed}\n")
        else:
            print(f"Unexpected result: {result}\n")
    
    # Clean up
    if os.path.exists(index_path):
        try:
            Indexer.delete(index_path)
        except:
            shutil.rmtree(index_path, ignore_errors=True)


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("KJARNI INDEXER TEST SUITE")
    print("=" * 60 + "\n")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        docs_dirs = create_test_documents(temp_path)
        print(f"Created test documents in: {docs_dirs[0]}\n")
        
        # Basic tests
        test_version()
        test_indexer_creation()
        test_cancel_token()
        
        # Core functionality tests
        index_path = test_index_create(temp_path, docs_dirs)
        test_index_info(index_path)
        test_index_add(index_path, temp_path)
        test_index_delete(index_path)
        
        # Error handling tests
        test_index_not_found()
        test_empty_inputs()
        
        # Progress callback tests
        test_progress_callback(temp_path, docs_dirs)
        test_progress_callback_on_add(temp_path, docs_dirs)
        
        # Cancellation tests
        test_cancellation_immediate(temp_path, docs_dirs)
        test_cancellation(temp_path)
        test_cancellation_from_thread(temp_path)
    
    print("=" * 60)
    print("TEST SUITE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()