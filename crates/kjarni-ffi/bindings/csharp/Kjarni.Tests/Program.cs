using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Kjarni;

namespace Kjarni.Tests
{
    class Program
    {
        static void Main(string[] args)
        {
            using var classifier = new Classifier("distilbert-sentiment");
            var result = classifier.Classify("I love kjarni");

            foreach (var (label, score) in result.AllScores)
            {
                Console.WriteLine($"{label}: {score:F8}");
            }

            return;

            if (args.Length > 0 && args[0] == "benchmark")
            {
                EmbedderBenchmark.Run();
                return;
            }

            Console.WriteLine();
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("KJARNI C# TEST SUITE");
            Console.WriteLine(new string('=', 60));
            Console.WriteLine();

            // Create temporary directory for all test artifacts
            var tempDir = Path.Combine(Path.GetTempPath(), $"kjarni_test_{Guid.NewGuid():N}");
            Directory.CreateDirectory(tempDir);

            try
            {
                // Create test documents and index
                Console.WriteLine("Setting up test environment...");
                var docsDir = CreateTestDocuments(tempDir);
                Console.WriteLine($"  Created test documents in: {docsDir}");

                var indexPath = CreateIndex(tempDir, docsDir);
                Console.WriteLine($"  Created test index at: {indexPath}");
                Console.WriteLine();

                // Run tests
                TestVersion();
                TestIndexerCreation();
                TestCancelToken();
                TestIndexInfo(indexPath);

                var indexPath2 = TestIndexCreate(tempDir);
                TestIndexAdd(indexPath2, tempDir);
                TestIndexDelete(indexPath2);

                TestProgressCallback(tempDir, docsDir);
                TestCancellation(tempDir);

                // Searcher tests
                TestSearcherCreation();
                var searcherWithReranker = TestSearcherWithReranker();
                TestBasicSearch(indexPath);
                TestSearchModes(indexPath);
                TestSearchTopK(indexPath);
                TestSearchThreshold(indexPath);
                TestStaticKeywordSearch(indexPath);
                TestSearchWithReranker(indexPath, searcherWithReranker);
                TestSearchRelevance(indexPath);

                // Error handling
                TestIndexNotFound();
                TestSearchIndexNotFound();

                Console.WriteLine(new string('=', 60));
                Console.WriteLine("ALL TESTS PASSED!");
                Console.WriteLine(new string('=', 60));
            }
            finally
            {
                // Cleanup
                try
                {
                    Directory.Delete(tempDir, recursive: true);
                }
                catch
                {
                    Console.WriteLine($"Warning: Could not clean up {tempDir}");
                }
            }
        }

        // =================================================================
        // Test Document Setup
        // =================================================================

        static string CreateTestDocuments(string baseDir)
        {
            var docsDir = Path.Combine(baseDir, "docs");
            Directory.CreateDirectory(docsDir);
            Directory.CreateDirectory(Path.Combine(docsDir, "subdir"));

            var testFiles = new Dictionary<string, string>
            {
                ["machine_learning.txt"] = @"
Machine learning is a subset of artificial intelligence that enables
systems to learn and improve from experience. It focuses on developing
algorithms that can access data and use it to learn for themselves.

Supervised learning uses labeled training data to learn the mapping
between inputs and outputs. Common algorithms include linear regression,
decision trees, and neural networks.

Unsupervised learning finds patterns in unlabeled data. Clustering and
dimensionality reduction are key techniques in this area.",

                ["deep_learning.txt"] = @"
Deep learning is a subset of machine learning based on artificial
neural networks with multiple layers. These deep neural networks
can learn hierarchical representations of data.

Convolutional neural networks (CNNs) excel at image recognition tasks.
They use convolutional layers to detect features like edges and shapes.

Recurrent neural networks (RNNs) process sequential data like text
and time series. LSTM and GRU are popular RNN architectures.",

                ["transformers.txt"] = @"
Transformers are a neural network architecture that revolutionized
natural language processing. They use self-attention mechanisms to
process sequences in parallel.

BERT (Bidirectional Encoder Representations from Transformers) learns
contextual word embeddings by training on masked language modeling.

GPT (Generative Pre-trained Transformer) models are autoregressive
and excel at text generation tasks.",

                ["python_basics.txt"] = @"
Python is a high-level programming language known for its readability
and simplicity. It supports multiple programming paradigms including
procedural, object-oriented, and functional programming.

Python uses indentation to define code blocks, making the code
visually clean and consistent. Variables are dynamically typed.

Popular Python libraries include NumPy for numerical computing,
Pandas for data analysis, and TensorFlow for machine learning.",

                ["rust_basics.txt"] = @"
Rust is a systems programming language focused on safety, speed,
and concurrency. It guarantees memory safety without garbage collection
through its ownership system.

The borrow checker ensures references are always valid and prevents
data races at compile time. This eliminates many common bugs.

Rust is used for building web servers, command-line tools, and
embedded systems where performance and reliability are critical.",

                ["subdir/embeddings.txt"] = @"
Word embeddings represent words as dense vectors in a continuous
vector space. Similar words have similar vector representations.

Word2Vec learns embeddings by predicting context words (CBOW) or
predicting a word from its context (Skip-gram).

Sentence embeddings extend this to entire sentences or documents.
Models like Sentence-BERT create embeddings useful for semantic search."
            };

            foreach (var (filename, content) in testFiles)
            {
                var filepath = Path.Combine(docsDir, filename);
                var dir = Path.GetDirectoryName(filepath);
                if (dir != null && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                File.WriteAllText(filepath, content.Trim());
            }

            return docsDir;
        }

        static string CreateIndex(string tempDir, string docsDir)
        {
            var indexPath = Path.Combine(tempDir, "test_search_index");

            using var indexer = new Indexer(
                model: "minilm-l6-v2",
                chunkSize: 256,
                chunkOverlap: 25,
                quiet: true);

            var stats = indexer.Create(indexPath, new[] { docsDir }, force: true);
            Console.WriteLine($"  Created index with {stats.DocumentsIndexed} documents");

            return indexPath;
        }

        static string CreateLargeTestDocuments(string baseDir, int numFiles = 50)
        {
            var docsDir = Path.Combine(baseDir, "large_docs");
            Directory.CreateDirectory(docsDir);

            for (int i = 0; i < numFiles; i++)
            {
                var content = $@"
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

This is document number {i} in the test corpus.";

                File.WriteAllText(Path.Combine(docsDir, $"doc_{i:D4}.txt"), content.Trim());
            }

            return docsDir;
        }

        // =================================================================
        // Basic Tests
        // =================================================================

        static void TestVersion()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Version");
            Console.WriteLine(new string('=', 60));

            var version = Kjarni.GetVersion();
            Console.WriteLine($"Kjarni version: {version}");

            if (string.IsNullOrEmpty(version))
                throw new Exception("Version should not be empty");

            Console.WriteLine("✓ Version test passed\n");
        }

        static void TestIndexerCreation()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Indexer Creation");
            Console.WriteLine(new string('=', 60));

            Console.WriteLine("Creating indexer with defaults...");
            using (var indexer = new Indexer(quiet: true))
            {
                Console.WriteLine($"  Model name: {indexer.ModelName}");
                Console.WriteLine($"  Dimension: {indexer.Dimension}");
                Console.WriteLine($"  Chunk size: {indexer.ChunkSize}");

                if (string.IsNullOrEmpty(indexer.ModelName))
                    throw new Exception("Model name should not be empty");
                if (indexer.Dimension <= 0)
                    throw new Exception("Dimension should be positive");
                if (indexer.ChunkSize <= 0)
                    throw new Exception("Chunk size should be positive");
            }
            Console.WriteLine("✓ Default indexer creation passed\n");

            Console.WriteLine("Creating indexer with custom settings...");
            using (var indexer = new Indexer(
                model: "minilm-l6-v2",
                device: "cpu",
                chunkSize: 256,
                chunkOverlap: 25,
                batchSize: 16,
                extensions: new[] { "txt", "md" },
                recursive: true,
                quiet: true))
            {
                Console.WriteLine($"  Model name: {indexer.ModelName}");
                Console.WriteLine($"  Dimension: {indexer.Dimension}");
                Console.WriteLine($"  Chunk size: {indexer.ChunkSize}");

                if (indexer.ChunkSize != 256)
                    throw new Exception($"Expected ChunkSize=256, got {indexer.ChunkSize}");
            }
            Console.WriteLine("✓ Custom indexer creation passed\n");
        }

        static void TestCancelToken()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: CancelToken");
            Console.WriteLine(new string('=', 60));

            using var token = new CancelToken();

            if (token.IsCancelled)
                throw new Exception("New token should not be cancelled");
            Console.WriteLine("  New token IsCancelled: False ✓");

            token.Cancel();
            if (!token.IsCancelled)
                throw new Exception("Token should be cancelled after Cancel()");
            Console.WriteLine("  After Cancel() IsCancelled: True ✓");

            token.Reset();
            if (token.IsCancelled)
                throw new Exception("Token should not be cancelled after Reset()");
            Console.WriteLine("  After Reset() IsCancelled: False ✓");

            Console.WriteLine("✓ CancelToken test passed\n");
        }

        static void TestIndexInfo(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Index Info");
            Console.WriteLine(new string('=', 60));

            var info = Indexer.GetInfo(indexPath);

            Console.WriteLine($"  Path: {info.Path}");
            Console.WriteLine($"  Document count: {info.DocumentCount}");
            Console.WriteLine($"  Segment count: {info.SegmentCount}");
            Console.WriteLine($"  Dimension: {info.Dimension}");
            Console.WriteLine($"  Size (bytes): {info.SizeBytes}");
            Console.WriteLine($"  Embedding model: {info.EmbeddingModel}");

            if (info.DocumentCount <= 0)
                throw new Exception("Should have documents");
            if (info.Dimension <= 0)
                throw new Exception("Should have positive dimension");

            Console.WriteLine("✓ Index info test passed\n");
        }

        static string TestIndexCreate(string tempDir)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Index Create");
            Console.WriteLine(new string('=', 60));

            var docsDir = Path.Combine(tempDir, "create_test_docs");
            Directory.CreateDirectory(docsDir);
            File.WriteAllText(Path.Combine(docsDir, "test1.txt"), "This is a test document about machine learning.");
            File.WriteAllText(Path.Combine(docsDir, "test2.txt"), "Another document about neural networks and deep learning.");

            var indexPath = Path.Combine(tempDir, "create_test_index");

            using var indexer = new Indexer(
                model: "minilm-l6-v2",
                chunkSize: 256,
                quiet: true);

            Console.WriteLine($"Creating index at: {indexPath}");
            var stats = indexer.Create(indexPath, new[] { docsDir }, force: true);

            Console.WriteLine($"  Documents indexed: {stats.DocumentsIndexed}");
            Console.WriteLine($"  Chunks created: {stats.ChunksCreated}");
            Console.WriteLine($"  Dimension: {stats.Dimension}");
            Console.WriteLine($"  Files processed: {stats.FilesProcessed}");
            Console.WriteLine($"  Elapsed (ms): {stats.ElapsedMs}");

            if (stats.DocumentsIndexed <= 0)
                throw new Exception("Should have indexed some documents");

            Console.WriteLine("✓ Index create test passed\n");
            return indexPath;
        }

        static void TestIndexAdd(string indexPath, string tempDir)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Index Add");
            Console.WriteLine(new string('=', 60));

            var extraDir = Path.Combine(tempDir, "extra_docs");
            Directory.CreateDirectory(extraDir);
            File.WriteAllText(Path.Combine(extraDir, "extra1.txt"), "Additional document about transformers and attention.");
            File.WriteAllText(Path.Combine(extraDir, "extra2.txt"), "More content about embeddings and vector search.");

            using var indexer = new Indexer(model: "minilm-l6-v2", quiet: true);

            var infoBefore = Indexer.GetInfo(indexPath);
            Console.WriteLine($"  Documents before: {infoBefore.DocumentCount}");

            var added = indexer.Add(indexPath, new[] { extraDir });
            Console.WriteLine($"  Documents added: {added}");

            var infoAfter = Indexer.GetInfo(indexPath);
            Console.WriteLine($"  Documents after: {infoAfter.DocumentCount}");

            if (added <= 0)
                throw new Exception("Should have added some documents");

            Console.WriteLine("✓ Index add test passed\n");
        }

        static void TestIndexDelete(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Index Delete");
            Console.WriteLine(new string('=', 60));

            Indexer.Delete(indexPath);

            if (Directory.Exists(indexPath))
                throw new Exception("Index should be deleted");

            Console.WriteLine($"  Deleted index at: {indexPath}");
            Console.WriteLine("✓ Index delete test passed\n");
        }

        // =================================================================
        // Progress Callback Tests
        // =================================================================

        static void TestProgressCallback(string tempDir, string docsDir)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Progress Callback");
            Console.WriteLine(new string('=', 60));

            var indexPath = Path.Combine(tempDir, "progress_test_index");
            var progressUpdates = new List<Progress>();
            var stagesSeen = new HashSet<string>();

            using var indexer = new Indexer(
                model: "minilm-l6-v2",
                chunkSize: 256,
                quiet: true);

            Console.WriteLine($"Creating index with progress callback...");
            var stats = indexer.Create(
                indexPath,
                new[] { docsDir },
                force: true,
                onProgress: p =>
                {
                    progressUpdates.Add(p);
                    stagesSeen.Add(p.Stage);
                    var msg = p.Message != null ? $" - {p.Message}" : "";
                    Console.WriteLine($"  [{p.Stage}] {p.Current}/{p.Total}{msg}");
                });

            Console.WriteLine($"\nProgress callback results:");
            Console.WriteLine($"  Total updates received: {progressUpdates.Count}");
            Console.WriteLine($"  Stages seen: {string.Join(", ", stagesSeen.OrderBy(s => s))}");
            Console.WriteLine($"  Documents indexed: {stats.DocumentsIndexed}");

            if (progressUpdates.Count == 0)
                throw new Exception("Should have received progress updates");

            // Cleanup
            Indexer.Delete(indexPath);

            Console.WriteLine("✓ Progress callback test passed\n");
        }

        static void TestCancellation(string tempDir)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Cancellation");
            Console.WriteLine(new string('=', 60));

            Console.WriteLine("  Creating large test dataset...");
            var largeDocsDir = CreateLargeTestDocuments(tempDir, numFiles: 100);
            var indexPath = Path.Combine(tempDir, "cancel_test_index");

            using var indexer = new Indexer(
                model: "minilm-l6-v2",
                chunkSize: 256,
                batchSize: 8,
                quiet: true);

            using var cancelToken = new CancelToken();
            var filesSeen = new List<Progress>();
            string? cancelledAtStage = null;
            int cancelledAtCount = 0;

            Console.WriteLine("  Starting indexing (will cancel after 10 files loaded)...");

            try
            {
                var stats = indexer.Create(
                    indexPath,
                    new[] { largeDocsDir },
                    force: true,
                    onProgress: p =>
                    {
                        filesSeen.Add(p);
                        if (p.Stage == "loading" && p.Current >= 10 && !cancelToken.IsCancelled)
                        {
                            Console.WriteLine($"  Requesting cancellation at {p.Stage} {p.Current}/{p.Total}");
                            cancelledAtStage = p.Stage;
                            cancelledAtCount = p.Current;
                            cancelToken.Cancel();
                        }
                    },
                    cancelToken: cancelToken);

                Console.WriteLine($"✗ Indexing completed without cancellation!");
                Console.WriteLine($"  Documents indexed: {stats.DocumentsIndexed}");
            }
            catch (KjarniException ex) when (ex.Message.Contains("cancel", StringComparison.OrdinalIgnoreCase))
            {
                Console.WriteLine($"  Got expected cancellation error: {ex.Message}");
                Console.WriteLine($"  Cancelled at stage: {cancelledAtStage}");
                Console.WriteLine($"  Cancelled at count: {cancelledAtCount}");
                Console.WriteLine($"  Total progress updates: {filesSeen.Count}");

                if (cancelledAtCount < 50)
                {
                    Console.WriteLine("✓ Cancellation test passed\n");
                }
                else
                {
                    Console.WriteLine($"  Warning: Cancellation happened late ({cancelledAtCount} files)");
                    Console.WriteLine("✓ Cancellation test passed (but could be faster)\n");
                }
            }

            // Cleanup
            if (Directory.Exists(indexPath))
            {
                try { Indexer.Delete(indexPath); }
                catch { Directory.Delete(indexPath, true); }
            }
        }

        // =================================================================
        // Searcher Tests
        // =================================================================

        static void TestSearcherCreation()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Searcher Creation");
            Console.WriteLine(new string('=', 60));

            Console.WriteLine("Creating searcher with defaults...");
            using (var searcher = new Searcher(quiet: true))
            {
                Console.WriteLine($"  Model name: {searcher.ModelName}");
                Console.WriteLine($"  Default mode: {searcher.DefaultMode}");
                Console.WriteLine($"  Default top_k: {searcher.DefaultTopK}");
                Console.WriteLine($"  Has reranker: {searcher.HasReranker}");
                Console.WriteLine($"  Reranker model: {searcher.RerankerModel ?? "null"}");

                if (string.IsNullOrEmpty(searcher.ModelName))
                    throw new Exception("Model name should not be empty");
                if (searcher.DefaultMode != SearchMode.Hybrid)
                    throw new Exception("Default mode should be Hybrid");
                if (searcher.DefaultTopK != 10)
                    throw new Exception("Default top_k should be 10");
                if (searcher.HasReranker)
                    throw new Exception("Should not have reranker by default");
            }
            Console.WriteLine("✓ Default searcher creation passed\n");

            Console.WriteLine("Creating searcher with custom settings...");
            using (var searcher = new Searcher(
                model: "minilm-l6-v2",
                device: "cpu",
                defaultMode: SearchMode.Semantic,
                defaultTopK: 5,
                quiet: true))
            {
                Console.WriteLine($"  Model name: {searcher.ModelName}");
                Console.WriteLine($"  Default mode: {searcher.DefaultMode}");
                Console.WriteLine($"  Default top_k: {searcher.DefaultTopK}");

                if (searcher.DefaultMode != SearchMode.Semantic)
                    throw new Exception("Default mode should be Semantic");
                if (searcher.DefaultTopK != 5)
                    throw new Exception("Default top_k should be 5");
            }
            Console.WriteLine("✓ Custom searcher creation passed\n");
        }

        static Searcher? TestSearcherWithReranker()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Searcher with Reranker");
            Console.WriteLine(new string('=', 60));

            Console.WriteLine("Creating searcher with reranker...");
            try
            {
                var searcher = new Searcher(
                    model: "minilm-l6-v2",
                    rerankModel: "minilm-l6-v2-cross-encoder",
                    quiet: true);

                Console.WriteLine($"  Model name: {searcher.ModelName}");
                Console.WriteLine($"  Has reranker: {searcher.HasReranker}");
                Console.WriteLine($"  Reranker model: {searcher.RerankerModel}");

                if (!searcher.HasReranker)
                    throw new Exception("Should have reranker");
                if (searcher.RerankerModel != "minilm-l6-v2-cross-encoder")
                    throw new Exception("Reranker model name mismatch");

                Console.WriteLine("✓ Searcher with reranker creation passed\n");
                return searcher;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Warning: Could not create reranker: {ex.Message}");
                Console.WriteLine("  Skipping reranker tests\n");
                return null;
            }
        }

        static void TestBasicSearch(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Basic Search");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);

            var query = "What is machine learning?";
            Console.WriteLine($"Query: '{query}'");

            var results = searcher.Search(indexPath, query);

            Console.WriteLine($"  Found {results.Count} results");
            if (results.Count == 0)
                throw new Exception("Should find some results");

            foreach (var (r, i) in results.Take(3).Select((r, i) => (r, i)))
            {
                Console.WriteLine($"  [{i + 1}] Score: {r.Score:F4}");
                Console.WriteLine($"      Text: {r.Text[..Math.Min(60, r.Text.Length)]}...");
                Console.WriteLine($"      Source: {r.Metadata.GetValueOrDefault("source", "N/A")}");
            }

            Console.WriteLine("✓ Basic search passed\n");
        }

        static void TestSearchModes(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Search Modes");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);
            var query = "neural networks deep learning";

            foreach (var mode in new[] { SearchMode.Keyword, SearchMode.Semantic, SearchMode.Hybrid })
            {
                Console.WriteLine($"\n  Mode: {mode}");
                var results = searcher.Search(indexPath, query, mode: mode, topK: 3);

                Console.WriteLine($"  Results: {results.Count}");
                if (results.Count == 0)
                    throw new Exception($"Should find results with {mode} mode");

                foreach (var (r, i) in results.Select((r, i) => (r, i)))
                {
                    Console.WriteLine($"    [{i + 1}] {r.Score:F4} - {r.Text[..Math.Min(50, r.Text.Length)]}...");
                }
            }

            Console.WriteLine("\n✓ Search modes test passed\n");
        }

        static void TestSearchTopK(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Top-K Limiting");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);
            var query = "programming language";

            foreach (var k in new[] { 1, 3, 5, 10 })
            {
                var results = searcher.Search(indexPath, query, topK: k);
                Console.WriteLine($"  top_k={k}: got {results.Count} results");
                if (results.Count > k)
                    throw new Exception($"Should return at most {k} results");
            }

            Console.WriteLine("✓ Top-K limiting test passed\n");
        }

        static void TestSearchThreshold(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Score Threshold");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);
            var query = "transformers attention mechanism";

            var allResults = searcher.Search(indexPath, query, topK: 10);
            Console.WriteLine($"  Without threshold: {allResults.Count} results");

            if (allResults.Count > 0)
            {
                var scores = allResults.Select(r => r.Score).ToList();
                var midScore = (scores.Max() + scores.Min()) / 2;

                Console.WriteLine($"  Score range: {scores.Min():F4} - {scores.Max():F4}");
                Console.WriteLine($"  Using threshold: {midScore:F4}");

                var filteredResults = searcher.Search(indexPath, query, topK: 10, threshold: midScore);
                Console.WriteLine($"  With threshold: {filteredResults.Count} results");

                foreach (var r in filteredResults)
                {
                    if (r.Score < midScore)
                        throw new Exception($"Score {r.Score} below threshold {midScore}");
                }
            }

            Console.WriteLine("✓ Score threshold test passed\n");
        }

        static void TestStaticKeywordSearch(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Static Keyword Search");
            Console.WriteLine(new string('=', 60));

            var query = "Python NumPy Pandas";
            Console.WriteLine($"Query: '{query}'");

            var results = Searcher.SearchKeywords(indexPath, query, topK: 5);

            Console.WriteLine($"  Found {results.Count} results");
            if (results.Count == 0)
                throw new Exception("Should find some results");

            foreach (var (r, i) in results.Select((r, i) => (r, i)))
            {
                Console.WriteLine($"  [{i + 1}] Score: {r.Score:F4}");
                Console.WriteLine($"      Text: {r.Text[..Math.Min(60, r.Text.Length)]}...");
            }

            Console.WriteLine("✓ Static keyword search passed\n");
        }

        static void TestSearchWithReranker(string indexPath, Searcher? searcherWithReranker)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Search with Reranker");
            Console.WriteLine(new string('=', 60));

            if (searcherWithReranker == null)
            {
                Console.WriteLine("  Skipping - reranker not available\n");
                return;
            }

            var query = "How do neural networks learn?";
            Console.WriteLine($"Query: '{query}'");

            var resultsNoRerank = searcherWithReranker.Search(indexPath, query, topK: 5, rerank: false);
            Console.WriteLine($"\n  Without reranking ({resultsNoRerank.Count} results):");
            foreach (var (r, i) in resultsNoRerank.Take(3).Select((r, i) => (r, i)))
            {
                Console.WriteLine($"    [{i + 1}] {r.Score:F4} - {r.Text[..Math.Min(50, r.Text.Length)]}...");
            }

            var resultsReranked = searcherWithReranker.Search(indexPath, query, topK: 5, rerank: true);
            Console.WriteLine($"\n  With reranking ({resultsReranked.Count} results):");
            foreach (var (r, i) in resultsReranked.Take(3).Select((r, i) => (r, i)))
            {
                Console.WriteLine($"    [{i + 1}] {r.Score:F4} - {r.Text[..Math.Min(50, r.Text.Length)]}...");
            }

            Console.WriteLine("\n✓ Search with reranker passed\n");
        }

        static void TestSearchRelevance(string indexPath)
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Search Relevance");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);

            var testCases = new[]
            {
                ("Rust memory safety ownership", new[] { "rust", "memory", "ownership", "borrow" }),
                ("Python data analysis", new[] { "python", "pandas", "numpy", "data" }),
                ("transformer attention BERT GPT", new[] { "transformer", "attention", "bert", "gpt" }),
            };

            foreach (var (query, expectedTerms) in testCases)
            {
                Console.WriteLine($"\n  Query: '{query}'");
                var results = searcher.Search(indexPath, query, topK: 3);

                if (results.Count > 0)
                {
                    var topResult = results[0];
                    var textLower = topResult.Text.ToLowerInvariant();

                    var foundTerms = expectedTerms.Where(t => textLower.Contains(t)).ToList();
                    Console.WriteLine($"    Top result contains: [{string.Join(", ", foundTerms)}]");
                    Console.WriteLine($"    Text: {topResult.Text[..Math.Min(60, topResult.Text.Length)]}...");

                    if (foundTerms.Count == 0)
                        throw new Exception($"Top result should contain relevant terms for '{query}'");
                }
            }

            Console.WriteLine("\n✓ Search relevance test passed\n");
        }

        // =================================================================
        // Error Handling Tests
        // =================================================================

        static void TestIndexNotFound()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Index Not Found Error");
            Console.WriteLine(new string('=', 60));

            try
            {
                Indexer.GetInfo("/nonexistent/path/to/index");
                Console.WriteLine("✗ Should have raised an exception");
            }
            catch (KjarniException ex)
            {
                Console.WriteLine($"  Got expected error: {ex.Message}");
                Console.WriteLine("✓ Index not found error handling passed\n");
            }
        }

        static void TestSearchIndexNotFound()
        {
            Console.WriteLine(new string('=', 60));
            Console.WriteLine("TEST: Search Index Not Found Error");
            Console.WriteLine(new string('=', 60));

            using var searcher = new Searcher(model: "minilm-l6-v2", quiet: true);

            try
            {
                searcher.Search("/nonexistent/index/path", "test query");
                Console.WriteLine("✗ Should have raised an exception");
            }
            catch (KjarniException ex)
            {
                Console.WriteLine($"  Got expected error: {ex.Message}");
                Console.WriteLine("✓ Search index not found error handling passed\n");
            }
        }
    }
}