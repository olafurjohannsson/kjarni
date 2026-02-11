using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class RerankerTests : IDisposable
    {
        private readonly Reranker _reranker;
        private readonly ITestOutputHelper _output;

        public RerankerTests(ITestOutputHelper output)
        {
            _output = output;
            _reranker = new Reranker(quiet: true);
        }

        public void Dispose() => _reranker.Dispose();

        [Fact]
        public void Score_RelevantDocHigherThanIrrelevant()
        {
            var relevant = _reranker.Score(
                "What is machine learning?",
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.");

            var irrelevant = _reranker.Score(
                "What is machine learning?",
                "The weather today is sunny with a high of 72 degrees.");

            _output.WriteLine($"Relevant: {relevant:F6}");
            _output.WriteLine($"Irrelevant: {irrelevant:F6}");

            Assert.True(relevant > irrelevant,
                $"Expected relevant ({relevant:F4}) > irrelevant ({irrelevant:F4})");
        }

        [Fact]
        public void Score_IsDeterministic()
        {
            var a = _reranker.Score("query", "document text");
            var b = _reranker.Score("query", "document text");

            Assert.Equal(a, b);
        }

        [Fact]
        public void Rerank_ReturnsAllDocuments()
        {
            var docs = new[]
            {
                "Completely unrelated: baking cookies.",
                "Machine learning uses algorithms to learn from data.",
                "The stock market fluctuated wildly today.",
            };

            var results = _reranker.Rerank("What is machine learning?", docs);

            Assert.Equal(3, results.Length);
        }

        [Fact]
        public void Rerank_OrdersByRelevance()
        {
            var docs = new[]
            {
                "The weather is nice today.",
                "Machine learning is a branch of artificial intelligence.",
                "I need to buy groceries later.",
                "Deep learning uses neural networks with many layers.",
            };

            var results = _reranker.Rerank("What is machine learning?", docs);

            _output.WriteLine("Reranked order:");
            foreach (var r in results)
                _output.WriteLine($"  [{r.Index}] {r.Score:F6} {r.Document}");

            // ML-related docs should rank above unrelated ones
            var topTwo = results.Take(2).Select(r => r.Index).ToArray();
            Assert.Contains(1, topTwo); // "Machine learning is a branch..."
            Assert.Contains(3, topTwo); // "Deep learning uses neural networks..."
        }

        [Fact]
        public void Rerank_ScoresDescending()
        {
            var docs = new[]
            {
                "Python is a programming language.",
                "Neural networks learn hierarchical representations.",
                "The cat sat on the mat.",
            };

            var results = _reranker.Rerank("deep learning neural networks", docs);

            for (int i = 0; i < results.Length - 1; i++)
            {
                Assert.True(results[i].Score >= results[i + 1].Score,
                    $"Results not sorted: [{i}] {results[i].Score:F4} < [{i + 1}] {results[i + 1].Score:F4}");
            }
        }

        [Fact]
        public void Rerank_DocumentFieldMatchesInput()
        {
            var docs = new[] { "First doc", "Second doc", "Third doc" };
            var results = _reranker.Rerank("query", docs);

            foreach (var r in results)
            {
                Assert.Equal(docs[r.Index], r.Document);
            }
        }

        [Fact]
        public void RerankTopK_ReturnsAtMostK()
        {
            var docs = new[]
            {
                "Doc one about ML.",
                "Doc two about weather.",
                "Doc three about cooking.",
                "Doc four about neural networks.",
                "Doc five about gardening.",
            };

            var results = _reranker.RerankTopK("machine learning", docs, k: 2);

            Assert.Equal(2, results.Length);
        }

        [Fact]
        public void RerankTopK_TopResultIsRelevant()
        {
            var docs = new[]
            {
                "Baking bread requires flour and water.",
                "Transformers revolutionized natural language processing.",
                "The sunset was beautiful yesterday evening.",
            };

            var results = _reranker.RerankTopK("NLP transformers attention", docs, k: 1);

            Assert.Single(results);
            Assert.Equal(1, results[0].Index); // Transformers doc

            _output.WriteLine($"Top result: [{results[0].Index}] {results[0].Score:F6} {results[0].Document}");
        }

        [Fact]
        public void Rerank_EmptyArray_ReturnsEmpty()
        {
            var results = _reranker.Rerank("query", Array.Empty<string>());
            Assert.Empty(results);
        }

        [Fact]
        public void Rerank_SingleDocument()
        {
            var results = _reranker.Rerank("query", new[] { "Only document" });

            Assert.Single(results);
            Assert.Equal(0, results[0].Index);
            Assert.Equal("Only document", results[0].Document);
        }

        [Fact]
        public void Score_UnicodeText()
        {
            var score = _reranker.Score(
                "¿Qué es el aprendizaje automático?",
                "El aprendizaje automático es una rama de la inteligencia artificial.");

            _output.WriteLine($"Spanish query-doc score: {score:F6}");
            Assert.False(float.IsNaN(score));
        }
        [Fact]
        public void Score_AfterDispose_Throws()
        {
            var reranker = new Reranker(quiet: true);
            reranker.Dispose();

            Assert.Throws<ObjectDisposedException>(() =>
                reranker.Score("query", "doc"));
        }

        [Fact]
        public void DoubleDispose_DoesNotThrow()
        {
            var reranker = new Reranker(quiet: true);
            reranker.Dispose();
            reranker.Dispose();
        }
    }
}