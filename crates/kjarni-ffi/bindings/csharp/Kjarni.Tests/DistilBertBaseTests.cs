using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    // =================================================================
    // DistilBERT Base Cased (768 dimensions)
    // =================================================================

    public class DistilBertBaseTests : IDisposable
    {
        private readonly Embedder _embedder;
        private readonly ITestOutputHelper _output;

        public DistilBertBaseTests(ITestOutputHelper output)
        {
            _output = output;
            _embedder = new Embedder(model: "distilbert-base", quiet: true);
        }

        public void Dispose() => _embedder.Dispose();

        [Fact]
        public void Dimension_Is768()
        {
            Assert.Equal(768, _embedder.Dim);
        }

        [Fact]
        public void Encode_ReturnsCorrectDimension()
        {
            var embedding = _embedder.Encode("Hello world");
            Assert.Equal(768, embedding.Length);
        }

        [Fact]
        public void Encode_IsL2Normalized()
        {
            var embedding = _embedder.Encode("Test normalization for distilbert.");
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));

            _output.WriteLine($"L2 norm: {norm:F8}");
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_IsDeterministic()
        {
            var a = _embedder.Encode("Deterministic test");
            var b = _embedder.Encode("Deterministic test");

            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a[i], b[i]);
        }

        [Fact]
        public void Similarity_RelatedHigherThanUnrelated()
        {
            var related = _embedder.Similarity("cat", "dog");
            var unrelated = _embedder.Similarity("cat", "quantum computing");

            _output.WriteLine($"cat-dog: {related:F6}");
            _output.WriteLine($"cat-quantum: {unrelated:F6}");

            Assert.True(related > unrelated,
                $"Expected cat-dog ({related:F4}) > cat-quantum ({unrelated:F4})");
        }

        [Fact]
        public void Similarity_IdenticalTextNearOne()
        {
            var score = _embedder.Similarity("machine learning", "machine learning");
            _output.WriteLine($"Identical: {score:F6}");

            Assert.True(score > 0.99f);
        }

        [Fact]
        public void EncodeBatch_MatchesSingleEncode()
        {
            var texts = new[] { "First", "Second", "Third" };
            var batch = _embedder.EncodeBatch(texts);

            Assert.Equal(3, batch.Length);
            for (int i = 0; i < texts.Length; i++)
            {
                var single = _embedder.Encode(texts[i]);
                // Cosine similarity should be ~1.0 even if padding causes tiny element-wise diffs
                var dot = single.Zip(batch[i], (a, b) => a * b).Sum();
                Assert.InRange(dot, 0.9999f, 1.0001f);
            }
        }

        [Fact]
        public void Encode_UnicodeText()
        {
            var embedding = _embedder.Encode("Héllo wörld café 日本語");
            Assert.Equal(768, embedding.Length);

            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }
    }
}