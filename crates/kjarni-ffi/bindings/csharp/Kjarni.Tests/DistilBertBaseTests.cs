using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
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
        public void Encode_FirstFiveValues()
        {
            var embedding = _embedder.Encode("Hello world");

            Assert.Equal( 0.02895f, embedding[0], 4);
            Assert.Equal( 0.00946f, embedding[1], 4);
            Assert.Equal( 0.08015f, embedding[2], 3);
            Assert.Equal( 0.02010f, embedding[3], 4);
            Assert.Equal(-0.03069f, embedding[4], 4);
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

        [Theory]
        [InlineData("cat", "dog", 0.9491f)]
        [InlineData("cat", "quantum computing", 0.7529f)]
        public void Similarity_ExactValues(string a, string b, float expected)
        {
            var score = _embedder.Similarity(a, b);
            _output.WriteLine($"{a} / {b}: {score:F6}");

            Assert.Equal(expected, score, 3);
        }

        [Fact]
        public void Similarity_IdenticalTextIsOne()
        {
            var score = _embedder.Similarity("machine learning", "machine learning");
            _output.WriteLine($"Identical: {score:F6}");

            Assert.Equal(1.0f, score, 3);
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
                var dot = single.Zip(batch[i], (a, b) => a * b).Sum();
                Assert.InRange(dot, 0.9999f, 1.0001f);
            }
        }

        [Fact]
        public void Encode_UnicodeText()
        {
            var embedding = _embedder.Encode("Bonjour le monde, cafe resume");
            Assert.Equal(768, embedding.Length);

            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }
    }
}