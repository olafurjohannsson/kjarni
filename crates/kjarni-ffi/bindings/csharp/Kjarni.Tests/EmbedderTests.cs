using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class EmbedderTests : IDisposable
    {
        private readonly Embedder _embedder;
        private readonly ITestOutputHelper _output;

        public EmbedderTests(ITestOutputHelper output)
        {
            _output = output;
            _embedder = new Embedder(model: "minilm-l6-v2", quiet: true);
        }

        public void Dispose() => _embedder.Dispose();

        [Fact]
        public void Dimension_Is384()
        {
            Assert.Equal(384, _embedder.Dim);
        }

        [Fact]
        public void Encode_ReturnsCorrectDimension()
        {
            var embedding = _embedder.Encode("Hello world");
            Assert.Equal(384, embedding.Length);
        }

        [Fact]
        public void Encode_FirstFiveValues()
        {
            var embedding = _embedder.Encode("Hello world");

            Assert.Equal(-0.03448f, embedding[0], 4);
            Assert.Equal( 0.03102f, embedding[1], 4);
            Assert.Equal( 0.00673f, embedding[2], 4);
            Assert.Equal( 0.02611f, embedding[3], 4);
            Assert.Equal(-0.03936f, embedding[4], 4);
        }

        [Fact]
        public void Encode_OutputIsL2Normalized()
        {
            var embedding = _embedder.Encode("The quick brown fox jumps over the lazy dog.");
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));

            _output.WriteLine($"L2 norm: {norm:F8}");
            Assert.InRange(norm, 0.999f, 1.001f);
        }

        [Theory]
        [InlineData("")]
        [InlineData("a")]
        [InlineData("This is a much longer sentence that contains many tokens and should still produce a normalized embedding vector.")]
        public void Encode_AlwaysNormalized(string text)
        {
            var embedding = _embedder.Encode(text);
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));

            _output.WriteLine($"Input length: {text.Length}, L2 norm: {norm:F8}");
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_IsDeterministic()
        {
            var a = _embedder.Encode("Deterministic inference test");
            var b = _embedder.Encode("Deterministic inference test");

            Assert.Equal(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a[i], b[i]);
        }

        [Theory]
        [InlineData("cat", "dog", 0.6606f)]
        [InlineData("cat", "quantum computing", 0.1080f)]
        [InlineData("dog", "puppy", 0.8040f)]
        [InlineData("dog", "car", 0.4756f)]
        [InlineData("dog", "quantum physics", 0.2157f)]
        [InlineData("hello", "world", 0.3454f)]
        [InlineData("machine learning", "cooking recipes", 0.2353f)]
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
        public void Similarity_SemanticPairsRankedCorrectly()
        {
            // dog/puppy (0.804) > dog/cat (0.661) > dog/car (0.476) > dog/quantum (0.216)
            var scores = new[]
            {
                ("dog", "puppy",           _embedder.Similarity("dog", "puppy")),
                ("dog", "cat",             _embedder.Similarity("dog", "cat")),
                ("dog", "car",             _embedder.Similarity("dog", "car")),
                ("dog", "quantum physics", _embedder.Similarity("dog", "quantum physics")),
            };

            foreach (var (a, b, score) in scores)
                _output.WriteLine($"{a} / {b}: {score:F6}");

            for (int i = 0; i < scores.Length - 1; i++)
            {
                Assert.True(scores[i].Item3 > scores[i + 1].Item3,
                    $"Expected '{scores[i].Item1}/{scores[i].Item2}' ({scores[i].Item3:F4}) > " +
                    $"'{scores[i + 1].Item1}/{scores[i + 1].Item2}' ({scores[i + 1].Item3:F4})");
            }
        }

        [Fact]
        public void EncodeBatch_MatchesSingleEncode()
        {
            var texts = new[] { "First sentence", "Second sentence", "Third sentence" };

            var batch = _embedder.EncodeBatch(texts);
            Assert.Equal(texts.Length, batch.Length);

            for (int i = 0; i < texts.Length; i++)
            {
                var single = _embedder.Encode(texts[i]);
                Assert.Equal(single.Length, batch[i].Length);

                for (int j = 0; j < single.Length; j++)
                    Assert.Equal(single[j], batch[i][j], precision: 5);
            }
        }

        [Fact]
        public void EncodeBatch_EmptyArray_ReturnsEmpty()
        {
            var result = _embedder.EncodeBatch(Array.Empty<string>());
            Assert.Empty(result);
        }

        [Fact]
        public void EncodeBatch_SingleElement()
        {
            var batch = _embedder.EncodeBatch(new[] { "Just one" });
            Assert.Single(batch);
            Assert.Equal(384, batch[0].Length);
        }

        [Fact]
        public void EncodeBatch_LargeBatch()
        {
            var texts = Enumerable.Range(0, 64)
                .Select(i => $"Sentence number {i} for batch processing test.")
                .ToArray();

            var batch = _embedder.EncodeBatch(texts);
            Assert.Equal(64, batch.Length);

            foreach (var embedding in batch)
            {
                Assert.Equal(384, embedding.Length);
                var norm = MathF.Sqrt(embedding.Sum(x => x * x));
                Assert.InRange(norm, 0.998f, 1.002f);
            }
        }

        [Fact]
        public void Encode_UnicodeText()
        {
            var embedding = _embedder.Encode("Bonjour le monde, cafe resume");
            Assert.Equal(384, embedding.Length);

            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_VeryLongText()
        {
            var longText = string.Join(" ", Enumerable.Repeat("This is a repeated sentence for testing purposes.", 100));
            var embedding = _embedder.Encode(longText);

            Assert.Equal(384, embedding.Length);
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_AfterDispose_Throws()
        {
            var embedder = new Embedder(model: "minilm-l6-v2", quiet: true);
            embedder.Dispose();

            Assert.Throws<ObjectDisposedException>(() => embedder.Encode("test"));
        }

        [Fact]
        public void DoubleDispose_DoesNotThrow()
        {
            var embedder = new Embedder(model: "minilm-l6-v2", quiet: true);
            embedder.Dispose();
            embedder.Dispose();
        }
    }
}