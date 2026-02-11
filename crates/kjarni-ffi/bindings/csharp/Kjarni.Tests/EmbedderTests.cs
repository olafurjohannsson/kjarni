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

        public void Dispose()
        {
            _embedder.Dispose();
        }

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
            var text = "Deterministic inference test";
            var a = _embedder.Encode(text);
            var b = _embedder.Encode(text);

            Assert.Equal(a.Length, b.Length);
            for (int i = 0; i < a.Length; i++)
            {
                Assert.Equal(a[i], b[i]);
            }
        }

        [Fact]
        public void Similarity_RelatedWordsHigherThanUnrelated()
        {
            var catDog = _embedder.Similarity("cat", "dog");
            var catQuantum = _embedder.Similarity("cat", "quantum computing");

            _output.WriteLine($"cat-dog: {catDog:F6}");
            _output.WriteLine($"cat-quantum: {catQuantum:F6}");

            Assert.True(catDog > catQuantum,
                $"Expected cat-dog ({catDog:F4}) > cat-quantum ({catQuantum:F4})");
        }

        [Fact]
        public void Similarity_IdenticalTextIsHighest()
        {
            var same = _embedder.Similarity("machine learning", "machine learning");
            var different = _embedder.Similarity("machine learning", "cooking recipes");

            _output.WriteLine($"identical: {same:F6}");
            _output.WriteLine($"different: {different:F6}");

            Assert.True(same > 0.99f, $"Identical text similarity should be ~1.0, got {same:F4}");
            Assert.True(same > different);
        }

        [Fact]
        public void Similarity_SemanticPairsRankedCorrectly()
        {
            var pairs = new[]
            {
                ("dog", "puppy"),
                ("dog", "cat"),
                ("dog", "car"),
                ("dog", "quantum physics"),
            };

            var scores = pairs.Select(p =>
            {
                var score = _embedder.Similarity(p.Item1, p.Item2);
                _output.WriteLine($"{p.Item1} - {p.Item2}: {score:F6}");
                return score;
            }).ToArray();

            // Each pair should be more similar than the next
            for (int i = 0; i < scores.Length - 1; i++)
            {
                Assert.True(scores[i] > scores[i + 1],
                    $"Expected '{pairs[i]}' ({scores[i]:F4}) > '{pairs[i + 1]}' ({scores[i + 1]:F4})");
            }
        }

        [Fact]
        public void Similarity_ReturnsValidRange()
        {
            var score = _embedder.Similarity("hello", "world");
            _output.WriteLine($"Score: {score:F6}");
            Assert.InRange(score, -1.0f, 1.0f);
        }

        [Fact]
        public void EncodeBatch_MatchesSingleEncode2()
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
                {
                    Assert.Equal(single[j], batch[i][j], precision: 5);
                }
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
            var embedding = _embedder.Encode("日本語テスト 🌍 émojis café");
            Assert.Equal(384, embedding.Length);

            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_VeryLongText()
        {
            // Most BERT models truncate at 512 tokens — should not crash
            var longText = string.Join(" ", Enumerable.Repeat("This is a repeated sentence for testing purposes.", 100));
            var embedding = _embedder.Encode(longText);

            Assert.Equal(384, embedding.Length);
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [Fact]
        public void Encode_DifferentTextsProduceDifferentEmbeddings()
        {
            var a = _embedder.Encode("The stock market crashed yesterday.");
            var b = _embedder.Encode("I love chocolate ice cream.");
            var diffs = a.Zip(b, (x, y) => MathF.Abs(x - y)).Sum();
            _output.WriteLine($"Total absolute difference: {diffs:F6}");

            Assert.True(diffs > 0.1f, "Different texts should produce different embeddings");
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
            embedder.Dispose(); // Should not throw
        }
    }
}