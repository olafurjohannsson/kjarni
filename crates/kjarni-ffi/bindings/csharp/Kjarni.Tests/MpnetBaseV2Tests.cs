using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    public class MpnetBaseV2Tests : IDisposable
    {
        private readonly Embedder _embedder;
        private readonly ITestOutputHelper _output;

        public MpnetBaseV2Tests(ITestOutputHelper output)
        {
            _output = output;
            _embedder = new Embedder(model: "mpnet-base-v2", quiet: true);
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
            var embedding = _embedder.Encode("Test normalization for mpnet.");
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
        public void Similarity_SemanticRankingCorrect()
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

            for (int i = 0; i < scores.Length - 1; i++)
            {
                Assert.True(scores[i] > scores[i + 1],
                    $"Expected '{pairs[i]}' ({scores[i]:F4}) > '{pairs[i + 1]}' ({scores[i + 1]:F4})");
            }
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
        public void EncodeBatch_AllNormalized()
        {
            var texts = Enumerable.Range(0, 16)
                .Select(i => $"Sentence {i} for batch test.")
                .ToArray();

            var batch = _embedder.EncodeBatch(texts);
            Assert.Equal(16, batch.Length);

            foreach (var embedding in batch)
            {
                Assert.Equal(768, embedding.Length);
                var norm = MathF.Sqrt(embedding.Sum(x => x * x));
                Assert.InRange(norm, 0.998f, 1.002f);
            }
        }
    }
}