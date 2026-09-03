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

        // These are torch's values, which the engine now reproduces.
        //
        // MPNet used to sit 4.7e-4 away from torch while every other encoder was at
        // 1e-7. The cause was `MpnetConfig::metadata()` hardcoding the tanh
        // approximation of GELU while its config.json asks for the exact erf form,
        // and parsing `hidden_act` into a field it then ignored. With that fixed
        // parity is 1.6e-7 and these fixtures are a real check against torch rather
        // than a recording of our own output.
        //
        // Compared with a tolerance rather than decimal places on purpose: the
        // first value lands 3e-7 from a 4-decimal rounding boundary, so
        // `Assert.Equal(.., 4)` would flip on any architecture that rounds the last
        // bit differently.
        [Fact]
        public void Encode_FirstFiveValues()
        {
            var embedding = _embedder.Encode("Hello world");

            Assert.Equal( 0.0262497f, embedding[0], 1e-5f);
            Assert.Equal( 0.0133956f, embedding[1], 1e-5f);
            Assert.Equal(-0.0045332f, embedding[2], 1e-5f);
            Assert.Equal(-0.0217914f, embedding[3], 1e-5f);
            Assert.Equal( 0.0545519f, embedding[4], 1e-5f);
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

        // torch gives 0.778325, 0.608123, 0.390584 and 0.250900 for these pairs;
        // the engine agrees to within 1e-5.
        [Theory]
        [InlineData("dog", "puppy", 0.778325f)]
        [InlineData("dog", "cat", 0.608123f)]
        [InlineData("dog", "car", 0.390585f)]
        [InlineData("dog", "quantum physics", 0.250894f)]
        public void Similarity_ExactValues(string a, string b, float expected)
        {
            var score = _embedder.Similarity(a, b);
            _output.WriteLine($"{a} / {b}: {score:F6}");

            Assert.Equal(expected, score, 1e-4f);
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
            // dog/puppy (0.718) > dog/cat (0.640) > dog/car (0.546) > dog/quantum (0.351)
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