using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Kjarni.Extensions.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace Kjarni.Extensions.AI.Tests
{
    public class KjarniEmbeddingGeneratorTests : IDisposable
    {
        private const string Model = "minilm-l6-v2";
        private const int Dim = 384;

        private readonly KjarniEmbeddingGenerator _generator;

        public KjarniEmbeddingGeneratorTests()
        {
            _generator = new KjarniEmbeddingGenerator(model: Model, quiet: true);
        }

        public void Dispose() => _generator.Dispose();

        [Fact]
        public async Task GenerateAsync_SingleValue_ReturnsOneEmbeddingOfModelDimension()
        {
            var result = await _generator.GenerateAsync(new[] { "Hello world" });

            Assert.Single(result);
            Assert.Equal(Dim, result[0].Vector.Length);
        }

        [Fact]
        public async Task GenerateAsync_FirstFiveValues()
        {
            // Same golden values as Kjarni.Tests.EmbedderTests.Encode_FirstFiveValues —
            // the adapter must forward the model output untouched.
            var result = await _generator.GenerateAsync(new[] { "Hello world" });
            float[] v = result[0].Vector.ToArray();

            Assert.Equal(-0.03448f, v[0], 4);
            Assert.Equal( 0.03102f, v[1], 4);
            Assert.Equal( 0.00673f, v[2], 4);
            Assert.Equal( 0.02611f, v[3], 4);
            Assert.Equal(-0.03936f, v[4], 4);
        }

        [Fact]
        public async Task GenerateAsync_PreservesInputOrder()
        {
            // Order is the adapter's contract: result[i] must correspond to inputs[i].
            // Asserted against direct per-input encodes rather than semantic distance,
            // so this is exact and cannot drift with the model.
            string[] inputs = { "doctor", "physician", "banana" };

            using var embedder = new Embedder(model: Model, quiet: true);
            var result = await _generator.GenerateAsync(inputs);

            Assert.Equal(inputs.Length, result.Count);

            for (int i = 0; i < inputs.Length; i++)
                AssertClose(embedder.Encode(inputs[i]), result[i].Vector.ToArray());
        }

        [Fact]
        public async Task GenerateAsync_MatchesUnderlyingEmbedderExactly()
        {
            // The adapter must not perturb the vectors it forwards.
            using var embedder = new Embedder(model: Model, quiet: true);
            float[] direct = embedder.Encode("semantic search without a vector database");

            var viaAdapter = await _generator.GenerateAsync(
                new[] { "semantic search without a vector database" });

            AssertClose(direct, viaAdapter[0].Vector.ToArray());
        }

        [Fact]
        public async Task GenerateAsync_MatchesEncodeBatch()
        {
            // Mirrors EmbedderTests.EncodeBatch_MatchesSingleEncode: batching through the
            // adapter must agree with EncodeBatch element-for-element.
            string[] texts = { "first document", "second document", "third document" };

            using var embedder = new Embedder(model: Model, quiet: true);
            float[][] direct = embedder.EncodeBatch(texts);

            var viaAdapter = await _generator.GenerateAsync(texts);

            Assert.Equal(direct.Length, viaAdapter.Count);
            for (int i = 0; i < direct.Length; i++)
                AssertClose(direct[i], viaAdapter[i].Vector.ToArray());
        }

        [Fact]
        public async Task GenerateAsync_OutputIsL2Normalized()
        {
            var result = await _generator.GenerateAsync(new[] { "Test normalization." });

            float norm = 0f;
            foreach (float x in result[0].Vector.ToArray()) norm += x * x;

            Assert.InRange(MathF.Sqrt(norm), 0.998f, 1.002f);
        }

        [Fact]
        public async Task GenerateAsync_IsDeterministic()
        {
            var a = await _generator.GenerateAsync(new[] { "Deterministic test" });
            var b = await _generator.GenerateAsync(new[] { "Deterministic test" });

            for (int i = 0; i < a[0].Vector.Length; i++)
                Assert.Equal(a[0].Vector.Span[i], b[0].Vector.Span[i]);
        }

        [Fact]
        public async Task GenerateAsync_EmptyInput_ReturnsEmptyResult()
        {
            var result = await _generator.GenerateAsync(Array.Empty<string>());
            Assert.Empty(result);
        }

        [Fact]
        public async Task GenerateAsync_NullInput_Throws()
        {
            await Assert.ThrowsAsync<ArgumentNullException>(
                () => _generator.GenerateAsync(null!));
        }

        [Fact]
        public async Task GenerateAsync_NullElement_Throws()
        {
            await Assert.ThrowsAsync<ArgumentException>(
                () => _generator.GenerateAsync(new string?[] { "ok", null }!));
        }

        [Fact]
        public async Task GenerateAsync_SetsModelIdOnEachEmbedding()
        {
            var result = await _generator.GenerateAsync(new[] { "a", "b" });

            Assert.All(result, e => Assert.Equal(Model, e.ModelId));
        }

        [Fact]
        public async Task GenerateAsync_MismatchedModelId_Throws()
        {
            var options = new EmbeddingGenerationOptions { ModelId = "mpnet-base-v2" };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => _generator.GenerateAsync(new[] { "x" }, options));
        }

        [Fact]
        public async Task GenerateAsync_MatchingModelId_Succeeds()
        {
            var options = new EmbeddingGenerationOptions { ModelId = Model };

            var result = await _generator.GenerateAsync(new[] { "x" }, options);

            Assert.Single(result);
        }

        [Fact]
        public async Task GenerateAsync_UnsupportedDimensions_Throws()
        {
            var options = new EmbeddingGenerationOptions { Dimensions = 128 };

            await Assert.ThrowsAsync<NotSupportedException>(
                () => _generator.GenerateAsync(new[] { "x" }, options));
        }

        [Fact]
        public async Task GenerateAsync_CancelledToken_Throws()
        {
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            await Assert.ThrowsAnyAsync<OperationCanceledException>(
                () => _generator.GenerateAsync(new[] { "x" }, cancellationToken: cts.Token));
        }

        [Fact]
        public async Task GenerateAsync_ConcurrentCalls_AreSerializedAndCorrect()
        {
            // The native embedder is not re-entrant; the adapter must serialize.
            string[] inputs = { "alpha", "beta", "gamma", "delta" };

            var tasks = Enumerable.Range(0, 8)
                .Select(_ => _generator.GenerateAsync(inputs))
                .ToArray();

            var results = await Task.WhenAll(tasks);

            Assert.All(results, r => Assert.Equal(4, r.Count));

            // Every run must produce identical vectors for the same input.
            var reference = results[0][0].Vector.ToArray();
            foreach (var r in results)
                for (int i = 0; i < reference.Length; i++)
                    Assert.Equal(reference[i], r[0].Vector.Span[i], 6);
        }

        [Fact]
        public void Dimensions_MatchesModel()
        {
            Assert.Equal(Dim, _generator.Dimensions);
        }

        [Fact]
        public void GetService_ReturnsMetadataWithProviderAndModel()
        {
            var metadata = _generator.GetService(typeof(EmbeddingGeneratorMetadata))
                as EmbeddingGeneratorMetadata;

            Assert.NotNull(metadata);
            Assert.Equal("kjarni", metadata!.ProviderName);
            Assert.Equal(Model, metadata.DefaultModelId);
            Assert.Equal(Dim, metadata.DefaultModelDimensions);
        }

        [Fact]
        public void GetService_ReturnsUnderlyingEmbedder()
        {
            Assert.IsType<Embedder>(_generator.GetService(typeof(Embedder)));
        }

        [Fact]
        public void GetService_ReturnsSelfForOwnType()
        {
            Assert.Same(_generator, _generator.GetService(typeof(KjarniEmbeddingGenerator)));
        }

        [Fact]
        public void GetService_WithServiceKey_ReturnsNull()
        {
            Assert.Null(_generator.GetService(typeof(EmbeddingGeneratorMetadata), "key"));
        }

        [Fact]
        public void GetService_UnknownType_ReturnsNull()
        {
            Assert.Null(_generator.GetService(typeof(string)));
        }

        [Fact]
        public void GetService_NullType_Throws()
        {
            Assert.Throws<ArgumentNullException>(() => _generator.GetService(null!));
        }

        [Fact]
        public async Task Disposed_GenerateAsync_Throws()
        {
            var g = new KjarniEmbeddingGenerator(model: Model, quiet: true);
            g.Dispose();

            await Assert.ThrowsAsync<ObjectDisposedException>(
                () => g.GenerateAsync(new[] { "x" }));
        }

        [Fact]
        public void Dispose_IsIdempotent()
        {
            var g = new KjarniEmbeddingGenerator(model: Model, quiet: true);
            g.Dispose();
            g.Dispose();
        }

        [Fact]
        public void Dispose_DoesNotDisposeBorrowedEmbedder()
        {
            using var embedder = new Embedder(model: Model, quiet: true);

            var g = embedder.AsEmbeddingGenerator(Model);   // ownsEmbedder: false
            g.Dispose();

            // The borrowed embedder must still be usable.
            Assert.Equal(Dim, embedder.Encode("still alive").Length);
        }

        [Fact]
        public async Task AsEmbeddingGenerator_ProducesWorkingGenerator()
        {
            using var embedder = new Embedder(model: Model, quiet: true);
            var g = embedder.AsEmbeddingGenerator(Model);

            var result = await g.GenerateAsync(new[] { "wrapped" });

            Assert.Single(result);
            Assert.Equal(Dim, result[0].Vector.Length);
        }

        [Fact]
        public async Task AddKjarniEmbeddingGenerator_ResolvesFromContainer()
        {
            var services = new ServiceCollection();
            services.AddKjarniEmbeddingGenerator(model: Model);

            using var provider = services.BuildServiceProvider();
            var generator = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

            var result = await generator.GenerateAsync(new[] { "from di" });

            Assert.Single(result);
            Assert.Equal(Dim, result[0].Vector.Length);
        }

        [Fact]
        public void AddKjarniEmbeddingGenerator_RegistersSingleton()
        {
            var services = new ServiceCollection();
            services.AddKjarniEmbeddingGenerator(model: Model);

            using var provider = services.BuildServiceProvider();
            var a = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
            var b = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();

            Assert.Same(a, b);
        }

        [Fact]
        public void AddKjarniEmbeddingGenerator_NullServices_Throws()
        {
            Assert.Throws<ArgumentNullException>(
                () => ((IServiceCollection)null!).AddKjarniEmbeddingGenerator());
        }

        /// <summary>
        /// Element-wise comparison with an absolute tolerance.
        /// </summary>
        /// <remarks>
        /// Deliberately not xUnit's <c>precision:</c> overload, which rounds both operands to N
        /// decimals before comparing — two values 1e-7 apart but straddling a rounding boundary
        /// (0.0153350 vs 0.0153349) fail that check despite being numerically equivalent for our
        /// purposes. Single-encode and batch-encode legitimately differ at that magnitude because
        /// the matmul accumulation order differs, so tolerance is the correct assertion here.
        /// </remarks>
        private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-5f)
        {
            Assert.Equal(expected.Length, actual.Length);

            for (int i = 0; i < expected.Length; i++)
            {
                float delta = MathF.Abs(expected[i] - actual[i]);
                Assert.True(
                    delta <= tolerance,
                    $"index {i}: expected {expected[i]:R}, got {actual[i]:R} (delta {delta:R} > {tolerance:R})");
            }
        }
    }
}
