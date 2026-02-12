using System;
using System.Linq;
using Xunit;
using Xunit.Abstractions;

namespace Kjarni.Tests
{
    /// <summary>
    /// GPU inference tests. These require an available GPU.
    /// Skip gracefully if no GPU is present.
    /// </summary>
    [Trait("Category", "GPU")]
    public class GpuEmbedderTests : IDisposable
    {
        private readonly Embedder _gpu;
        private readonly Embedder _cpu;
        private readonly ITestOutputHelper _output;

        public GpuEmbedderTests(ITestOutputHelper output)
        {
            _output = output;
            try
            {
                _gpu = new Embedder(model: "minilm-l6-v2", device: "gpu", quiet: true);
                _cpu = new Embedder(model: "minilm-l6-v2", device: "cpu", quiet: true);
            }
            catch (KjarniException ex) when (ex.ErrorCode == KjarniErrorCode.GpuUnavailable)
            {
                Skip.If(true, "No GPU available");
                throw;
            }
        }

        public void Dispose()
        {
            _gpu?.Dispose();
            _cpu?.Dispose();
        }

        [SkippableFact]
        public void Encode_GpuMatchesCpu()
        {
            var gpuResult = _gpu.Encode("Hello world");
            var cpuResult = _cpu.Encode("Hello world");

            Assert.Equal(cpuResult.Length, gpuResult.Length);

            var similarity = Embedder.CosineSimilarity(gpuResult, cpuResult);
            _output.WriteLine($"GPU-CPU cosine similarity: {similarity:F8}");

            Assert.True(similarity > 0.99f,
                $"GPU and CPU results should be nearly identical, got similarity {similarity:F4}");
        }

        [SkippableFact]
        public void Encode_GpuDimensionCorrect()
        {
            var embedding = _gpu.Encode("Test GPU dimensions");
            Assert.Equal(384, embedding.Length);
        }

        [SkippableFact]
        public void Encode_GpuIsNormalized()
        {
            var embedding = _gpu.Encode("Test GPU normalization");
            var norm = MathF.Sqrt(embedding.Sum(x => x * x));

            _output.WriteLine($"GPU L2 norm: {norm:F8}");
            Assert.InRange(norm, 0.998f, 1.002f);
        }

        [SkippableFact]
        public void Encode_GpuIsDeterministic()
        {
            var a = _gpu.Encode("Deterministic GPU test");
            var b = _gpu.Encode("Deterministic GPU test");

            var similarity = Embedder.CosineSimilarity(a, b);
            Assert.True(similarity > 0.9999f);
        }

        [SkippableFact]
        public void Similarity_GpuSemanticRanking()
        {
            var catDog = _gpu.Similarity("cat", "dog");
            var catQuantum = _gpu.Similarity("cat", "quantum computing");

            _output.WriteLine($"GPU cat-dog: {catDog:F6}");
            _output.WriteLine($"GPU cat-quantum: {catQuantum:F6}");

            Assert.True(catDog > catQuantum);
        }

        [SkippableFact]
        public void EncodeBatch_GpuMatchesCpu()
        {
            var texts = new[] { "First sentence", "Second sentence", "Third sentence" };

            var gpuBatch = _gpu.EncodeBatch(texts);
            var cpuBatch = _cpu.EncodeBatch(texts);

            Assert.Equal(cpuBatch.Length, gpuBatch.Length);

            for (int i = 0; i < texts.Length; i++)
            {
                var similarity = Embedder.CosineSimilarity(gpuBatch[i], cpuBatch[i]);
                _output.WriteLine($"Text {i} GPU-CPU similarity: {similarity:F8}");

                Assert.True(similarity > 0.99f,
                    $"Text {i}: GPU-CPU similarity too low: {similarity:F4}");
            }
        }

        [SkippableFact]
        public void EncodeBatch_GpuLargeBatch()
        {
            var texts = Enumerable.Range(0, 64)
                .Select(i => $"Sentence number {i} for GPU batch test.")
                .ToArray();

            var batch = _gpu.EncodeBatch(texts);
            Assert.Equal(64, batch.Length);

            foreach (var embedding in batch)
            {
                Assert.Equal(384, embedding.Length);
                var norm = MathF.Sqrt(embedding.Sum(x => x * x));
                Assert.InRange(norm, 0.998f, 1.002f);
            }
        }
    }

    [Trait("Category", "GPU")]
    public class GpuClassifierTests : IDisposable
    {
        private readonly Classifier _gpu;
        private readonly Classifier _cpu;
        private readonly ITestOutputHelper _output;

        public GpuClassifierTests(ITestOutputHelper output)
        {
            _output = output;
            try
            {
                _gpu = new Classifier("distilbert-sentiment", device: "gpu", quiet: true);
                _cpu = new Classifier("distilbert-sentiment", quiet: true);
            }
            catch (KjarniException ex) when (ex.ErrorCode == KjarniErrorCode.GpuUnavailable)
            {
                Skip.If(true, "No GPU available");
                throw;
            }
        }

        public void Dispose()
        {
            _gpu?.Dispose();
            _cpu?.Dispose();
        }

        [SkippableFact]
        public void Classify_GpuMatchesCpu()
        {
            var gpuResult = _gpu.Classify("I love this product!");
            var cpuResult = _cpu.Classify("I love this product!");

            _output.WriteLine($"GPU: {gpuResult}");
            _output.WriteLine($"CPU: {cpuResult}");

            Assert.Equal(cpuResult.Label, gpuResult.Label);
            Assert.InRange(MathF.Abs(gpuResult.Score - cpuResult.Score), 0f, 0.01f);
        }

        [SkippableFact]
        public void Classify_GpuPositive()
        {
            var result = _gpu.Classify("This is amazing!");
            _output.WriteLine($"GPU result: {result}");

            Assert.Equal("POSITIVE", result.Label);
            Assert.True(result.Score > 0.9f);
        }

        [SkippableFact]
        public void Classify_GpuNegative()
        {
            var result = _gpu.Classify("This is terrible and awful.");
            _output.WriteLine($"GPU result: {result}");

            Assert.Equal("NEGATIVE", result.Label);
            Assert.True(result.Score > 0.9f);
        }

        [SkippableFact]
        public void Classify_GpuScoresSumToOne()
        {
            var result = _gpu.Classify("Test sentence for GPU");
            var sum = result.AllScores.Sum(s => s.Score);

            _output.WriteLine($"GPU score sum: {sum:F8}");
            Assert.InRange(sum, 0.99f, 1.01f);
        }
    }
}