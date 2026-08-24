using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace Kjarni.Extensions.AI
{
    /// <summary>
    /// A <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/> backed by a local Kjarni
    /// <see cref="Embedder"/>. Runs entirely in-process: no HTTP, no Python, no cloud.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Because inference is local and CPU/GPU bound rather than I/O bound, generation runs
    /// synchronously on the calling thread and returns an already-completed task. There is no
    /// thread-pool hop, so callers see the cost directly rather than through a hidden queue.
    /// </para>
    /// <para>
    /// The underlying native embedder is not re-entrant, so concurrent calls to
    /// <see cref="GenerateAsync"/> are serialized internally. A single instance is safe to
    /// register as a singleton and share across requests.
    /// </para>
    /// </remarks>
    public sealed class KjarniEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
    {
        private const string ProviderNameConstant = "kjarni";

        private readonly Embedder _embedder;
        private readonly bool _ownsEmbedder;
        private readonly EmbeddingGeneratorMetadata _metadata;
        private readonly SemaphoreSlim _gate = new SemaphoreSlim(1, 1);
        private readonly int _dimensions;
        private readonly string _modelId;

        private bool _disposed;

        /// <summary>
        /// Creates a generator that owns its own <see cref="Embedder"/>.
        /// </summary>
        /// <param name="model">Kjarni model name, e.g. <c>minilm-l6-v2</c>.</param>
        /// <param name="device"><c>"cpu"</c> or <c>"gpu"</c>.</param>
        /// <param name="cacheDir">Model cache directory, or <see langword="null"/> for the default.</param>
        /// <param name="normalize">L2-normalize the returned vectors.</param>
        /// <param name="quiet">Suppress model download/progress output.</param>
        public KjarniEmbeddingGenerator(
            string model = "minilm-l6-v2",
            string device = "cpu",
            string? cacheDir = null,
            bool normalize = true,
            bool quiet = true)
            : this(
                new Embedder(model: model, device: device, cacheDir: cacheDir, normalize: normalize, quiet: quiet),
                modelId: model,
                ownsEmbedder: true)
        {
        }

        /// <summary>
        /// Wraps an existing <see cref="Embedder"/>.
        /// </summary>
        /// <param name="embedder">The embedder to delegate to.</param>
        /// <param name="modelId">Model identifier reported through metadata.</param>
        /// <param name="ownsEmbedder">
        /// When <see langword="true"/>, disposing this generator also disposes <paramref name="embedder"/>.
        /// </param>
        public KjarniEmbeddingGenerator(Embedder embedder, string? modelId = null, bool ownsEmbedder = false)
        {
            _embedder = embedder ?? throw new ArgumentNullException(nameof(embedder));
            _ownsEmbedder = ownsEmbedder;
            _modelId = modelId ?? "unknown";
            _dimensions = embedder.Dim;
            _metadata = new EmbeddingGeneratorMetadata(
                providerName: ProviderNameConstant,
                providerUri: null,
                defaultModelId: _modelId,
                defaultModelDimensions: _dimensions);
        }

        /// <summary>The embedding dimension produced by the underlying model.</summary>
        public int Dimensions => _dimensions;

        /// <inheritdoc />
        public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values,
            EmbeddingGenerationOptions? options = null,
            CancellationToken cancellationToken = default)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));
            ThrowIfDisposed();

            if (options?.ModelId is string requested &&
                !string.Equals(requested, _modelId, StringComparison.Ordinal))
            {
                throw new NotSupportedException(
                    $"This generator is bound to model '{_modelId}'. Kjarni loads one model per " +
                    $"instance, so '{requested}' cannot be served here. Construct a separate " +
                    $"{nameof(KjarniEmbeddingGenerator)} for that model.");
            }

            if (options?.Dimensions is int requestedDims && requestedDims != _dimensions)
            {
                throw new NotSupportedException(
                    $"Model '{_modelId}' produces {_dimensions}-dimensional embeddings and does not " +
                    $"support Matryoshka truncation to {requestedDims}.");
            }

            var inputs = values as string[] ?? values.ToArray();
            var result = new GeneratedEmbeddings<Embedding<float>>(inputs.Length);
            if (inputs.Length == 0) return result;

            for (int i = 0; i < inputs.Length; i++)
            {
                if (inputs[i] is null)
                    throw new ArgumentException($"Input at index {i} is null.", nameof(values));
            }

            cancellationToken.ThrowIfCancellationRequested();

            await _gate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                ThrowIfDisposed();
                cancellationToken.ThrowIfCancellationRequested();

                float[][] vectors = _embedder.EncodeBatch(inputs);

                for (int i = 0; i < vectors.Length; i++)
                {
                    result.Add(new Embedding<float>(vectors[i])
                    {
                        ModelId = _modelId,
                        CreatedAt = DateTimeOffset.UtcNow,
                    });
                }
            }
            finally
            {
                _gate.Release();
            }

            return result;
        }

        /// <inheritdoc />
        public object? GetService(Type serviceType, object? serviceKey = null)
        {
            if (serviceType is null) throw new ArgumentNullException(nameof(serviceType));
            if (serviceKey is not null) return null;

            if (serviceType == typeof(EmbeddingGeneratorMetadata)) return _metadata;
            if (serviceType == typeof(Embedder)) return _embedder;
            if (serviceType.IsInstanceOfType(this)) return this;

            return null;
        }

        /// <inheritdoc />
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _gate.Dispose();
            if (_ownsEmbedder) _embedder.Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(KjarniEmbeddingGenerator));
        }
    }
}
