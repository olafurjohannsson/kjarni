using System;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;

namespace Kjarni.Extensions.AI
{
    /// <summary>
    /// Extension methods for exposing a Kjarni <see cref="Embedder"/> as an
    /// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>.
    /// </summary>
    public static class KjarniEmbeddingGeneratorExtensions
    {
        /// <summary>
        /// Wraps this <see cref="Embedder"/> in an <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>.
        /// </summary>
        /// <param name="embedder">The embedder to wrap.</param>
        /// <param name="modelId">Model identifier reported through metadata.</param>
        /// <param name="ownsEmbedder">
        /// When <see langword="true"/>, disposing the returned generator also disposes
        /// <paramref name="embedder"/>. Defaults to <see langword="false"/> so the caller
        /// keeps ownership of an embedder it created.
        /// </param>
        public static IEmbeddingGenerator<string, Embedding<float>> AsEmbeddingGenerator(
            this Embedder embedder,
            string? modelId = null,
            bool ownsEmbedder = false)
            => new KjarniEmbeddingGenerator(embedder, modelId, ownsEmbedder);

        /// <summary>
        /// Registers a local Kjarni embedding generator as a singleton
        /// <see cref="IEmbeddingGenerator{TInput, TEmbedding}"/>.
        /// </summary>
        /// <remarks>
        /// Registered as a singleton because model weights are loaded once and are expensive to
        /// re-initialize. The generator serializes concurrent calls internally, so sharing one
        /// instance across requests is safe.
        /// </remarks>
        /// <param name="services">The service collection.</param>
        /// <param name="model">Kjarni model name, e.g. <c>minilm-l6-v2</c>.</param>
        /// <param name="device"><c>"cpu"</c> or <c>"gpu"</c>.</param>
        /// <param name="cacheDir">Model cache directory, or <see langword="null"/> for the default.</param>
        /// <param name="normalize">L2-normalize the returned vectors.</param>
        /// <param name="quiet">Suppress model download/progress output.</param>
        public static IServiceCollection AddKjarniEmbeddingGenerator(
            this IServiceCollection services,
            string model = "minilm-l6-v2",
            string device = "cpu",
            string? cacheDir = null,
            bool normalize = true,
            bool quiet = true)
        {
            if (services is null) throw new ArgumentNullException(nameof(services));

            services.AddSingleton<IEmbeddingGenerator<string, Embedding<float>>>(
                _ => new KjarniEmbeddingGenerator(model, device, cacheDir, normalize, quiet));

            return services;
        }
    }
}
