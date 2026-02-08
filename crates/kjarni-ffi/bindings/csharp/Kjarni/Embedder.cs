using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Text embedding model.
    /// </summary>
    public class Embedder : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Create a new Embedder.
        /// </summary>
        /// <param name="model">Model name (e.g., "minilm-l6-v2")</param>
        /// <param name="device">"cpu" or "gpu"</param>
        /// <param name="cacheDir">Directory to cache models</param>
        /// <param name="normalize">L2-normalize embeddings</param>
        /// <param name="quiet">Suppress progress output</param>
        public Embedder(
            string? model = null,
            string device = "cpu",
            string? cacheDir = null,
            bool normalize = true,
            bool quiet = false)
        {
            var config = Native.kjarni_embedder_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Normalize = normalize ? 1 : 0;
            config.Quiet = quiet ? 1 : 0;

            IntPtr modelNamePtr = IntPtr.Zero;
            IntPtr cacheDirPtr = IntPtr.Zero;

            try
            {
                if (model != null)
                {
                    modelNamePtr = Marshal.StringToCoTaskMemUTF8(model);
                    config.ModelName = modelNamePtr;
                }

                if (cacheDir != null)
                {
                    cacheDirPtr = Marshal.StringToCoTaskMemUTF8(cacheDir);
                    config.CacheDir = cacheDirPtr;
                }

                var err = Native.kjarni_embedder_new(ref config, out _handle);
                Native.CheckError(err);
            }
            finally
            {
                if (modelNamePtr != IntPtr.Zero) Marshal.FreeCoTaskMem(modelNamePtr);
                if (cacheDirPtr != IntPtr.Zero) Marshal.FreeCoTaskMem(cacheDirPtr);
            }
        }

        /// <summary>
        /// Encode a single text to an embedding vector.
        /// </summary>
        public float[] Encode(string text)
        {
            ThrowIfDisposed();

            var err = Native.kjarni_embedder_encode(_handle, text, out var result);
            Native.CheckError(err);

            try
            {
                return result.ToArray();
            }
            finally
            {
                Native.kjarni_float_array_free(result);
            }
        }

        /// <summary>
        /// Encode multiple texts to embedding vectors.
        /// </summary>
        public float[][] EncodeBatch(string[] texts)
        {
            ThrowIfDisposed();

            if (texts.Length == 0) return Array.Empty<float[]>();


            using var utf8Texts = new Utf8StringArray(texts);
            var err = Native.kjarni_embedder_encode_batch(_handle, utf8Texts.Pointers, (UIntPtr)utf8Texts.Length, out var result);
            Native.CheckError(err);

            try
            {
                return result.ToArray();
            }
            finally
            {
                Native.kjarni_float_2d_array_free(result);
            }
        }

        /// <summary>
        /// Compute cosine similarity between two texts.
        /// </summary>
        public float Similarity(string text1, string text2)
        {
            ThrowIfDisposed();

            var err = Native.kjarni_embedder_similarity(_handle, text1, text2, out var result);
            Native.CheckError(err);

            return result;
        }

        public static float CosineSimilarity(float[] a, float[] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException($"Vector dimensions must match: {a.Length} vs {b.Length}");
            return Native.kjarni_cosine_similarity(a, b, (nuint)a.Length);
        }

        /// <summary>
        /// Get the embedding dimension.
        /// </summary>
        public int Dim
        {
            get
            {
                ThrowIfDisposed();
                return (int)Native.kjarni_embedder_dim(_handle);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Native.kjarni_embedder_free(_handle);
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Embedder));
        }

        ~Embedder()
        {
            Dispose();
        }
    }
}