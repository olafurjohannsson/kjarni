using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Summarizes article-length text.
    /// </summary>
    /// <example>
    /// <code>
    /// using var summarizer = new Summarizer();
    /// Console.WriteLine(summarizer.Summarize(article));
    /// </code>
    /// </example>
    public class Summarizer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Load a summarization model, downloading it on first use.
        /// </summary>
        /// <param name="model">Model name, e.g. "distilbart-cnn" (the default) or "bart-large-cnn".</param>
        /// <param name="minLength">Minimum summary length in tokens; 0 uses the model's default.</param>
        /// <param name="maxLength">Maximum summary length in tokens; 0 uses the model's default.</param>
        /// <param name="device">Compute device - "cpu" or "gpu".</param>
        /// <param name="cacheDir">Directory to cache downloaded models.</param>
        /// <param name="quiet">Suppress progress output.</param>
        /// <param name="modelPath">A local folder holding a BART or T5 model in the Hugging Face layout, with weights as safetensors or pytorch_model.bin. Used instead of <paramref name="model"/> when given.</param>
        public Summarizer(
            string? model = null,
            int minLength = 0,
            int maxLength = 0,
            string device = "cpu",
            string? cacheDir = null,
            bool quiet = false,
            string? modelPath = null)
        {
            if (minLength < 0) throw new ArgumentOutOfRangeException(nameof(minLength));
            if (maxLength < 0) throw new ArgumentOutOfRangeException(nameof(maxLength));

            var config = Native.kjarni_summarizer_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.MinLength = (UIntPtr)minLength;
            config.MaxLength = (UIntPtr)maxLength;
            config.Quiet = quiet ? 1 : 0;

            var modelPtr = Native.Utf8OrNull(model);
            var cacheDirPtr = Native.Utf8OrNull(cacheDir);
            var modelPathPtr = Native.Utf8OrNull(modelPath);
            try
            {
                if (modelPtr != IntPtr.Zero) config.ModelName = modelPtr;
                config.CacheDir = cacheDirPtr;
                config.ModelPath = modelPathPtr;
                Native.CheckError(Native.kjarni_summarizer_new(ref config, out _handle));
            }
            finally
            {
                Marshal.FreeCoTaskMem(modelPtr);
                Marshal.FreeCoTaskMem(cacheDirPtr);
                Marshal.FreeCoTaskMem(modelPathPtr);
            }
        }

        /// <summary>
        /// Summarize <paramref name="text"/>.
        /// </summary>
        public string Summarize(string text)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_summarizer_summarize(_handle, text, out var result));
            return Native.TakeString(result);
        }

        /// <summary>Releases the native model. The object cannot be used afterwards.</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_summarizer_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>Releases the native model if <see cref="Dispose"/> was not called.</summary>
        ~Summarizer()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Summarizer));
        }
    }
}
