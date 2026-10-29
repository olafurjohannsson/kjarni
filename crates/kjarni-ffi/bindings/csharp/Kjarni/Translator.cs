using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Translates text between languages.
    /// </summary>
    /// <example>
    /// <code>
    /// using var translator = new Translator(from: "en", to: "de");
    /// Console.WriteLine(translator.Translate("The house is small."));
    /// </code>
    /// </example>
    public class Translator : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Load a translation model, downloading it on first use.
        /// </summary>
        /// <param name="model">Model name, e.g. "flan-t5-base" (the default) or "flan-t5-large".</param>
        /// <param name="from">Default source language ("en", "english", ...), used when a call passes none.</param>
        /// <param name="to">Default target language, used when a call passes none.</param>
        /// <param name="device">Compute device - "cpu" or "gpu".</param>
        /// <param name="cacheDir">Directory to cache downloaded models.</param>
        /// <param name="quiet">Suppress progress output.</param>
        public Translator(
            string? model = null,
            string? from = null,
            string? to = null,
            string device = "cpu",
            string? cacheDir = null,
            bool quiet = false)
        {
            var config = Native.kjarni_translator_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Quiet = quiet ? 1 : 0;

            var modelPtr = Native.Utf8OrNull(model);
            var cacheDirPtr = Native.Utf8OrNull(cacheDir);
            var fromPtr = Native.Utf8OrNull(from);
            var toPtr = Native.Utf8OrNull(to);
            try
            {
                if (modelPtr != IntPtr.Zero) config.ModelName = modelPtr;
                config.CacheDir = cacheDirPtr;
                config.From = fromPtr;
                config.To = toPtr;
                Native.CheckError(Native.kjarni_translator_new(ref config, out _handle));
            }
            finally
            {
                Marshal.FreeCoTaskMem(modelPtr);
                Marshal.FreeCoTaskMem(cacheDirPtr);
                Marshal.FreeCoTaskMem(fromPtr);
                Marshal.FreeCoTaskMem(toPtr);
            }
        }

        /// <summary>
        /// Translate <paramref name="text"/>. Languages left null fall back to the
        /// defaults given at construction; with neither, this throws with
        /// <see cref="KjarniErrorCode.InvalidConfig"/>.
        /// </summary>
        public string Translate(string text, string? from = null, string? to = null)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_translator_translate(_handle, text, from, to, out var result));
            return Native.TakeString(result);
        }

        /// <summary>Releases the native model. The object cannot be used afterwards.</summary>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_translator_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>Releases the native model if <see cref="Dispose"/> was not called.</summary>
        ~Translator()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Translator));
        }
    }
}
