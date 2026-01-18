using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Text classification model.
    /// </summary>
    public class Classifier : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Create a new Classifier.
        /// </summary>
        public Classifier(
            string model = "sentiment",
            string device = "cpu",
            string? cacheDir = null,
            string[]? labels = null,
            bool multiLabel = false,
            bool quiet = false)
        {
            var config = Native.kjarni_classifier_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.MultiLabel = multiLabel ? 1 : 0;
            config.Quiet = quiet ? 1 : 0;

            IntPtr modelNamePtr = IntPtr.Zero;
            IntPtr cacheDirPtr = IntPtr.Zero;
            IntPtr labelsPtr = IntPtr.Zero;
            IntPtr[] labelPtrs = Array.Empty<IntPtr>();

            try
            {
                modelNamePtr = Marshal.StringToCoTaskMemUTF8(model);
                config.ModelName = modelNamePtr;

                if (cacheDir != null)
                {
                    cacheDirPtr = Marshal.StringToCoTaskMemUTF8(cacheDir);
                    config.CacheDir = cacheDirPtr;
                }

                if (labels != null && labels.Length > 0)
                {
                    labelPtrs = new IntPtr[labels.Length];
                    for (int i = 0; i < labels.Length; i++)
                    {
                        labelPtrs[i] = Marshal.StringToCoTaskMemUTF8(labels[i]);
                    }
                    labelsPtr = Marshal.AllocCoTaskMem(IntPtr.Size * labels.Length);
                    Marshal.Copy(labelPtrs, 0, labelsPtr, labels.Length);
                    config.Labels = labelsPtr;
                    config.NumLabels = (UIntPtr)labels.Length;
                }

                var err = Native.kjarni_classifier_new(ref config, out _handle);
                Native.CheckError(err);
            }
            finally
            {
                if (modelNamePtr != IntPtr.Zero) Marshal.FreeCoTaskMem(modelNamePtr);
                if (cacheDirPtr != IntPtr.Zero) Marshal.FreeCoTaskMem(cacheDirPtr);
                if (labelsPtr != IntPtr.Zero) Marshal.FreeCoTaskMem(labelsPtr);
                foreach (var ptr in labelPtrs)
                {
                    if (ptr != IntPtr.Zero) Marshal.FreeCoTaskMem(ptr);
                }
            }
        }

        /// <summary>
        /// Classify a single text.
        /// </summary>
        /// <returns>Array of (label, score) tuples sorted by score.</returns>
        public (string Label, float Score)[] Classify(string text)
        {
            ThrowIfDisposed();

            var err = Native.kjarni_classifier_classify(_handle, text, out var result);
            Native.CheckError(err);

            try
            {
                return result.ToArray();
            }
            finally
            {
                Native.kjarni_class_results_free(result);
            }
        }

        /// <summary>
        /// Get number of classification labels.
        /// </summary>
        public int NumLabels
        {
            get
            {
                ThrowIfDisposed();
                return (int)Native.kjarni_classifier_num_labels(_handle);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                Native.kjarni_classifier_free(_handle);
                _disposed = true;
            }
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Classifier));
        }
    }
}