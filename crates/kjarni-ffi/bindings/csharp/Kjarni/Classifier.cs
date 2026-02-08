using System;
using System.Linq;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Result of classifying a single text.
    /// </summary>
    public class ClassificationResult
    {
        /// <summary>The predicted label (highest score).</summary>
        public string Label { get; }

        /// <summary>Confidence score for the predicted label (0.0 - 1.0).</summary>
        public float Score { get; }

        /// <summary>All labels with their scores, sorted by score descending.</summary>
        public IReadOnlyList<(string Label, float Score)> AllScores { get; }

        internal ClassificationResult((string Label, float Score)[] scores)
        {
            AllScores = scores;
            Label = scores[0].Label;
            Score = scores[0].Score;
        }

        /// <summary>Get the top K predictions.</summary>
        public IEnumerable<(string Label, float Score)> TopK(int k)
            => AllScores.Take(k);

        /// <summary>Check if the top prediction exceeds a confidence threshold.</summary>
        public bool IsConfident(float threshold)
            => Score >= threshold;

        /// <summary>Get predictions above a threshold.</summary>
        public IEnumerable<(string Label, float Score)> AboveThreshold(float threshold)
            => AllScores.Where(x => x.Score >= threshold);

        public override string ToString()
            => $"{Label} ({Score * 100:F1}%)";
    }

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
        public ClassificationResult Classify(string text)
        {
            ThrowIfDisposed();

            var err = Native.kjarni_classifier_classify(_handle, text, out var result);
            Native.CheckError(err);

            try
            {
                return new ClassificationResult(result.ToArray());
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
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_classifier_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Classifier()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(Classifier));
        }
    }
}