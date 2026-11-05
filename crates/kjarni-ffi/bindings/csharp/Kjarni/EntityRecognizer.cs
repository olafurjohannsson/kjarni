using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// One named entity. <see cref="Start"/> and <see cref="End"/> index the input
    /// string, so <c>text.Substring(Start, End - Start) == Text</c>.
    /// </summary>
    public sealed class Entity
    {
        /// <summary>The entity type, such as PER or Location.</summary>
        public string Label { get; }
        /// <summary>The entity's text, exactly as it appears in the input.</summary>
        public string Text { get; }
        /// <summary>Where the entity starts in the input string.</summary>
        public int Start { get; }
        /// <summary>Where it ends in the input string, exclusive.</summary>
        public int End { get; }
        /// <summary>The mean probability of its tokens' labels.</summary>
        public float Score { get; }

        internal Entity(string label, string text, int start, int end, float score)
        {
            Label = label;
            Text = text;
            Start = start;
            End = end;
            Score = score;
        }

        /// <inheritdoc/>
        public override string ToString() => $"{Label} \"{Text}\" ({Score:F3})";
    }

    /// <summary>How tokens become entities, as transformers' aggregation_strategy.</summary>
    public enum Aggregation
    {
        /// <summary>A word takes the label of its first token, so it is never split.</summary>
        First = 0,
        /// <summary>Each token keeps its own label; neighbours with the same tag join.</summary>
        Simple = 1,
        /// <summary>A word takes the label of its most confident token.</summary>
        Max = 2,
        /// <summary>A word takes the label with the highest probability averaged over it.</summary>
        Average = 3,
    }

    /// <summary>
    /// Finds names of people, places, organizations and other entities with a token
    /// classification model. The entities are those transformers' pipeline("ner")
    /// gives for the same <see cref="Aggregation"/>, with the same spans and scores.
    /// </summary>
    public class EntityRecognizer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>Create an EntityRecognizer.</summary>
        /// <param name="model">Model name; the default is "distilbert-ner", for English.</param>
        /// <param name="device">"cpu" or "gpu"</param>
        /// <param name="cacheDir">Directory to cache models</param>
        /// <param name="aggregation">How tokens become entities</param>
        /// <param name="quiet">Suppress progress output</param>
        /// <param name="modelPath">A local folder holding any *ForTokenClassification model, such as one downloaded from the Hugging Face Hub. Used instead of <paramref name="model"/> when given.</param>
        public EntityRecognizer(
            string? model = null,
            string device = "cpu",
            string? cacheDir = null,
            Aggregation aggregation = Aggregation.First,
            bool quiet = false,
            string? modelPath = null)
        {
            var config = Native.kjarni_ner_config_default();
            config.Device = device == "gpu" ? KjarniDevice.Gpu : KjarniDevice.Cpu;
            config.Aggregation = (int)aggregation;
            config.Quiet = quiet ? 1 : 0;

            var owned = new List<IntPtr>();
            IntPtr Utf8(string s)
            {
                var p = Marshal.StringToCoTaskMemUTF8(s);
                owned.Add(p);
                return p;
            }

            try
            {
                if (model != null) config.ModelName = Utf8(model);
                if (cacheDir != null) config.CacheDir = Utf8(cacheDir);
                if (modelPath != null) config.ModelPath = Utf8(modelPath);
                Native.CheckError(Native.kjarni_ner_new(ref config, out _handle));
            }
            finally
            {
                foreach (var p in owned) Marshal.FreeCoTaskMem(p);
            }
        }

        /// <summary>The entities in <paramref name="text"/>, in order.</summary>
        public Entity[] Recognize(string text)
        {
            ThrowIfDisposed();
            var textPtr = Marshal.StringToCoTaskMemUTF8(text);
            try
            {
                Native.CheckError(Native.kjarni_ner_recognize(_handle, textPtr, out var list));
                try
                {
                    var len = (int)list.Len;
                    var result = new Entity[len];
                    var size = Marshal.SizeOf<Native.KjarniEntity>();
                    for (int i = 0; i < len; i++)
                    {
                        var e = Marshal.PtrToStructure<Native.KjarniEntity>(IntPtr.Add(list.Entities, i * size));
                        result[i] = new Entity(
                            Marshal.PtrToStringUTF8(e.Label) ?? "",
                            Marshal.PtrToStringUTF8(e.Text) ?? "",
                            Utf16Index(text, (long)e.Start),
                            Utf16Index(text, (long)e.End),
                            e.Score);
                    }
                    return result;
                }
                finally
                {
                    Native.kjarni_entities_free(list);
                }
            }
            finally
            {
                Marshal.FreeCoTaskMem(textPtr);
            }
        }

        /// <summary>
        /// The engine counts Unicode code points; a .NET string counts UTF-16 units,
        /// and characters outside the Basic Multilingual Plane, such as emoji, take two.
        /// </summary>
        internal static int Utf16Index(string text, long codePoints)
        {
            int i = 0;
            for (long n = 0; n < codePoints && i < text.Length; n++)
                i += char.IsHighSurrogate(text[i]) && i + 1 < text.Length ? 2 : 1;
            return i;
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(EntityRecognizer));
        }

        /// <inheritdoc/>
        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_ner_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        /// <summary>Frees the native handle if Dispose was not called.</summary>
        ~EntityRecognizer() => Dispose();
    }

    internal static partial class Native
    {
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniEntity
        {
            public IntPtr Label;
            public IntPtr Text;
            public UIntPtr Start;
            public UIntPtr End;
            public float Score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniEntities
        {
            public IntPtr Entities;
            public UIntPtr Len;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniNerConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public int Aggregation;
            public int Quiet;
        }

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniNerConfig kjarni_ner_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_ner_new(ref KjarniNerConfig config, out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_ner_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniErrorCode kjarni_ner_recognize(IntPtr handle, IntPtr text, out KjarniEntities entities);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_entities_free(KjarniEntities entities);
    }
}
