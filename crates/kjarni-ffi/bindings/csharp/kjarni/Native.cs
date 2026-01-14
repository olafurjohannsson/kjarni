using System;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace Kjarni
{
    internal static class Native
    {
        // Cross-platform library name resolution
        private const string LibName = "kjarni_ffi";

        // Static constructor to set up library resolver
        static Native()
        {
            NativeLibrary.SetDllImportResolver(typeof(Native).Assembly, ImportResolver);
        }

        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (libraryName != LibName)
            {
                return IntPtr.Zero;
            }

            // Try platform-specific names
            string[] candidates;
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                candidates = new[] { "kjarni_ffi.dll", "libkjarni_ffi.dll" };
            }
            else if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                candidates = new[] { "libkjarni_ffi.dylib", "kjarni_ffi.dylib" };
            }
            else
            {
                candidates = new[] { "libkjarni_ffi.so", "kjarni_ffi.so" };
            }

            // Search paths
            var assemblyDir = Path.GetDirectoryName(assembly.Location) ?? ".";
            var searchDirs = new[]
            {
                assemblyDir,
                Path.Combine(assemblyDir, "runtimes", GetRuntimeIdentifier(), "native"),
                Path.Combine(assemblyDir, "native"),
                Environment.CurrentDirectory,
            };

            foreach (var dir in searchDirs)
            {
                foreach (var candidate in candidates)
                {
                    var fullPath = Path.Combine(dir, candidate);
                    if (File.Exists(fullPath))
                    {
                        return NativeLibrary.Load(fullPath);
                    }
                }
            }

            // Fall back to system search
            foreach (var candidate in candidates)
            {
                if (NativeLibrary.TryLoad(candidate, out var handle))
                {
                    return handle;
                }
            }

            return IntPtr.Zero;
        }

        private static string GetRuntimeIdentifier()
        {
            var arch = RuntimeInformation.OSArchitecture switch
            {
                Architecture.X64 => "x64",
                Architecture.Arm64 => "arm64",
                Architecture.X86 => "x86",
                _ => "x64"
            };

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return $"win-{arch}";
            if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return $"osx-{arch}";
            return $"linux-{arch}";
        }

        // ... rest of the declarations stay the same ...
        
        // =================================================================
        // Error Codes (same as before)
        // =================================================================

        public enum KjarniError
        {
            Ok = 0,
            NullPointer = 1,
            InvalidUtf8 = 2,
            ModelNotFound = 3,
            LoadFailed = 4,
            InferenceFailed = 5,
            GpuUnavailable = 6,
            InvalidConfig = 7,
            Cancelled = 8,
            Timeout = 9,
            StreamEnded = 10,
            Unknown = 255,
        }

        public enum KjarniDevice
        {
            Cpu = 0,
            Gpu = 1,
        }

        // =================================================================
        // Structures (same as before, plus Reranker)
        // =================================================================

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniFloatArray
        {
            public IntPtr Data;
            public UIntPtr Len;

            public float[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Data == IntPtr.Zero) return Array.Empty<float>();
                var result = new float[len];
                Marshal.Copy(Data, result, 0, len);
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniFloat2DArray
        {
            public IntPtr Data;
            public UIntPtr Rows;
            public UIntPtr Cols;

            public float[][] ToArray()
            {
                var rows = (int)Rows;
                var cols = (int)Cols;
                if (rows == 0 || cols == 0 || Data == IntPtr.Zero)
                    return Array.Empty<float[]>();

                var flat = new float[rows * cols];
                Marshal.Copy(Data, flat, 0, rows * cols);

                var result = new float[rows][];
                for (int i = 0; i < rows; i++)
                {
                    result[i] = new float[cols];
                    Array.Copy(flat, i * cols, result[i], 0, cols);
                }
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniEmbedderConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public int Normalize;
            public int Quiet;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassResult
        {
            public IntPtr Label;
            public float Score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassResults
        {
            public IntPtr Results;
            public UIntPtr Len;

            public (string Label, float Score)[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Results == IntPtr.Zero)
                    return Array.Empty<(string, float)>();

                var result = new (string, float)[len];
                var structSize = Marshal.SizeOf<KjarniClassResult>();

                for (int i = 0; i < len; i++)
                {
                    var ptr = IntPtr.Add(Results, i * structSize);
                    var item = Marshal.PtrToStructure<KjarniClassResult>(ptr);
                    var label = item.Label != IntPtr.Zero
                        ? Marshal.PtrToStringUTF8(item.Label) ?? ""
                        : "";
                    result[i] = (label, item.Score);
                }
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniClassifierConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public IntPtr Labels;
            public UIntPtr NumLabels;
            public int MultiLabel;
            public int Quiet;
        }

        // Reranker structures
        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankResult
        {
            public UIntPtr Index;
            public float Score;
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankResults
        {
            public IntPtr Results;
            public UIntPtr Len;

            public (int Index, float Score)[] ToArray()
            {
                var len = (int)Len;
                if (len == 0 || Results == IntPtr.Zero)
                    return Array.Empty<(int, float)>();

                var result = new (int, float)[len];
                var structSize = Marshal.SizeOf<KjarniRerankResult>();

                for (int i = 0; i < len; i++)
                {
                    var ptr = IntPtr.Add(Results, i * structSize);
                    var item = Marshal.PtrToStructure<KjarniRerankResult>(ptr);
                    result[i] = ((int)item.Index, item.Score);
                }
                return result;
            }
        }

        [StructLayout(LayoutKind.Sequential)]
        public struct KjarniRerankerConfig
        {
            public KjarniDevice Device;
            public IntPtr CacheDir;
            public IntPtr ModelName;
            public IntPtr ModelPath;
            public int Quiet;
        }

        // =================================================================
        // Function Declarations
        // =================================================================

        // Error handling
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_last_error_message();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_clear_error();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_error_name(KjarniError err);

        // Global
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_init();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr kjarni_version();

        // Memory
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_float_array_free(KjarniFloatArray arr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_float_2d_array_free(KjarniFloat2DArray arr);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_string_free(IntPtr s);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_class_results_free(KjarniClassResults results);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_rerank_results_free(KjarniRerankResults results);

        // Embedder
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniEmbedderConfig kjarni_embedder_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_embedder_new(
            ref KjarniEmbedderConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_embedder_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_embedder_encode(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out KjarniFloatArray result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_embedder_encode_batch(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPUTF8Str)] string[] texts,
            UIntPtr numTexts,
            out KjarniFloat2DArray result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_embedder_similarity(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text1,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text2,
            out float result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_embedder_dim(IntPtr handle);

        // Classifier
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniClassifierConfig kjarni_classifier_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_classifier_new(
            ref KjarniClassifierConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_classifier_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_classifier_classify(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            out KjarniClassResults result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern UIntPtr kjarni_classifier_num_labels(IntPtr handle);

        // Reranker
        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniRerankerConfig kjarni_reranker_config_default();

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_reranker_new(
            ref KjarniRerankerConfig config,
            out IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void kjarni_reranker_free(IntPtr handle);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_reranker_score(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string document,
            out float result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_reranker_rerank(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPUTF8Str)] string[] documents,
            UIntPtr numDocs,
            out KjarniRerankResults result);

        [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
        public static extern KjarniError kjarni_reranker_rerank_top_k(
            IntPtr handle,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string query,
            [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.LPUTF8Str)] string[] documents,
            UIntPtr numDocs,
            UIntPtr topK,
            out KjarniRerankResults result);

        // =================================================================
        // Helpers
        // =================================================================

        public static void CheckError(KjarniError err)
        {
            if (err != KjarniError.Ok)
            {
                var msgPtr = kjarni_last_error_message();
                var msg = msgPtr != IntPtr.Zero
                    ? Marshal.PtrToStringUTF8(msgPtr)
                    : $"Error: {err}";
                throw new KjarniException(msg ?? "Unknown error", err);
            }
        }

        public static string GetVersion()
        {
            var ptr = kjarni_version();
            return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "unknown" : "unknown";
        }
    }

    public class KjarniException : Exception
    {
        public Native.KjarniError ErrorCode { get; }

        public KjarniException(string message, Native.KjarniError code)
            : base(message)
        {
            ErrorCode = code;
        }
    }
}