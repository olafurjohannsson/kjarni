using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;

namespace EdgeGpt
{
    /// <summary>
    /// Exception thrown by EdgeGPT operations
    /// </summary>
    public class EdgeGptException : Exception
    {
        public EdgeGptException(string message) : base(message) { }
    }

    /// <summary>
    /// Managed wrapper around the native EdgeGPTHandle (FFI)
    /// </summary>
    public sealed class EdgeGpt : IDisposable
    {
        private IntPtr _handle;

#if WINDOWS
        private const string LIB_NAME = "edgegpt.dll";
#elif OSX
        private const string LIB_NAME = "libedgegpt.dylib";
#else
        private const string LIB_NAME = "libedgegpt.so";
#endif

        // ========== FFI declarations ==========

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr edge_gpt_new_cpu();

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr edge_gpt_new_gpu();

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void edge_gpt_free(IntPtr handle);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_encode(
            IntPtr handle,
            string text,
            out IntPtr outEmbedding,
            out UIntPtr outLen
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_encode_batch(
            IntPtr handle,
            IntPtr[] texts,
            UIntPtr numTexts,
            out IntPtr outEmbeddings,
            out IntPtr outLens,
            out UIntPtr embeddingDim
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_similarity(
            IntPtr handle,
            string text1,
            string text2,
            out float outSimilarity
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_rerank(
            IntPtr handle,
            string query,
            IntPtr[] documents,
            UIntPtr numDocs,
            out IntPtr outIndices,
            out IntPtr outScores
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_generate(
            IntPtr handle,
            string prompt,
            out IntPtr outText
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern int edge_gpt_summarize(
            IntPtr handle,
            string text,
            out IntPtr outSummary
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void edge_gpt_free_float_array(IntPtr data, UIntPtr len);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void edge_gpt_free_batch_embeddings(
            IntPtr embeddings,
            IntPtr lens,
            UIntPtr numTexts,
            UIntPtr embeddingDim
        );

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void edge_gpt_free_usize_array(IntPtr data, UIntPtr len);

        [DllImport(LIB_NAME, CallingConvention = CallingConvention.Cdecl)]
        private static extern void edge_gpt_free_string(IntPtr s);

        // ========== Construction / Destruction ==========

        /// <summary>
        /// Initialize EdgeGPT on CPU (Default)
        /// </summary>
        public EdgeGpt() : this("cpu") { }

        /// <summary>
        /// Initialize EdgeGPT on specified device ("cpu" or "gpu")
        /// </summary>
        public EdgeGpt(string device)
        {
            if (device.ToLower() == "gpu")
            {
                _handle = edge_gpt_new_gpu();
                if (_handle == IntPtr.Zero)
                    throw new EdgeGptException("Failed to create EdgeGPT instance (GPU). Check WGPU support.");
            }
            else
            {
                _handle = edge_gpt_new_cpu();
                if (_handle == IntPtr.Zero)
                    throw new EdgeGptException("Failed to create EdgeGPT instance (CPU)");
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                edge_gpt_free(_handle);
                _handle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~EdgeGpt()
        {
            Dispose();
        }

        private static void CheckError(int err, string operation)
        {
            if (err != 0)
                throw new EdgeGptException($"{operation} failed with error code: {err}");
        }

        // ========== Encode single text ==========

        public float[] Encode(string text)
        {
            int err = edge_gpt_encode(_handle, text, out IntPtr ptr, out UIntPtr len);
            CheckError(err, "encode");

            ulong length = len.ToUInt64();
            float[] result = new float[length];
            Marshal.Copy(ptr, result, 0, (int)length);
            edge_gpt_free_float_array(ptr, len);

            return result;
        }

        // ========== Encode batch of texts ==========

        public List<float[]> EncodeBatch(IEnumerable<string> texts)
        {
            string[] textArray = texts.ToArray();
            IntPtr[] textPtrs = textArray.Select(Marshal.StringToHGlobalAnsi).ToArray();

            try
            {
                int err = edge_gpt_encode_batch(
                    _handle,
                    textPtrs,
                    (UIntPtr)textArray.Length,
                    out IntPtr embeddingsPtr,
                    out IntPtr lensPtr,
                    out UIntPtr embeddingDimPtr
                );
                CheckError(err, "encode_batch");

                ulong numTexts = (ulong)textArray.Length;
                ulong embeddingDim = embeddingDimPtr.ToUInt64();

                List<float[]> result = new((int)numTexts);
                for (ulong i = 0; i < numTexts; i++)
                {
                    IntPtr embeddingPtr = Marshal.ReadIntPtr(embeddingsPtr, (int)((long)i * IntPtr.Size));
                    float[] embedding = new float[embeddingDim];
                    Marshal.Copy(embeddingPtr, embedding, 0, (int)embeddingDim);
                    result.Add(embedding);
                }

                edge_gpt_free_batch_embeddings(embeddingsPtr, lensPtr, (UIntPtr)numTexts, embeddingDimPtr);
                return result;
            }
            finally
            {
                foreach (IntPtr p in textPtrs)
                    Marshal.FreeHGlobal(p);
            }
        }

        // ========== Compute similarity ==========

        public float Similarity(string text1, string text2)
        {
            int err = edge_gpt_similarity(_handle, text1, text2, out float sim);
            CheckError(err, "similarity");
            return sim;
        }

        // ========== Rerank documents ==========

        public List<(ulong Index, float Score)> Rerank(string query, IEnumerable<string> documents)
        {
            string[] docs = documents.ToArray();
            IntPtr[] docPtrs = docs.Select(Marshal.StringToHGlobalAnsi).ToArray();

            try
            {
                int err = edge_gpt_rerank(
                    _handle,
                    query,
                    docPtrs,
                    (UIntPtr)docs.Length,
                    out IntPtr indicesPtr,
                    out IntPtr scoresPtr
                );
                CheckError(err, "rerank");

                ulong len = (ulong)docs.Length;
                ulong[] indices = new ulong[len];
                float[] scores = new float[len];

                // Copy indices (size_t)
                if (IntPtr.Size == 8)
                {
                    for (ulong i = 0; i < len; i++)
                        indices[i] = (ulong)Marshal.ReadInt64(indicesPtr, (int)(i * 8));
                }
                else
                {
                    for (ulong i = 0; i < len; i++)
                        indices[i] = (uint)Marshal.ReadInt32(indicesPtr, (int)(i * 4));
                }

                Marshal.Copy(scoresPtr, scores, 0, (int)len);

                edge_gpt_free_usize_array(indicesPtr, (UIntPtr)len);
                edge_gpt_free_float_array(scoresPtr, (UIntPtr)len);

                List<(ulong, float)> result = new((int)len);
                for (int i = 0; i < (int)len; i++)
                    result.Add((indices[i], scores[i]));

                return result;
            }
            finally
            {
                foreach (IntPtr p in docPtrs)
                    Marshal.FreeHGlobal(p);
            }
        }

        // ========== Generation ==========

        public string Generate(string prompt)
        {
            int err = edge_gpt_generate(_handle, prompt, out IntPtr outPtr);
            CheckError(err, "generate");

            string result = Marshal.PtrToStringAnsi(outPtr);
            edge_gpt_free_string(outPtr);
            return result;
        }

        public string Summarize(string text)
        {
            int err = edge_gpt_summarize(_handle, text, out IntPtr outPtr);
            CheckError(err, "summarize");

            string result = Marshal.PtrToStringAnsi(outPtr);
            edge_gpt_free_string(outPtr);
            return result;
        }
    }
}
