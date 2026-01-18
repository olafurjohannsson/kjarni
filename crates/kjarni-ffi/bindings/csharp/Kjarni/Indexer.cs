using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// Statistics returned after indexing operations.
    /// </summary>
    public record IndexStats(
        int DocumentsIndexed,
        int ChunksCreated,
        int Dimension,
        long SizeBytes,
        int FilesProcessed,
        int FilesSkipped,
        long ElapsedMs
    );

    /// <summary>
    /// Information about an existing index.
    /// </summary>
    public record IndexInfo(
        string Path,
        int DocumentCount,
        int SegmentCount,
        int Dimension,
        long SizeBytes,
        string? EmbeddingModel
    );

    /// <summary>
    /// Progress update during indexing operations.
    /// </summary>
    public record Progress(
        string Stage,
        int Current,
        int Total,
        string? Message
    );

    /// <summary>
    /// Token to cancel long-running operations.
    /// </summary>
    public class CancelToken : IDisposable
    {
        internal IntPtr Handle { get; private set; }
        private bool _disposed;

        public CancelToken()
        {
            Handle = Native.kjarni_cancel_token_new();
        }

        /// <summary>
        /// Request cancellation of the associated operation.
        /// </summary>
        public void Cancel()
        {
            if (!_disposed && Handle != IntPtr.Zero)
            {
                Native.kjarni_cancel_token_cancel(Handle);
            }
        }

        /// <summary>
        /// Check if cancellation has been requested.
        /// </summary>
        public bool IsCancelled
        {
            get
            {
                if (_disposed || Handle == IntPtr.Zero) return false;
                return Native.kjarni_cancel_token_is_cancelled(Handle);
            }
        }

        /// <summary>
        /// Reset the token for reuse with another operation.
        /// </summary>
        public void Reset()
        {
            if (!_disposed && Handle != IntPtr.Zero)
            {
                Native.kjarni_cancel_token_reset(Handle);
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (Handle != IntPtr.Zero)
                {
                    Native.kjarni_cancel_token_free(Handle);
                    Handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~CancelToken()
        {
            Dispose();
        }
    }

    /// <summary>
    /// Document indexer for RAG applications.
    /// </summary>
    /// <example>
    /// <code>
    /// using var indexer = new Indexer(model: "minilm-l6-v2");
    /// var stats = indexer.Create("my_index", new[] { "docs/" });
    /// Console.WriteLine($"Indexed {stats.DocumentsIndexed} documents");
    /// </code>
    /// </example>
    public class Indexer : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Create a new Indexer.
        /// </summary>
        /// <param name="model">Embedding model name (default: "minilm-l6-v2")</param>
        /// <param name="device">Compute device - "cpu" or "gpu"</param>
        /// <param name="cacheDir">Directory to cache downloaded models</param>
        /// <param name="chunkSize">Maximum chunk size in characters</param>
        /// <param name="chunkOverlap">Overlap between chunks in characters</param>
        /// <param name="batchSize">Batch size for embedding operations</param>
        /// <param name="extensions">File extensions to include (null = defaults)</param>
        /// <param name="excludePatterns">Glob patterns to exclude</param>
        /// <param name="recursive">Recurse into subdirectories</param>
        /// <param name="includeHidden">Include hidden files</param>
        /// <param name="maxFileSize">Skip files larger than this in bytes</param>
        /// <param name="quiet">Suppress progress output</param>
        public Indexer(
            string model = "minilm-l6-v2",
            string device = "cpu",
            string? cacheDir = null,
            int chunkSize = 512,
            int chunkOverlap = 50,
            int batchSize = 32,
            IEnumerable<string>? extensions = null,
            IEnumerable<string>? excludePatterns = null,
            bool recursive = true,
            bool includeHidden = false,
            int maxFileSize = 10 * 1024 * 1024,
            bool quiet = false)
        {
            var config = Native.kjarni_indexer_config_default();

            config.Device = device.ToLowerInvariant() == "gpu"
                ? KjarniDevice.Gpu
                : KjarniDevice.Cpu;

            var modelBytes = System.Text.Encoding.UTF8.GetBytes(model + "\0");
            var modelHandle = GCHandle.Alloc(modelBytes, GCHandleType.Pinned);

            GCHandle? cacheDirHandle = null;
            GCHandle? extensionsHandle = null;
            GCHandle? excludeHandle = null;

            try
            {
                config.ModelName = modelHandle.AddrOfPinnedObject();
                config.ChunkSize = (UIntPtr)chunkSize;
                config.ChunkOverlap = (UIntPtr)chunkOverlap;
                config.BatchSize = (UIntPtr)batchSize;
                config.Recursive = recursive ? 1 : 0;
                config.IncludeHidden = includeHidden ? 1 : 0;
                config.MaxFileSize = (UIntPtr)maxFileSize;
                config.Quiet = quiet ? 1 : 0;

                if (cacheDir != null)
                {
                    var cacheDirBytes = System.Text.Encoding.UTF8.GetBytes(cacheDir + "\0");
                    cacheDirHandle = GCHandle.Alloc(cacheDirBytes, GCHandleType.Pinned);
                    config.CacheDir = cacheDirHandle.Value.AddrOfPinnedObject();
                }

                if (extensions != null)
                {
                    var extStr = string.Join(",", extensions) + "\0";
                    var extBytes = System.Text.Encoding.UTF8.GetBytes(extStr);
                    extensionsHandle = GCHandle.Alloc(extBytes, GCHandleType.Pinned);
                    config.Extensions = extensionsHandle.Value.AddrOfPinnedObject();
                }

                if (excludePatterns != null)
                {
                    var exclStr = string.Join(",", excludePatterns) + "\0";
                    var exclBytes = System.Text.Encoding.UTF8.GetBytes(exclStr);
                    excludeHandle = GCHandle.Alloc(exclBytes, GCHandleType.Pinned);
                    config.ExcludePatterns = excludeHandle.Value.AddrOfPinnedObject();
                }

                var err = Native.kjarni_indexer_new(ref config, out _handle);
                Native.CheckError(err);
            }
            finally
            {
                modelHandle.Free();
                cacheDirHandle?.Free();
                extensionsHandle?.Free();
                excludeHandle?.Free();
            }
        }

        /// <summary>
        /// Create a new index from files/directories.
        /// </summary>
        /// <param name="indexPath">Path where the index will be created</param>
        /// <param name="inputs">List of file or directory paths to index</param>
        /// <param name="force">Overwrite existing index</param>
        /// <param name="onProgress">Optional progress callback</param>
        /// <param name="cancelToken">Optional cancellation token</param>
        /// <returns>IndexStats with indexing statistics</returns>
        public IndexStats Create(
            string indexPath,
            IEnumerable<string> inputs,
            bool force = false,
            Action<Progress>? onProgress = null,
            CancelToken? cancelToken = null)
        {
            using var utf8Inputs = new Utf8StringArray(inputs);
            if (utf8Inputs.Length == 0)
            {
                throw new ArgumentException("No input paths specified", nameof(inputs));
            }

            Native.KjarniIndexStats stats;

            if (onProgress == null && cancelToken == null)
            {
                // Simple path
                var err = Native.kjarni_indexer_create(
                    _handle,
                    indexPath,
                    utf8Inputs.Pointers,
                    (UIntPtr)utf8Inputs.Length,
                    force ? 1 : 0,
                    out stats);
                Native.CheckError(err);
            }
            else
            {
                // Callback path
                Native.KjarniProgressCallback? callback = null;
                if (onProgress != null)
                {
                    callback = (progress, userData) =>
                    {
                        var stage = progress.Stage switch
                        {
                            KjarniProgressStage.Scanning => "scanning",
                            KjarniProgressStage.Loading => "loading",
                            KjarniProgressStage.Embedding => "embedding",
                            KjarniProgressStage.Writing => "writing",
                            KjarniProgressStage.Committing => "committing",
                            _ => "unknown"
                        };

                        var message = progress.Message != IntPtr.Zero
                            ? Marshal.PtrToStringUTF8(progress.Message)
                            : null;

                        onProgress(new Progress(
                            stage,
                            (int)progress.Current,
                            (int)progress.Total,
                            message));
                    };
                }

                var err = Native.kjarni_indexer_create_with_callback(
                    _handle,
                    indexPath,
                    utf8Inputs.Pointers,
                    (UIntPtr)utf8Inputs.Length,
                    force ? 1 : 0,
                    callback,
                    IntPtr.Zero,
                    cancelToken?.Handle ?? IntPtr.Zero,
                    out stats);
                Native.CheckError(err);
            }

            return new IndexStats(
                (int)stats.DocumentsIndexed,
                (int)stats.ChunksCreated,
                (int)stats.Dimension,
                (long)stats.SizeBytes,
                (int)stats.FilesProcessed,
                (int)stats.FilesSkipped,
                (long)stats.ElapsedMs);
        }

        /// <summary>
        /// Add documents to an existing index.
        /// </summary>
        /// <param name="indexPath">Path to existing index</param>
        /// <param name="inputs">List of file or directory paths to add</param>
        /// <param name="onProgress">Optional progress callback</param>
        /// <param name="cancelToken">Optional cancellation token</param>
        /// <returns>Number of documents added</returns>
        public int Add(
            string indexPath,
            IEnumerable<string> inputs,
            Action<Progress>? onProgress = null,
            CancelToken? cancelToken = null)
        {
            using var utf8Inputs = new Utf8StringArray(inputs);
            if (utf8Inputs.Length == 0)
            {
                return 0;
            }

            UIntPtr docsAdded;

            if (onProgress == null && cancelToken == null)
            {
                // Simple path
                var err = Native.kjarni_indexer_add(
                    _handle,
                    indexPath,
                    utf8Inputs.Pointers,
                    (UIntPtr)utf8Inputs.Length,
                    out docsAdded);
                Native.CheckError(err);
            }
            else
            {
                // Callback path
                Native.KjarniProgressCallback? callback = null;
                if (onProgress != null)
                {
                    callback = (progress, userData) =>
                    {
                        var stage = progress.Stage switch
                        {
                            KjarniProgressStage.Scanning => "scanning",
                            KjarniProgressStage.Loading => "loading",
                            KjarniProgressStage.Embedding => "embedding",
                            KjarniProgressStage.Writing => "writing",
                            KjarniProgressStage.Committing => "committing",
                            _ => "unknown"
                        };

                        var message = progress.Message != IntPtr.Zero
                            ? Marshal.PtrToStringUTF8(progress.Message)
                            : null;

                        onProgress(new Progress(
                            stage,
                            (int)progress.Current,
                            (int)progress.Total,
                            message));
                    };
                }

                var err = Native.kjarni_indexer_add_with_callback(
                    _handle,
                    indexPath,
                    utf8Inputs.Pointers,
                    (UIntPtr)utf8Inputs.Length,
                    callback,
                    IntPtr.Zero,
                    cancelToken?.Handle ?? IntPtr.Zero,
                    out docsAdded);
                Native.CheckError(err);
            }

            return (int)docsAdded;
        }

        /// <summary>
        /// Get information about an existing index.
        /// </summary>
        /// <param name="indexPath">Path to the index directory</param>
        /// <returns>IndexInfo with index statistics</returns>
        public static IndexInfo GetInfo(string indexPath)
        {
            var err = Native.kjarni_index_info(indexPath, out var info);
            Native.CheckError(err);

            var result = new IndexInfo(
                info.Path != IntPtr.Zero ? Marshal.PtrToStringUTF8(info.Path) ?? "" : "",
                (int)info.DocumentCount,
                (int)info.SegmentCount,
                (int)info.Dimension,
                (long)info.SizeBytes,
                info.EmbeddingModel != IntPtr.Zero ? Marshal.PtrToStringUTF8(info.EmbeddingModel) : null);

            Native.kjarni_index_info_free(info);
            return result;
        }

        /// <summary>
        /// Delete an index.
        /// </summary>
        /// <param name="indexPath">Path to the index to delete</param>
        public static void Delete(string indexPath)
        {
            var err = Native.kjarni_index_delete(indexPath);
            Native.CheckError(err);
        }

        /// <summary>
        /// Get the embedding model name used by this indexer.
        /// </summary>
        public string ModelName
        {
            get
            {
                var ptr = Native.kjarni_indexer_model_name(_handle);
                return ptr != IntPtr.Zero ? Marshal.PtrToStringUTF8(ptr) ?? "" : "";
            }
        }

        /// <summary>
        /// Get the embedding dimension produced by the model.
        /// </summary>
        public int Dimension => (int)Native.kjarni_indexer_dimension(_handle);

        /// <summary>
        /// Get the configured chunk size in characters.
        /// </summary>
        public int ChunkSize => (int)Native.kjarni_indexer_chunk_size(_handle);

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_indexer_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Indexer()
        {
            Dispose();
        }
    }
}