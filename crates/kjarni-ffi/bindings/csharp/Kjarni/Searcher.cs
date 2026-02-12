using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Kjarni
{
    /// <summary>
    /// Search mode determining how queries are processed.
    /// </summary>
    public enum SearchMode
    {
        /// <summary>BM25 keyword search</summary>
        Keyword = 0,
        /// <summary>Embedding-based semantic search</summary>
        Semantic = 1,
        /// <summary>Combined keyword + semantic (recommended)</summary>
        Hybrid = 2,
    }

    /// <summary>
    /// A single search result.
    /// </summary>
    public record SearchResult(
        float Score,
        int DocumentId,
        string Text,
        Dictionary<string, object> Metadata
    );

    /// <summary>
    /// Document searcher for RAG applications.
    /// </summary>
    /// <example>
    /// <code>
    /// using var searcher = new Searcher(model: "minilm-l6-v2");
    /// var results = searcher.Search("my_index", "What is machine learning?");
    /// foreach (var r in results)
    /// {
    ///     Console.WriteLine($"{r.Score:F4}: {r.Text[..50]}...");
    /// }
    /// </code>
    /// </example>
    public class Searcher : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Create a new Searcher.
        /// </summary>
        /// <param name="model">Embedding model name (must match the model used to create the index)</param>
        /// <param name="device">Compute device - "cpu" or "gpu"</param>
        /// <param name="cacheDir">Directory to cache downloaded models</param>
        /// <param name="rerankerModel">Optional cross-encoder model for reranking</param>
        /// <param name="defaultMode">Default search mode</param>
        /// <param name="defaultTopK">Default number of results</param>
        /// <param name="quiet">Suppress progress output</param>
        public Searcher(
            string model = "minilm-l6-v2",
            string device = "cpu",
            string? cacheDir = null,
            string? rerankerModel = null,
            SearchMode defaultMode = SearchMode.Hybrid,
            int defaultTopK = 10,
            bool quiet = false)
        {
            var config = Native.kjarni_searcher_config_default();

            config.Device = device.ToLowerInvariant() == "gpu"
                ? KjarniDevice.Gpu
                : KjarniDevice.Cpu;

            config.DefaultMode = (KjarniSearchMode)defaultMode;
            config.DefaultTopK = (UIntPtr)defaultTopK;
            config.Quiet = quiet ? 1 : 0;

            var modelBytes = System.Text.Encoding.UTF8.GetBytes(model + "\0");
            var modelHandle = GCHandle.Alloc(modelBytes, GCHandleType.Pinned);

            GCHandle? cacheDirHandle = null;
            GCHandle? rerankHandle = null;

            try
            {
                config.ModelName = modelHandle.AddrOfPinnedObject();

                if (cacheDir != null)
                {
                    var cacheDirBytes = System.Text.Encoding.UTF8.GetBytes(cacheDir + "\0");
                    cacheDirHandle = GCHandle.Alloc(cacheDirBytes, GCHandleType.Pinned);
                    config.CacheDir = cacheDirHandle.Value.AddrOfPinnedObject();
                }

                if (rerankerModel != null)
                {
                    var rerankBytes = System.Text.Encoding.UTF8.GetBytes(rerankerModel + "\0");
                    rerankHandle = GCHandle.Alloc(rerankBytes, GCHandleType.Pinned);
                    config.RerankModel = rerankHandle.Value.AddrOfPinnedObject();
                }

                var err = Native.kjarni_searcher_new(ref config, out _handle);
                Native.CheckError(err);
            }
            finally
            {
                modelHandle.Free();
                cacheDirHandle?.Free();
                rerankHandle?.Free();
            }
        }

        /// <summary>
        /// Search an index.
        /// </summary>
        /// <param name="indexPath">Path to the index directory</param>
        /// <param name="query">Search query string</param>
        /// <param name="mode">Search mode (null = use default)</param>
        /// <param name="topK">Number of results (null = use default)</param>
        /// <param name="rerank">Use reranker (null = auto)</param>
        /// <param name="threshold">Minimum score threshold</param>
        /// <param name="sourcePattern">Filter by source file pattern (glob)</param>
        /// <param name="filterKey">Metadata key to filter on</param>
        /// <param name="filterValue">Required value for filterKey</param>
        /// <returns>List of SearchResult objects</returns>
        public List<SearchResult> Search(
            string indexPath,
            string query,
            SearchMode? mode = null,
            int? topK = null,
            bool? rerank = null,
            float? threshold = null,
            string? sourcePattern = null,
            string? filterKey = null,
            string? filterValue = null)
        {
            var options = Native.kjarni_search_options_default();

            if (mode.HasValue)
            {
                options.Mode = (int)mode.Value;
            }
            if (topK.HasValue)
            {
                options.TopK = (UIntPtr)topK.Value;
            }
            if (rerank.HasValue)
            {
                options.UseReranker = rerank.Value ? 1 : 0;
            }
            if (threshold.HasValue)
            {
                options.Threshold = threshold.Value;
            }

            GCHandle? sourceHandle = null;
            GCHandle? keyHandle = null;
            GCHandle? valueHandle = null;

            try
            {
                if (sourcePattern != null)
                {
                    var bytes = System.Text.Encoding.UTF8.GetBytes(sourcePattern + "\0");
                    sourceHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                    options.SourcePattern = sourceHandle.Value.AddrOfPinnedObject();
                }

                if (filterKey != null)
                {
                    var bytes = System.Text.Encoding.UTF8.GetBytes(filterKey + "\0");
                    keyHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                    options.FilterKey = keyHandle.Value.AddrOfPinnedObject();
                }

                if (filterValue != null)
                {
                    var bytes = System.Text.Encoding.UTF8.GetBytes(filterValue + "\0");
                    valueHandle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
                    options.FilterValue = valueHandle.Value.AddrOfPinnedObject();
                }

                var err = Native.kjarni_searcher_search_with_options(
                    _handle,
                    indexPath,
                    query,
                    ref options,
                    out var results);
                Native.CheckError(err);

                return ParseResults(results);
            }
            finally
            {
                sourceHandle?.Free();
                keyHandle?.Free();
                valueHandle?.Free();
            }
        }

        /// <summary>
        /// Static keyword search (BM25) - no embedding model needed.
        /// </summary>
        /// <param name="indexPath">Path to the index directory</param>
        /// <param name="query">Search query string</param>
        /// <param name="topK">Maximum number of results</param>
        /// <returns>List of SearchResult objects</returns>
        public static List<SearchResult> SearchKeywords(
            string indexPath,
            string query,
            int topK = 10)
        {
            var err = Native.kjarni_search_keywords(
                indexPath,
                query,
                (UIntPtr)topK,
                out var results);
            Native.CheckError(err);

            return ParseResults(results);
        }

        private static List<SearchResult> ParseResults(Native.KjarniSearchResults results)
        {
            var output = new List<SearchResult>();

            if (results.Results != IntPtr.Zero && (int)results.Len > 0)
            {
                var len = (int)results.Len;
                var structSize = Marshal.SizeOf<Native.KjarniSearchResult>();

                for (int i = 0; i < len; i++)
                {
                    var ptr = IntPtr.Add(results.Results, i * structSize);
                    var item = Marshal.PtrToStructure<Native.KjarniSearchResult>(ptr);

                    var text = item.Text != IntPtr.Zero
                        ? Marshal.PtrToStringUTF8(item.Text) ?? ""
                        : "";

                    var metadata = new Dictionary<string, object>();
                    if (item.MetadataJson != IntPtr.Zero)
                    {
                        var json = Marshal.PtrToStringUTF8(item.MetadataJson);
                        if (!string.IsNullOrEmpty(json))
                        {
                            try
                            {
                                var parsed = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json);
                                if (parsed != null)
                                {
                                    foreach (var kvp in parsed)
                                    {
                                        metadata[kvp.Key] = kvp.Value.ValueKind switch
                                        {
                                            JsonValueKind.String => kvp.Value.GetString() ?? "",
                                            JsonValueKind.Number => kvp.Value.GetDouble(),
                                            JsonValueKind.True => true,
                                            JsonValueKind.False => false,
                                            _ => kvp.Value.ToString()
                                        };
                                    }
                                }
                            }
                            catch
                            {
                                // Ignore JSON parse errors
                            }
                        }
                    }

                    output.Add(new SearchResult(
                        item.Score,
                        (int)item.DocumentId,
                        text,
                        metadata));
                }

                Native.kjarni_search_results_free(results);
            }

            return output;
        }

        /// <summary>
        /// Check if a reranker is configured.
        /// </summary>
        public bool HasReranker => Native.kjarni_searcher_has_reranker(_handle);

        /// <summary>
        /// Get the default search mode.
        /// </summary>
        public SearchMode DefaultMode => (SearchMode)Native.kjarni_searcher_default_mode(_handle);

        /// <summary>
        /// Get the default number of results.
        /// </summary>
        public int DefaultTopK => (int)Native.kjarni_searcher_default_top_k(_handle);

        /// <summary>
        /// Get the embedding model name used by this searcher.
        /// </summary>
        public string ModelName
        {
            get
            {
                var required = (int)Native.kjarni_searcher_model_name(_handle, IntPtr.Zero, UIntPtr.Zero);
                if (required == 0) return "";

                var buf = Marshal.AllocHGlobal(required);
                try
                {
                    Native.kjarni_searcher_model_name(_handle, buf, (UIntPtr)required);
                    return Marshal.PtrToStringUTF8(buf) ?? "";
                }
                finally
                {
                    Marshal.FreeHGlobal(buf);
                }
            }
        }

        /// <summary>
        /// Get the reranker model name, or null if no reranker is configured.
        /// </summary>
        public string? RerankerModel
        {
            get
            {
                var required = (int)Native.kjarni_searcher_reranker_model(_handle, IntPtr.Zero, UIntPtr.Zero);
                if (required == 0) return null;

                var buf = Marshal.AllocHGlobal(required);
                try
                {
                    Native.kjarni_searcher_reranker_model(_handle, buf, (UIntPtr)required);
                    return Marshal.PtrToStringUTF8(buf);
                }
                finally
                {
                    Marshal.FreeHGlobal(buf);
                }
            }
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_searcher_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~Searcher()
        {
            Dispose();
        }
    }
}