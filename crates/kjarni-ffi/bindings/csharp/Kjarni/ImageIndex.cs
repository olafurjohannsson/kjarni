using System;
using System.Runtime.InteropServices;

namespace Kjarni
{
    /// <summary>
    /// An image matching a search.
    /// </summary>
    /// <param name="Path">The image file.</param>
    /// <param name="Score">Cosine similarity; small even for good matches, so only the order matters.</param>
    public record ImageHit(string Path, float Score);

    /// <summary>
    /// What scanning a directory did.
    /// </summary>
    /// <param name="Added">Images indexed.</param>
    /// <param name="Skipped">Files that could not be decoded.</param>
    public record ScanReport(int Added, int Skipped);

    /// <summary>
    /// Photos searched by description, with CLIP. Searching is safe from several
    /// threads; adding is not.
    /// </summary>
    /// <example>
    /// <code>
    /// using var index = new ImageIndex();
    /// index.AddDirectory("~/Pictures");
    /// foreach (var hit in index.Search("a dog on a beach", 5))
    ///     Console.WriteLine(hit.Path);
    /// </code>
    /// </example>
    public class ImageIndex : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        /// <summary>
        /// Load a CLIP model.
        /// </summary>
        /// <param name="modelDir">A CLIP model directory; null uses clip-vit-base-32, downloaded on first use.</param>
        public ImageIndex(string? modelDir = null)
        {
            Native.CheckError(Native.kjarni_image_index_new(modelDir, out _handle));
        }

        /// <summary>
        /// Number of images indexed.
        /// </summary>
        public int Count
        {
            get
            {
                ThrowIfDisposed();
                return (int)Native.kjarni_image_index_len(_handle);
            }
        }

        /// <summary>
        /// Index one JPEG or PNG.
        /// </summary>
        public void Add(string path)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_image_index_add(_handle, path));
        }

        /// <summary>
        /// Index every JPEG and PNG under <paramref name="dir"/>.
        /// </summary>
        public ScanReport AddDirectory(string dir)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_image_index_add_directory(_handle, dir, out var report));
            return new ScanReport((int)report.Added, (int)report.Skipped);
        }

        /// <summary>
        /// The <paramref name="topK"/> images best matching <paramref name="query"/>.
        /// </summary>
        public ImageHit[] Search(string query, int topK = 10)
        {
            ThrowIfDisposed();
            if (topK < 0) throw new ArgumentOutOfRangeException(nameof(topK));
            Native.CheckError(Native.kjarni_image_index_search(_handle, query, (UIntPtr)topK, out var results));
            try
            {
                var len = (int)results.Len;
                var hits = new ImageHit[len];
                var size = Marshal.SizeOf<Native.KjarniImageHit>();
                for (int i = 0; i < len; i++)
                {
                    var h = Marshal.PtrToStructure<Native.KjarniImageHit>(IntPtr.Add(results.Hits, i * size));
                    var path = h.Path == IntPtr.Zero ? "" : Marshal.PtrToStringUTF8(h.Path) ?? "";
                    hits[i] = new ImageHit(path, h.Score);
                }
                return hits;
            }
            finally
            {
                Native.kjarni_image_results_free(results);
            }
        }

        /// <summary>
        /// Write the vectors (not the images) to <paramref name="path"/>.
        /// </summary>
        public void Save(string path)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_image_index_save(_handle, path));
        }

        /// <summary>
        /// Restore vectors written by <see cref="Save"/>; returns how many.
        /// </summary>
        public int Load(string path)
        {
            ThrowIfDisposed();
            Native.CheckError(Native.kjarni_image_index_load(_handle, path, out var count));
            return (int)count;
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                if (_handle != IntPtr.Zero)
                {
                    Native.kjarni_image_index_free(_handle);
                    _handle = IntPtr.Zero;
                }
                _disposed = true;
            }
            GC.SuppressFinalize(this);
        }

        ~ImageIndex()
        {
            Dispose();
        }

        private void ThrowIfDisposed()
        {
            if (_disposed)
                throw new ObjectDisposedException(nameof(ImageIndex));
        }
    }
}
