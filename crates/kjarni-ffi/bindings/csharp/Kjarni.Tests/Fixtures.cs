using System;
using System.IO;

namespace Kjarni.Tests
{
    /// <summary>
    /// Paths to the repo's shared test fixtures, found by walking up from the test
    /// binary so the tests run from any working directory.
    /// </summary>
    internal static class Fixtures
    {
        public static string RepoRoot { get; } = FindRoot();

        public static string SpeechWav => Path.Combine(RepoRoot, "crates", "kjarni", "tests", "fixtures", "speech.wav");

        public static string Photos => Path.Combine(RepoRoot, "tests", "fixtures", "clip", "photos");

        private static string FindRoot()
        {
            var dir = new DirectoryInfo(AppContext.BaseDirectory);
            while (dir != null && !File.Exists(Path.Combine(dir.FullName, "crates", "kjarni-ffi", "Cargo.toml")))
                dir = dir.Parent;
            return dir?.FullName ?? throw new InvalidOperationException("repo root not found above " + AppContext.BaseDirectory);
        }
    }

    /// <summary>
    /// Tests needing a model of several hundred megabytes skip when
    /// KJARNI_SKIP_HEAVY is set, so CI can run the rest without the downloads.
    /// Call first, before anything loads a model.
    /// </summary>
    internal static class Heavy
    {
        public static void SkipIfSet() =>
            Xunit.Skip.If(!string.IsNullOrEmpty(Environment.GetEnvironmentVariable("KJARNI_SKIP_HEAVY")),
                "KJARNI_SKIP_HEAVY is set");
    }

    /// <summary>
    /// A scratch directory, emptied first and removed on dispose.
    /// </summary>
    internal sealed class TempDir : IDisposable
    {
        public string Path { get; }

        public TempDir(string name)
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), name);
            if (Directory.Exists(Path)) Directory.Delete(Path, true);
            Directory.CreateDirectory(Path);
        }

        public void Dispose()
        {
            try { Directory.Delete(Path, true); } catch (IOException) { }
        }
    }
}
