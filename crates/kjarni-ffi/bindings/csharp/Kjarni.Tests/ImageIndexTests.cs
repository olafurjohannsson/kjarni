using System;
using System.IO;
using Xunit;

namespace Kjarni.Tests
{
    public class ImageIndexTests : IDisposable
    {
        // Per test, since tests add to it; created after the skip check.
        private ImageIndex? _lazy;
        private ImageIndex _index => _lazy ??= new ImageIndex();

        public void Dispose() => _lazy?.Dispose();

        [SkippableTheory]
        [InlineData("a sandy beach under a blue sky", "beach_holiday")]
        [InlineData("a dense green forest with trees", "deep_forest")]
        [InlineData("a red car", "red_car")]
        [InlineData("a snowy mountain peak", "snowy_mountain")]
        public void ADescriptionFindsTheRightPhoto(string query, string expected)
        {
            Heavy.SkipIfSet();
            var report = _index.AddDirectory(Fixtures.Photos);
            Assert.True(report.Added >= 4);
            Assert.Equal(0, report.Skipped);

            var hits = _index.Search(query, 2);
            Assert.Equal(2, hits.Length);
            Assert.Contains(expected, hits[0].Path);
        }

        [SkippableFact]
        public void SaveAndLoadRoundTrip()
        {
            Heavy.SkipIfSet();
            Assert.Equal(0, _index.Count);
            _index.AddDirectory(Fixtures.Photos);

            using var dir = new TempDir("kjarni_cs_images");
            var saved = Path.Combine(dir.Path, "photos.idx");
            _index.Save(saved);

            using var restored = new ImageIndex();
            Assert.Equal(_index.Count, restored.Load(saved));
            Assert.Contains("red_car", restored.Search("a red car", 1)[0].Path);
        }

        [SkippableFact]
        public void MissingPhotoIsAnError()
        {
            Heavy.SkipIfSet();
            Assert.Throws<KjarniException>(() => _index.Add("/no/such/photo.jpg"));
        }
    }
}
