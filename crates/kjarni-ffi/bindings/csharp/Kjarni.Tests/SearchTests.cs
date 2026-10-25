using System;
using System.IO;
using System.Linq;
using Xunit;

namespace Kjarni.Tests
{
    public class SearchTests : IDisposable
    {
        private readonly TempDir _docs = new TempDir("kjarni_cs_docs");
        private readonly string _index;

        public SearchTests()
        {
            File.WriteAllText(Path.Combine(_docs.Path, "cats.txt"), "Cats are small carnivorous mammals that purr and chase mice.");
            File.WriteAllText(Path.Combine(_docs.Path, "rust.md"), "Rust is a systems programming language focused on memory safety.");
            File.WriteAllText(Path.Combine(_docs.Path, "bread.txt"), "Sourdough bread is leavened with a starter of wild yeast.");
            File.WriteAllText(Path.Combine(_docs.Path, "skip.log"), "This file has an extension that is not indexed.");
            _index = Path.Combine(_docs.Path, "index");
        }

        public void Dispose() => _docs.Dispose();

        private static bool FromFile(SearchResult r, string name) =>
            r.Metadata.Values.Any(v => v?.ToString()?.Contains(name) == true);

        [Fact]
        public void IndexAddAndSearch()
        {
            using var indexer = new Indexer(extensions: new[] { "txt", "md" }, quiet: true);
            var stats = indexer.Create(_index, new[] { _docs.Path });
            Assert.Equal(3, stats.DocumentsIndexed);
            Assert.Equal(384, stats.Dimension);

            var again = Assert.Throws<KjarniException>(() => indexer.Create(_index, new[] { _docs.Path }));
            Assert.NotEqual(KjarniErrorCode.Ok, again.ErrorCode);
            indexer.Create(_index, new[] { _docs.Path }, force: true);

            var tea = Path.Combine(_docs.Path, "tea.txt");
            File.WriteAllText(tea, "Green tea is brewed from unoxidised leaves of the tea plant.");
            Assert.True(indexer.Add(_index, new[] { tea }) >= 1);

            using var searcher = new Searcher(quiet: true);
            Assert.False(searcher.HasReranker);

            var hits = searcher.Search(_index, "feline pets");
            Assert.NotEmpty(hits);
            Assert.Contains("cats", hits[0].Text, StringComparison.OrdinalIgnoreCase);
            Assert.True(FromFile(hits[0], "cats.txt"));

            var semantic = searcher.Search(_index, "a drink made from leaves", mode: SearchMode.Semantic, topK: 1);
            Assert.Single(semantic);
            Assert.Contains("tea", semantic[0].Text, StringComparison.OrdinalIgnoreCase);

            var keyword = searcher.Search(_index, "sourdough", mode: SearchMode.Keyword);
            Assert.NotEmpty(keyword);
            Assert.Contains("sourdough", keyword[0].Text, StringComparison.OrdinalIgnoreCase);

            var filtered = searcher.Search(_index, "programming", sourcePattern: "*.md");
            Assert.NotEmpty(filtered);
            Assert.All(filtered, r => Assert.True(FromFile(r, "rust.md")));

            Assert.Throws<KjarniException>(() => searcher.Search(Path.Combine(_docs.Path, "nope"), "x"));
        }
    }
}
