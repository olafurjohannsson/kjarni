// Tests for the C++ wrapper.
//
// `kjarni.hpp` is 576 lines of lifetime and pointer handling over a C ABI, and none
// of it was covered: the example demonstrates the API but asserts nothing, and CI
// never compiled C++ at all. A regenerated header could have broken every consumer
// silently.
//
// No test framework, on purpose. This directory is something people copy from, and a
// dependency on Catch2 or GoogleTest would be a worse first impression than a
// hundred lines of harness.
//
// Run under sanitizers to make the ownership assertions mean something:
//
//   cmake -B build -DCMAKE_BUILD_TYPE=Debug -DKJARNI_SANITIZE=ON
//   cmake --build build && ctest --test-dir build --output-on-failure

#include "kjarni.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void ok(bool condition, const char* what) {
    ++checks;
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "  FAIL  %s\n", what);
}

void ok_near(float actual, float expected, float tol, const char* what) {
    ++checks;
    if (std::fabs(actual - expected) <= tol) return;
    ++failures;
    std::fprintf(stderr, "  FAIL  %s (got %.4f, expected %.4f ± %.4f)\n", what, actual,
                 expected, tol);
}

void section(const char* name) { std::fprintf(stderr, "%s\n", name); }

// ─── Compile-time contracts ──────────────────────────────────────
//
// The Qt wrapper stores these in std::optional and moves them out of the factory
// functions, so these are load-bearing rather than incidental. Copying any of them
// would double-free the underlying handle.

static_assert(std::is_move_constructible_v<kjarni::Embedder>);
static_assert(std::is_move_constructible_v<kjarni::Classifier>);
static_assert(std::is_move_constructible_v<kjarni::Reranker>);
static_assert(std::is_move_constructible_v<kjarni::Chat>);
static_assert(std::is_move_constructible_v<kjarni::Embedding>);
static_assert(std::is_move_constructible_v<kjarni::Indexer>);
static_assert(std::is_move_constructible_v<kjarni::Searcher>);
static_assert(std::is_move_constructible_v<kjarni::Transcriber>);
static_assert(std::is_move_constructible_v<kjarni::Summarizer>);
static_assert(std::is_move_constructible_v<kjarni::Translator>);
static_assert(std::is_move_constructible_v<kjarni::ImageIndex>);

static_assert(!std::is_copy_constructible_v<kjarni::Embedder>);
static_assert(!std::is_copy_constructible_v<kjarni::Classifier>);
static_assert(!std::is_copy_constructible_v<kjarni::Reranker>);
static_assert(!std::is_copy_constructible_v<kjarni::Chat>);
static_assert(!std::is_copy_constructible_v<kjarni::Embedding>);
static_assert(!std::is_copy_constructible_v<kjarni::Indexer>);
static_assert(!std::is_copy_constructible_v<kjarni::Searcher>);
static_assert(!std::is_copy_constructible_v<kjarni::Transcriber>);
static_assert(!std::is_copy_constructible_v<kjarni::Summarizer>);
static_assert(!std::is_copy_constructible_v<kjarni::Translator>);
static_assert(!std::is_copy_constructible_v<kjarni::ImageIndex>);

// A Result must not silently decay to bool in a condition where the value was meant.
static_assert(std::is_same_v<kjarni::Result<int>, std::expected<int, kjarni::Error>>);

// ─── Errors ──────────────────────────────────────────────────────

void test_errors() {
    section("errors");

    // A model that cannot exist: the failure must be a value, not a crash or an
    // empty success.
    auto bad = kjarni::Embedder::create({.model = "definitely-not-a-model-xyz"});
    ok(!bad.has_value(), "unknown model fails");
    ok(!bad.error().message().empty(), "error carries a message");
    ok(bad.error().code() != KJARNI_ERROR_CODE_OK, "error carries a non-ok code");

    // The C API keeps one global "last error" slot. The message must have been
    // captured when the failure happened, not read lazily afterwards, or a later
    // unrelated failure would overwrite it.
    const std::string first = bad.error().message();
    auto second_bad = kjarni::Classifier::create({.model = "also-not-real-abc"});
    ok(!second_bad.has_value(), "second failure also fails");
    ok(bad.error().message() == first, "earlier error message survives a later failure");
}

// ─── Embedder ────────────────────────────────────────────────────

void test_embedder() {
    section("embedder");

    auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    ok(embedder.has_value(), "embedder loads");
    if (!embedder) return;

    ok(embedder->dimensions() == 384, "MiniLM-L6-v2 reports 384 dimensions");

    auto vec = embedder->encode("Hello world");
    ok(vec.has_value(), "encode succeeds");
    if (!vec) return;

    ok(vec->size() == 384, "embedding has the model's dimension");

    // Golden values from Kjarni.Tests.EmbedderTests.Encode_FirstFiveValues. The
    // wrapper must forward the model's output untouched; if these drift, the C++
    // layer is transforming something it should not.
    ok_near(vec->values()[0], -0.03448f, 1e-3f, "first value matches the reference");
    ok_near(vec->values()[1], 0.03102f, 1e-3f, "second value matches the reference");
    ok_near(vec->values()[4], -0.03936f, 1e-3f, "fifth value matches the reference");

    // normalize defaults to true.
    float norm = 0.f;
    for (float v : vec->values()) norm += v * v;
    ok_near(std::sqrt(norm), 1.0f, 1e-3f, "embeddings are unit length by default");

    // The span borrows; to_vector copies. Both must see the same data.
    auto copied = vec->to_vector();
    ok(copied.size() == vec->size(), "to_vector preserves size");
    ok(copied[0] == vec->values()[0], "to_vector preserves contents");

    // Range-for over the embedding.
    std::size_t counted = 0;
    for (float v : *vec) { (void)v; ++counted; }
    ok(counted == 384, "embedding is iterable");
}

void test_embedder_batch() {
    section("embedder: batch");

    auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (!embedder) { ok(false, "embedder loads"); return; }

    const std::vector<std::string> texts{"cat", "dog", "quantum physics"};
    auto batch = embedder->encode(texts);
    ok(batch.has_value(), "batch encode succeeds");
    if (!batch) return;

    ok(batch->size() == 3, "one vector per input");
    ok((*batch)[0].size() == 384, "batch vectors have the model dimension");

    // Order is the contract: result[i] corresponds to texts[i]. Callers index into
    // their own arrays by position, so a reordering here would be silent corruption.
    for (std::size_t i = 0; i < texts.size(); ++i) {
        auto alone = embedder->encode(texts[i]);
        if (!alone) { ok(false, "single encode succeeds"); continue; }
        ok_near(kjarni::cosine((*batch)[i], alone->values()), 1.0f, 1e-3f,
                "batch position matches encoding that text alone");
    }

    // Empty input must not crash or misreport.
    auto empty = embedder->encode(std::vector<std::string>{});
    ok(empty.has_value(), "empty batch succeeds");
    if (empty) ok(empty->empty(), "empty batch returns nothing");
}

void test_similarity() {
    section("similarity");

    auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (!embedder) { ok(false, "embedder loads"); return; }

    auto query = embedder->encode("How do I get my money back?");
    auto related = embedder->encode("What is your refund policy?");
    auto unrelated = embedder->encode("The weather in Reykjavik is unpredictable.");
    if (!query || !related || !unrelated) { ok(false, "encodes succeed"); return; }

    const float hit = kjarni::cosine(*query, *related);
    const float miss = kjarni::cosine(*query, *unrelated);

    ok(hit > miss + 0.2f, "the refund sentence ranks well above the weather one");
    ok(kjarni::cosine(*query, *query) > 0.999f, "a vector is identical to itself");

    // Guards the argument order: kjarni_cosine_similarity takes one length, not one
    // per vector, and getting that wrong silently reads past the end.
    ok(std::fabs(kjarni::cosine(*query, *related) - kjarni::cosine(*related, *query)) < 1e-5f,
       "cosine is symmetric");
}

// ─── Classifier ──────────────────────────────────────────────────

void test_classifier() {
    section("classifier");

    auto classifier = kjarni::Classifier::create({.model = "distilbert-sentiment"});
    ok(classifier.has_value(), "classifier loads");
    if (!classifier) return;

    ok(classifier->num_labels() == 2, "sentiment model has two labels");

    auto labels = classifier->classify("This is absolutely wonderful.");
    ok(labels.has_value(), "classify succeeds");
    if (!labels) return;

    ok(labels->size() == 2, "every label is returned");
    ok(labels->front().name == "POSITIVE", "positive text classifies as POSITIVE");
    ok(labels->front().score > 0.9f, "with high confidence");

    // Sorted highest first, so front() is the prediction.
    ok(labels->front().score >= labels->back().score, "labels come back sorted");

    float total = 0.f;
    for (const auto& l : *labels) total += l.score;
    ok_near(total, 1.0f, 0.01f, "softmax scores sum to one");

    auto top = classifier->top("Worst purchase of my life.");
    ok(top.has_value() && top->has_value(), "top returns a label");
    if (top && *top) ok((*top)->name == "NEGATIVE", "negative text classifies as NEGATIVE");
}

// ─── Reranker ────────────────────────────────────────────────────

void test_reranker() {
    section("reranker");

    auto reranker = kjarni::Reranker::create();
    ok(reranker.has_value(), "reranker loads");
    if (!reranker) return;

    const std::vector<std::string> docs{
        "Our office is open until 5pm on weekdays.",
        "To end your plan, go to Settings and choose Close account.",
        "Cancellation of orders is handled by the warehouse team.",
        "You can update your billing address at any time.",
    };

    auto ranked = reranker->rerank("how do I cancel my subscription", docs);
    ok(ranked.has_value(), "rerank succeeds");
    if (!ranked) return;

    ok(ranked->size() == docs.size(), "every document is scored");

    // The paraphrase, which shares no words with the query, must beat the decoy that
    // repeats both "cancel" and "subscription".
    ok(ranked->front().index == 1, "the paraphrase ranks first, not the keyword decoy");

    for (std::size_t i = 1; i < ranked->size(); ++i) {
        ok((*ranked)[i - 1].score >= (*ranked)[i].score, "results are sorted descending");
    }

    // index must point into the caller's array, which is how metadata is recovered.
    for (const auto& r : *ranked) {
        ok(r.index < docs.size(), "index is within the input range");
    }

    auto top2 = reranker->rerank("how do I cancel my subscription", docs, 2);
    ok(top2.has_value() && top2->size() == 2, "top-k limits the result count");
    if (top2 && !top2->empty()) {
        ok(top2->front().index == ranked->front().index, "top-k agrees with the full ranking");
    }

    // Scores are raw logits: negative is normal, and only order is meaningful.
    auto strong = reranker->score("capital of Iceland",
                                  "Reykjavik is the capital and largest city of Iceland.");
    auto weak = reranker->score("capital of Iceland", "Bananas are a good source of potassium.");
    ok(strong.has_value() && weak.has_value(), "pair scoring succeeds");
    if (strong && weak) ok(*strong > *weak, "a relevant pair outscores an irrelevant one");

    auto empty = reranker->rerank("query", std::vector<std::string>{});
    ok(empty.has_value(), "reranking nothing succeeds");
    if (empty) ok(empty->empty(), "reranking nothing returns nothing");
}

// ─── Ownership ───────────────────────────────────────────────────

void test_ownership() {
    section("ownership");

    // Repeated create/destroy: under ASan with leak detection this catches a handle
    // that is never freed, and a double free would abort here.
    for (int i = 0; i < 3; ++i) {
        auto e = kjarni::Embedder::create({.model = "minilm-l6-v2"});
        if (!e) { ok(false, "embedder loads in loop"); return; }
        auto v = e->encode("scope test");
        if (!v) { ok(false, "encode in loop"); return; }
    }
    ok(true, "repeated construction and destruction is clean");

    // Moving must transfer ownership exactly once: the moved-from object must not
    // free the handle the moved-to object now owns.
    auto original = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (!original) { ok(false, "embedder loads"); return; }

    kjarni::Embedder moved = std::move(*original);
    auto after_move = moved.encode("still works after a move");
    ok(after_move.has_value(), "a moved-to embedder still works");

    // Same for the owned array type.
    auto vec = moved.encode("owned array");
    if (!vec) { ok(false, "encode succeeds"); return; }
    const float first = vec->values()[0];
    kjarni::Embedding relocated = std::move(*vec);
    ok(relocated.values()[0] == first, "a moved-to embedding keeps its data");

    // Storing in an optional, which is what the Qt wrapper does.
    std::optional<kjarni::Embedder> held;
    auto fresh = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (fresh) held.emplace(std::move(*fresh));
    ok(held.has_value(), "an embedder can live in std::optional");
    if (held) ok(held->encode("in an optional").has_value(), "and still works there");
    held.reset();
    ok(!held.has_value(), "resetting the optional destroys it cleanly");
}

// ─── Fixtures ────────────────────────────────────────────────────

namespace fs = std::filesystem;

/// The repo root, found from this file so the tests run from any directory.
fs::path repo_root() { return fs::path{__FILE__}.parent_path() / "../../../.."; }

/// A scratch directory, emptied first and removed when it goes out of scope.
struct TempDir {
    fs::path path;
    explicit TempDir(const char* name) : path(fs::temp_directory_path() / name) {
        fs::remove_all(path);
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

void write_file(const fs::path& p, const char* text) { std::ofstream{p} << text; }

bool contains(std::string haystack, std::string_view needle) {
    for (auto& c : haystack) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return haystack.find(needle) != std::string::npos;
}

// ─── Indexer and searcher ────────────────────────────────────────

void test_index_and_search() {
    section("indexer + searcher");

    TempDir docs{"kjarni_cpp_docs"};
    write_file(docs.path / "cats.txt", "Cats are small carnivorous mammals that purr and chase mice.");
    write_file(docs.path / "rust.md", "Rust is a systems programming language focused on memory safety.");
    write_file(docs.path / "bread.txt", "Sourdough bread is leavened with a starter of wild yeast.");
    write_file(docs.path / "skip.log", "This file has an extension that is not indexed.");
    const std::string index = (docs.path / "index").string();

    auto indexer = kjarni::Indexer::create({.extensions = {"txt", "md"}});
    if (!indexer) { ok(false, "indexer loads"); return; }

    const std::vector<std::string> inputs{docs.path.string()};
    auto stats = indexer->create_index(index, inputs);
    ok(stats.has_value(), "an index is built");
    if (!stats) return;
    ok(stats->documents_indexed == 3, "only the listed extensions are indexed");
    ok(stats->dimension == 384, "minilm vectors have 384 dimensions");

    auto again = indexer->create_index(index, inputs);
    ok(!again.has_value(), "building over an existing index fails without force");
    ok(indexer->create_index(index, inputs, /*force=*/true).has_value(), "force replaces it");

    write_file(docs.path / "tea.txt", "Green tea is brewed from unoxidised leaves of the tea plant.");
    const std::vector<std::string> extra{(docs.path / "tea.txt").string()};
    auto added = indexer->add(index, extra);
    ok(added.has_value() && *added >= 1, "a file can be added to an existing index");

    auto searcher = kjarni::Searcher::create();
    if (!searcher) { ok(false, "searcher loads"); return; }
    ok(!searcher->has_reranker(), "no reranker unless one is asked for");

    auto hits = searcher->search(index, "feline pets");
    ok(hits.has_value() && !hits->empty(), "a search returns hits");
    if (hits && !hits->empty()) ok(contains(hits->front().text, "cats"), "the cat document ranks first");
    if (hits && !hits->empty()) ok(contains(hits->front().metadata_json, "cats.txt"), "hits carry their source");

    auto semantic = searcher->search(index, "a drink made from leaves",
                                     {.mode = kjarni::SearchMode::semantic, .top_k = 1});
    ok(semantic.has_value() && semantic->size() == 1, "top_k limits the results");
    if (semantic && !semantic->empty()) ok(contains(semantic->front().text, "tea"), "an added document is found");

    auto keyword = searcher->search(index, "sourdough", {.mode = kjarni::SearchMode::keyword});
    ok(keyword.has_value() && !keyword->empty(), "keyword search finds an exact word");
    if (keyword && !keyword->empty()) ok(contains(keyword->front().text, "sourdough"), "and ranks it first");

    auto filtered = searcher->search(index, "programming", {.source_pattern = "*.md"});
    ok(filtered.has_value(), "a source pattern is accepted");
    if (filtered) {
        bool all_md = !filtered->empty();
        for (const auto& h : *filtered) all_md = all_md && contains(h.metadata_json, "rust.md");
        ok(all_md, "a source pattern keeps only matching files");
    }

    auto missing = searcher->search((docs.path / "nope").string(), "x");
    ok(!missing.has_value(), "searching a missing index fails");
}

// ─── Transcriber ─────────────────────────────────────────────────

void test_transcriber() {
    section("transcriber");

    auto t = kjarni::Transcriber::create({.language = "en"});
    if (!t) { ok(false, "whisper loads"); return; }

    const auto wav = repo_root() / "crates/kjarni/tests/fixtures/speech.wav";
    auto r = t->transcribe_file(wav.string());
    ok(r.has_value(), "a wav file is transcribed");
    if (r) {
        ok(!r->text.empty(), "the transcript has text");
        ok(r->language == "en", "the given language is reported");
        ok(r->duration_secs > 0.f, "the duration is reported");
        ok(!r->segments.empty(), "segments are returned");
        bool ordered = true;
        for (const auto& s : r->segments) ordered = ordered && s.end >= s.start;
        ok(ordered, "every segment ends after it starts");
    }

    // A second of silence at 8 kHz exercises resampling and must not fail.
    const std::vector<float> silence(8000, 0.f);
    ok(t->transcribe(silence, 8000).has_value(), "PCM at another sample rate is accepted");
    ok(!t->transcribe(silence, 0).has_value(), "a zero sample rate is rejected");
    ok(!t->transcribe_file("/no/such/file.wav").has_value(), "a missing file is an error");
}

// ─── Summarizer and translator ───────────────────────────────────

void test_summarizer() {
    section("summarizer");

    auto s = kjarni::Summarizer::create();
    if (!s) { ok(false, "summarizer loads"); return; }

    const std::string article =
        "The city council voted on Tuesday to approve a new public library in the "
        "downtown district. The library will include a children's wing, a maker space "
        "with 3D printers, and a cafe. Construction is expected to begin next spring and "
        "take about eighteen months. Council members said the project would be funded "
        "through a combination of municipal bonds and private donations. Several "
        "residents spoke in favour of the plan at the meeting, citing the lack of quiet "
        "study spaces in the area. A small group opposed it, arguing the money would be "
        "better spent on road repairs. The mayor said the library would become a "
        "cornerstone of the community and help revitalise the downtown area, which has "
        "seen several businesses close over the past few years.";
    auto out = s->summarize(article);
    ok(out.has_value() && !out->empty(), "an article is summarized");
    if (out) ok(out->size() < article.size(), "the summary is shorter than the article");

    ok(!kjarni::Summarizer::create({.model = "no-such-model"}).has_value(), "an unknown model fails");
}

void test_translator() {
    section("translator");

    auto t = kjarni::Translator::create({.from = "en", .to = "de"});
    if (!t) { ok(false, "translator loads"); return; }

    auto out = t->translate("The house is small.");
    ok(out.has_value() && contains(*out, "haus"), "default languages translate en to de");

    auto explicit_langs = t->translate("The house is small.", "English", "German");
    ok(explicit_langs.has_value() && contains(*explicit_langs, "haus"), "languages can be named per call");

    auto bare = kjarni::Translator::create();
    if (bare) {
        auto err = bare->translate("hello");
        ok(!err.has_value() && err.error().code() == KJARNI_ERROR_CODE_INVALID_CONFIG,
           "no languages anywhere is an invalid config");
    }
}

// ─── Image search ────────────────────────────────────────────────

void test_image_index() {
    section("image index");

    auto idx = kjarni::ImageIndex::create();
    if (!idx) { ok(false, "CLIP loads"); return; }
    ok(idx->size() == 0, "a new index is empty");

    const auto photos = repo_root() / "tests/fixtures/clip/photos";
    auto report = idx->add_directory(photos.string());
    ok(report.has_value() && report->added >= 4, "a directory of photos is indexed");
    if (report) ok(report->skipped == 0, "every photo decodes");

    const std::pair<const char*, const char*> cases[] = {
        {"a sandy beach under a blue sky", "beach_holiday"},
        {"a dense green forest with trees", "deep_forest"},
        {"a red car", "red_car"},
        {"a snowy mountain peak", "snowy_mountain"},
    };
    for (const auto& [query, expected] : cases) {
        auto hits = idx->search(query, 2);
        ok(hits.has_value() && hits->size() == 2, "top_k limits image hits");
        if (hits && !hits->empty()) ok(contains(hits->front().path, expected), query);
    }

    TempDir dir{"kjarni_cpp_images"};
    const std::string saved = (dir.path / "photos.idx").string();
    ok(idx->save(saved).has_value(), "the index saves");

    auto restored = kjarni::ImageIndex::create();
    if (!restored) { ok(false, "a second index loads"); return; }
    auto count = restored->load(saved);
    ok(count.has_value() && *count == idx->size(), "a saved index restores every vector");
    auto hits = restored->search("a red car", 1);
    ok(hits && !hits->empty() && contains(hits->front().path, "red_car"), "a restored index searches the same");

    ok(!idx->add("/no/such/photo.jpg").has_value(), "a missing photo is an error");
}

// ─── Version ─────────────────────────────────────────────────────

void test_version() {
    section("version");
    ok(!kjarni::version().empty(), "version is reported");
}

}  // namespace

int main() {
    test_errors();
    test_embedder();
    test_embedder_batch();
    test_similarity();
    test_classifier();
    test_reranker();
    test_ownership();
    test_index_and_search();
    test_transcriber();
    test_summarizer();
    test_translator();
    test_image_index();
    test_version();

    std::fprintf(stderr, "\n%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
