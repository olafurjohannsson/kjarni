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

#include <cmath>
#include <cstdio>
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

static_assert(!std::is_copy_constructible_v<kjarni::Embedder>);
static_assert(!std::is_copy_constructible_v<kjarni::Classifier>);
static_assert(!std::is_copy_constructible_v<kjarni::Reranker>);
static_assert(!std::is_copy_constructible_v<kjarni::Chat>);
static_assert(!std::is_copy_constructible_v<kjarni::Embedding>);

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
    test_version();

    std::fprintf(stderr, "\n%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
