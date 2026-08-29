// Kjarni from C++23.
//
// Everything here goes through `kjarni.hpp`, so there is no manual free anywhere:
// handles are unique_ptrs, arrays are move-only owners, and errors are values.
//
//   cmake -B build && cmake --build build && ./build/example

#include "kjarni.hpp"

#include <print>
#include <string>
#include <vector>

namespace {

/// Prints the diagnostic and returns non-zero, for `return fail(...)` at a call site.
int fail(std::string_view what, const kjarni::Error& e) {
    std::println(stderr, "{} failed: {}", what, e.message());
    return 1;
}

int embeddings() {
    std::println("── Embeddings ──");

    auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (!embedder) return fail("embedder", embedder.error());

    std::println("dimensions: {}", embedder->dimensions());

    // One call, one batch: markedly faster than looping.
    const std::vector<std::string> texts{
        "How do I get my money back?",
        "What is your refund policy?",
        "The weather in Reykjavik is unpredictable.",
    };

    auto vectors = embedder->encode(texts);
    if (!vectors) return fail("encode batch", vectors.error());

    const auto& query = (*vectors)[0];
    for (std::size_t i = 1; i < vectors->size(); ++i) {
        std::println("  {:.4f}  {}", kjarni::cosine(query, (*vectors)[i]), texts[i]);
    }

    // A single embedding borrows the C buffer rather than copying it.
    auto one = embedder->encode("semantic search, locally");
    if (!one) return fail("encode", one.error());
    std::println("  first 4: [{:.4f}, {:.4f}, {:.4f}, {:.4f}, ...]",
                 one->values()[0], one->values()[1], one->values()[2], one->values()[3]);

    return 0;
}

int classification() {
    std::println("\n── Classification ──");

    auto classifier = kjarni::Classifier::create({.model = "distilbert-sentiment"});
    if (!classifier) return fail("classifier", classifier.error());

    for (std::string_view text : {"This is absolutely wonderful.",
                                  "Worst purchase of my life."}) {
        auto top = classifier->top(text);
        if (!top) return fail("classify", top.error());
        if (*top) std::println("  {:<8} {:.4f}  {}", (*top)->name, (*top)->score, text);
    }

    return 0;
}

int reranking() {
    std::println("\n── Reranking ──");

    auto reranker = kjarni::Reranker::create();
    if (!reranker) return fail("reranker", reranker.error());

    const std::vector<std::string> documents{
        "Our office is open until 5pm on weekdays.",
        "To end your plan, go to Settings and choose Close account.",
        "Cancellation of orders is handled by the warehouse team.",
        "You can update your billing address at any time.",
    };

    auto ranked = reranker->rerank("how do I cancel my subscription", documents);
    if (!ranked) return fail("rerank", ranked.error());

    // `index` points back into `documents`, so the caller keeps its own metadata.
    for (const auto& r : *ranked) {
        std::println("  {:>7.2f}  {}", r.score, documents[r.index]);
    }
    std::println("  (raw logits: negative is normal, only the order means anything)");

    return 0;
}

int chat() {
    std::println("\n── Chat ──");

    auto chat = kjarni::Chat::create({
        .model = "llama3.2-1b-instruct",
        .system_prompt = "You are terse. Answer in one short sentence.",
    });
    if (!chat) {
        std::println("  skipped: {}", chat.error().message());
        return 0;  // not fatal: the model may simply not be downloaded
    }

    std::println("  context window: {} tokens", chat->context_size());

    auto reply = chat->send("What is the capital of Iceland?", kjarni::Generation::greedy(48));
    if (!reply) return fail("send", reply.error());
    std::println("  {}", *reply);

    // Streaming takes any callable; returning false stops generation early.
    std::print("  streaming: ");
    int tokens = 0;
    auto streamed = chat->stream(
        "Count from one to five.", kjarni::Generation::greedy(32),
        [&tokens](std::string_view token) {
            std::print("{}", token);
            return ++tokens < 40;
        });
    if (!streamed) return fail("stream", streamed.error());
    std::println("\n  ({} tokens)", tokens);

    return 0;
}

}  // namespace

int main() {
    std::println("kjarni {}\n", kjarni::version());

    if (int rc = embeddings()) return rc;
    if (int rc = classification()) return rc;
    if (int rc = reranking()) return rc;
    if (int rc = chat()) return rc;

    std::println("\nok");
    return 0;
}
