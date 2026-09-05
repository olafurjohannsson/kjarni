// The same program as hello.cpp, written against the C header instead of the
// C++23 one, for codebases that cannot move to C++23 yet.
//
// kjarni.hpp needs C++23 for std::expected. kjarni.h needs nothing: it is the
// plain C ABI every language binding in this project is built on, and it works
// from C++11 upward. What you give up is the RAII and the std::expected returns,
// which is about fifteen lines to put back.
//
// Build (Linux x86_64), from a directory holding libkjarni_ffi.so and the headers:
//
//   g++ -std=c++11 hello_c_api.cpp -I. -L. -lkjarni_ffi -Wl,-rpath,'$ORIGIN' -o hello_c_api
//
// -std=c++17 and -std=c++23 build it unchanged.

#include "kjarni.h"

#include <cstdio>
#include <memory>
#include <vector>

namespace {

// Handles are freed with their matching _free function. A unique_ptr with a
// custom deleter makes that happen on every return path, including early ones.
struct EmbedderDeleter {
    void operator()(KjarniEmbedder* p) const { kjarni_embedder_free(p); }
};
using EmbedderPtr = std::unique_ptr<KjarniEmbedder, EmbedderDeleter>;

// Any KjarniFloatArray you receive is yours to free once you have copied out of it.
std::vector<float> encode(KjarniEmbedder* emb, const char* text) {
    KjarniFloatArray arr;
    if (kjarni_embedder_encode(emb, text, &arr) != KJARNI_ERROR_CODE_OK) {
        std::fprintf(stderr, "encode failed: %s\n", kjarni_last_error_message());
        return std::vector<float>();
    }
    std::vector<float> out(arr.data, arr.data + arr.len);
    kjarni_float_array_free(arr);
    return out;
}

} // namespace

int main() {
    KjarniEmbedderConfig cfg = kjarni_embedder_config_default();
    cfg.model_name = "minilm-l6-v2";
    cfg.quiet = 1;

    KjarniEmbedder* raw = NULL;
    if (kjarni_embedder_new(&cfg, &raw) != KJARNI_ERROR_CODE_OK) {
        // Read the error immediately: it reports the most recent failure process wide.
        std::fprintf(stderr, "could not load model: %s\n", kjarni_last_error_message());
        return 1;
    }
    EmbedderPtr embedder(raw);

    const std::vector<float> question  = encode(embedder.get(), "How do I get my money back?");
    const std::vector<float> related   = encode(embedder.get(), "What is your refund policy?");
    const std::vector<float> unrelated = encode(embedder.get(), "The weather in Reykjavik is unpredictable.");
    if (question.empty() || related.empty() || unrelated.empty()) return 1;

    std::printf("related:   %.4f\n",
                kjarni_cosine_similarity(question.data(), related.data(), question.size()));
    std::printf("unrelated: %.4f\n",
                kjarni_cosine_similarity(question.data(), unrelated.data(), question.size()));
}
