// The smallest useful Kjarni program: load an embedding model, compare three
// sentences by meaning. See example.cpp for classification, reranking and chat.
//
// Build (Linux x86_64), from a directory holding libkjarni_ffi.so and the headers:
//
//   g++ -std=c++23 hello.cpp -I. -L. -lkjarni_ffi -Wl,-rpath,'$ORIGIN' -o hello
//
#include "kjarni.hpp"

#include <print>

int main() {
    // Downloaded once and cached under ~/.cache/kjarni, then loaded from disk.
    auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
    if (!embedder) {
        std::println("{}", embedder.error().message());
        return 1;
    }

    auto question = embedder->encode("How do I get my money back?");
    auto related = embedder->encode("What is your refund policy?");
    auto unrelated = embedder->encode("The weather in Reykjavik is unpredictable.");

    // No shared words with the question, but the same meaning.
    std::println("related:   {:.4f}", kjarni::cosine(*question, *related));
    std::println("unrelated: {:.4f}", kjarni::cosine(*question, *unrelated));
}
