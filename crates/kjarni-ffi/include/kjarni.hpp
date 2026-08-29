// Kjarni for C++23.
//
// A header-only wrapper over `kjarni.h`. Nothing here allocates on its own account:
// every handle is a `std::unique_ptr` with the matching C deleter, and every array
// the C API hands back is owned by a move-only RAII type. Leaking or double-freeing
// takes deliberate effort.
//
// Errors are values. Fallible calls return `std::expected<T, kjarni::Error>` rather
// than throwing, so a caller decides whether a missing model is exceptional. Nothing
// in this header throws except `std::bad_alloc`.
//
//   #include "kjarni.hpp"
//
//   auto embedder = kjarni::Embedder::create({.model = "minilm-l6-v2"});
//   if (!embedder) return std::println("{}", embedder.error().message()), 1;
//
//   auto vec = embedder->encode("semantic search, locally");
//   std::println("{} dimensions", vec->size());
//
// Requires C++23 for std::expected. For C++20, swap std::expected for a
// std::variant or exceptions; everything else here is C++20.

#ifndef KJARNI_HPP
#define KJARNI_HPP

#include "kjarni.h"

#include <algorithm>
#include <cstddef>
#include <expected>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace kjarni {

// ─── Errors ──────────────────────────────────────────────────────

/// A failed call, carrying the library's own diagnostic text.
///
/// The message is captured at construction rather than read lazily: the C API keeps
/// one global "last error", so reading it later would report whatever failed most
/// recently instead of what failed here.
class Error {
public:
    explicit Error(KjarniErrorCode code)
        : code_(code), message_(capture()) {}

    [[nodiscard]] KjarniErrorCode code() const noexcept { return code_; }
    [[nodiscard]] const std::string& message() const noexcept { return message_; }

private:
    static std::string capture() {
        const char* msg = kjarni_last_error_message();
        return msg ? std::string(msg) : std::string("no detail available");
    }

    KjarniErrorCode code_;
    std::string message_;
};

/// The result of any fallible Kjarni call.
template <typename T>
using Result = std::expected<T, Error>;

namespace detail {

/// Turns a C status code into a `Result`, running `produce` only on success.
template <typename F>
auto check(KjarniErrorCode code, F&& produce) -> Result<decltype(produce())> {
    if (code != KJARNI_ERROR_CODE_OK) return std::unexpected(Error{code});
    return std::forward<F>(produce)();
}

inline std::expected<void, Error> check(KjarniErrorCode code) {
    if (code != KJARNI_ERROR_CODE_OK) return std::unexpected(Error{code});
    return {};
}

/// `const char*` array backing a `std::span<const std::string>`, for the C calls
/// that take one.
class CStrings {
public:
    explicit CStrings(std::span<const std::string> items) {
        ptrs_.reserve(items.size());
        for (const auto& s : items) ptrs_.push_back(s.c_str());
    }
    [[nodiscard]] const char* const* data() const noexcept { return ptrs_.data(); }
    [[nodiscard]] std::size_t size() const noexcept { return ptrs_.size(); }

private:
    std::vector<const char*> ptrs_;
};

/// Deleter pairing a C handle with its free function, so `unique_ptr` needs no
/// custom class per type.
template <auto Fn>
struct Deleter {
    template <typename T>
    void operator()(T* p) const noexcept {
        if (p) Fn(p);
    }
};

template <typename T, auto Fn>
using Handle = std::unique_ptr<T, Deleter<Fn>>;

/// Owns one of the C API's array structs and frees it with `Fn`.
///
/// Move-only: two copies would each free the same buffer.
template <typename Owned, auto Fn>
class Owning {
public:
    Owning() = default;
    explicit Owning(Owned raw) noexcept : raw_(raw) {}

    ~Owning() { reset(); }

    Owning(const Owning&) = delete;
    Owning& operator=(const Owning&) = delete;

    Owning(Owning&& other) noexcept : raw_(std::exchange(other.raw_, Owned{})) {}
    Owning& operator=(Owning&& other) noexcept {
        if (this != &other) {
            reset();
            raw_ = std::exchange(other.raw_, Owned{});
        }
        return *this;
    }

    [[nodiscard]] const Owned& raw() const noexcept { return raw_; }
    Owned* out() noexcept { return &raw_; }

private:
    void reset() noexcept {
        Fn(raw_);
        raw_ = Owned{};
    }

    Owned raw_{};
};

/// Trampoline letting any C++ callable serve as a `KjarniStreamCallbackFn`.
template <typename F>
bool stream_trampoline(const char* text, void* user_data) {
    return (*static_cast<F*>(user_data))(std::string_view{text ? text : ""});
}

}  // namespace detail

// ─── Owned results ───────────────────────────────────────────────

/// A single embedding. Borrowed as a `std::span`, so no copy is made.
class Embedding {
public:
    explicit Embedding(KjarniFloatArray arr) noexcept : arr_(arr) {}

    [[nodiscard]] std::span<const float> values() const noexcept {
        return {arr_.raw().data, arr_.raw().len};
    }
    [[nodiscard]] std::size_t size() const noexcept { return arr_.raw().len; }
    [[nodiscard]] const float* data() const noexcept { return arr_.raw().data; }

    [[nodiscard]] auto begin() const noexcept { return values().begin(); }
    [[nodiscard]] auto end() const noexcept { return values().end(); }

    /// Copies into an owning vector, for when the embedding must outlive this object.
    [[nodiscard]] std::vector<float> to_vector() const {
        return {values().begin(), values().end()};
    }

private:
    detail::Owning<KjarniFloatArray, kjarni_float_array_free> arr_;
};

/// One predicted label.
struct Label {
    std::string name;
    float score{};
};

/// One reranked document: an index into the input, and its score.
struct Ranked {
    std::size_t index{};
    float score{};
};

/// One search hit.
struct Hit {
    float score{};
    std::size_t document_id{};
    std::string text;
    std::string metadata_json;
};

// ─── Cosine similarity ───────────────────────────────────────────

/// Cosine similarity between two embeddings.
/// Vectors of unequal length compare over the shorter prefix, which is what the C
/// function does; mismatched dimensions almost always mean two different models.
[[nodiscard]] inline float cosine(std::span<const float> a, std::span<const float> b) noexcept {
    return kjarni_cosine_similarity(a.data(), b.data(), std::min(a.size(), b.size()));
}

[[nodiscard]] inline float cosine(const Embedding& a, const Embedding& b) noexcept {
    return cosine(a.values(), b.values());
}

// ─── Embedder ────────────────────────────────────────────────────

/// Turns text into vectors.
/// Designated-initialiser options: `Embedder::create({.model = "mpnet-base-v2"})`.
struct EmbedderOptions {
    std::string model = "minilm-l6-v2";
    std::string cache_dir;      ///< empty = default location
    bool gpu = false;
    bool normalize = true;
    bool quiet = true;
};

class Embedder {
public:
    using Options = EmbedderOptions;

    [[nodiscard]] static Result<Embedder> create(const EmbedderOptions& opts = {}) {
        KjarniEmbedderConfig cfg = kjarni_embedder_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = opts.cache_dir.empty() ? nullptr : opts.cache_dir.c_str();
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.normalize = opts.normalize ? 1 : 0;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniEmbedder* raw = nullptr;
        if (auto e = detail::check(kjarni_embedder_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Embedder{raw};
    }

    [[nodiscard]] Result<Embedding> encode(std::string_view text) const {
        KjarniFloatArray arr{};
        const std::string owned{text};
        if (auto e = detail::check(kjarni_embedder_encode(handle_.get(), owned.c_str(), &arr));
            !e) {
            return std::unexpected(e.error());
        }
        return Embedding{arr};
    }

    /// Encodes a batch in one call, which is markedly faster than looping.
    [[nodiscard]] Result<std::vector<std::vector<float>>> encode(
        std::span<const std::string> texts) const {
        // An empty std::vector's data() may be null, and the C API rejects a null
        // array before it reaches its own zero-length branch. Nothing to encode is
        // not an error, so answer it here.
        if (texts.empty()) return std::vector<std::vector<float>>{};

        detail::CStrings ptrs{texts};
        detail::Owning<KjarniFloat2DArray, kjarni_float_2d_array_free> out;

        if (auto e = detail::check(kjarni_embedder_encode_batch(
                handle_.get(), ptrs.data(), ptrs.size(), out.out()));
            !e) {
            return std::unexpected(e.error());
        }

        const auto& raw = out.raw();
        std::vector<std::vector<float>> result;
        result.reserve(raw.rows);
        for (std::size_t r = 0; r < raw.rows; ++r) {
            const float* row = raw.data + r * raw.cols;
            result.emplace_back(row, row + raw.cols);
        }
        return result;
    }

    [[nodiscard]] std::size_t dimensions() const noexcept {
        return kjarni_embedder_dim(handle_.get());
    }

private:
    explicit Embedder(KjarniEmbedder* raw) noexcept : handle_(raw) {}
    detail::Handle<KjarniEmbedder, kjarni_embedder_free> handle_;
};

// ─── Classifier ──────────────────────────────────────────────────

/// Sentiment, emotion and toxicity classification.
struct ClassifierOptions {
    std::string model = "distilbert-sentiment";
    std::string cache_dir;
    bool gpu = false;
    bool quiet = true;
};

class Classifier {
public:
    using Options = ClassifierOptions;

    [[nodiscard]] static Result<Classifier> create(const ClassifierOptions& opts = {}) {
        KjarniClassifierConfig cfg = kjarni_classifier_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = opts.cache_dir.empty() ? nullptr : opts.cache_dir.c_str();
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniClassifier* raw = nullptr;
        if (auto e = detail::check(kjarni_classifier_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Classifier{raw};
    }

    /// Every label with its score, highest first.
    [[nodiscard]] Result<std::vector<Label>> classify(std::string_view text) const {
        detail::Owning<KjarniClassResults, kjarni_class_results_free> out;
        const std::string owned{text};

        if (auto e = detail::check(
                kjarni_classifier_classify(handle_.get(), owned.c_str(), out.out()));
            !e) {
            return std::unexpected(e.error());
        }

        std::vector<Label> labels;
        labels.reserve(out.raw().len);
        for (std::size_t i = 0; i < out.raw().len; ++i) {
            const auto& r = out.raw().results[i];
            labels.push_back({r.label ? r.label : "", r.score});
        }
        return labels;
    }

    /// The highest-scoring label, or `std::nullopt` if the model returned nothing.
    [[nodiscard]] Result<std::optional<Label>> top(std::string_view text) const {
        auto labels = classify(text);
        if (!labels) return std::unexpected(labels.error());
        if (labels->empty()) return std::optional<Label>{};
        return std::optional<Label>{std::move(labels->front())};
    }

    [[nodiscard]] std::size_t num_labels() const noexcept {
        return kjarni_classifier_num_labels(handle_.get());
    }

private:
    explicit Classifier(KjarniClassifier* raw) noexcept : handle_(raw) {}
    detail::Handle<KjarniClassifier, kjarni_classifier_free> handle_;
};

// ─── Reranker ────────────────────────────────────────────────────

/// Cross-encoder reranking: rescores candidates you already have.
struct RerankerOptions {
    std::string model = "minilm-l6-v2-cross-encoder";
    std::string cache_dir;
    bool gpu = false;
    bool quiet = true;
};

class Reranker {
public:
    using Options = RerankerOptions;

    [[nodiscard]] static Result<Reranker> create(const RerankerOptions& opts = {}) {
        KjarniRerankerConfig cfg = kjarni_reranker_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = opts.cache_dir.empty() ? nullptr : opts.cache_dir.c_str();
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniReranker* raw = nullptr;
        if (auto e = detail::check(kjarni_reranker_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Reranker{raw};
    }

    /// Scores one query-document pair. Raw logits: negative values are normal, and
    /// only the ordering is meaningful.
    [[nodiscard]] Result<float> score(std::string_view query, std::string_view document) const {
        const std::string q{query}, d{document};
        float out = 0.f;
        if (auto e = detail::check(
                kjarni_reranker_score(handle_.get(), q.c_str(), d.c_str(), &out));
            !e) {
            return std::unexpected(e.error());
        }
        return out;
    }

    /// Reorders `documents`, best first. `Ranked::index` points back into the input,
    /// so the caller keeps its own IDs and metadata.
    [[nodiscard]] Result<std::vector<Ranked>> rerank(std::string_view query,
                                                     std::span<const std::string> documents) const {
        return collect(query, documents, std::nullopt);
    }

    /// Reranks and keeps only the best `k`.
    [[nodiscard]] Result<std::vector<Ranked>> rerank(std::string_view query,
                                                     std::span<const std::string> documents,
                                                     std::size_t k) const {
        return collect(query, documents, k);
    }

private:
    explicit Reranker(KjarniReranker* raw) noexcept : handle_(raw) {}

    [[nodiscard]] Result<std::vector<Ranked>> collect(std::string_view query,
                                                      std::span<const std::string> documents,
                                                      std::optional<std::size_t> k) const {
        // See Embedder::encode: a null array trips the C API's pointer check.
        if (documents.empty()) return std::vector<Ranked>{};

        const std::string q{query};
        detail::CStrings ptrs{documents};
        detail::Owning<KjarniRerankResults, kjarni_rerank_results_free> out;

        const auto code =
            k ? kjarni_reranker_rerank_top_k(handle_.get(), q.c_str(), ptrs.data(), ptrs.size(),
                                             *k, out.out())
              : kjarni_reranker_rerank(handle_.get(), q.c_str(), ptrs.data(), ptrs.size(),
                                       out.out());

        if (auto e = detail::check(code); !e) return std::unexpected(e.error());

        std::vector<Ranked> ranked;
        ranked.reserve(out.raw().len);
        for (std::size_t i = 0; i < out.raw().len; ++i) {
            ranked.push_back({out.raw().results[i].index, out.raw().results[i].score});
        }
        return ranked;
    }

    detail::Handle<KjarniReranker, kjarni_reranker_free> handle_;
};

// ─── Chat ────────────────────────────────────────────────────────

/// Sampling controls. Defaults come from the model.
struct Generation {
    std::optional<float> temperature;
    std::optional<int> top_k;
    std::optional<float> top_p;
    std::optional<int> max_new_tokens;

    /// Deterministic output: same prompt, same answer.
    ///
    /// Fields are assigned rather than designated-initialised so that a partial
    /// initialiser list does not warn under `-Wextra` in consumers' builds.
    [[nodiscard]] static Generation greedy(int max_tokens = 512) {
        Generation g;
        g.temperature = 0.f;
        g.max_new_tokens = max_tokens;
        return g;
    }

    [[nodiscard]] KjarniGenerationConfig to_c() const noexcept {
        KjarniGenerationConfig cfg = kjarni_generation_config_default();
        if (temperature) {
            cfg.temperature = *temperature;
            cfg.do_sample = *temperature > 0.f ? 1 : 0;
        }
        if (top_k) cfg.top_k = *top_k;
        if (top_p) cfg.top_p = *top_p;
        if (max_new_tokens) cfg.max_new_tokens = *max_new_tokens;
        return cfg;
    }
};

/// A local language model.
///
/// Generation is compute-bound and the underlying model is not re-entrant, so one
/// instance serves one request at a time. Share it behind your own queue rather than
/// creating several.
struct ChatOptions {
    std::string model = "llama3.2-3b-instruct";
    std::string system_prompt;
    bool gpu = false;
    bool quiet = true;
};

class Chat {
public:
    using Options = ChatOptions;

    [[nodiscard]] static Result<Chat> create(const ChatOptions& opts = {}) {
        KjarniChatConfig cfg = kjarni_chat_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.system_prompt = opts.system_prompt.empty() ? nullptr : opts.system_prompt.c_str();
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniChat* raw = nullptr;
        if (auto e = detail::check(kjarni_chat_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Chat{raw};
    }

    /// Generates a full response before returning.
    [[nodiscard]] Result<std::string> send(std::string_view message) const {
        const std::string owned{message};
        char* out = nullptr;
        if (auto e = detail::check(
                kjarni_chat_send(handle_.get(), owned.c_str(), nullptr, &out));
            !e) {
            return std::unexpected(e.error());
        }
        return take(out);
    }

    [[nodiscard]] Result<std::string> send(std::string_view message,
                                           const Generation& gen) const {
        const std::string owned{message};
        KjarniGenerationConfig cfg = gen.to_c();
        char* out = nullptr;
        if (auto e = detail::check(
                kjarni_chat_send(handle_.get(), owned.c_str(), &cfg, &out));
            !e) {
            return std::unexpected(e.error());
        }
        return take(out);
    }

    /// Streams tokens as they are produced.
    ///
    /// `on_token` is any callable taking `std::string_view`. Return `false` from it
    /// to stop generation early rather than running to completion and discarding the
    /// result.
    template <typename F>
    [[nodiscard]] std::expected<void, Error> stream(std::string_view message,
                                                    F&& on_token) const {
        const std::string owned{message};
        auto fn = std::forward<F>(on_token);
        return detail::check(kjarni_chat_stream(handle_.get(), owned.c_str(), nullptr,
                                                &detail::stream_trampoline<decltype(fn)>,
                                                &fn, nullptr));
    }

    template <typename F>
    [[nodiscard]] std::expected<void, Error> stream(std::string_view message,
                                                    const Generation& gen,
                                                    F&& on_token) const {
        const std::string owned{message};
        KjarniGenerationConfig cfg = gen.to_c();
        auto fn = std::forward<F>(on_token);
        return detail::check(kjarni_chat_stream(handle_.get(), owned.c_str(), &cfg,
                                                &detail::stream_trampoline<decltype(fn)>,
                                                &fn, nullptr));
    }

    /// The model's context window, in tokens.
    [[nodiscard]] std::size_t context_size() const noexcept {
        return kjarni_chat_context_size(handle_.get());
    }

private:
    explicit Chat(KjarniChat* raw) noexcept : handle_(raw) {}

    /// Adopts a `char*` the C API allocated and hands back an owning string.
    static std::string take(char* raw) {
        if (!raw) return {};
        std::string s{raw};
        kjarni_string_free(raw);
        return s;
    }

    detail::Handle<KjarniChat, kjarni_chat_free> handle_;
};

/// Library version, e.g. "0.1.0".
[[nodiscard]] inline std::string_view version() noexcept {
    const char* v = kjarni_version();
    return v ? std::string_view{v} : std::string_view{};
}

}  // namespace kjarni

#endif  // KJARNI_HPP
