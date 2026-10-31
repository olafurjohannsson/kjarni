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

// Image search is part of the library's default build. Define KJARNI_NO_IMAGE_IO
// when linking a library built with --no-default-features, which leaves it out.
// A kjarni.h included before this file, without KJARNI_IMAGE_IO, has already left out
// the declarations the image wrapper needs, so the wrapper is left out too.
#if defined(KJARNI_FFI_H) && !defined(KJARNI_IMAGE_IO) && !defined(KJARNI_NO_IMAGE_IO)
#define KJARNI_NO_IMAGE_IO
#endif
#if !defined(KJARNI_NO_IMAGE_IO) && !defined(KJARNI_IMAGE_IO)
#define KJARNI_IMAGE_IO
#endif

#include "kjarni.h"

#include <algorithm>
#include <cstdint>
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

    /// The error code.
    [[nodiscard]] KjarniErrorCode code() const noexcept { return code_; }
    /// A description of the failure, captured from the library when the error was created.
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

/// Adopts a `char*` the C API allocated and returns an owning string.
inline std::string adopt(char* raw) {
    if (!raw) return {};
    std::string s{raw};
    kjarni_string_free(raw);
    return s;
}

/// Joins items with commas, for the C config fields that take a list that way.
inline std::string join(std::span<const std::string> items) {
    std::string out;
    for (const auto& item : items) {
        if (!out.empty()) out += ',';
        out += item;
    }
    return out;
}

/// Null for an empty string: the C API reads null as "use the default".
inline const char* or_null(const std::string& s) noexcept { return s.empty() ? nullptr : s.c_str(); }

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

    /// The vector.
    [[nodiscard]] std::span<const float> values() const noexcept {
        return {arr_.raw().data, arr_.raw().len};
    }
    /// The number of dimensions.
    [[nodiscard]] std::size_t size() const noexcept { return arr_.raw().len; }
    /// A pointer to the first element, valid for the lifetime of this object.
    [[nodiscard]] const float* data() const noexcept { return arr_.raw().data; }

    /// An iterator to the first element.
    [[nodiscard]] auto begin() const noexcept { return values().begin(); }
    /// An iterator past the last element.
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

    /// Loads the model, downloading it on first use.
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

    /// The embedding of one text.
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

    /// The number of dimensions the model produces.
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

    /// Loads the model, downloading it on first use.
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

    /// Every label with its score for each text, in order. One call for many
    /// texts is much faster than calling `classify` for each.
    [[nodiscard]] Result<std::vector<std::vector<Label>>> classify(std::span<const std::string> texts) const {
        if (texts.empty()) return std::vector<std::vector<Label>>{};
        detail::CStrings ptrs{texts};
        detail::Owning<KjarniClassBatch, kjarni_class_batch_free> out;
        if (auto e = detail::check(kjarni_classifier_classify_batch(handle_.get(), ptrs.data(), ptrs.size(), out.out()));
            !e) {
            return std::unexpected(e.error());
        }
        std::vector<std::vector<Label>> all;
        all.reserve(out.raw().len);
        for (std::size_t i = 0; i < out.raw().len; ++i) {
            const auto& r = out.raw().items[i];
            auto& labels = all.emplace_back();
            labels.reserve(r.len);
            for (std::size_t j = 0; j < r.len; ++j) {
                labels.push_back({r.results[j].label ? r.results[j].label : "", r.results[j].score});
            }
        }
        return all;
    }

    /// The highest-scoring label, or `std::nullopt` if the model returned nothing.
    [[nodiscard]] Result<std::optional<Label>> top(std::string_view text) const {
        auto labels = classify(text);
        if (!labels) return std::unexpected(labels.error());
        if (labels->empty()) return std::optional<Label>{};
        return std::optional<Label>{std::move(labels->front())};
    }

    /// The number of labels the model predicts.
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

    /// Loads the model, downloading it on first use.
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
    /// Load the 4-bit GGUF weights instead of full precision: a fraction of the
    /// memory and several times faster to generate on a CPU.
    bool quantized = false;
};

class Chat {
public:
    using Options = ChatOptions;

    /// Loads the model, downloading it on first use.
    [[nodiscard]] static Result<Chat> create(const ChatOptions& opts = {}) {
        KjarniChatConfig cfg = kjarni_chat_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.system_prompt = opts.system_prompt.empty() ? nullptr : opts.system_prompt.c_str();
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;
        cfg.quantized = opts.quantized ? 1 : 0;

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

    /// Sends one message with these generation settings and returns the reply.
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

    /// Streams a reply with these generation settings. `on_token` works as in the overload above.
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

// ─── Indexer ─────────────────────────────────────────────────────

/// What an indexing run did.
struct IndexStats {
    std::size_t documents_indexed{};
    std::size_t chunks_created{};
    std::size_t dimension{};
    std::uint64_t size_bytes{};
    std::size_t files_processed{};
    std::size_t files_skipped{};
    std::uint64_t elapsed_ms{};
};

struct IndexerOptions {
    std::string model = "minilm-l6-v2";
    std::string cache_dir;
    /// Characters per chunk; 0 keeps the library's default.
    std::size_t chunk_size = 0;
    /// Characters shared by neighbouring chunks; 0 keeps the library's default.
    std::size_t chunk_overlap = 0;
    /// File extensions to index, e.g. {"md", "txt"}; empty keeps the defaults.
    std::vector<std::string> extensions;
    /// Names or `*` patterns to skip, e.g. {"node_modules", "*.min.js"}.
    std::vector<std::string> exclude;
    bool recursive = true;
    bool include_hidden = false;
    bool gpu = false;
    bool quiet = true;
};

/// Builds a search index on disk from files and folders.
class Indexer {
public:
    using Options = IndexerOptions;

    /// Loads the embedding model, downloading it on first use.
    [[nodiscard]] static Result<Indexer> create(const IndexerOptions& opts = {}) {
        KjarniIndexerConfig cfg = kjarni_indexer_config_default();
        const std::string extensions = detail::join(opts.extensions);
        const std::string exclude = detail::join(opts.exclude);
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = detail::or_null(opts.cache_dir);
        if (opts.chunk_size) cfg.chunk_size = opts.chunk_size;
        if (opts.chunk_overlap) cfg.chunk_overlap = opts.chunk_overlap;
        cfg.extensions = detail::or_null(extensions);
        cfg.exclude_patterns = detail::or_null(exclude);
        cfg.recursive = opts.recursive ? 1 : 0;
        cfg.include_hidden = opts.include_hidden ? 1 : 0;
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniIndexer* raw = nullptr;
        if (auto e = detail::check(kjarni_indexer_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Indexer{raw};
    }

    /// Indexes `inputs` (files and folders) into a new index at `index_path`.
    /// `force` replaces an index already there.
    [[nodiscard]] Result<IndexStats> create_index(std::string_view index_path,
                                                  std::span<const std::string> inputs,
                                                  bool force = false) const {
        const std::string path{index_path};
        detail::CStrings ptrs{inputs};
        KjarniIndexStats stats{};
        if (auto e = detail::check(kjarni_indexer_create(handle_.get(), path.c_str(), ptrs.data(),
                                                         ptrs.size(), force ? 1 : 0, &stats));
            !e) {
            return std::unexpected(e.error());
        }
        return IndexStats{stats.documents_indexed, stats.chunks_created, stats.dimension,
                          stats.size_bytes,        stats.files_processed, stats.files_skipped,
                          stats.elapsed_ms};
    }

    /// Adds `inputs` to an existing index; returns how many chunks were added.
    [[nodiscard]] Result<std::size_t> add(std::string_view index_path,
                                          std::span<const std::string> inputs) const {
        const std::string path{index_path};
        detail::CStrings ptrs{inputs};
        std::size_t added = 0;
        if (auto e = detail::check(
                kjarni_indexer_add(handle_.get(), path.c_str(), ptrs.data(), ptrs.size(), &added));
            !e) {
            return std::unexpected(e.error());
        }
        return added;
    }

private:
    explicit Indexer(KjarniIndexer* raw) noexcept : handle_(raw) {}

    detail::Handle<KjarniIndexer, kjarni_indexer_free> handle_;
};

// ─── Searcher ────────────────────────────────────────────────────

enum class SearchMode { keyword = KJARNI_SEARCH_MODE_KEYWORD, semantic = KJARNI_SEARCH_MODE_SEMANTIC, hybrid = KJARNI_SEARCH_MODE_HYBRID };

struct SearcherOptions {
    /// Must be the model the index was built with.
    std::string model = "minilm-l6-v2";
    /// A cross-encoder for reranking, e.g. "minilm-l6-v2-cross-encoder"; empty for none.
    std::string rerank_model;
    std::string cache_dir;
    SearchMode mode = SearchMode::hybrid;
    std::size_t top_k = 10;
    bool gpu = false;
    bool quiet = true;
};

/// Per-query overrides; anything left unset uses the searcher's defaults.
struct SearchOptions {
    std::optional<SearchMode> mode;
    std::size_t top_k = 0;
    std::optional<bool> rerank;
    /// Drop results scoring below this.
    float threshold = 0.f;
    /// Only files whose path matches this `*` pattern.
    std::string source_pattern;
    std::string filter_key;
    std::string filter_value;
};

/// Searches an index built by `Indexer`, by keyword, meaning or both.
class Searcher {
public:
    using Options = SearcherOptions;

    /// Loads the models, downloading them on first use.
    [[nodiscard]] static Result<Searcher> create(const SearcherOptions& opts = {}) {
        KjarniSearcherConfig cfg = kjarni_searcher_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.rerank_model = detail::or_null(opts.rerank_model);
        cfg.cache_dir = detail::or_null(opts.cache_dir);
        cfg.default_mode = static_cast<KjarniSearchMode>(opts.mode);
        if (opts.top_k) cfg.default_top_k = opts.top_k;
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniSearcher* raw = nullptr;
        if (auto e = detail::check(kjarni_searcher_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Searcher{raw};
    }

    /// Searches an index with the searcher's default mode and result count.
    [[nodiscard]] Result<std::vector<Hit>> search(std::string_view index_path,
                                                  std::string_view query) const {
        const std::string path{index_path}, q{query};
        detail::Owning<KjarniSearchResults, kjarni_search_results_free> out;
        if (auto e = detail::check(kjarni_searcher_search(handle_.get(), path.c_str(), q.c_str(), out.out()));
            !e) {
            return std::unexpected(e.error());
        }
        return hits(out.raw());
    }

    /// Searches an index with these options for this call.
    [[nodiscard]] Result<std::vector<Hit>> search(std::string_view index_path, std::string_view query,
                                                  const SearchOptions& options) const {
        const std::string path{index_path}, q{query};
        KjarniSearchOptions o = kjarni_search_options_default();
        if (options.mode) o.mode = static_cast<std::int32_t>(*options.mode);
        if (options.top_k) o.top_k = options.top_k;
        if (options.rerank) o.use_reranker = *options.rerank ? 1 : 0;
        o.threshold = options.threshold;
        o.source_pattern = detail::or_null(options.source_pattern);
        o.filter_key = detail::or_null(options.filter_key);
        o.filter_value = detail::or_null(options.filter_value);
        detail::Owning<KjarniSearchResults, kjarni_search_results_free> out;
        if (auto e = detail::check(kjarni_searcher_search_with_options(handle_.get(), path.c_str(),
                                                                        q.c_str(), &o, out.out()));
            !e) {
            return std::unexpected(e.error());
        }
        return hits(out.raw());
    }

    /// Whether this searcher was created with a reranker.
    [[nodiscard]] bool has_reranker() const noexcept { return kjarni_searcher_has_reranker(handle_.get()); }

private:
    explicit Searcher(KjarniSearcher* raw) noexcept : handle_(raw) {}

    static std::vector<Hit> hits(const KjarniSearchResults& raw) {
        std::vector<Hit> out;
        out.reserve(raw.len);
        for (std::size_t i = 0; i < raw.len; ++i) {
            const auto& r = raw.results[i];
            out.push_back({r.score, r.document_id, r.text ? r.text : "",
                           r.metadata_json ? r.metadata_json : ""});
        }
        return out;
    }

    detail::Handle<KjarniSearcher, kjarni_searcher_free> handle_;
};

// ─── Transcriber ─────────────────────────────────────────────────

/// One timed piece of a transcript, in seconds from the start.
struct Segment {
    float start{};
    float end{};
    std::string text;
};

struct Transcript {
    std::string text;
    /// ISO code of the language transcribed: the one given, or the one detected.
    std::string language;
    float duration_secs{};
    std::vector<Segment> segments;
};

struct TranscriberOptions {
    std::string model = "whisper-small";
    /// ISO code such as "en" or "is"; empty detects it from the audio.
    std::string language;
    /// Translate the speech into English instead of transcribing it.
    bool translate = false;
    bool gpu = false;
    bool quiet = true;
};

/// Speech to text with Whisper.
class Transcriber {
public:
    using Options = TranscriberOptions;

    /// Loads the model, downloading it on first use.
    [[nodiscard]] static Result<Transcriber> create(const TranscriberOptions& opts = {}) {
        KjarniTranscriberConfig cfg = kjarni_transcriber_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.language = detail::or_null(opts.language);
        cfg.translate = opts.translate ? 1 : 0;
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniTranscriber* raw = nullptr;
        if (auto e = detail::check(kjarni_transcriber_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Transcriber{raw};
    }

    /// Transcribes a wav, mp3, flac or ogg file.
    [[nodiscard]] Result<Transcript> transcribe_file(std::string_view path) const {
        const std::string p{path};
        detail::Owning<KjarniTranscription, kjarni_transcription_free> out;
        if (auto e = detail::check(kjarni_transcriber_transcribe_file(handle_.get(), p.c_str(), out.out()));
            !e) {
            return std::unexpected(e.error());
        }
        return transcript(out.raw());
    }

    /// Transcribes mono samples in -1..1 at any sample rate.
    [[nodiscard]] Result<Transcript> transcribe(std::span<const float> samples,
                                                std::uint32_t sample_rate) const {
        detail::Owning<KjarniTranscription, kjarni_transcription_free> out;
        if (auto e = detail::check(kjarni_transcriber_transcribe_pcm(
                handle_.get(), samples.data(), samples.size(), sample_rate, out.out()));
            !e) {
            return std::unexpected(e.error());
        }
        return transcript(out.raw());
    }

private:
    explicit Transcriber(KjarniTranscriber* raw) noexcept : handle_(raw) {}

    static Transcript transcript(const KjarniTranscription& raw) {
        Transcript t{raw.text ? raw.text : "", raw.language ? raw.language : "", raw.duration_secs, {}};
        t.segments.reserve(raw.len);
        for (std::size_t i = 0; i < raw.len; ++i) {
            const auto& s = raw.segments[i];
            t.segments.push_back({s.start, s.end, s.text ? s.text : ""});
        }
        return t;
    }

    detail::Handle<KjarniTranscriber, kjarni_transcriber_free> handle_;
};

// ─── Summarizer ──────────────────────────────────────────────────

struct SummarizerOptions {
    std::string model = "distilbart-cnn";
    std::string cache_dir;
    /// Summary length bounds in tokens; 0 keeps the model's defaults.
    std::size_t min_length = 0;
    std::size_t max_length = 0;
    bool gpu = false;
    bool quiet = true;
};

/// Summarizes article-length text.
class Summarizer {
public:
    using Options = SummarizerOptions;

    /// Loads the model, downloading it on first use.
    [[nodiscard]] static Result<Summarizer> create(const SummarizerOptions& opts = {}) {
        KjarniSummarizerConfig cfg = kjarni_summarizer_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = detail::or_null(opts.cache_dir);
        cfg.min_length = opts.min_length;
        cfg.max_length = opts.max_length;
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniSummarizer* raw = nullptr;
        if (auto e = detail::check(kjarni_summarizer_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Summarizer{raw};
    }

    /// A summary of the text.
    [[nodiscard]] Result<std::string> summarize(std::string_view text) const {
        const std::string t{text};
        char* out = nullptr;
        if (auto e = detail::check(kjarni_summarizer_summarize(handle_.get(), t.c_str(), &out)); !e) {
            return std::unexpected(e.error());
        }
        return detail::adopt(out);
    }

private:
    explicit Summarizer(KjarniSummarizer* raw) noexcept : handle_(raw) {}

    detail::Handle<KjarniSummarizer, kjarni_summarizer_free> handle_;
};

// ─── Translator ──────────────────────────────────────────────────

struct TranslatorOptions {
    std::string model = "flan-t5-base";
    std::string cache_dir;
    /// Default languages for `translate(text)`, e.g. "en" and "de".
    std::string from;
    std::string to;
    bool gpu = false;
    bool quiet = true;
};

/// Translates text between languages.
class Translator {
public:
    using Options = TranslatorOptions;

    /// Loads the model, downloading it on first use.
    [[nodiscard]] static Result<Translator> create(const TranslatorOptions& opts = {}) {
        KjarniTranslatorConfig cfg = kjarni_translator_config_default();
        cfg.model_name = opts.model.c_str();
        cfg.cache_dir = detail::or_null(opts.cache_dir);
        cfg.from = detail::or_null(opts.from);
        cfg.to = detail::or_null(opts.to);
        cfg.device = opts.gpu ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        cfg.quiet = opts.quiet ? 1 : 0;

        KjarniTranslator* raw = nullptr;
        if (auto e = detail::check(kjarni_translator_new(&cfg, &raw)); !e) {
            return std::unexpected(e.error());
        }
        return Translator{raw};
    }

    /// Translates between the default languages given at creation.
    [[nodiscard]] Result<std::string> translate(std::string_view text) const {
        return run(text, nullptr, nullptr);
    }

    /// Translates between these languages for this call, such as "en" and "de".
    [[nodiscard]] Result<std::string> translate(std::string_view text, std::string_view from,
                                                std::string_view to) const {
        const std::string f{from}, t{to};
        return run(text, f.c_str(), t.c_str());
    }

private:
    explicit Translator(KjarniTranslator* raw) noexcept : handle_(raw) {}

    [[nodiscard]] Result<std::string> run(std::string_view text, const char* from, const char* to) const {
        const std::string t{text};
        char* out = nullptr;
        if (auto e = detail::check(kjarni_translator_translate(handle_.get(), t.c_str(), from, to, &out)); !e) {
            return std::unexpected(e.error());
        }
        return detail::adopt(out);
    }

    detail::Handle<KjarniTranslator, kjarni_translator_free> handle_;
};

#if defined(KJARNI_IMAGE_IO)
// ─── Image search ────────────────────────────────────────────────

struct ImageHit {
    std::string path;
    /// Cosine similarity; small even for good matches, so only the order matters.
    float score{};
};

struct ScanReport {
    std::size_t added{};
    /// Files that could not be decoded.
    std::size_t skipped{};
};

struct ImageIndexOptions {
    /// A CLIP model directory; empty uses clip-vit-base-32, downloaded on first use.
    std::string model_dir;
};

/// Photos searched by description, with CLIP. Not safe to share across threads
/// while adding.
class ImageIndex {
public:
    using Options = ImageIndexOptions;

    /// Loads the CLIP model, downloading it on first use.
    [[nodiscard]] static Result<ImageIndex> create(const ImageIndexOptions& opts = {}) {
        KjarniImageIndex* raw = nullptr;
        if (auto e = detail::check(kjarni_image_index_new(detail::or_null(opts.model_dir), &raw)); !e) {
            return std::unexpected(e.error());
        }
        return ImageIndex{raw};
    }

    /// Adds one image file.
    [[nodiscard]] std::expected<void, Error> add(std::string_view path) {
        const std::string p{path};
        return detail::check(kjarni_image_index_add(handle_.get(), p.c_str()));
    }

    /// Adds every JPEG and PNG under `dir`.
    [[nodiscard]] Result<ScanReport> add_directory(std::string_view dir) {
        const std::string d{dir};
        KjarniScanReport report{};
        if (auto e = detail::check(kjarni_image_index_add_directory(handle_.get(), d.c_str(), &report)); !e) {
            return std::unexpected(e.error());
        }
        return ScanReport{report.added, report.skipped};
    }

    /// The images that best match a description, best first.
    [[nodiscard]] Result<std::vector<ImageHit>> search(std::string_view query, std::size_t top_k = 10) const {
        const std::string q{query};
        detail::Owning<KjarniImageResults, kjarni_image_results_free> out;
        if (auto e = detail::check(kjarni_image_index_search(handle_.get(), q.c_str(), top_k, out.out())); !e) {
            return std::unexpected(e.error());
        }
        std::vector<ImageHit> hits;
        hits.reserve(out.raw().len);
        for (std::size_t i = 0; i < out.raw().len; ++i) {
            hits.push_back({out.raw().hits[i].path ? out.raw().hits[i].path : "", out.raw().hits[i].score});
        }
        return hits;
    }

    /// Writes the vectors (not the images) to `path`.
    [[nodiscard]] std::expected<void, Error> save(std::string_view path) const {
        const std::string p{path};
        return detail::check(kjarni_image_index_save(handle_.get(), p.c_str()));
    }

    /// Restores vectors saved by `save`; returns how many.
    [[nodiscard]] Result<std::size_t> load(std::string_view path) {
        const std::string p{path};
        std::size_t count = 0;
        if (auto e = detail::check(kjarni_image_index_load(handle_.get(), p.c_str(), &count)); !e) {
            return std::unexpected(e.error());
        }
        return count;
    }

    /// The number of images in the index.
    [[nodiscard]] std::size_t size() const noexcept { return kjarni_image_index_len(handle_.get()); }

private:
    explicit ImageIndex(KjarniImageIndex* raw) noexcept : handle_(raw) {}

    detail::Handle<KjarniImageIndex, kjarni_image_index_free> handle_;
};
#endif  // KJARNI_IMAGE_IO

/// Library version, e.g. "0.1.0".
[[nodiscard]] inline std::string_view version() noexcept {
    const char* v = kjarni_version();
    return v ? std::string_view{v} : std::string_view{};
}

}  // namespace kjarni

#endif  // KJARNI_HPP
