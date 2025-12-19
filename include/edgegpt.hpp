//! edgegpt.hpp - C++ wrapper for EdgeGPT
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <cmath> // For std::sqrt

extern "C"
{
#include "edgegpt.h"
}

namespace edgegpt
{

    /// Exception thrown by EdgeGPT operations
    class EdgeGPTException : public std::runtime_error
    {
    public:
        explicit EdgeGPTException(const std::string &msg) : std::runtime_error(msg) {}
    };

    /// RAII wrapper for EdgeGPT handle
    class EdgeGPT
    {
    private:
        EdgeGPTHandle *handle_;

        void check_error(EdgeGPTError err, const std::string &operation)
        {
            if (err != EDGE_GPT_SUCCESS)
            { // Changed from EdgeGPTError::Success
                throw EdgeGPTException(operation + " failed with error code: " + std::to_string(static_cast<int>(err)));
            }
        }

    public:
        /// Create CPU-based EdgeGPT instance
        EdgeGPT() : handle_(edge_gpt_new_cpu())
        {
            if (handle_ == nullptr)
            {
                throw EdgeGPTException("Failed to create EdgeGPT instance");
            }
        }

        /// Create EdgeGPT instance specifying device ("cpu" or "gpu")
        explicit EdgeGPT(const std::string& device) {
            if (device == "gpu") {
                handle_ = edge_gpt_new_gpu();
                if (handle_ == nullptr) {
                    throw EdgeGPTException("Failed to create EdgeGPT instance (GPU). Check WGPU support.");
                }
            } else {
                handle_ = edge_gpt_new_cpu();
                if (handle_ == nullptr) {
                    throw EdgeGPTException("Failed to create EdgeGPT instance (CPU)");
                }
            }
        }

        /// Destructor
        ~EdgeGPT()
        {
            if (handle_ != nullptr)
            {
                edge_gpt_free(handle_);
            }
        }

        // Delete copy constructor and assignment
        EdgeGPT(const EdgeGPT &) = delete;
        EdgeGPT &operator=(const EdgeGPT &) = delete;

        // Allow move
        EdgeGPT(EdgeGPT &&other) noexcept : handle_(other.handle_)
        {
            other.handle_ = nullptr;
        }

        EdgeGPT &operator=(EdgeGPT &&other) noexcept
        {
            if (this != &other)
            {
                if (handle_ != nullptr)
                {
                    edge_gpt_free(handle_);
                }
                handle_ = other.handle_;
                other.handle_ = nullptr;
            }
            return *this;
        }

        /// Encode a single text
        std::vector<float> encode(const std::string &text)
        {
            float *embedding_ptr = nullptr;
            size_t len = 0;

            auto err = edge_gpt_encode(handle_, text.c_str(), &embedding_ptr, &len);
            check_error(err, "encode");

            std::vector<float> result(embedding_ptr, embedding_ptr + len);
            edge_gpt_free_float_array(embedding_ptr, len);

            return result;
        }

        /// Encode a batch of texts
        std::vector<std::vector<float>> encode_batch(const std::vector<std::string> &texts)
        {
            std::vector<const char *> c_texts;
            c_texts.reserve(texts.size());
            for (const auto &text : texts)
            {
                c_texts.push_back(text.c_str());
            }

            float **embeddings_ptr = nullptr;
            size_t *lens_ptr = nullptr;
            size_t embedding_dim = 0;

            auto err = edge_gpt_encode_batch(
                handle_,
                c_texts.data(),
                texts.size(),
                &embeddings_ptr,
                &lens_ptr,
                &embedding_dim);
            check_error(err, "encode_batch");

            std::vector<std::vector<float>> result;
            result.reserve(texts.size());

            for (size_t i = 0; i < texts.size(); ++i)
            {
                result.emplace_back(embeddings_ptr[i], embeddings_ptr[i] + embedding_dim);
            }

            edge_gpt_free_batch_embeddings(embeddings_ptr, lens_ptr, texts.size(), embedding_dim);

            return result;
        }

        /// Compute similarity between two texts
        float similarity(const std::string &text1, const std::string &text2)
        {
            float sim = 0.0f;
            auto err = edge_gpt_similarity(handle_, text1.c_str(), text2.c_str(), &sim);
            check_error(err, "similarity");
            return sim;
        }

        /// Rerank documents by relevance to query
        std::vector<std::pair<size_t, float>> rerank(
            const std::string &query,
            const std::vector<std::string> &documents)
        {
            std::vector<const char *> c_docs;
            c_docs.reserve(documents.size());
            for (const auto &doc : documents)
            {
                c_docs.push_back(doc.c_str());
            }

            size_t *indices_ptr = nullptr;
            float *scores_ptr = nullptr;

            auto err = edge_gpt_rerank(
                handle_,
                query.c_str(),
                c_docs.data(),
                documents.size(),
                &indices_ptr,
                &scores_ptr);
            check_error(err, "rerank");

            std::vector<std::pair<size_t, float>> result;
            result.reserve(documents.size());

            for (size_t i = 0; i < documents.size(); ++i)
            {
                result.emplace_back(indices_ptr[i], scores_ptr[i]);
            }

            edge_gpt_free_usize_array(indices_ptr, documents.size());
            edge_gpt_free_float_array(scores_ptr, documents.size());

            return result;
        }
        /// Generate text continuation (Llama/GPT)
        std::string generate(const std::string &prompt)
        {
            char *out_ptr = nullptr;
            auto err = edge_gpt_generate(handle_, prompt.c_str(), &out_ptr);
            check_error(err, "generate");

            std::string result(out_ptr);
            edge_gpt_free_string(out_ptr);
            return result;
        }

        /// Summarize text (BART/T5)
        std::string summarize(const std::string &text)
        {
            char *out_ptr = nullptr;
            auto err = edge_gpt_summarize(handle_, text.c_str(), &out_ptr);
            check_error(err, "summarize");

            std::string result(out_ptr);
            edge_gpt_free_string(out_ptr);
            return result;
        }
    };

} // namespace edgegpt