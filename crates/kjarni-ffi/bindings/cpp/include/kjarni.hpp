//! kjarni.hpp - C++ wrapper for Kjarni FFI
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <utility>

// Include the C header
extern "C" {
#include "kjarni.h"
}

namespace kjarni
{

// =============================================================================
// Exception
// =============================================================================

/// Exception thrown by Kjarni operations
class KjarniException : public std::runtime_error
{
public:
    int error_code;

    KjarniException( const std::string& msg, int code = -1 )
        : std::runtime_error( msg ), error_code( code )
    {
    }

    static KjarniException from_last_error( const std::string& operation, KjarniError code )
    {
        const char* msg = kjarni_last_error_message();
        std::string full_msg = operation + " failed: ";
        if( msg )
        {
            full_msg += msg;
        }
        else
        {
            full_msg += kjarni_error_name( code );
        }
        return KjarniException( full_msg, static_cast< int >( code ) );
    }
};

// =============================================================================
// Helper functions
// =============================================================================

namespace detail
{
inline void check_error( KjarniError err, const std::string& operation )
{
    if( err != KJARNI_ERROR_OK )
    {
        throw KjarniException::from_last_error( operation, err );
    }
}

inline std::vector< float > float_array_to_vector( KjarniFloatArray& arr )
{
    std::vector< float > result;
    if( arr.data && arr.len > 0 )
    {
        result.assign( arr.data, arr.data + arr.len );
    }
    kjarni_float_array_free( arr );
    return result;
}

inline std::vector< std::vector< float > > float_2d_array_to_vector( KjarniFloat2DArray& arr )
{
    std::vector< std::vector< float > > result;
    if( arr.data && arr.rows > 0 && arr.cols > 0 )
    {
        result.reserve( arr.rows );
        for( size_t i = 0; i < arr.rows; ++i )
        {
            result.emplace_back(
                arr.data + i * arr.cols,
                arr.data + ( i + 1 ) * arr.cols
                );
        }
    }
    kjarni_float_2d_array_free( arr );
    return result;
}
}

// =============================================================================
// Version
// =============================================================================

inline std::string version()
{
    return kjarni_version();
}

// =============================================================================
// Embedder
// =============================================================================

/// Configuration for Embedder
struct EmbedderConfig {
    std::string model;
    std::string device = "cpu";
    std::string cache_dir;
    std::string model_path;
    bool normalize = true;
    bool quiet = false;
};

/// Text embedding model
class Embedder
{
private:
    KjarniEmbedder* handle_ = nullptr;

public:
    /// Create with default settings
    Embedder()
    {
        KjarniEmbedderConfig config = kjarni_embedder_config_default();
        detail::check_error(
            kjarni_embedder_new( &config, &handle_ ),
            "Embedder creation"
            );
    }

    /// Create with configuration
    explicit Embedder( const EmbedderConfig& cfg )
    {
        KjarniEmbedderConfig config = kjarni_embedder_config_default();
        config.device = ( cfg.device == "gpu" ) ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        config.normalize = cfg.normalize ? 1 : 0;
        config.quiet = cfg.quiet ? 1 : 0;

        if( !cfg.model.empty() )
        {
            config.model_name = cfg.model.c_str();
        }
        if( !cfg.cache_dir.empty() )
        {
            config.cache_dir = cfg.cache_dir.c_str();
        }
        if( !cfg.model_path.empty() )
        {
            config.model_path = cfg.model_path.c_str();
        }

        detail::check_error(
            kjarni_embedder_new( &config, &handle_ ),
            "Embedder creation"
            );
    }

    ~Embedder()
    {
        if( handle_ )
        {
            kjarni_embedder_free( handle_ );
        }
    }

    // Non-copyable
    Embedder( const Embedder& ) = delete;
    Embedder& operator=( const Embedder& ) = delete;

    // Movable
    Embedder( Embedder&& other ) noexcept : handle_( other.handle_ )
    {
        other.handle_ = nullptr;
    }

    Embedder& operator=( Embedder&& other ) noexcept
    {
        if( this != &other )
        {
            if( handle_ )
            {
                kjarni_embedder_free( handle_ );
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /// Encode a single text
    std::vector< float > encode( const std::string& text )
    {
        KjarniFloatArray result;
        detail::check_error(
            kjarni_embedder_encode( handle_, text.c_str(), &result ),
            "encode"
            );
        return detail::float_array_to_vector( result );
    }

    /// Encode multiple texts
    std::vector< std::vector< float > > encode_batch( const std::vector< std::string >& texts )
    {
        if( texts.empty() )
        {
            return {};
        }

        std::vector< const char* > c_texts;
        c_texts.reserve( texts.size() );
        for( const auto& t : texts )
        {
            c_texts.push_back( t.c_str() );
        }

        KjarniFloat2DArray result;
        detail::check_error(
            kjarni_embedder_encode_batch( handle_, c_texts.data(), texts.size(), &result ),
            "encode_batch"
            );
        return detail::float_2d_array_to_vector( result );
    }

    /// Compute cosine similarity between two texts
    float similarity( const std::string& text1, const std::string& text2 )
    {
        float result;
        detail::check_error(
            kjarni_embedder_similarity( handle_, text1.c_str(), text2.c_str(), &result ),
            "similarity"
            );
        return result;
    }

    /// Get embedding dimension
    size_t dim() const
    {
        return kjarni_embedder_dim( handle_ );
    }
};

// =============================================================================
// Classifier
// =============================================================================

/// Single classification result
struct ClassResult {
    std::string label;
    float score;
};

/// Configuration for Classifier
struct ClassifierConfig {
    std::string model = "sentiment";
    std::string device = "cpu";
    std::string cache_dir;
    std::string model_path;
    std::vector< std::string > labels;
    bool multi_label = false;
    bool quiet = false;
};

/// Text classification model
class Classifier
{
private:
    KjarniClassifier* handle_ = nullptr;

public:
    /// Create with default settings (sentiment)
    Classifier() : Classifier( ClassifierConfig{} )
    {
    }

    /// Create with model name
    explicit Classifier( const std::string& model )
    {
        ClassifierConfig cfg;
        cfg.model = model;
        *this = Classifier( cfg );
    }

    /// Create with configuration
    explicit Classifier( const ClassifierConfig& cfg )
    {
        KjarniClassifierConfig config = kjarni_classifier_config_default();
        config.device = ( cfg.device == "gpu" ) ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        config.multi_label = cfg.multi_label ? 1 : 0;
        config.quiet = cfg.quiet ? 1 : 0;

        if( !cfg.model.empty() )
        {
            config.model_name = cfg.model.c_str();
        }
        if( !cfg.cache_dir.empty() )
        {
            config.cache_dir = cfg.cache_dir.c_str();
        }
        if( !cfg.model_path.empty() )
        {
            config.model_path = cfg.model_path.c_str();
        }

        // Handle custom labels
        std::vector< const char* > c_labels;
        if( !cfg.labels.empty() )
        {
            c_labels.reserve( cfg.labels.size() );
            for( const auto& l : cfg.labels )
            {
                c_labels.push_back( l.c_str() );
            }
            config.labels = c_labels.data();
            config.num_labels = c_labels.size();
        }

        detail::check_error(
            kjarni_classifier_new( &config, &handle_ ),
            "Classifier creation"
            );
    }

    ~Classifier()
    {
        if( handle_ )
        {
            kjarni_classifier_free( handle_ );
        }
    }

    // Non-copyable
    Classifier( const Classifier& ) = delete;
    Classifier& operator=( const Classifier& ) = delete;

    // Movable
    Classifier( Classifier&& other ) noexcept : handle_( other.handle_ )
    {
        other.handle_ = nullptr;
    }

    Classifier& operator=( Classifier&& other ) noexcept
    {
        if( this != &other )
        {
            if( handle_ )
            {
                kjarni_classifier_free( handle_ );
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /// Classify a single text
    std::vector< ClassResult > classify( const std::string& text )
    {
        KjarniClassResults results;
        detail::check_error(
            kjarni_classifier_classify( handle_, text.c_str(), &results ),
            "classify"
            );

        std::vector< ClassResult > output;
        if( results.results && results.len > 0 )
        {
            output.reserve( results.len );
            for( size_t i = 0; i < results.len; ++i )
            {
                output.push_back( {
                        results.results[ i ].label ? results.results[ i ].label : "",
                        results.results[ i ].score
                    } );
            }
        }
        kjarni_class_results_free( results );
        return output;
    }

    /// Get number of labels
    size_t num_labels() const
    {
        return kjarni_classifier_num_labels( handle_ );
    }
};

// =============================================================================
// Reranker
// =============================================================================

/// Single rerank result
struct RerankResult {
    size_t index;
    float score;
    std::string document;
};

/// Configuration for Reranker
struct RerankerConfig {
    std::string model;
    std::string device = "cpu";
    std::string cache_dir;
    std::string model_path;
    bool quiet = false;
};

/// Text reranking model using cross-encoders
class Reranker
{
private:
    KjarniReranker* handle_ = nullptr;

public:
    /// Create with default settings
    Reranker()
    {
        KjarniRerankerConfig config = kjarni_reranker_config_default();
        detail::check_error(
            kjarni_reranker_new( &config, &handle_ ),
            "Reranker creation"
            );
    }

    /// Create with model name
    explicit Reranker( const std::string& model )
    {
        RerankerConfig cfg;
        cfg.model = model;
        *this = Reranker( cfg );
    }

    /// Create with configuration
    explicit Reranker( const RerankerConfig& cfg )
    {
        KjarniRerankerConfig config = kjarni_reranker_config_default();
        config.device = ( cfg.device == "gpu" ) ? KJARNI_DEVICE_GPU : KJARNI_DEVICE_CPU;
        config.quiet = cfg.quiet ? 1 : 0;

        if( !cfg.model.empty() )
        {
            config.model_name = cfg.model.c_str();
        }
        if( !cfg.cache_dir.empty() )
        {
            config.cache_dir = cfg.cache_dir.c_str();
        }
        if( !cfg.model_path.empty() )
        {
            config.model_path = cfg.model_path.c_str();
        }

        detail::check_error(
            kjarni_reranker_new( &config, &handle_ ),
            "Reranker creation"
            );
    }

    ~Reranker()
    {
        if( handle_ )
        {
            kjarni_reranker_free( handle_ );
        }
    }

    // Non-copyable
    Reranker( const Reranker& ) = delete;
    Reranker& operator=( const Reranker& ) = delete;

    // Movable
    Reranker( Reranker&& other ) noexcept : handle_( other.handle_ )
    {
        other.handle_ = nullptr;
    }

    Reranker& operator=( Reranker&& other ) noexcept
    {
        if( this != &other )
        {
            if( handle_ )
            {
                kjarni_reranker_free( handle_ );
            }
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    /// Score a single query-document pair
    float score( const std::string& query, const std::string& document )
    {
        float result;
        detail::check_error(
            kjarni_reranker_score( handle_, query.c_str(), document.c_str(), &result ),
            "score"
            );
        return result;
    }

    /// Rerank documents by relevance to query
    std::vector< RerankResult > rerank(
        const std::string& query,
        const std::vector< std::string >& documents )
    {
        if( documents.empty() )
        {
            return {};
        }

        std::vector< const char* > c_docs;
        c_docs.reserve( documents.size() );
        for( const auto& d : documents )
        {
            c_docs.push_back( d.c_str() );
        }

        KjarniRerankResults results;
        detail::check_error(
            kjarni_reranker_rerank( handle_, query.c_str(), c_docs.data(), documents.size(), &results ),
            "rerank"
            );

        std::vector< RerankResult > output;
        if( results.results && results.len > 0 )
        {
            output.reserve( results.len );
            for( size_t i = 0; i < results.len; ++i )
            {
                output.push_back( {
                        results.results[ i ].index,
                        results.results[ i ].score,
                        documents[ results.results[ i ].index ]
                    } );
            }
        }
        kjarni_rerank_results_free( results );
        return output;
    }

    /// Rerank and return top-k results
    std::vector< RerankResult > rerank_top_k(
        const std::string& query,
        const std::vector< std::string >& documents,
        size_t k )
    {
        if( documents.empty() )
        {
            return {};
        }

        std::vector< const char* > c_docs;
        c_docs.reserve( documents.size() );
        for( const auto& d : documents )
        {
            c_docs.push_back( d.c_str() );
        }

        KjarniRerankResults results;
        detail::check_error(
            kjarni_reranker_rerank_top_k( handle_, query.c_str(), c_docs.data(), documents.size(), k, &results ),
            "rerank_top_k"
            );

        std::vector< RerankResult > output;
        if( results.results && results.len > 0 )
        {
            output.reserve( results.len );
            for( size_t i = 0; i < results.len; ++i )
            {
                output.push_back( {
                        results.results[ i ].index,
                        results.results[ i ].score,
                        documents[ results.results[ i ].index ]
                    } );
            }
        }
        kjarni_rerank_results_free( results );
        return output;
    }
};

} // namespace kjarni