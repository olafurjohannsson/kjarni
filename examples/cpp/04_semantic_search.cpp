//! Example 4: Semantic search in a document corpus

#include "edgegpt.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cmath>

int main() {
    std::cout << "EdgeGPT C++ Example 4: Semantic Search\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        // Corpus of documents
        std::vector<std::string> corpus = {
            "The solar system contains eight planets orbiting the sun",
            "Machine learning algorithms can recognize patterns in data",
            "Pizza is a popular Italian dish with cheese and toppings",
            "Neural networks are inspired by biological neurons",
            "The Pacific Ocean is the largest body of water on Earth",
            "Python is a high-level programming language",
            "Photosynthesis converts light energy into chemical energy",
            "Deep learning has revolutionized computer vision tasks"
        };
        
        std::string query = "artificial intelligence and programming";
        
        std::cout << "Corpus: " << corpus.size() << " documents\n";
        std::cout << "Query: \"" << query << "\"\n\n";
        
        std::cout << "Searching...\n\n";
        
        // Encode query and corpus
        auto query_emb = edge_gpt.encode(query);
        auto corpus_embs = edge_gpt.encode_batch(corpus);
        
        // Compute similarities
        std::vector<std::pair<size_t, float>> results;
        for (size_t i = 0; i < corpus_embs.size(); ++i) {
            float dot = 0.0f, norm_q = 0.0f, norm_d = 0.0f;
            for (size_t j = 0; j < query_emb.size(); ++j) {
                dot += query_emb[j] * corpus_embs[i][j];
                norm_q += query_emb[j] * query_emb[j];
                norm_d += corpus_embs[i][j] * corpus_embs[i][j];
            }
            float sim = dot / (std::sqrt(norm_q) * std::sqrt(norm_d));
            results.push_back({i, sim});
        }
        
        // Sort by similarity
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Display top 5 results
        std::cout << "Top 5 Results:\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (size_t i = 0; i < std::min(size_t(5), results.size()); ++i) {
            size_t idx = results[i].first;
            float score = results[i].second;
            
            std::cout << (i + 1) << ". [Score: " << std::fixed << std::setprecision(4) << score << "]\n";
            std::cout << "   " << corpus[idx] << "\n\n";
        }
        
        std::cout << "✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}