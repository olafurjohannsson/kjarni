//! Example 5: Document reranking with cross-encoder

#include "edgegpt.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "EdgeGPT C++ Example 5: Document Reranking\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        std::string query = "How does machine learning work?";
        
        std::vector<std::string> documents = {
            "Machine learning algorithms learn patterns from data to make predictions",
            "The weather forecast predicts rain for tomorrow afternoon",
            "Neural networks use layers of interconnected nodes to process information",
            "I enjoy cooking pasta with tomato sauce and basil",
            "Supervised learning requires labeled training data",
            "The movie received excellent reviews from critics",
            "Deep learning is a subset of machine learning using neural networks"
        };
        
        std::cout << "Query: \"" << query << "\"\n\n";
        std::cout << "Documents (" << documents.size() << "):\n";
        for (size_t i = 0; i < documents.size(); ++i) {
            std::cout << "  [" << i << "] " << documents[i] << "\n";
        }
        
        std::cout << "\nReranking with cross-encoder...\n\n";
        
        auto ranked = edge_gpt.rerank(query, documents);
        
        std::cout << "Reranked Results:\n";
        std::cout << std::string(60, '-') << "\n";
        
        for (size_t rank = 0; rank < ranked.size(); ++rank) {
            size_t idx = ranked[rank].first;
            float score = ranked[rank].second;
            
            std::cout << "Rank " << (rank + 1) << ": ";
            std::cout << "[Score: " << std::fixed << std::setprecision(4) << score << "]\n";
            std::cout << "  [" << idx << "] " << documents[idx] << "\n\n";
        }
        
        std::cout << "✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}