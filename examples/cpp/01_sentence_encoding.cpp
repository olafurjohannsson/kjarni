//! Example 1: Basic sentence encoding

#include "edgegpt.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

int main() {
    std::cout << "EdgeGPT C++ Example 1: Sentence Encoding\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        // Create EdgeGPT instance
        std::cout << "Creating EdgeGPT instance...\n";
        edgegpt::EdgeGPT edge_gpt;
        
        // Encode a sentence
        std::string text = "The quick brown fox jumps over the lazy dog";
        std::cout << "\nEncoding text: \"" << text << "\"\n";
        
        auto embedding = edge_gpt.encode(text);
        
        std::cout << "\n✓ Successfully encoded!\n";
        std::cout << "  Embedding dimension: " << embedding.size() << "\n";
        std::cout << "  First 10 values: ";
        
        std::cout << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < std::min(size_t(10), embedding.size()); ++i) {
            std::cout << embedding[i] << " ";
        }
        std::cout << "\n";
        
        // Compute L2 norm
        float norm = 0.0f;
        for (float val : embedding) {
            norm += val * val;
        }
        norm = std::sqrt(norm);
        
        std::cout << "  L2 norm: " << norm << " (should be ~1.0 for normalized embeddings)\n";
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}