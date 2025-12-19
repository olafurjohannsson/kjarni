//! Example 2: Batch encoding for efficiency

#include "edgegpt.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    std::cout << "EdgeGPT C++ Example 2: Batch Encoding\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        std::vector<std::string> sentences = {
            "Artificial intelligence is transforming the world",
            "Machine learning enables computers to learn from data",
            "Deep learning uses neural networks with many layers",
            "Natural language processing helps computers understand text",
            "Computer vision allows machines to interpret images"
        };
        
        std::cout << "Encoding " << sentences.size() << " sentences...\n\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        auto embeddings = edge_gpt.encode_batch(sentences);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "✓ Batch encoding completed in " << duration.count() << " ms\n";
        std::cout << "  Average: " << (duration.count() / float(sentences.size())) << " ms per sentence\n\n";
        
        std::cout << "Results:\n";
        for (size_t i = 0; i < embeddings.size(); ++i) {
            std::cout << "  [" << i << "] Dimension: " << embeddings[i].size() << " | First 3: ";
            std::cout << std::fixed << std::setprecision(4);
            for (size_t j = 0; j < std::min(size_t(3), embeddings[i].size()); ++j) {
                std::cout << embeddings[i][j] << " ";
            }
            std::cout << "\n";
        }
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}