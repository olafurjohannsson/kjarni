//! Example 3: Computing semantic similarity

#include "edgegpt.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "EdgeGPT C++ Example 3: Semantic Similarity\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        // Example 1: Similar sentences
        std::string text1 = "I love programming in C++";
        std::string text2 = "C++ is my favorite programming language";
        
        std::cout << "Text 1: \"" << text1 << "\"\n";
        std::cout << "Text 2: \"" << text2 << "\"\n";
        
        float sim = edge_gpt.similarity(text1, text2);
        std::cout << "Similarity: " << std::fixed << std::setprecision(4) << sim << "\n\n";
        
        // Example 2: Different topics
        std::string text3 = "The weather is sunny today";
        
        std::cout << "Text 1: \"" << text1 << "\"\n";
        std::cout << "Text 3: \"" << text3 << "\"\n";
        
        sim = edge_gpt.similarity(text1, text3);
        std::cout << "Similarity: " << std::fixed << std::setprecision(4) << sim << "\n\n";
        
        // Similarity matrix
        std::cout << "Similarity Matrix:\n";
        std::vector<std::string> texts = {text1, text2, text3};
        
        std::cout << std::setw(10) << "" << " | ";
        for (size_t i = 0; i < texts.size(); ++i) {
            std::cout << std::setw(8) << "Text " + std::to_string(i+1) << " | ";
        }
        std::cout << "\n" << std::string(60, '-') << "\n";
        
        for (size_t i = 0; i < texts.size(); ++i) {
            std::cout << std::setw(10) << ("Text " + std::to_string(i+1)) << " | ";
            for (size_t j = 0; j < texts.size(); ++j) {
                float similarity = edge_gpt.similarity(texts[i], texts[j]);
                std::cout << std::setw(8) << std::fixed << std::setprecision(4) << similarity << " | ";
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