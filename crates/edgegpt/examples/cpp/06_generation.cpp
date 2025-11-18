//! Example 6: Text Generation (Llama/GPT)

#include "edgegpt.hpp"
#include <iostream>
#include <string>

int main() {
    std::cout << "EdgeGPT C++ Example 6: Text Generation\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        // Initialize with GPU if available for generation
        std::cout << "Loading Text Generator (Llama/GPT)...\n";
        edgegpt::EdgeGPT edge_gpt("gpu"); 
        
        std::string prompt = "The future of artificial intelligence is";
        
        std::cout << "Prompt: \"" << prompt << "\"\n";
        std::cout << "Generating...\n\n";
        
        // Generate text
        std::string output = edge_gpt.generate(prompt);
        
        std::cout << "--- Generated Output ---\n";
        std::cout << prompt << output << "\n";
        std::cout << "------------------------\n";
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}