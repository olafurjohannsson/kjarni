//! Example 9: RAG Pipeline (Chat with Data)

#include "edgegpt.hpp"
#include <iostream>
#include <sstream>

int main() {
    std::cout << "EdgeGPT C++ Example 9: Local RAG Pipeline\n";
    std::cout << "-----------------------------------------\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt("gpu"); // Need GPU for generation
        
        // 1. Knowledge Base
        std::string knowledge_base = R"(
            Kjarni is a high-performance inference engine written in Rust.
            It supports Llama, GPT, BERT, and BART models.
            It uses WGPU for cross-platform GPU acceleration.
            It is designed to be the 'SQLite of AI', offering a single library for all AI tasks.
            It has bindings for C#, C++, Go, and Python.
        )";
        
        // 2. Split and Index
        std::cout << "Indexing knowledge base...\n";
        auto chunks = edge_gpt.split_text(knowledge_base, 100, 20); // Chunk size 100 chars
        auto index = edge_gpt.build_index(chunks);
        
        // 3. User Question
        std::string question = "What models does Kjarni support?";
        std::cout << "\nQuestion: " << question << "\n";
        
        // 4. Retrieval
        auto results = edge_gpt.search(index, question, 1); // Get top 1 chunk
        std::string context = results[0].text;
        std::cout << "Retrieved Context: \"" << context << "\"\n";
        
        // 5. Augmented Generation
        std::stringstream prompt;
        prompt << "Context: " << context << "\n"
               << "Question: " << question << "\n"
               << "Answer:";
               
        std::cout << "\nGenerating Answer...\n";
        std::string answer = edge_gpt.generate(prompt.str());
        
        std::cout << "\n>> " << answer << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}