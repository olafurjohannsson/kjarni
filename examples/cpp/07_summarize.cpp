//! Example 7: Text Summarization

#include "edgegpt.hpp"
#include <iostream>
#include <string>

int main() {
    std::cout << "EdgeGPT C++ Example 7: Summarization\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        std::cout << "Loading Seq2Seq Model (BART)...\n";
        edgegpt::EdgeGPT edge_gpt("cpu"); // CPU is often fine for single summaries
        
        std::string article = R"(
            Rust is a multi-paradigm, general-purpose programming language that emphasizes performance, 
            type safety, and concurrency. It enforces memory safety—meaning that all references point 
            to valid memory—without using a garbage collector. To simultaneously enforce memory safety 
            and prevent data races, its 'borrow checker' tracks the object lifetime of all references 
            in a program during compilation. Rust was influenced by languages like C++, Haskell, and Erlang.
            It has gained significant popularity in systems programming, web assembly, and embedded devices
            due to its ability to provide high-level abstractions with low-level control.
        )";
        
        std::cout << "Original Text (" << article.length() << " chars):\n" << article << "\n\n";
        std::cout << "Summarizing...\n";
        
        std::string summary = edge_gpt.summarize(article);
        
        std::cout << "\n--- Summary ---\n";
        std::cout << summary << "\n";
        std::cout << "---------------\n";
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}