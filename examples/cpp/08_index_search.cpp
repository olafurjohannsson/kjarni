//! Example 8: Building and Searching an Index (RAG Core)

#include "edgegpt.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "EdgeGPT C++ Example 8: Indexing & Search\n";
    std::cout << std::string(60, '=') << "\n\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        // 1. Data Ingestion
        std::vector<std::string> documents = {
            "Rust guarantees memory safety without a garbage collector.",
            "Python is dynamically typed and garbage collected.",
            "C++ allows manual memory management and pointer arithmetic.",
            "Go uses a concurrent garbage collector.",
            "Java runs on a virtual machine with automatic memory management."
        };
        
        std::cout << "Indexing " << documents.size() << " documents...\n";
        
        // 2. Build Index (Computes embeddings + BM25 automatically)
        auto index = edge_gpt.build_index(documents);
        
        // 3. Search
        std::string query = "memory safety no gc";
        std::cout << "\nQuery: \"" << query << "\"\n";
        
        // Hybrid Search (Vector + Keyword)
        auto results = edge_gpt.search(index, query, 3);
        
        std::cout << "\nTop 3 Results:\n";
        for (const auto& res : results) {
            std::cout << "- [Score: " << res.score << "] " << res.text << "\n";
        }
        
        // 4. Save/Load (Persistence)
        std::cout << "\nSaving index to 'memory_langs.json'...\n";
        edge_gpt.save_index(index, "memory_langs.json");
        
        std::cout << "\n✓ Example completed successfully!\n";
        
    } catch (const edgegpt::EdgeGPTException& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}