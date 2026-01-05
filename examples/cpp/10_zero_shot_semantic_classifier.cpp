//! Example 10: Zero-Shot Semantic Classifier

#include "edgegpt.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    std::cout << "EdgeGPT C++ Example 10: Zero-Shot Classification\n";
    std::cout << "----------------------------------------------\n";
    
    try {
        edgegpt::EdgeGPT edge_gpt;
        
        // Define categories (labels)
        std::vector<std::string> labels = {"Sports", "Technology", "Politics", "Cooking"};
        
        // Pre-compute label embeddings
        std::cout << "Encoding labels...\n";
        auto label_embs = edge_gpt.encode_batch(labels);
        
        // Input text to classify
        std::string input = "The new GPU architecture delivers 2x performance per watt.";
        std::cout << "\nInput: \"" << input << "\"\n";
        
        // Encode input
        auto input_emb = edge_gpt.encode(input);
        
        // Find best match
        int best_idx = -1;
        float best_score = -1.0f;
        
        std::cout << "\nScores:\n";
        for (size_t i = 0; i < labels.size(); ++i) {
            // Compute similarity between input and label
            float score = edge_gpt.similarity_vec(input_emb, label_embs[i]);
            
            std::cout << "  " << labels[i] << ": " << score << "\n";
            
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }
        
        std::cout << "\n>> Classification: " << labels[best_idx] << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
    }
    return 0;
}