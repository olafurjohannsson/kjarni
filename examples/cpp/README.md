# EdgeGPT C++ Examples

Modern C++17 examples demonstrating EdgeGPT functionality.

## Building
```bash
# Build the Rust library first
cd ../../
cargo build --release --features c-bindings

# Build C++ examples
cd examples/cpp
mkdir build && cd build
cmake ..
make
```

## Running Examples
```bash
# From build directory
./01_sentence_encoding
./02_batch_encoding
./03_similarity
./04_semantic_search
./05_reranking
```

## Examples

1. **01_sentence_encoding.cpp** - Encode a single sentence and inspect the embedding
2. **02_batch_encoding.cpp** - Efficiently encode multiple sentences at once
3. **03_similarity.cpp** - Compute semantic similarity between texts
4. **04_semantic_search.cpp** - Search for relevant documents in a corpus
5. **05_reranking.cpp** - Rerank search results using cross-encoder

## API Overview
```cpp
#include "edgegpt.hpp"

// Create instance
edgegpt::EdgeGPT edge_gpt;

// Encode single text
auto embedding = edge_gpt.encode("Hello world");

// Encode batch
std::vector<std::string> texts = {"Text 1", "Text 2"};
auto embeddings = edge_gpt.encode_batch(texts);

// Compute similarity
float sim = edge_gpt.similarity("Text A", "Text B");

// Rerank documents
auto ranked = edge_gpt.rerank(query, documents);
```

## Requirements

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.15+
- EdgeGPT Rust library (built with c-bindings feature)