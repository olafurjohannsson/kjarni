# EdgeGPT Examples

This directory contains examples for using EdgeGPT in different programming languages.

## Available Languages

- **[C++](cpp/)** - Modern C++17 examples using the C++ wrapper
- **[C](c/)** - Pure C examples using the C FFI
- **[Python](python/)** - Python examples using PyO3 bindings
- **[Rust](rust/)** - Native Rust examples

## Quick Start

### C++
```bash
cd cpp
mkdir build && cd build
cmake ..
make
./01_sentence_encoding
```

### C
```bash
cd c
make
./01_basic_encoding
```

### Python
```bash
cd python
pip install -r requirements.txt
python 01_sentence_encoding.py
```

### Rust
```bash
cargo run --example 01_sentence_encoding
```

## Examples Overview

Each language folder contains numbered examples:

1. **Basic Encoding** - Encode a single sentence
2. **Batch Encoding** - Encode multiple sentences efficiently
3. **Similarity** - Compute semantic similarity between texts
4. **Semantic Search** - Find similar documents in a corpus
5. **Reranking** - Rerank search results for better relevance

## Building the Library

Before running examples, build the EdgeGPT library:
```bash
# From the repository root
cd crates/edgegpt

# For C/C++
cargo build --release --features c-bindings

# For Python
maturin develop --release --features python
```