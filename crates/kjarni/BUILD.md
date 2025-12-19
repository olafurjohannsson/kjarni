# Building EdgeGPT

## Rust Library

### Build the Rust library:
```bash
# Debug build
cargo build

# Release build (recommended)
cargo build --release

# With Python bindings
cargo build --release --features python

# With C bindings
cargo build --release --features c-bindings
```

## C/C++ Examples

### Prerequisites:
- CMake 3.15+
- C++17 compatible compiler
- Built Rust library (see above)

### Build steps:
```bash
cd crates/edgegpt

# Build the Rust library first
cargo build --release --features c-bindings

# Create build directory
mkdir -p build
cd build

# Configure
cmake ..

# Build
cmake --build .

# Run example
./edgegpt_example
```

## Python Bindings

### Build Python wheel:
```bash
# Install maturin
pip install maturin

# Build and install in development mode
cd crates/edgegpt
maturin develop --release --features python

# Or build a wheel
maturin build --release --features python
```

### Test Python bindings:
```python
import edgegpt

# Create instance
edge = edgegpt.EdgeGPT(device="cpu")

# Encode text
embedding = edge.encode("Hello, world!")
print(f"Embedding dimension: {len(embedding)}")

# Compute similarity
sim = edge.similarity("I love ML", "Machine learning is great")
print(f"Similarity: {sim:.4f}")

# Rerank documents
query = "What is AI?"
docs = [
    "Artificial intelligence is...",
    "The weather is nice",
    "ML is a subset of AI"
]
ranked = edge.rerank(query, docs)
for idx, score in ranked:
    print(f"[{idx}] {docs[idx]}: {score:.4f}")
```
```

## Updated directory structure
```
crates/edgegpt/
├── BUILD.md              # NEW: Build instructions
├── CMakeLists.txt        # NEW: CMake build file
├── Cargo.toml
├── examples/
│   └── example.cpp       # NEW: C++ example
├── include/
│   ├── edgegpt.h
│   └── edgegpt.hpp
└── src/
    ├── cross_encoder_api.rs
    ├── edge_gpt.rs
    ├── ffi/
    │   ├── c.rs
    │   ├── mod.rs
    │   ├── python.rs
    │   └── types.rs
    ├── lib.rs
    ├── model_manager.rs
    ├── sentence_encoder_api.rs
    ├── test_encoder.py
    └── utils.rs