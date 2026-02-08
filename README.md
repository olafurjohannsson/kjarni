# Kjarni

A native AI inference engine written in Rust. Runs transformer models on CPU and GPU without Python, ONNX, or CUDA.

Currently ships as a [C# NuGet package](https://www.nuget.org/packages/Kjarni) for .NET developers.

## What It Does

```csharp
dotnet add package Kjarni
```

```csharp
using Kjarni;

// Sentiment analysis
using var classifier = new Classifier("distilbert-sentiment");
var result = classifier.Classify("I love this product!");
// POSITIVE (99.9%)

// Content moderation
using var toxicity = new Classifier("toxic-bert");
var toxic = toxicity.Classify("You are an idiot");
// toxic: 72%, insult: 24%

// Semantic search
using var embedder = new Embedder("minilm-l6-v2");
var query = embedder.Encode("how do I get my money back?");
// Matches "Refunds are processed within 5-7 business days" — zero keyword overlap

// Full RAG pipeline
using var indexer = new Indexer(model: "minilm-l6-v2");
indexer.Create("my_index", new[] { "docs/" });

using var searcher = new Searcher(model: "minilm-l6-v2");
var results = searcher.Search("my_index", "how do returns work?");
```

Models download automatically on first use. No API keys. No GPU required. Your data never leaves the machine.

## Architecture

Kjarni loads HuggingFace models directly from safetensors files using memory-mapped I/O. All inference runs natively — no Python runtime, no ONNX conversion step, no external dependencies.

**Shipping now:**

- Hand-tuned SIMD kernels — AVX2/FMA on x86, NEON on ARM
- GPU inference via WebGPU with custom WGSL compute shaders
- Zero-copy model loading from safetensors via mmap
- BF16 compute path for models that ship in bfloat16
- Hybrid search — BM25 keyword search combined with semantic vector search

**In progress:**

- Quantization — Q4_K, Q6_K, Q8_0 for reduced memory usage
- Speculative decoding for faster text generation
- Chat and summarization APIs for the NuGet package
- Additional language bindings (Go)

### Key implementation details

- **CPU inference** with hand-tuned SIMD kernels — AVX2/FMA on x86, NEON on ARM
- **GPU inference** via WebGPU with custom WGSL compute shaders for matmul, attention, normalization, and RoPE
- **Quantization support** — Q4_K, Q6_K, Q8_0 for reduced memory usage
- **BF16 compute path** for models that ship in bfloat16
- **Zero-copy model loading** from safetensors via mmap
- **Custom tokenizers** — WordPiece (BERT), BPE (GPT/Llama), with HuggingFace tokenizer.json support
- **Beam search and sampling** — temperature, top-k, top-p, min-p, repetition penalty
- **Hybrid search** — BM25 keyword search combined with semantic vector search

## Models

| Task | Model | Arch | Size |
|------|-------|------|------|
| Embeddings | `minilm-l6-v2` | BERT | 90MB |
| Embeddings | `mpnet-base-v2` | MPNet | 420MB |
| Embeddings | `distilbert-base` | DistilBERT | 260MB |
| Classification | `distilbert-sentiment` | DistilBERT | 268MB |
| Classification | `roberta-sentiment` | RoBERTa | 499MB |
| Classification | `bert-sentiment-multilingual` | BERT | 681MB |
| Classification | `distilroberta-emotion` | RoBERTa | 329MB |
| Classification | `toxic-bert` | BERT | 438MB |
| Reranking | `minilm-l6-v2-cross-encoder` | BERT | 90MB |

Additional architectures supported in the engine: Llama, Qwen2, Mistral, Phi-3, T5, BART, Whisper, NomicBERT. These will ship in future NuGet releases.

## Project Structure

```
crates/
├── kjarni/              # High-level API (Embedder, Classifier, Searcher, etc.)
├── kjarni-transformers/  # Core engine — models, kernels, GPU shaders
├── kjarni-ffi/          # C ABI + language bindings
│   └── bindings/
│       └── csharp/      # .NET NuGet package
├── kjarni-cli/          # Command-line interface
└── kjarni-examples/     # Rust examples
```

## Building from Source

```bash
# Build the CLI
cargo build --release -p kjarni-cli

# Build the shared library (.so / .dll / .dylib)
cargo build --release -p kjarni-ffi

# Run tests
cargo test
```

## CLI

Kjarni also ships as a command-line tool:

```bash
# Classify text
kjarni classify "I love this" --model distilbert-sentiment

# Generate embeddings
kjarni embed "Hello world" --model minilm-l6-v2

# Index and search documents
kjarni index create my_index docs/
kjarni search my_index "how do returns work?"

# Rerank results
echo -e "relevant doc\nirrelevant doc" | kjarni rerank "my query"
```

## Platform Support

| Platform | CPU | GPU |
|----------|-----|-----|
| Linux x64 | ✅ | ✅ (Vulkan) |
| Windows x64 | ✅ | ✅ (DX12/Vulkan) |
| macOS ARM64 | Planned | Planned (Metal) |
| macOS x64 | Planned | Planned |

## License

MIT