<p align="center">
  <a href="https://kjarni.ai"><img src="https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/assets/kjarni-banner.png" alt="kjarni: local AI inference in a single native library" width="600"></a>
</p>

# Kjarni

**Run transformer models inside your program, in any language. Same output as
PyTorch, no Python, no server.**

Kjarni is a transformer inference library written in Rust: embeddings,
classification, reranking, search, chat, speech to text, summarization and
translation, on CPU or GPU inside your process. This crate is its Rust API; the
same engine ships for Python, C#, Go, C++ and the browser.

```toml
[dependencies]
kjarni = "0.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use kjarni::Embedder;

let embedder = Embedder::new("minilm-l6-v2").await?;
println!("{}", embedder.similarity("doctor", "physician").await?);  // 0.8598
println!("{}", embedder.similarity("doctor", "banana").await?);     // 0.3379
```

Models load from HuggingFace safetensors and GGUF directly. The first call downloads
and caches under `~/.cache/kjarni`; nothing leaves the machine.

## What it implements

BERT-style encoders, cross-encoders, Llama-family decoders (Llama, Qwen2, Mistral,
Phi-3), T5, BART and Whisper. Quantised weights in Q4_K, Q6_K, Q8_0 and Q8_K, on CPU
with AVX2/FMA and NEON kernels or on GPU through WebGPU, which means no CUDA toolkit.

For an arbitrary research model, convert it and use ONNX Runtime. For embeddings,
classification, reranking or chat inside a program that has to ship somewhere, this
is one dependency with no toolchain.

## The workspace

| Crate | What it is |
|---|---|
| `kjarni` | The public API. Start here. |
| `kjarni-transformers` | The engine: kernels, model graphs, CPU and GPU backends |
| `kjarni-models` | Model definitions and the registry |
| `kjarni-search` | Vector search primitives |
| `kjarni-rag` | Retrieval-augmented generation |
| `kjarni-cli` | The `kjarni` command line tool |
| `kjarni-ffi` | The C ABI every other language binding is built on |

## How fast it is

An i7-13700 and an RTX A2000 12 GB, each engine in its own process. On this NVIDIA card, engines using CUDA are well ahead of Kjarni's WebGPU; what WebGPU gives you is the same code on NVIDIA, AMD, Intel and Apple hardware with nothing to install.

![Embeddings per second on the CPU](https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/bench/charts/matrix-embeddings-cpu-light.png)

![Embeddings per second on the GPU](https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/bench/charts/matrix-embeddings-gpu-light.png)

![Speech to text speed on CPU and GPU](https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/bench/charts/matrix-whisper-light.png)

![Text generation on the CPU](https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/bench/charts/matrix-generation-cpu-light.png)

![Text generation on the GPU](https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/bench/charts/matrix-generation-gpu-light.png)

These are the Rust engine's own numbers measured through its bindings; the Rust API and the C ABI take the same time, 3.43 and 3.44 ms for one short MiniLM embedding. Method and every number: [BENCHMARKS.md](https://github.com/olafurjohannsson/kjarni/blob/main/bench/BENCHMARKS.md).

## The same engine elsewhere

One engine, one set of kernels, the same numbers from every language.

- [kjarni.ai](https://kjarni.ai) - documentation and worked examples
- [GitHub](https://github.com/olafurjohannsson/kjarni) - source, issues, releases
- [NuGet](https://www.nuget.org/packages/Kjarni) - from C#
- [PyPI](https://pypi.org/project/kjarni/) - from Python
- [npm](https://www.npmjs.com/package/kjarni-wasm) - in the browser, via WebAssembly
- [Go module](https://pkg.go.dev/github.com/olafurjohannsson/kjarni-go) - from Go
- [Why I Built a Native ML Inference Engine in Rust](https://kjarni.ai/blog/nativeinference/) - the reasoning
- [Semantic Search in C++](https://kjarni.ai/blog/cppinference/) - the C ABI used directly
- [ML from the Command Line](https://kjarni.ai/blog/cli/) - the same models as a UNIX tool

## License

MIT or Apache-2.0.
