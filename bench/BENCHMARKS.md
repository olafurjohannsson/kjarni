# Benchmarks

Kjarni against the usual ways of running the same models: PyTorch (through `transformers` and `sentence-transformers`), ONNX Runtime, faster-whisper and llama.cpp. Same weights, same inputs, same machine, CPU only.

Reproduce with `bench/compare/run_encoders.py` and `bench/compare/run_rest.py`; `make_charts.py` and `make_report.py` turn the results into this page.

## Method

- **Machine:** Intel Core i7-13700 (8 performance and 8 efficiency cores, 24 threads), 64 GB, Linux. Idle: each run waits for a load average under 1.
- **Isolation:** every engine runs in its own fresh process, so load time and memory are its own.
- **Memory:** *memory after* is what the process holds once the work is done; *peak* is the most it held at any point, usually while loading.
- **Timing:** minimum of repeated calls after warm-up, the least noisy estimate of what each engine can do. Medians are in the raw JSON.
- **Threads:** each engine's default. Kjarni uses rayon's pool; PyTorch and ONNX Runtime use the physical core count. For the LLMs, both engines get the better of 8 and 16 threads for each of prompt processing and generation.
- **Agreement:** encoder outputs are compared with PyTorch's; the largest difference is listed with each row. Generated text is recorded so the engines can be checked to have done the same work.
- **Versions:** torch 2.13.0+cpu, transformers 5.16.1, sentence-transformers 6.0.1, onnxruntime 1.31.0, faster-whisper 1.2.1, ctranslate2 4.8.4, open_clip 3.3.0, llama.cpp b11901.

## Embeddings, reranking and classification

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/encoders-throughput-dark.svg">
  <img alt="Batch throughput" src="charts/encoders-throughput-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/encoders-load-dark.svg">
  <img alt="Load time" src="charts/encoders-load-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/encoders-memory-dark.svg">
  <img alt="Memory after" src="charts/encoders-memory-light.svg">
</picture>

| Task | Engine | Batch of 64 | One text | Load | Memory after | Peak memory | Max diff vs PyTorch |
|---|---|--:|--:|--:|--:|--:|--:|
| Embed · all-MiniLM-L6-v2 | **Kjarni** | 205 ms (312/s) | 2.3 ms | 94 ms | 271 MB | 290 MB | 2e-07 |
| Embed · all-MiniLM-L6-v2 | PyTorch | 217 ms (295/s) | 3.5 ms | 2,434 ms | 649 MB | 738 MB | reference |
| Embed · all-MiniLM-L6-v2 | ONNX Runtime | 182 ms (351/s) | 1.5 ms | 674 ms | 412 MB | 409 MB | 2e-07 |
| Embed · all-mpnet-base-v2 | **Kjarni** | 1274 ms (50/s) | 24.2 ms | 229 ms | 708 MB | 749 MB | 2e-07 |
| Embed · all-mpnet-base-v2 | PyTorch | 1702 ms (38/s) | 29.5 ms | 2,449 ms | 1,036 MB | 1,178 MB | reference |
| Embed · all-mpnet-base-v2 | ONNX Runtime | 1087 ms (59/s) | 16.2 ms | 1,168 ms | 946 MB | 944 MB | 3e-07 |
| Rerank · ms-marco-MiniLM-L6 | **Kjarni** | 246 ms (261/s) | 9.1 ms | 113 ms | 308 MB | 323 MB | 4e-06 |
| Rerank · ms-marco-MiniLM-L6 | PyTorch | 273 ms (234/s) | 7.1 ms | 2,428 ms | 616 MB | 762 MB | reference |
| Rerank · ms-marco-MiniLM-L6 | ONNX Runtime | 247 ms (259/s) | 3.8 ms | 672 ms | 377 MB | 375 MB | 6e-06 |
| Classify · DistilBERT SST-2 | **Kjarni** | 157 ms (408/s) | 12.3 ms | 217 ms | 399 MB | 502 MB | 1e-07 |
| Classify · DistilBERT SST-2 | PyTorch | 223 ms (287/s) | 12.0 ms | 2,141 ms | 686 MB | 745 MB | reference |
| Classify · DistilBERT SST-2 | ONNX Runtime | 153 ms (418/s) | 8.9 ms | 872 ms | 412 MB | 439 MB | 1e-07 |

## Speech to text

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/whisper-dark.svg">
  <img alt="Whisper speed" src="charts/whisper-light.svg">
</picture>

whisper-small, English, greedy decoding, CPU.

| Audio | Engine | Time | Faster than real time | Load | Memory after | Peak memory |
|---|---|--:|--:|--:|--:|--:|
| 11 s | **Kjarni** | 1.99 s | 5.5× | 441 ms | 1,082 MB | 1,246 MB |
| 11 s | faster-whisper | 2.08 s | 5.3× | 1,482 ms | 1,676 MB | 2,120 MB |
| 11 s | faster-whisper-int8 | 1.12 s | 9.9× | 1,109 ms | 724 MB | 825 MB |
| 11 s | openai-whisper | 1.57 s | 7.0× | 2,083 ms | 1,393 MB | 1,665 MB |
| 58 s | **Kjarni** | 4.99 s | 11.5× | 460 ms | 1,102 MB | 1,430 MB |
| 58 s | faster-whisper | 8.28 s | 6.9× | 1,657 ms | 1,667 MB | 2,120 MB |
| 58 s | faster-whisper-int8 | 4.20 s | 13.7× | 1,150 ms | 728 MB | 829 MB |
| 58 s | openai-whisper | 6.98 s | 8.2× | 2,090 ms | 1,429 MB | 1,665 MB |

## Summarization and translation

distilbart-cnn-12-6 and flan-t5-base, each with its checkpoint's own generation settings.

| Task | Engine | Time | Load | Memory after | Peak memory | Output |
|---|---|--:|--:|--:|--:|---|
| summarize | **Kjarni** | 2,787 ms | 569 ms | 1,280 MB | 1,329 MB | The library will include a children's wing, a maker space with 3D printers, and a cafe … |
| summarize | PyTorch | 2,446 ms | 2,426 ms | 1,697 MB | 1,745 MB | The library will include a children's wing, a maker space with 3D printers, and a cafe … |
| translate | **Kjarni** | 898 ms | 565 ms | 1,049 MB | 1,247 MB | Das Haus ist klein, aber das Garten hinter es ist groß und voll von alten Bäumen. |
| translate | PyTorch | 646 ms | 2,378 ms | 1,391 MB | 1,388 MB | Das Haus ist klein, aber das Garten hinter sie ist groß und voller alten Apple-Trees. |

## Image search (CLIP)

ViT-B/32 with the LAION-2B weights, indexing 64 photos and running one text query.

| Engine | Images per second | Query | Load | Memory after | Peak memory |
|---|--:|--:|--:|--:|--:|
| **Kjarni** | 57.2 | 37.8 ms | 304 ms | 775 MB | 865 MB |
| open_clip | 52.2 | 16.9 ms | 3,460 ms | 1,176 MB | 1,679 MB |

## Text generation

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-generation-dark.svg">
  <img alt="Generation speed" src="charts/decode-generation-light.svg">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/decode-prompt-dark.svg">
  <img alt="Prompt processing" src="charts/decode-prompt-light.svg">
</picture>

The same Q4_K_M GGUF file in both engines. llama.cpp is timed with its own `llama-bench` (pp512, tg128); Kjarni through its Python binding, from a 512-token prompt and 128 greedy tokens.

| Model | Engine | Prompt processing | Generation |
|---|---|--:|--:|
| qwen2.5-0.5b-instruct | **Kjarni** | 615 tok/s | 36.1 tok/s |
| qwen2.5-0.5b-instruct | llama.cpp | 611 tok/s | 45.9 tok/s |
| llama3.2-1b-instruct | **Kjarni** | 282 tok/s | 19.0 tok/s |
| llama3.2-1b-instruct | llama.cpp | 424 tok/s | 27.2 tok/s |

