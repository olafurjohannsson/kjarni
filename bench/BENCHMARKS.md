# Benchmarks

Kjarni against the usual ways of running the same models: PyTorch (through `transformers` and `sentence-transformers`), ONNX Runtime, faster-whisper and llama.cpp. Same weights, same inputs, same machine, CPU only.

Reproduce with `bench/compare/run_encoders.py` and `bench/compare/run_rest.py`; `make_charts.py` and `make_report.py` turn the results into this page.

## Method

- **Machine:** Intel Core i7-13700 (8 performance and 8 efficiency cores, 24 threads), 64 GB, Linux. Idle: each run waits for a load average under 1.
- **Isolation:** every engine runs in its own fresh process, so load time and memory are its own.
- **Timing:** minimum of repeated calls after warm-up, the least noisy estimate of what each engine can do. Medians are in the raw JSON.
- **Threads:** each engine's default. Kjarni uses rayon's pool; PyTorch and ONNX Runtime use the physical core count. For the LLMs, both engines get the better of 8 and 16 threads.
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
  <img alt="Peak memory" src="charts/encoders-memory-light.svg">
</picture>

| Task | Engine | Batch of 64 | One text | Load | Peak memory | Max diff vs PyTorch |
|---|---|--:|--:|--:|--:|--:|
| Embed · all-MiniLM-L6-v2 | **Kjarni** | 217 ms (294/s) | 2.4 ms | 86 ms | 376 MB | 2e-07 |
| Embed · all-MiniLM-L6-v2 | PyTorch | 236 ms (271/s) | 5.5 ms | 3,517 ms | 710 MB | reference |
| Embed · all-MiniLM-L6-v2 | ONNX Runtime | 207 ms (310/s) | 1.4 ms | 531 ms | 416 MB | 2e-07 |
| Embed · all-mpnet-base-v2 | **Kjarni** | 1365 ms (47/s) | 25.8 ms | 228 ms | 1,164 MB | 2e-07 |
| Embed · all-mpnet-base-v2 | PyTorch | 1878 ms (34/s) | 29.8 ms | 2,485 ms | 1,173 MB | reference |
| Embed · all-mpnet-base-v2 | ONNX Runtime | 1261 ms (51/s) | 20.7 ms | 1,153 ms | 938 MB | 3e-07 |
| Rerank · ms-marco-MiniLM-L6 | **Kjarni** | 260 ms (247/s) | 9.2 ms | 114 ms | 409 MB | 4e-06 |
| Rerank · ms-marco-MiniLM-L6 | PyTorch | 301 ms (212/s) | 9.2 ms | 2,466 ms | 743 MB | reference |
| Rerank · ms-marco-MiniLM-L6 | ONNX Runtime | 271 ms (236/s) | 3.8 ms | 698 ms | 374 MB | 6e-06 |
| Classify · DistilBERT SST-2 | **Kjarni** | 162 ms (396/s) | 13.2 ms | 212 ms | 664 MB | 1e-07 |
| Classify · DistilBERT SST-2 | PyTorch | 218 ms (293/s) | 9.2 ms | 2,105 ms | 728 MB | reference |
| Classify · DistilBERT SST-2 | ONNX Runtime | 145 ms (443/s) | 10.7 ms | 825 ms | 439 MB | 1e-07 |

## Speech to text

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="charts/whisper-dark.svg">
  <img alt="Whisper speed" src="charts/whisper-light.svg">
</picture>

whisper-small, English, greedy decoding, CPU.

| Audio | Engine | Time | Faster than real time | Load | Peak memory |
|---|---|--:|--:|--:|--:|
| 11 s | **Kjarni** | 2.13 s | 5.2× | 680 ms | 2,905 MB |
| 11 s | faster-whisper | 2.26 s | 4.9× | 1,650 ms | 2,119 MB |
| 11 s | faster-whisper-int8 | 1.14 s | 9.7× | 1,139 ms | 825 MB |
| 11 s | openai-whisper | 1.75 s | 6.3× | 2,440 ms | 1,663 MB |
| 58 s | **Kjarni** | 5.66 s | 10.2× | 681 ms | 3,087 MB |
| 58 s | faster-whisper | 9.42 s | 6.1× | 1,477 ms | 2,120 MB |
| 58 s | faster-whisper-int8 | 4.39 s | 13.1× | 1,237 ms | 829 MB |
| 58 s | openai-whisper | 8.15 s | 7.1× | 2,044 ms | 1,665 MB |

## Summarization and translation

distilbart-cnn-12-6 and flan-t5-base, each with its checkpoint's own generation settings.

| Task | Engine | Time | Load | Peak memory | Output |
|---|---|--:|--:|--:|---|
| summarize | **Kjarni** | 3,511 ms | 893 ms | 3,464 MB | The library will include a children's wing, a maker space with 3D printers, and a cafe … |
| summarize | PyTorch | 2,135 ms | 2,423 ms | 1,728 MB | The library will include a children's wing, a maker space with 3D printers, and a cafe … |
| translate | **Kjarni** | 1,090 ms | 679 ms | 2,344 MB | Das Haus ist klein, aber das Garten hinter es ist groß und voll von alten Bäumen. |
| translate | PyTorch | 700 ms | 2,406 ms | 1,374 MB | Das Haus ist klein, aber das Garten hinter sie ist groß und voller alten Apple-Trees. |

## Image search (CLIP)

ViT-B/32 with the LAION-2B weights, indexing 64 photos and running one text query.

| Engine | Images per second | Query | Load | Peak memory |
|---|--:|--:|--:|--:|
| **Kjarni** | 54.4 | 26.8 ms | 329 ms | 2,816 MB |
| open_clip | 49.8 | 17.6 ms | 3,626 ms | 1,667 MB |

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
| qwen2.5-0.5b-instruct | **Kjarni** | 472 tok/s | 32.1 tok/s |
| qwen2.5-0.5b-instruct | llama.cpp | 607 tok/s | 41.8 tok/s |
| llama3.2-1b-instruct | **Kjarni** | 238 tok/s | 18.0 tok/s |
| llama3.2-1b-instruct | llama.cpp | 426 tok/s | 22.3 tok/s |

