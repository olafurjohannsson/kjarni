# Changelog

## 0.2.6

### Faster batches for reranking, classification and embeddings

Measured on a batch of 64 texts against PyTorch and ONNX Runtime; the full method
and results are in [bench/BENCHMARKS.md](bench/BENCHMARKS.md).

- **Reranking and classification group inputs by length.** Both padded a whole batch
  to its longest member and used the encoder's allocating forward pass. They now
  use the same length-bucketed, buffered path as embeddings: reranking 64 documents
  went from 561 ms to 246 ms, faster than PyTorch's 273 ms.
- **Batch classification in the C API.** `kjarni_classifier_classify_batch` runs many
  texts through the model together, and every binding uses it: `classify_batch` in
  Python, `ClassifyBatch` in C# and Go, `classify(span)` in C++. 64 texts went from
  681 ms, one at a time, to 157 ms, against PyTorch's 223 ms.
- **Encoder buffers are reused across calls.** Each forward pass allocated and zeroed
  a couple of hundred megabytes of scratch space, 40 ms of a 260 ms MiniLM batch.
  Whisper and the BART and T5 encoders reuse them too.
- **faer 0.24.** Matrix multiplies on encoder shapes are 1.1 to 1.7 times faster.
- The embedding layer norm and the position and token-type additions run in
  parallel instead of on one thread.

### Prompts are processed more than ten times faster

A 500-token prompt took Qwen2.5 0.5B ten seconds before the first token; it now
takes about one. Prompt processing on the same GGUF files went from 41 to 615
tokens a second (llama.cpp: 611) on Qwen2.5 0.5B and from 20 to 282 on Llama 3.2
1B (llama.cpp: 424).

- **Prompts use the streaming attention path.** It was reserved for an all-ones
  mask, but Llama-family models pass a causal mask for a prompt, which asks for
  exactly what that path does. Every prompt took the path that materialises the
  full score tensor instead: 330 ms per layer on a 500-token prompt, against 14.
- **Quantized and BF16 weights are multiplied as tiles for prompts.** The
  per-row kernels, right for decoding one token, read and dequantized every
  weight once per prompt token. From 16 rows up, tiles are dequantized once and
  multiplied with faer.
- **Only the last position's logits are computed for a prompt**, the only ones
  used: on a 152k vocabulary, 70 GFLOP saved per 500-token prompt.

### Models hold one copy of their weights

Whisper-small went from 2.8 GB to 1.1 GB once loaded, distilbart-cnn from 3.5 GB
to 1.3 GB and CLIP from 1.3 GB to 0.65 GB: about the size of their weight files,
and less than the same models take in PyTorch. Three things held extra copies:

- **The mmap cache never let go.** It held every weight file mapped for the life
  of the process, after its weights had been copied out and even after the model
  was dropped. It holds weak references now.
- **`StdFeedForward` stored its matrices twice**, in public fields beside the
  layers that use them. In Whisper and BART the feed-forward is most of the model.
- **The token embedding table was loaded three times** for encoder-decoder
  models, by the encoder, the decoder and the pipeline. `ModelWeights` now hands
  out one shared copy (`get_shared_array2`).
- **Loading peaked at twice the model.** The mapped file stayed resident while
  its weights were copied out of it. Each tensor's pages are now released as soon
  as it is copied, so Whisper-small peaks at 1.28 GB instead of 2.97 GB.

A test loads Whisper-small and distilbart-cnn and checks each holds under 1.3
times its weight file once loaded, and peaks under 1.5 times while loading; they
are at 1.03 and 1.05.

### Whisper keeps every sentence on long audio

Audio was cut into fixed 30-second windows, wherever the 30 seconds fell. Whisper
drops an unfinished phrase at the end of a window and guesses at the fragment that
opens the next, so a sentence spanning a cut went missing. A cut could also send a
window into a loop to the 448-token limit: a 54-second recording took 30 seconds.

- **Windows are cut at pauses**: the latest clear pause in a window's last five
  seconds.
- **Whisper's timestamp rules** from OpenAI's decoder are applied, so segments carry
  real start and end times instead of one segment per 30-second window.
- **Text invented in a window's padding is dropped.** Past the end of the speech,
  Whisper carries on into the silence, continuing a phrase it expects. A segment
  that starts there, and `[BLANK_AUDIO]`, are discarded, and no segment opens in the
  window's last half-second.

Transcribing long audio takes about a quarter longer, from the timestamp tokens.

### Quantized chat in every language, and Qwen from GGUF

- **`quantized`** (C: `KjarniChatConfig.quantized`; `quantized=True` in Python and
  C#, `WithQuantized` in Go, `.quantized` in C++) loads a model's 4-bit GGUF weights:
  a fraction of the memory, and several times faster to generate on a CPU. Only the
  CLI could reach them before.
- **Qwen GGUF files load and generate correctly.** Three bugs stood in the way, each
  enough on its own: Q5_0 tensors, which Q4_K_M files use wherever a row is not a
  multiple of 256, were rejected; the Q/K row permutation llama.cpp applies only to
  Llama files was undone for every architecture; and the attention biases were never
  mapped to their GGUF names, so Qwen ran without them. Q5_0 and Q4_0 are now
  converted to Q8_0 at load, which is exact.
- `KjarniChatConfig.model_path` was ignored. It is now rejected with a message, until
  loading from a path is supported.

### Fixes

- `kjarni_embedder_encode_batch` always mean-pooled and normalised, ignoring the
  embedder's `normalize` setting, so a batch disagreed with encoding the same texts
  one at a time. It now uses the same path.
- `Classifier::classify_batch` applied sigmoid to probabilities instead of logits in
  multi-label mode, scored custom labels as zero, and dropped any text with nothing
  above the threshold, shifting every later result onto the wrong text. It now does
  exactly what classifying each text alone does.
- Four C API results freed their arrays with the wrong size: class results and string
  arrays as a single element, and collections whose capacity could exceed their
  length. All are boxed slices of exactly their length now.
- Releases publish `SHA256SUMS` and a signed build-provenance attestation for every
  shipped library (`gh attestation verify`).

## 0.2.5

### Every capability in every language

Speech to text, summarization, translation and image search were in the engine and
on the command line but not in the C ABI, so no binding could reach them. They are
now `kjarni_transcriber_*`, `kjarni_summarizer_*`, `kjarni_translator_*` and
`kjarni_image_index_*`, and every binding wraps all of them:

| Added | C++ | C# | Go | Python |
| ----- | :-: | :-: | :-: | :----: |
| Transcriber, Summarizer, Translator, ImageIndex | ✓ | ✓ | ✓ | ✓ |
| Indexer, Searcher | ✓ | | | |
| Chat (with streaming, history, conversations) | | | | ✓ |
| Chat with caller-held history | | | ✓ | |
| Indexer `Add`, `Recreate`, extensions, excludes, chunking | | | ✓ | |
| Search options and result metadata | | | ✓ | |

The same input now gives byte-identical output from C++, C#, Go and Python for
transcription, summaries, translation, image and hybrid search scores, and greedy
chat.

`image-io` is now a default feature of `kjarni-ffi`, so the published libraries
include image search; `--no-default-features` leaves it out, and each binding
reports that rather than failing to load. `kjarni_image_index_new` with a null model
directory now uses clip-vit-base-32, downloaded on first use.

### The Go binding leaked every result

C functions that take a struct by value, such as
`kjarni_float_array_free(KjarniFloatArray)`, were called through purego with a
pointer to the struct. That is the Windows x64 convention; on Linux the struct
travels in registers, so the free read the Go pointer as its data field and an
unrelated register as its length, found zero, and did nothing. Every embedding,
classification, rerank and search result leaked: about 28MB per 300 batches of 64
embeddings. A new test measures RSS across that loop.

### The Go binding panicked after about two thousand streamed replies

Each `Stream` call created a purego callback. purego never frees one and allows about
two thousand per process, so a long-running chat server panicked. All streams now
share one callback and are told apart by `user_data`.

### Smaller fixes

- Go: `Indexer.Create` with no inputs indexed out of range and panicked; it now
  returns an error. `Searcher.Search` requested the reranker on every query whether
  or not one was configured; it now uses the searcher's default.
- Go: `goString` copied byte by byte through two loops; it is one copy now.
- Python: the in-repo library path pointed at `crates/target/release`, one directory
  short, so a development checkout silently loaded whatever `libkjarni_ffi` was on
  the system path instead.
- Python: embeddings were copied out of the engine one float at a time through the
  interpreter; they are sliced in C now.
- Python now has a test suite, run in CI. The C++ job downloads the larger models
  and runs every capability; the C#, Go and Python jobs set `KJARNI_SKIP_HEAVY` and
  skip those, rather than each caching its own 3.5GB.

The browser build is unchanged: Whisper, the seq2seq models and CLIP are native only.

## 0.2.4

### `kjarni mcp`: Kjarni as tools for AI agents

`kjarni mcp` serves Kjarni to any MCP client (Claude Code, Cursor, Claude Desktop,
VS Code) as tools that run locally: `search_folder` (semantic and hybrid search over
a folder of documents or code, indexed on first use and kept up to date),
`transcribe`, `classify`, `rerank` and `similarity`. Models load once per session,
and indexes live in Kjarni's cache, never in the folders searched.

```bash
claude mcp add kjarni -- kjarni mcp
```

### Whisper is four times faster and detects the language

Three minutes of English went from 64 s to 16 s on CPU.

- **The mel spectrogram** computed a naive DFT per bin. It is now two matrix
  multiplies against cos/sin tables built once: 1.5 s to 15 ms per chunk, and 100
  times closer to HuggingFace's features (worst difference 1.7e-3 to 1.7e-5).
- **The convolutional front end** is im2col and a matmul instead of scalar loops,
  with Whisper's exact erf GELU in place of the tanh approximation. Encoder output
  against HuggingFace: worst difference 0.27 to 0.0025.
- **The encoder** runs the buffered path, about three times faster on 1,500
  positions.
- **Single-query attention** gets one task per head and reads the cache by rows,
  with bit-identical results: cross-attention 10.9 s to 4.9 s on three minutes.
- **Up to four chunks decode together**, sharing each read of the weights. The
  output is identical and decoding takes half the time.
- **The language is detected** from the first chunk. Auto-detect was a stub that
  always answered English.
- A greedy decode stuck repeating a phrase ends its chunk after four repeats.

### Faster CPU encoders, and multilingual-e5-small

- Embedding batches are grouped by token length, so each text is padded to its
  neighbours rather than to the longest in the call.
- Residual add and layer norm are fused and run in parallel across rows. GELU, the
  head transposes and the fused QKV split run in parallel too.
- Per 200-word chunk on a 24-thread i7-13700: multilingual-e5-small 72 to 41 ms,
  mpnet 272 to 148 ms, bge-m3 539 to 352 ms. Vectors match sentence-transformers to
  2e-7.
- **multilingual-e5-small** (intfloat, MIT, about 100 languages) joins the registry.
  Texts take a `query: ` or `passage: ` prefix.
- CLIP: `embed_image_tensors` embeds a batch of images about four times faster per
  image than one at a time, and the patch projection is one matmul (60 to 2 ms per
  image).

### Batched decoding masked the wrong tokens

The decoder multiplied the causal mask into the padding mask and passed the result
as `[batch, key]`. When the batch size equalled the prompt length, each sequence was
masked by a causal row, and the first saw only its first token. Self-attention
already applies the causal mask, so only a real `[batch, total_len]` padding mask
is passed now.

### Smaller fixes

- The indexer honours exclusions: a pattern such as `node_modules`,
  `node_modules/**` or `*.min.js` matches any component of a path, and an excluded
  or hidden directory is never entered, rather than walked and filtered file by
  file.
- `LinearLayer::matmul` with a few rows ran on one thread; it uses the same kernels
  as `matmul_noalloc` now. `matmul_noalloc` asserts its output shape, since its
  kernels write through raw pointers and a wrong-sized buffer was silent memory
  corruption.
- A Qt example, `crates/kjarni-ffi/examples/qt`, searches help articles from QML.

### Windows no longer needs the Visual C++ Redistributable

`kjarni_ffi.dll` and `kjarni.exe` imported `VCRUNTIME140.dll`, the Visual C++
Redistributable. Most machines have it; a clean install or a minimal VM does not, and
there the failure did not point at the cause: .NET reported
`DllNotFoundException: Unable to load DLL 'kjarni_ffi'` with the file sitting next to
the executable, because Windows reports a missing dependency of a DLL as the DLL itself
failing to load.

The C runtime is now linked statically for `x86_64-pc-windows-msvc`
(`.cargo/config.toml`). The binaries import only libraries that ship with Windows,
including the Universal CRT, which is part of Windows 10 and later. Release CI checks
the import table and fails if `VCRUNTIME` comes back.

### Indexing progress callbacks work under IL2CPP

`Indexer.Create` and `Indexer.Add` passed `onProgress` to native code as a lambda.
IL2CPP, Unity's ahead-of-time backend, can only hand native code a delegate to a static
method, so passing `onProgress` there threw. Progress now goes through one static
callback marked `[MonoPInvokeCallback]`, with the handler carried in `user_data`, the
same way chat streaming already worked. Nothing changes for callers.

### The netstandard2.1 package has no dependencies

`Kjarni` for netstandard2.1 depended on `System.Text.Json` 8.0.5, used in exactly one
place: reading the metadata attached to each search result. Unity cannot restore NuGet
dependencies, so that one call meant hand-installing `System.Text.Json` and its chain
(`System.Memory`, `System.Buffers`, `Microsoft.Bcl.AsyncInterfaces` and the rest)
before the package would load.

The metadata is always a flat object of string keys and string values, since the
native side serialises a `HashMap<String, String>`, so a small internal parser now
reads it instead. Both targets have no package dependencies, and using Kjarni from
Unity is two files: `Kjarni.dll` and the native library. `SearchResult.Metadata` is
unchanged.

## 0.2.3

### A panic in the engine no longer kills the host process

The workspace release profile set `panic = "abort"`. Cargo has no per-package panic
strategy, so that applied to `kjarni-ffi` as well — the cdylib that NuGet, the Go
module and the C header all load *into someone else's process*. Any panic reachable
through any of the 80 `extern "C"` entry points terminated that process: a .NET
service, a Go binary, a Qt application. No error code, no message, nothing the caller
could have handled. `kjarni-ffi/src` had 38 `unwrap`/`expect`/`panic!` sites of its
own, and everything the engine can panic on sits behind them.

`panic = "abort"` is gone from the profile, and the 66 fallible entry points now run
their bodies inside a `catch_unwind` barrier (`kjarni-ffi/src/panic.rs`). A panic
becomes the new `KJARNI_ERROR_CODE_PANIC` (11) with the entry point's name and the
panic message on `kjarni_last_error_message()`, which is what every other failure in
the C ABI already looks like. The remaining 14 entry points return compile-time
constants and are left unguarded, which the module documents; guarding
`kjarni_last_error_message` in particular would be circular.

This is a backstop, not a licence to panic — a panic reaching it is still a bug. It
only decides whether the caller gets to hear about it.

Building `kjarni-ffi` under `panic = "abort"` is now a compile error rather than a
silently defeated barrier, since `catch_unwind` cannot intercept an abort. That check
is what would have caught the original setting.

`KJARNI_ERROR_CODE_PANIC = 11` is additive — no existing code is renumbered — and is
mirrored in the C header, the C# `KjarniErrorCode`, Go's `ErrPanic`, and Python's
`PANIC`. The cost of the profile change is a larger binary and landing pads the
optimiser must keep.

### An error message with an interior nul is no longer discarded

`set_last_error` built its `CString` with `CString::new(msg).ok()`, which yields
`None` on an interior nul and cleared the stored message rather than replacing it, so
`kjarni_last_error_message()` returned NULL for an error that had just been reported.
It now truncates at the first nul and keeps the leading, most specific part. For an ordinary error that lost a diagnostic; for a caught panic it
would have lost the only evidence the panic happened.

## 0.1.4

### Encoders now truncate where sentence-transformers does

Encoder models read `max_seq_length` from `sentence_bert_config.json` when the model
ships one, falling back to `max_position_embeddings` as before. For
`all-MiniLM-L6-v2` that changes the default from 512 to **256**, and for
`all-mpnet-base-v2` from 512 to **384**.

This is a behaviour change for long inputs. Anything under the limit is unaffected,
but text above it now produces the same embedding the reference Python
implementation produces, where previously it did not. The default RAG chunk of 1000
characters measures at a median of 286 tokens on ordinary prose, so most indexed
chunks were in the affected range. **Rebuild indexes built with an earlier version**
if you index documents longer than 256 tokens.

Pass `ModelLoadConfig::max_sequence_length` to override.

### Asking for more positions than a model has is now an error

Requesting a sequence length above the model's position-embedding table used to
succeed. Tokens past the table received no position embedding at all and the encoder
returned a normal-looking vector that was quietly wrong. This is now refused at load
with a message naming the request and the ceiling. A length derived from a config
file is clamped with a warning instead, so a malformed config cannot make a model
unloadable.

### `.kjq` carries the tuned sequence length

`scripts/quantize_model.py` folds `max_seq_length` into the config it packs, and the
byte loader reads it back. Without this the browser had no way to learn the number,
since reading `sentence_bert_config.json` needs a filesystem, and WASM would have
truncated at 512 while every native binding truncated at 256. **Regenerate `.kjq`
files** built with an earlier version.

### Vector search is faster and its ordering is now defined

`VectorStore::search` keeps a bounded top-k rather than sorting every candidate and
truncating, hoists the query norm out of the inner loop, and runs in parallel above
2048 vectors. On 384-dimensional vectors, best of five on a 24-core machine:

| Vectors | Before | After |
| ------- | ------ | ----- |
| 10,000 | 2.6 ms | 0.2 ms |
| 100,000 | 24.9 ms | 4.9 ms |
| 1,000,000 | 255 ms | 56 ms |

Results are unchanged. Ties now break on ascending index, which makes the ordering
total: without that the parallel scan could return a different permutation of
equally-scoring documents between runs.

WASM keeps the serial path, since rayon there needs a threaded build.

### A TypeScript package for the browser, published to npm

`npm i kjarni-wasm` gets embeddings, classification, reranking and chat running
in a Web Worker, so loading a model or generating a reply never freezes the page.
Every method returns a promise, and chat is an async iterator.

```ts
const kjarni = await Kjarni.load({ encoder: "/models/minilm-l6-v2-q8.kjq" });
for await (const piece of kjarni.chat("Explain RAG")) out.textContent += piece;
```

The raw wasm-bindgen bundle is still in `kjarni-wasm.tar.gz` for anyone who
would rather drive it directly.

### Browser generation actually streams

`WasmChat` ran the whole decode loop to completion and only then drained the
channel the engine had been streaming into, so callers could show nothing until
the reply was finished. Polling the loop together with its own drain fixes that
on wasm's single thread, and the channel goes back to being a small buffer that
applies real back-pressure rather than one sized to hold an entire response.

`generate_stream` exposes it to JavaScript.

### The WASM bundle is now tested in a browser

Everything testing `kjarni-wasm` ran the Rust natively, which covers the engine
and misses the artifact entirely: a broken import, a bundle compiled from stale
sources, a binding that only fails through wasm-bindgen's glue. A new CI job
loads the built bundle in Chromium and checks it encodes text correctly.

This was not hypothetical. The live demo served a bundle containing neither
classification nor chat for days while every test stayed green.

### C++ is built and tested in CI

The C ABI header had never been compiled by CI, and shipped in every release archive
unverified. A new job regenerates `kjarni.h` with cbindgen and diffs it against the
committed copy, compiles it as C11 under `-Wall -Wextra -Werror`, and runs the C++
wrapper's 61 checks under AddressSanitizer and UndefinedBehaviorSanitizer.

### Other

- `kjarni-wasm`'s 18 binding tests now run in CI; they were excluded when the crate
  could not build natively.
- The `.kjq` test fixtures are exported in CI from cached weights. They previously
  looked for a sibling checkout that CI does not have, so they skipped and reported
  green while testing nothing.
- `quantize_model.py` tolerates an unreadable `sentence_bert_config.json` rather than
  aborting, which is what a truncated or redirected download leaves behind.
