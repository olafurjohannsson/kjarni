# Kjarni in the browser

`index.html` covers every capability the WASM bindings expose, in the same order as
the [C++](../../kjarni-ffi/examples/cpp) and C# examples: embeddings, reranking,
classification, search and chat.

## Running it

Build the bundle, then serve the crate directory — ES modules and WASM will not load
over `file://`:

```bash
cd crates/kjarni-wasm
RUSTFLAGS='-C target-feature=+simd128' \
  wasm-pack build --release --target web -- --no-default-features

python3 -m http.server 8000
# open http://localhost:8000/examples/
```

The page expects `../pkg/kjarni_wasm.js` from that build, and reads `.kjq` models from
the sibling website checkout. Point `MODELS` at wherever yours live.

## Models

| Capability | Model | Size |
| ---------- | ----- | ---- |
| Embeddings, search | all-MiniLM-L6-v2 | 23 MB |
| Reranking | ms-marco-MiniLM-L-6-v2 | 24 MB |
| Classification | distilbert-sentiment | 68 MB |
| Chat | qwen2.5-0.5b-instruct | 501 MB |

Each loads on demand rather than up front, so opening the page costs nothing until
you click. See [`../scripts/README.md`](../scripts/README.md) for creating `.kjq`
files of your own.

## The one thing to get right

**Inference blocks the thread it runs on.** Encoding a handful of sentences takes
milliseconds and is fine on the main thread. Chat takes seconds and is not — the tab
freezes for the whole generation.

This example calls everything directly so the code stays readable, with a
`yieldToPaint()` before each call so the pending state renders first. That is a
mitigation, not a fix. **A real application runs the model in a Web Worker**, posts
prompts to it, and receives tokens back as messages. Nothing in the bindings prevents
this; they are ordinary synchronous calls that work the same inside a worker.

## API

```js
import init, { WasmModel, WasmReranker, WasmClassifier, WasmChat,
                WasmIndexBuilder, WasmSearch } from "../pkg/kjarni_wasm.js";
await init();

const model = WasmModel.from_quantized(bytes);
const flat  = model.encode(["first", "second"], true);   // one flat Float32Array

const reranker = WasmReranker.load(bytes);
reranker.rerank(query, documents, limit);                // [{ index, score, text }]
reranker.score(query, document);                         // a raw logit

const classifier = WasmClassifier.load(bytes);
classifier.classify(text);                               // [{ label, score, index }]
classifier.labels();

const chat = WasmChat.load(bytes, "qwen2.5-0.5b-instruct");
chat.generate(prompt, maxNewTokens, temperature);        // blocks until complete
chat.context_size();
```

`encode` returns every vector concatenated into one array; slice by dimension:

```js
const dim = flat.length / texts.length;
const vec = (i) => flat.subarray(i * dim, (i + 1) * dim);
```

Search builds an index in the page, then queries it three ways:

```js
const builder = WasmIndexBuilder.new(modelBytes);
docs.forEach((text, i) => builder.add_chunk(text, Array.from(vec(i)), "source", i));

const searcher = WasmSearch.load(modelBytes, builder.finish());
searcher.search_keywords(query, 5);   // BM25
searcher.search_semantic(query, 5);   // embeddings
searcher.search(query, 5);            // hybrid, the default
```

## Tests

The bindings are covered by `cargo test -p kjarni-wasm`, which runs natively rather
than in a browser — the crate builds for the host precisely so its behaviour can be
asserted without a headless browser in the loop.
