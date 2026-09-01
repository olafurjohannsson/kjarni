# kjarni-wasm

Local AI inference in the browser. Embeddings, classification, reranking, semantic
search and chat, running in a Web Worker so the page never freezes.

```bash
npm i kjarni-wasm
```

Or without a bundler, straight from a CDN:

```html
<script type="module">
  import { Kjarni } from "https://cdn.jsdelivr.net/npm/kjarni-wasm@0.1.4/dist/index.js";
</script>
```

## Copy this and it runs

One file, nothing installed. Save it as `quickstart.html`, serve the directory,
open it: a real model downloads and ranks three sentences by meaning.

```bash
curl -sO https://raw.githubusercontent.com/olafurjohannsson/kjarni/main/crates/kjarni-wasm/examples/quickstart.html
python3 -m http.server 8000
# then open http://localhost:8000/quickstart.html
```

A server is needed because ES modules do not load over `file://`. Any static
server will do.

```html
<pre id="out">starting…</pre>
<script type="module">
  import init, { WasmModel } from "https://cdn.jsdelivr.net/npm/kjarni-wasm@0.1.4/pkg/kjarni_wasm.js";
  await init();

  const url = "https://huggingface.co/olafuraron/all-MiniLM-L6-v2-q8/resolve/main/all-MiniLM-L6-v2-q8.kjq";
  const model = WasmModel.from_quantized(new Uint8Array(await (await fetch(url)).arrayBuffer()));

  const texts = [
    "How do I get my money back?",
    "What is your refund policy?",
    "The weather in Reykjavik is unpredictable.",
  ];

  // Normalised, so a dot product is the cosine similarity.
  const flat = model.encode(texts, true);
  const dim = flat.length / texts.length;
  const vec = (i) => flat.subarray(i * dim, (i + 1) * dim);
  const cosine = (a, b) => a.reduce((sum, x, i) => sum + x * b[i], 0);

  out.textContent = `related:   ${cosine(vec(0), vec(1)).toFixed(4)}\n`
                  + `unrelated: ${cosine(vec(0), vec(2)).toFixed(4)}`;
</script>
```

```
related:   0.5268
unrelated: -0.0470
```

This imports the wasm module directly rather than the `Kjarni` wrapper, so it runs
on the main thread with no Worker: fine for a page that encodes a few sentences,
not for chat. Use `Kjarni` for anything larger, and see the Chat section.

Scores differ slightly from the native library (0.5510 / -0.0630) because the
browser model is int8-quantised. The ranking is the same, which is what a search
uses.

## Quick tour

```ts
import { Kjarni } from "kjarni-wasm";

const kjarni = await Kjarni.load({
  encoder: "https://huggingface.co/olafuraron/all-MiniLM-L6-v2-q8/resolve/main/all-MiniLM-L6-v2-q8.kjq",
});

const [doctor, physician] = await kjarni.encode(["doctor", "physician"]);
console.log(Kjarni.cosine(doctor, physician));   // 0.8598

// or in one call
console.log(await kjarni.similarity("doctor", "physician"));
```

That URL is a real, public 23 MB file. It downloads once, the browser caches it, and nothing else leaves the machine. See [Models](#models) for the rest, and serve them from your own origin in production rather than hotlinking.

Every method returns a promise and the work happens on a worker thread. A model
loads or a reply generates while the page keeps painting.

## Why a worker

Inference is compute-bound and synchronous. On the main thread, encoding a batch
freezes the tab for as long as it takes, and generating a chat reply freezes it for
seconds. That is the single most common way a browser AI feature ends up feeling
broken.

This package puts all of it on a worker and gives you promises, so you never have
to think about it.

## Loading models

Models are `.kjq` files: config, tokenizer and int8 weights in one download.

```ts
const kjarni = await Kjarni.load({
  encoder:    model("all-MiniLM-L6-v2-q8"),
  classifier: model("distilbert-sentiment-q8"),
  reranker:   model("ms-marco-MiniLM-L-6-v2-q8"),
  onProgress: (capability, loaded, total) => {
    console.log(`${capability}: ${Math.round((loaded / total) * 100)}%`);
  },
});
```

Load only what you use. Each model is a real download, from 23 MB for MiniLM to
several hundred for a chat model, so `onProgress` is worth wiring to a progress bar.

Bytes are transferred to the worker rather than copied, which matters: copying a
500 MB model is a visible pause and doubles peak memory.

## Classification

```ts
const [result] = await kjarni.classify("This is wonderful");
console.log(result.label, result.score);   // POSITIVE 0.9999

const batch = await kjarni.classify(["Great service", "Worst purchase ever"]);
console.log(await kjarni.labels());        // ["NEGATIVE", "POSITIVE"]
```

## Reranking

Scores query and document together, which is markedly more accurate than comparing
embeddings. Indices point back into your input, so you keep your own ids.

```ts
const ranked = await kjarni.rerank("how do I cancel?", documents, 5);
for (const { index, score } of ranked) console.log(score, documents[index]);
```

## Chat

A 0.5B language model runs in the tab, and tokens arrive as they are decoded.

```ts
const kjarni = await Kjarni.load({
  chat: model("qwen05b-q8"),
  chatModelId: "qwen2.5-0.5b-instruct",
});

let out = "";
for await (const piece of kjarni.chat("Explain RAG in one sentence.")) out += piece;
```

This works because the weights stay block-quantised in memory. wasm32 caps a
single allocation at `isize::MAX`, which is 2 GB, and Qwen2.5 0.5B expanded to f32
needs 1.98 GB, so a loader that dequantises up front traps on load. The `.kjq`
container carries block-quantised weights the engine reads in place, which is
500 MB rather than 1.98 GB, and decodes faster than f32 rather than slower.

Expect a few tokens per second on a laptop CPU. Run it in a Web Worker: generation
occupies the thread until it returns, so on the main thread the tab stops painting
for the whole reply.

`chat()` completes a prompt; it does not keep turns. Multi-turn conversation, where
the model's own chat template is applied and earlier turns are remembered, exists
on `WasmChat` (`send_stream`, `clear_history`) and is not yet surfaced through this
wrapper.

## Cleaning up

```ts
await kjarni.close();
```

Wasm handles are not reachable by the JavaScript garbage collector, so dropping the
reference leaks the weights. `close()` frees the models and stops the worker.

## Requirements

A browser with WebAssembly and module workers: Chrome and Edge 80+, Firefox 114+,
Safari 15+. No WebGPU required, inference runs on the CPU.

## Models

`.kjq` is a single file holding config, tokenizer and int8 weights. These are
public and hosted on Hugging Face, one repository per model, with the repository
and file names matching:

```ts
const model = (name: string) =>
  `https://huggingface.co/olafuraron/${name}/resolve/main/${name}.kjq`;
```

| Pass as | Name | Size | What it does |
| ------- | ---- | ---- | ------------ |
| `encoder` | `all-MiniLM-L6-v2-q8` | 23 MB | embeddings, similarity, search |
| `reranker` | `ms-marco-MiniLM-L-6-v2-q8` | 23 MB | cross-encoder reranking |
| `classifier` | `distilbert-sentiment-q8` | 68 MB | positive / negative sentiment |
| `chat` | `qwen05b-q8` | 508 MB | a local language model, block-quantised so it fits |

Hugging Face serves these with permissive CORS, so a browser can fetch them from
any origin. For production, copy what you need and serve it from your own origin:
you control caching, and your page does not depend on someone else's uptime.

Load only what you use, and wire `onProgress` to something visible: these are real
downloads, and a silent minute reads as a broken page.

## The same engine elsewhere

Kjarni is one Rust engine behind several packages. The browser is one target; if
you need the server half of the same product:

- **[Live demo](https://kjarni.ai/demo/)**: this package, running in a page
- **[Kjarni](https://www.nuget.org/packages/Kjarni)** and
  **[Kjarni.Extensions.AI](https://www.nuget.org/packages/Kjarni.Extensions.AI)**
  on NuGet for C#, including `IEmbeddingGenerator` and `IChatClient`
- **[Go](https://pkg.go.dev/github.com/olafurjohannsson/kjarni-go)**,
  **[C++](https://github.com/olafurjohannsson/kjarni/tree/main/crates/kjarni-ffi/examples/cpp)**
  and a CLI that reads stdin and writes JSON
- **[Source and issues](https://github.com/olafurjohannsson/kjarni)**

## Making your own `.kjq`

See [the format guide](../scripts/README.md). Any BERT-family encoder,
cross-encoder, classifier or small decoder from Hugging Face converts with one
script.
