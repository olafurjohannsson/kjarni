# kjarni-wasm

Local AI inference in the browser. Embeddings, classification, reranking and chat,
running in a Web Worker so the page never freezes.

```bash
npm i kjarni-wasm
```

Or without a bundler, straight from a CDN:

```html
<script type="module">
  import { Kjarni } from "https://cdn.jsdelivr.net/npm/kjarni-wasm@0.1.4/dist/index.js";
</script>
```

## Quick tour

```ts
import { Kjarni } from "kjarni-wasm";

const kjarni = await Kjarni.load({ encoder: "/models/minilm-l6-v2-q8.kjq" });

const [doctor, physician] = await kjarni.encode(["doctor", "physician"]);
console.log(Kjarni.cosine(doctor, physician));   // 0.8598

// or in one call
console.log(await kjarni.similarity("doctor", "physician"));
```

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
  encoder:    "/models/minilm-l6-v2-q8.kjq",
  classifier: "/models/distilbert-sentiment-q8.kjq",
  reranker:   "/models/ms-marco-MiniLM-L-6-v2-q8.kjq",
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

```ts
const kjarni = await Kjarni.load({
  chat: "/models/qwen05b-q8.kjq",
  chatModelId: "qwen2.5-0.5b-instruct",
});

for await (const piece of kjarni.chat("Explain RAG in one sentence.")) {
  output.textContent += piece;
}
```

`temperature: 0` selects greedy decoding, which is deterministic. Note that the
underlying binding currently generates to completion before returning, so the reply
arrives as one piece. The `for await` shape is the API either way, and it will
stream incrementally without a change on your side once token callbacks land.

## Cleaning up

```ts
await kjarni.close();
```

Wasm handles are not reachable by the JavaScript garbage collector, so dropping the
reference leaks the weights. `close()` frees the models and stops the worker.

## Requirements

A browser with WebAssembly and module workers: Chrome and Edge 80+, Firefox 114+,
Safari 15+. No WebGPU required, inference runs on the CPU.

## Making your own `.kjq`

See [the format guide](../scripts/README.md). Any BERT-family encoder,
cross-encoder, classifier or small decoder from Hugging Face converts with one
script.
