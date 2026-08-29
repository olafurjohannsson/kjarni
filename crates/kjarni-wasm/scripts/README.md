# Creating `.kjq` models

`.kjq` is Kjarni's single-file model container: config, tokenizer and weights in one
file, with the weights quantised to int8. It exists because shipping a model over a
network is dominated by transfer size and request count.

```
all-MiniLM-L6-v2     88 MB safetensors + config + tokenizer  →  23 MB, one request
qwen2.5-0.5b         1.9 GB                                  →  478 MB
distilbert-sentiment 255 MB                                  →  68 MB
```

Roughly **3.9x smaller**, and one `fetch` instead of three.

## Making one

You need a directory containing `model.safetensors`, `config.json` and
`tokenizer.json` — which is exactly what Kjarni's model cache already holds:

```bash
ls ~/.cache/kjarni/                      # or the path in KJARNI_CACHE_DIR
kjarni model download minilm-l6-v2       # if it is not there yet
```

The script needs `numpy` and `safetensors`. Neither is a Kjarni dependency, so use a
throwaway virtualenv rather than installing them globally:

```bash
python3 -m venv /tmp/kjq && /tmp/kjq/bin/pip install numpy safetensors

/tmp/kjq/bin/python quantize_model.py \
    --model-dir ~/.cache/kjarni/sentence-transformers_all-MiniLM-L6-v2 \
    --output all-MiniLM-L6-v2-q8.kjq
```

It prints what it did:

```
Tensors quantized (int8): 40
Tensors kept (f32):       64
Original weights size:    255.4 MB
Output file size:         64.7 MB
Compression ratio:        3.9x
```

Add `--verify` to reload the result and compare tensors against the source.

## Using one

In the browser, every binding takes `.kjq` bytes directly:

```js
const bytes = new Uint8Array(await (await fetch("/models/model-q8.kjq")).arrayBuffer());

const model      = WasmModel.from_quantized(bytes);          // embeddings
const reranker   = WasmReranker.load(bytes);                 // cross-encoder
const classifier = WasmClassifier.load(bytes);               // sentiment, emotion, toxicity
const chat       = WasmChat.load(bytes, "qwen2.5-0.5b-instruct");
```

From Rust, `kjarni_transformers::weights::kjq::unpack` returns the safetensors bytes,
config and tokenizer, which the loaders accept:

```rust
let unpacked = kjq::unpack(&bytes)?;
let encoder: SentenceEncoder = EncoderLoader::load_from_bytes(
    &unpacked.safetensors,
    &unpacked.config_json,
    unpacked.tokenizer_json.as_bytes(),
    Device::Cpu, None, None,
)?;
```

**Not yet wired into the CLI.** `ModelWeights::new` recognises `.gguf` but not
`.kjq`, so `kjarni embed --model-path foo.kjq` does not work. Teaching it the
extension would give the CLI, the C# bindings and everything else `.kjq` for free.

## Things worth knowing

**Quantisation is a transport concern, not a runtime one.** Tensors are dequantised
back to f32 while loading, so nothing downstream needs int8 kernels and the engine
runs the same code path as it would for safetensors.

**It costs some accuracy.** Measured against the same model loaded from f32
safetensors, with the inference code held constant:

| Text | cosine vs f32 |
| ---- | ------------- |
| "Hello world" | 0.948 |
| "The cat sat on the mat" | 0.975 |
| "Reykjavik is the capital of Iceland" | 0.976 |

Ranking is unaffected — a refund query still separates cleanly from an unrelated
sentence — but 0.948 is lower than int8 usually costs. Scaling is per-tensor;
per-channel would likely recover most of the gap at the same file size. Prefer
safetensors natively, where the bandwidth saving buys nothing.

**Not everything gets quantised.** The script keeps small tensors (layer norms,
biases) at f32, because quantising them costs accuracy and saves almost nothing. That
is why the ratio is ~3.9x rather than 4x.

**bfloat16 works.** numpy has no bf16 and `safetensors` will not hand one back, so
bf16 checkpoints used to fail outright — which covered most modern chat models,
including Qwen and Llama. The script now reads those tensors raw and widens them to
f32. bf16 shares f32's exponent and truncates the mantissa, so this is exact.

## Format

```
magic     "KJQ1"                     4 bytes
config    u32 length + JSON          config.json verbatim
tokenizer u32 length + JSON          tokenizer.json verbatim
tensors   u32 count, then per tensor:
            u32 name length + name
            u32 rank + u32 per dimension
            u8  quantized flag
            if quantized: f32 scale + i8 per element
            else:         f32 per element
```

All integers little-endian. Dequantisation is `value = q as f32 * scale`, one scale
per tensor.

The reader is `kjarni_transformers::weights::kjq`, which has tests covering
dequantisation, shapes, bad magic bytes and truncation. If you change the writer here,
change that module too — they are the two halves of one format.
