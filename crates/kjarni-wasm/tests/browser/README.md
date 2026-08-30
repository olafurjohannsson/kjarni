# Browser tests

Loads the built WASM bundle in a real Chromium and checks it works.

```bash
KJARNI_KJQ_DIR=/tmp/kjq node run.mjs
```

## Why this exists

The other twenty tests for `kjarni-wasm` run the Rust natively. That covers the
engine and misses the artifact: a broken import, a bundle compiled from stale
sources, a binding that only fails through wasm-bindgen's glue. None of those show
up in a native test run.

That is not hypothetical. The live demo served a bundle with no `WasmClassifier`
and no `WasmChat` in it for days while every test was green, because nothing
anywhere loaded the thing people actually download.

Pointed at that stale bundle, this fails the way you would want:

```
FAIL  harness threw: page.evaluate: TypeError: window.__run is not a function
      page errors: The requested module '/pkg/kjarni_wasm.js' does not
                   provide an export named 'WasmChat'
```

## What it checks

- The module graph loads with no page errors
- Every class the bundle is supposed to export is a function, which is the check
  that catches a stale build
- MiniLM returns 384 dimensions
- `doctor` and `physician` score above 0.7, `doctor` and `banana` below 0.4, and
  the first is well clear of the second

The semantic assertions matter more than they look. Dimensions alone would pass on
a bundle returning zeros.

## Setup

Needs `crates/kjarni-wasm/pkg` (from `wasm-pack build --target web`) and a
directory of `.kjq` fixtures:

```bash
python ../../scripts/quantize_model.py \
    --model-dir ~/.cache/kjarni/sentence-transformers_all-MiniLM-L6-v2 \
    --output /tmp/kjq/all-MiniLM-L6-v2-q8.kjq

npm install && npx playwright install chromium
```

Models are served from a local directory rather than fetched from Hugging Face:
this job is about the bundle, and a network hiccup should not turn into a red
build.

`KJARNI_PKG_DIR` overrides which bundle is loaded, which is how the stale-bundle
behaviour above was verified.

## No test framework

Same reasoning as the C++ tests. This is a hundred lines of driver against a
dependency people have to install anyway; a framework on top would be more to
install and no clearer to read.
