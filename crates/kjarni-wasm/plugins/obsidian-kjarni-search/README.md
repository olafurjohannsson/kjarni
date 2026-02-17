# Kjarni Search — Obsidian Plugin

Hybrid semantic + keyword search with cross-encoder reranking. Fully local, no API keys, no cloud.

## Setup

### 1. Build the plugin

```bash
cd obsidian-kjarni-search
npm install
npm run build
```

### 2. Build the WASM package

```bash
cd /path/to/kjarni/crates/kjarni-wasm
wasm-pack build --target web
```

### 3. Install into your vault

Copy files into your vault's plugin directory:

```
YOUR_VAULT/.obsidian/plugins/kjarni-search/
├── main.js           ← from plugin build
├── manifest.json     ← from plugin root
├── styles.css        ← from plugin root
├── pkg/
│   ├── kjarni_wasm.js          ← from wasm-pack build
│   └── kjarni_wasm_bg.wasm     ← from wasm-pack build
└── models/
    ├── encoder.kjq              ← all-MiniLM-L6-v2 quantized
    └── reranker.kjq             ← ms-marco-MiniLM-L-6-v2 quantized
```

Quick copy script:

```bash
VAULT="$HOME/your-vault"
PLUGIN="$VAULT/.obsidian/plugins/kjarni-search"

mkdir -p "$PLUGIN/pkg" "$PLUGIN/models"

# Plugin files
cp main.js manifest.json styles.css "$PLUGIN/"

# WASM
cp /path/to/kjarni/crates/kjarni-wasm/pkg/kjarni_wasm.js "$PLUGIN/pkg/"
cp /path/to/kjarni/crates/kjarni-wasm/pkg/kjarni_wasm_bg.wasm "$PLUGIN/pkg/"

# Models
cp /path/to/all-MiniLM-L6-v2/model_q8.kjq "$PLUGIN/models/encoder.kjq"
cp /path/to/ms-marco-MiniLM-L-6-v2/model_q8.kjq "$PLUGIN/models/reranker.kjq"
```

### 4. Enable the plugin

Open Obsidian → Settings → Community plugins → Enable "Kjarni Search"

The plugin will automatically index your vault on first load.

## Usage

- **Cmd+Shift+K** (or Ctrl+Shift+K) — Open search modal
- **Cmd+P → "Kjarni: Reindex vault"** — Rebuild index after major changes

## How it works

1. On first load, splits all markdown files into chunks (~1000 chars)
2. Encodes each chunk into a 384-dimensional embedding using all-MiniLM-L6-v2
3. Builds a hybrid index (BM25 inverted index + vector store)
4. Saves index to disk for fast subsequent loads
5. Search queries run BM25 + semantic similarity fusion, then cross-encoder reranking