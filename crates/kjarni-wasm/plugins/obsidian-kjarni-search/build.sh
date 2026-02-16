#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WASM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Kjarni Obsidian Plugin Setup ==="
echo "WASM root: $WASM_ROOT"
echo "Plugin dir: $SCRIPT_DIR"

# 1. Build WASM
echo ""
echo "--- Building WASM ---"
cd "$WASM_ROOT"
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release --target nodejs -- --no-default-features
echo "WASM built."

# 2. Build plugin
echo ""
echo "--- Building plugin ---"
cd "$SCRIPT_DIR"
npm install --silent
npm run build
echo "Plugin built."

# 3. Copy WASM pkg
echo ""
echo "--- Copying WASM files ---"
mkdir -p "$SCRIPT_DIR/pkg"
cp "$WASM_ROOT/pkg/kjarni_wasm.js" "$SCRIPT_DIR/pkg/"
cp "$WASM_ROOT/pkg/kjarni_wasm_bg.wasm" "$SCRIPT_DIR/pkg/"
echo "Copied pkg/"

# 4. Copy models
echo ""
echo "--- Copying models ---"
mkdir -p "$SCRIPT_DIR/models"
cp "$WASM_ROOT/examples/all-MiniLM-L6-v2/model_q8.kjq" "$SCRIPT_DIR/models/encoder.kjq"
cp "$WASM_ROOT/examples/ms-marco-MiniLM-L-6-v2/model_q8.kjq" "$SCRIPT_DIR/models/reranker.kjq"
echo "Copied models/"

# 5. Summary
echo ""
echo "=== Done ==="
echo ""
echo "Plugin contents:"
du -sh "$SCRIPT_DIR/main.js" "$SCRIPT_DIR/pkg/"* "$SCRIPT_DIR/models/"*
echo ""
echo "To install into a vault:"
echo "  ./install.sh /path/to/your/vault"