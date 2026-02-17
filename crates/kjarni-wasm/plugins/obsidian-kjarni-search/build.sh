#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WASM_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Kjarni Obsidian Plugin Build ==="

# 1. Build WASM (web target)
echo "--- Building WASM (--target web) ---"
cd "$WASM_ROOT"
RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release --target web -- --no-default-features

# 2. Build plugin (main.js + worker.js + encoder-worker.js)
echo "--- Building plugin ---"
cd "$SCRIPT_DIR"
npm install --silent
npm run build

# 3. Package for release
echo "--- Creating release package ---"
RELEASE_DIR="$SCRIPT_DIR/release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR/kjarni-search/pkg"

cp "$SCRIPT_DIR/main.js"            "$RELEASE_DIR/kjarni-search/"
cp "$SCRIPT_DIR/worker.js"          "$RELEASE_DIR/kjarni-search/"
cp "$SCRIPT_DIR/encoder-worker.js"  "$RELEASE_DIR/kjarni-search/"
cp "$SCRIPT_DIR/manifest.json"      "$RELEASE_DIR/kjarni-search/"
cp "$SCRIPT_DIR/styles.css"         "$RELEASE_DIR/kjarni-search/"
cp "$WASM_ROOT/pkg/kjarni_wasm_bg.wasm" "$RELEASE_DIR/kjarni-search/pkg/"

cd "$RELEASE_DIR"
tar -czf "$SCRIPT_DIR/kjarni-search.tar.gz" kjarni-search/
cd "$SCRIPT_DIR"
rm -rf "$RELEASE_DIR"

echo ""
echo "=== Done ==="
ls -lh "$SCRIPT_DIR/kjarni-search.tar.gz"
echo ""
echo "Contents:"
tar -tzf "$SCRIPT_DIR/kjarni-search.tar.gz"
echo ""
echo "Models auto-download from kjarni.ai on first launch."
