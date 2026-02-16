#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAULT="${1:?Usage: ./install.sh /path/to/your/vault}"
PLUGIN="$VAULT/.obsidian/plugins/kjarni-search"

echo "Installing Kjarni Search to: $PLUGIN"

mkdir -p "$PLUGIN/pkg" "$PLUGIN/models"

cp "$SCRIPT_DIR/main.js" "$PLUGIN/"
cp "$SCRIPT_DIR/manifest.json" "$PLUGIN/"
cp "$SCRIPT_DIR/styles.css" "$PLUGIN/"
cp "$SCRIPT_DIR/pkg/kjarni_wasm.js" "$PLUGIN/pkg/"
cp "$SCRIPT_DIR/pkg/kjarni_wasm_bg.wasm" "$PLUGIN/pkg/"
cp "$SCRIPT_DIR/models/encoder.kjq" "$PLUGIN/models/"
cp "$SCRIPT_DIR/models/reranker.kjq" "$PLUGIN/models/"

echo ""
echo "Installed. Files:"
du -sh "$PLUGIN/main.js" "$PLUGIN/pkg/"* "$PLUGIN/models/"*
echo ""
echo "Open Obsidian → Settings → Community plugins → Enable 'Kjarni Search'"