#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VAULT="${1:?Usage: ./install.sh /path/to/your/vault}"
PLUGIN="$VAULT/.obsidian/plugins/kjarni-search"

echo "Installing Kjarni Search to: $PLUGIN"

mkdir -p "$PLUGIN/pkg"

cp "$SCRIPT_DIR/main.js"            "$PLUGIN/"
cp "$SCRIPT_DIR/worker.js"          "$PLUGIN/"
cp "$SCRIPT_DIR/encoder-worker.js"  "$PLUGIN/"
cp "$SCRIPT_DIR/manifest.json"      "$PLUGIN/"
cp "$SCRIPT_DIR/styles.css"         "$PLUGIN/"
cp "$SCRIPT_DIR/pkg/kjarni_wasm_bg.wasm" "$PLUGIN/pkg/"

echo "Installed. Models auto-download on first launch."
echo "Open Obsidian → Settings → Community plugins → Enable 'Kjarni Search'"
