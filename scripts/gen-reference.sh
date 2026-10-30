#!/usr/bin/env bash
#
# API reference for every language, as Markdown.
#
#   ./scripts/gen-reference.sh [out-dir]      default: target/reference
#
# Writes <out>/python/kjarni.md, <out>/dotnet/*.md, <out>/js/**/*.md,
# <out>/go/kjarni.md, <out>/rust/*.md, <out>/c/*.md and <out>/cpp/kjarni.md, plus a
# VERSION file. The website imports these into its own design.
#
# Fails if any public item in the Python, .NET, JavaScript, Go, C or C++ APIs has no
# doc comment, so the reference cannot quietly fall behind the API. Needs python3,
# dotnet, node, go and cargo.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="${1:-target/reference}"
WORK=target/reference-tools
rm -rf "$OUT"
mkdir -p "$OUT" "$WORK"
OUT="$(cd "$OUT" && pwd)"

# ── Python: griffe reads the source, so no native library is needed ──
if [ ! -x "$WORK/venv/bin/python" ]; then
  python3 -m venv "$WORK/venv"
  "$WORK/venv/bin/pip" install -q "griffe==1.*" "griffe2md==1.*"
fi
"$WORK/venv/bin/python" -I scripts/reference_python.py \
  crates/kjarni-ffi/bindings/python "$OUT/python/kjarni.md"

# ── .NET: CS1591 as an error is the doc-comment check; docfx writes the pages ──
CS=crates/kjarni-ffi/bindings/csharp
for proj in Kjarni/Kjarni.csproj Kjarni.Extensions.AI/Kjarni.Extensions.AI.csproj; do
  dotnet build "$CS/$proj" -c Release -f net8.0 -warnaserror:CS1591 \
    -o "$WORK/dotnet-build/$(dirname "$proj")" -nologo -v quiet
done
if [ ! -x "$WORK/docfx" ]; then
  dotnet tool install docfx --version "2.*" --tool-path "$WORK" >/dev/null
fi
cat > "$WORK/docfx.json" <<EOF
{
  "metadata": [{
    "src": [{
      "src": "$(pwd)/$CS",
      "files": ["Kjarni/Kjarni.csproj", "Kjarni.Extensions.AI/Kjarni.Extensions.AI.csproj"]
    }],
    "dest": "$OUT/dotnet",
    "outputFormat": "markdown",
    "properties": { "TargetFramework": "net8.0" }
  }]
}
EOF
"$WORK/docfx" metadata "$WORK/docfx.json" --warningsAsErrors
echo "dotnet: $(find "$OUT/dotnet" -name '*.md' | wc -l) pages -> $OUT/dotnet"

# ── JavaScript: typedoc rejects undocumented exports ──
TS=crates/kjarni-wasm/ts
(
  cd "$TS"
  npx --yes -p typescript@5.6 -p typedoc@0.26 -p typedoc-plugin-markdown@4.2 \
    typedoc --plugin typedoc-plugin-markdown \
      --entryPoints src/index.ts --tsconfig tsconfig.json \
      --readme none --excludePrivate --excludeInternal \
      --validation.notDocumented --treatWarningsAsErrors --skipErrorChecking \
      --out "$OUT/js" --logLevel Warn
)
echo "js: $(find "$OUT/js" -name '*.md' | wc -l) pages -> $OUT/js"

# ── Go: go/doc reads the source; undocumented exports fail ──
go run scripts/reference_go.go crates/kjarni-ffi/bindings/go "$OUT/go/kjarni.md"

# ── Rust: RUSTC_BOOTSTRAP lets a stable toolchain write rustdoc's JSON ──
RUSTC_BOOTSTRAP=1 RUSTDOCFLAGS="-Z unstable-options --output-format json" \
  cargo doc -p kjarni --no-deps --quiet --target-dir "$WORK/rustdoc"
python3 -I scripts/reference_native.py rust "$WORK/rustdoc/doc/kjarni.json" . "$OUT/rust"

# ── C and C++: the headers are the source ──
python3 -I scripts/reference_native.py c crates/kjarni-ffi/include/kjarni.h "$OUT/c"
python3 -I scripts/reference_native.py cpp crates/kjarni-ffi/include/kjarni.hpp "$OUT/cpp/kjarni.md"

grep -m1 -E '^version *= *"' Cargo.toml | cut -d'"' -f2 > "$OUT/VERSION"
echo "reference for $(cat "$OUT/VERSION") in $OUT"
