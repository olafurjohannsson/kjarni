#!/bin/bash
# kjarni-ffi/scripts/build.sh

set -e
cd "$(dirname "$0")/.."

cargo build --release --features c-bindings

# Create lib directory with library
mkdir -p dist/lib
cp ../../target/release/libkjarni_ffi.so dist/lib/ 2>/dev/null || true
cp ../../target/release/libkjarni_ffi.dylib dist/lib/ 2>/dev/null || true
cp ../../target/release/kjarni_ffi.dll dist/lib/ 2>/dev/null || true

# Copy header
mkdir -p dist/include
cp include/kjarni.h dist/include/

echo "Built library to dist/"
echo "Set LD_LIBRARY_PATH=$PWD/dist/lib or install to /usr/local/lib"
export LD_LIBRARY_PATH=$PWD/dist/lib:$LD_LIBRARY_PATH
# Install library (optional, avoids LD_LIBRARY_PATH)
#sudo cp target/release/libkjarni_ffi.so /usr/local/lib/
#sudo ldconfig