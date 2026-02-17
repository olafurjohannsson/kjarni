RUSTFLAGS="-C target-feature=+simd128" wasm-pack build --release --target web -- --no-default-features
