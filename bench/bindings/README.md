# Binding overhead

The same MiniLM embedding of one short text, called through each language's
binding against the same `libkjarni_ffi`, on the CPU. A short text is the worst
case for a binding: the model does the least work, so crossing the boundary is the
largest share of the call. C is the baseline, calling the library directly.

    python3 bench/bindings/run.py

Needs gcc, Go and the .NET SDK, plus `cargo build --release -p kjarni-ffi`. Writes
`bench/compare/results/bindings.json`.
