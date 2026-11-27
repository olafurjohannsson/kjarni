# Contributing to Kjarni

Thanks for helping. The easiest places to start are the issues labelled
[`good first issue`](https://github.com/kjarni-ai/kjarni/labels/good%20first%20issue),
and adding a model: [#140](https://github.com/kjarni-ai/kjarni/issues/140)
walks through it.

## Before you start

Comment on the issue you want to work on, so two people don't do the same work.
For anything larger than a fix (a new architecture, a kernel, a change to an API),
open an issue first and say how you plan to do it.

## Building

```bash
cargo build --release
```

The workspace builds on Linux, macOS and Windows with a recent stable Rust. The
GPU backend uses WebGPU (Vulkan, Metal or DirectX 12), so no CUDA toolkit is needed.

## Tests

Many tests load real models: they download weights on first run, take a lot of
memory, and must not run in parallel with each other.

```bash
./test-everything.sh                 # the whole suite, one test at a time
cargo test -p kjarni-rag             # one crate
cargo test -- --test-threads=1       # anything that loads a model
```

Tests marked `#[ignore]` need models in the local cache or reference files; the
script runs them with `--include-ignored`.

## What a pull request needs

- **A test for every fix**, one that fails without the fix. For model output, the
  reference comes from PyTorch / `transformers`, checked in rather than computed
  in the test.
- **Outputs keep matching PyTorch.** Full-precision weights must give the same
  results as `transformers`, on CPU and GPU. Quantised weights may differ on the
  GPU, but `KJARNI_GPU_EXACT=1` must still match the CPU.
- **No speed regressions.** For a change to a kernel or a hot path, compare against
  `main` on the README models (alternate the two builds on an idle machine) and
  put the numbers in the PR.
- **The checks CI runs:** `cargo fmt --all`, `cargo clippy --workspace`, and both
  WASM builds (CPU-only and WebGPU).
- **Comments say what the code does,** not its history; history goes in the commit
  message.
- **A line in `CHANGELOG.md`** for anything a user would notice.

## Benchmarks

Numbers in the README and `bench/BENCHMARKS.md` come from `bench/compare/`. When you
quote a comparison, both engines run the same weights at their best settings, and
the machine is named. Results from other hardware are welcome:
[#153](https://github.com/kjarni-ai/kjarni/issues/153).

## License

Kjarni is dual-licensed under MIT and Apache-2.0. Contributions are accepted under
the same terms.
