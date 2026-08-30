# Changelog

## Unreleased

### A panic in the engine no longer kills the host process

The workspace release profile set `panic = "abort"`. Cargo has no per-package panic
strategy, so that applied to `kjarni-ffi` as well — the cdylib that NuGet, the Go
module and the C header all load *into someone else's process*. Any panic reachable
through any of the 80 `extern "C"` entry points terminated that process: a .NET
service, a Go binary, a Qt application. No error code, no message, nothing the caller
could have handled. `kjarni-ffi/src` had 38 `unwrap`/`expect`/`panic!` sites of its
own, and everything the engine can panic on sits behind them.

`panic = "abort"` is gone from the profile, and the 66 fallible entry points now run
their bodies inside a `catch_unwind` barrier (`kjarni-ffi/src/panic.rs`). A panic
becomes the new `KJARNI_ERROR_CODE_PANIC` (11) with the entry point's name and the
panic message on `kjarni_last_error_message()`, which is what every other failure in
the C ABI already looks like. The remaining 14 entry points return compile-time
constants and are left unguarded, which the module documents; guarding
`kjarni_last_error_message` in particular would be circular.

This is a backstop, not a licence to panic — a panic reaching it is still a bug. It
only decides whether the caller gets to hear about it.

Building `kjarni-ffi` under `panic = "abort"` is now a compile error rather than a
silently defeated barrier, since `catch_unwind` cannot intercept an abort. That check
is what would have caught the original setting.

`KJARNI_ERROR_CODE_PANIC = 11` is additive — no existing code is renumbered — and is
mirrored in the C header, the C# `KjarniErrorCode`, Go's `ErrPanic`, and Python's
`PANIC`. The cost of the profile change is a larger binary and landing pads the
optimiser must keep.

### An error message with an interior nul is no longer discarded

`set_last_error` built its `CString` with `CString::new(msg).ok()`, which yields
`None` on an interior nul and cleared the stored message rather than replacing it, so
`kjarni_last_error_message()` returned NULL for an error that had just been reported.
It now truncates at the first nul and keeps the leading, most specific part. For an ordinary error that lost a diagnostic; for a caught panic it
would have lost the only evidence the panic happened.

## 0.1.4

### Encoders now truncate where sentence-transformers does

Encoder models read `max_seq_length` from `sentence_bert_config.json` when the model
ships one, falling back to `max_position_embeddings` as before. For
`all-MiniLM-L6-v2` that changes the default from 512 to **256**, and for
`all-mpnet-base-v2` from 512 to **384**.

This is a behaviour change for long inputs. Anything under the limit is unaffected,
but text above it now produces the same embedding the reference Python
implementation produces, where previously it did not. The default RAG chunk of 1000
characters measures at a median of 286 tokens on ordinary prose, so most indexed
chunks were in the affected range. **Rebuild indexes built with an earlier version**
if you index documents longer than 256 tokens.

Pass `ModelLoadConfig::max_sequence_length` to override.

### Asking for more positions than a model has is now an error

Requesting a sequence length above the model's position-embedding table used to
succeed. Tokens past the table received no position embedding at all and the encoder
returned a normal-looking vector that was quietly wrong. This is now refused at load
with a message naming the request and the ceiling. A length derived from a config
file is clamped with a warning instead, so a malformed config cannot make a model
unloadable.

### `.kjq` carries the tuned sequence length

`scripts/quantize_model.py` folds `max_seq_length` into the config it packs, and the
byte loader reads it back. Without this the browser had no way to learn the number,
since reading `sentence_bert_config.json` needs a filesystem, and WASM would have
truncated at 512 while every native binding truncated at 256. **Regenerate `.kjq`
files** built with an earlier version.

### Vector search is faster and its ordering is now defined

`VectorStore::search` keeps a bounded top-k rather than sorting every candidate and
truncating, hoists the query norm out of the inner loop, and runs in parallel above
2048 vectors. On 384-dimensional vectors, best of five on a 24-core machine:

| Vectors | Before | After |
| ------- | ------ | ----- |
| 10,000 | 2.6 ms | 0.2 ms |
| 100,000 | 24.9 ms | 4.9 ms |
| 1,000,000 | 255 ms | 56 ms |

Results are unchanged. Ties now break on ascending index, which makes the ordering
total: without that the parallel scan could return a different permutation of
equally-scoring documents between runs.

WASM keeps the serial path, since rayon there needs a threaded build.

### C++ is built and tested in CI

The C ABI header had never been compiled by CI, and shipped in every release archive
unverified. A new job regenerates `kjarni.h` with cbindgen and diffs it against the
committed copy, compiles it as C11 under `-Wall -Wextra -Werror`, and runs the C++
wrapper's 61 checks under AddressSanitizer and UndefinedBehaviorSanitizer.

### Other

- `kjarni-wasm`'s 18 binding tests now run in CI; they were excluded when the crate
  could not build natively.
- The `.kjq` test fixtures are exported in CI from cached weights. They previously
  looked for a sibling checkout that CI does not have, so they skipped and reported
  green while testing nothing.
- `quantize_model.py` tolerates an unreadable `sentence_bert_config.json` rather than
  aborting, which is what a truncated or redirected download leaves behind.
