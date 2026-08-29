# Changelog

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
