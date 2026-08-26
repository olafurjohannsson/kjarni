# Test fixtures

## speech.wav

A ~11 second excerpt of John F. Kennedy's 1961 inaugural address:

> And so my fellow Americans, ask not what your country can do for you,
> ask what you can do for your country.

16-bit PCM, mono, 16 kHz — the format Whisper expects, so no resampling
happens before the model sees it.

**Provenance:** taken from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)
(`samples/jfk.wav`), where it serves the same purpose. The recording is a work
of the United States federal government and is in the public domain, which is
why it is safe to redistribute inside this repository.

Used by the `#[ignore]`d integration tests in
`crates/kjarni/src/transcriber/tests.rs`, which assert that the transcription
contains a known phrase. Content assertions are deliberate: a decode that
produces fluent nonsense still passes an "is not empty" check.

Run them with:

```bash
cargo test -p kjarni transcriber -- --ignored
```

They need the `whisper-small` weights, which download on first use.
