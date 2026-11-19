# Test fixtures

## gettysburg_40s.mp3

The first 40 seconds of a LibriVox reading of the Gettysburg Address: the
LibriVox notice, the title and date, then "Four score and seven years ago our
fathers brought forth on this continent a new nation…". Mono, 16 kHz, 32 kb/s
MP3.

whisper-small ends the first 30-second window after the notice, five seconds
before the window's audio does, so it checks that speech after an early end is
still transcribed.

**Provenance:** `gettysburg_address_64kb.mp3` from
[archive.org/details/gettysburg_johng_librivox](https://archive.org/details/gettysburg_johng_librivox).
LibriVox recordings are in the public domain, as is Lincoln's text.

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
