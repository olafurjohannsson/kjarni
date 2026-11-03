window.BENCHMARK_DATA = {
  "lastUpdate": 1793746298915,
  "repoUrl": "https://github.com/olafurjohannsson/kjarni",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "856712dc673b09350ae259ab0de12e5a211ed914",
          "message": "Update the benchmark action so custom results are accepted (#97)",
          "timestamp": "2026-11-03T08:54:18Z",
          "tree_id": "2b1168e3329b9601b90551c34b1a4a504e8c69ba",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/856712dc673b09350ae259ab0de12e5a211ed914"
        },
        "date": 1793697339491,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 482.89,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.28,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.002"
          },
          {
            "name": "macos-arm64 embed",
            "value": 442.25,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.898"
          },
          {
            "name": "macos-arm64 decode",
            "value": 6.36,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.790"
          },
          {
            "name": "windows-x64 embed",
            "value": 469.74,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.38,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "cc6ba6229703ef5e89edc70fe37954f970233d4c",
          "message": "Add NEON kernels for quantized weights, RoPE and norms (#98)\n\n* Add NEON kernels for quantized weights, RoPE and norms\n\nQ8_0, Q4_K and Q6_K ran scalar on ARM. Adds those, a 4x3 f32 tile and\ncolumn-parallel batches, and gives f32 and bf16 four running sums. Fixes the\nbatched f32 paths leaving output unwritten without AVX2. Runs the library tests\non macOS in CI.\n\n* test",
          "timestamp": "2026-11-03T09:25:14Z",
          "tree_id": "2db2a3ecbd11d61678d94c1bd6c968538b83f817",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/cc6ba6229703ef5e89edc70fe37954f970233d4c"
        },
        "date": 1793698720417,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 691.35,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.010"
          },
          {
            "name": "linux-x64 decode",
            "value": 43.67,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.011"
          },
          {
            "name": "macos-arm64 embed",
            "value": 569.15,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.006"
          },
          {
            "name": "macos-arm64 decode",
            "value": 45.25,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 4.594"
          },
          {
            "name": "windows-x64 embed",
            "value": 467.27,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.987"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.73,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.023"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6d2bb01c127f9a5be192886a2efd5ff8d32b06f7",
          "message": "Release/0.4.2 (#99)\n\n* Index greyscale JPEGs instead of skipping them\n\nThe JPEG decoder returns one channel for a single-component image even when\nRGB is requested, so black-and-white photos failed the size check and were\nleft out of the image index. Expands them to RGB, with a test on a real\ngreyscale JPEG.\n\n* Keep generating when a character spans several tokens\n\nA token that only starts a multi-byte character decoded to U+FFFD and was sent,\nand the next step skipped the model, so replies stopped at the first such\ncharacter. The unfinished character is now held back until it is complete, and\nthe model sees every token. Speculative decoding decodes the generated run as a\nwhole for the same reason.\n\n* Release 0.4.2\n\n* Format the multi-byte character test",
          "timestamp": "2026-11-03T12:53:25Z",
          "tree_id": "a52d8b6fca01e87166cdb0684ac9392f30b1083f",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/6d2bb01c127f9a5be192886a2efd5ff8d32b06f7"
        },
        "date": 1793712123768,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 386.5,
            "unit": "sentences/s",
            "extra": "CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz; vs previous commit: 1.004"
          },
          {
            "name": "linux-x64 decode",
            "value": 21.65,
            "unit": "tokens/s",
            "extra": "CPU: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 401.65,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.986"
          },
          {
            "name": "macos-arm64 decode",
            "value": 25.11,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.971"
          },
          {
            "name": "windows-x64 embed",
            "value": 471.11,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.998"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.65,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.013"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8809fdeffa58e4492025366e42f083d46e7d6636",
          "message": "Keep model download messages off stdout (#100)\n\nThe banner printed before a model download went to stdout and ignored\nquiet, so the first `kjarni similarity -q` on a machine printed the\nbanner ahead of the score, and library callers such as ImageIndex wrote\nit into the host program's stdout. It now goes to stderr and is skipped\nwhen quiet is set.",
          "timestamp": "2026-11-03T13:12:40Z",
          "tree_id": "ccef7f1bd1c3abd168ab6f9be284b080f336f32e",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/8809fdeffa58e4492025366e42f083d46e7d6636"
        },
        "date": 1793712623823,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 487.48,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.52,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 712.66,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.024"
          },
          {
            "name": "macos-arm64 decode",
            "value": 39.91,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 3.579"
          },
          {
            "name": "windows-x64 embed",
            "value": 741.97,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 173 Stepping 1, GenuineIntel; vs previous commit: 0.987"
          },
          {
            "name": "windows-x64 decode",
            "value": 19.69,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 173 Stepping 1, GenuineIntel; vs previous commit: 0.993"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8809fdeffa58e4492025366e42f083d46e7d6636",
          "message": "Keep model download messages off stdout (#100)\n\nThe banner printed before a model download went to stdout and ignored\nquiet, so the first `kjarni similarity -q` on a machine printed the\nbanner ahead of the score, and library callers such as ImageIndex wrote\nit into the host program's stdout. It now goes to stderr and is skipped\nwhen quiet is set.",
          "timestamp": "2026-11-03T13:12:40Z",
          "tree_id": "ccef7f1bd1c3abd168ab6f9be284b080f336f32e",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/8809fdeffa58e4492025366e42f083d46e7d6636"
        },
        "date": 1793713597840,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 481.98,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.994"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.46,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.004"
          },
          {
            "name": "macos-arm64 embed",
            "value": 407.9,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.009"
          },
          {
            "name": "macos-arm64 decode",
            "value": 23.56,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.989"
          },
          {
            "name": "windows-x64 embed",
            "value": 553.89,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.002"
          },
          {
            "name": "windows-x64 decode",
            "value": 18.29,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.001"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "olafurjohannss@gmail.com",
            "name": "Ólafur Aron Jóhannsson",
            "username": "olafurjohannsson"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b0ec58ffaadf1b7d9ddc964983b524b70f39d27",
          "message": "Release/0.4.3 (#101)\n\n* Decode MP3, FLAC and Ogg Vorbis for transcription\n\nThe docs, CLI help and every binding listed these formats, but the symphonia\nfeature they needed was an empty placeholder, so release builds failed on\nanything but WAV. Decodes them with symphonia, pure Rust like hound, and shares\nthe WAV path's mixdown and resampling. Tested against a WAV tone written in all\nfour formats: FLAC matches sample for sample, MP3 and Vorbis keep length, rate\nand pitch.\n\n* Print timestamps from kjarni transcribe without --no-stream\n\nStreamed tokens carry no times, so --timestamps was ignored unless streaming\nwas also turned off. The flag now waits for the full result. The --model-path\nhelp, two sentences run together, now says the flag is not supported yet.\n\n* Count documents written in indexing progress\n\nEmbedding progress was reported before each batch was embedded, so the last\nreport was the count before the final batch, and a small index printed\n\"Indexed 0 documents\". It is now reported after each batch is written.\n\n* Name a model the Python Reranker accepts in its docstring\n\n\"cross-encoder-minilm\" is rejected as an unknown model. Adds the first\nReranker test, which loads the documented model and pins its scores.\n\n* Load the library on Windows in the Go binding\n\npurego has Dlopen and Dlsym only on Unix, so the module did not compile for\nWindows. Windows now uses LoadLibrary and GetProcAddress.\n\n* Test the Go binding on Windows in CI\n\ngo-test ran only on Linux, which is how a Windows build failure shipped. Every\npush now vets the module for Windows, and build-windows runs the Go unit and\nintegration tests against the DLL it built. The Python job downloads the\nreranker model its new test needs.\n\n* Release 0.4.3",
          "timestamp": "2026-11-03T22:28:52Z",
          "tree_id": "be2820876e93b30cdc6e7e00131ead3a235ff659",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/3b0ec58ffaadf1b7d9ddc964983b524b70f39d27"
        },
        "date": 1793746297784,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 486.73,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.990"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.74,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.010"
          },
          {
            "name": "macos-arm64 embed",
            "value": 371.25,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.051"
          },
          {
            "name": "macos-arm64 decode",
            "value": 19.06,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.987"
          },
          {
            "name": "windows-x64 embed",
            "value": 535.31,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.985"
          },
          {
            "name": "windows-x64 decode",
            "value": 25.33,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
          }
        ]
      }
    ]
  }
}