window.BENCHMARK_DATA = {
  "lastUpdate": 1795694226634,
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
        "date": 1793746352717,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 666.4,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.994"
          },
          {
            "name": "linux-x64 decode",
            "value": 41.21,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.986"
          },
          {
            "name": "macos-arm64 embed",
            "value": 362.48,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.984"
          },
          {
            "name": "macos-arm64 decode",
            "value": 20.16,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.943"
          },
          {
            "name": "windows-x64 embed",
            "value": 469.69,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.995"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.55,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.026"
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
          "id": "4c8446a51171abc24ffd3afafc4979e2e9d7b130",
          "message": "0.4.4: search_images and folder index sync in MCP, Qwen BOS and GPT-2 CPU fi (#103)",
          "timestamp": "2026-11-04T10:32:59Z",
          "tree_id": "453b347c3f78198fc92f7dd3eac454b8acbcac6c",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/4c8446a51171abc24ffd3afafc4979e2e9d7b130"
        },
        "date": 1793789558073,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 484.72,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.6,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "macos-arm64 embed",
            "value": 649.27,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.084"
          },
          {
            "name": "macos-arm64 decode",
            "value": 44.56,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.032"
          },
          {
            "name": "windows-x64 embed",
            "value": 565.6,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 0.995"
          },
          {
            "name": "windows-x64 decode",
            "value": 19.36,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.021"
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
          "id": "8fa0d68a24cbfe189082adafbe3ff20534fbe62a",
          "message": "Release 0.4.4 (#104)",
          "timestamp": "2026-11-04T11:36:38Z",
          "tree_id": "d0fa8543f928ab04ae6714dc8af316b49336f12f",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/8fa0d68a24cbfe189082adafbe3ff20534fbe62a"
        },
        "date": 1793793496158,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 434.99,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 31.18,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.002"
          },
          {
            "name": "macos-arm64 embed",
            "value": 500.22,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.035"
          },
          {
            "name": "macos-arm64 decode",
            "value": 29.04,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.756"
          },
          {
            "name": "windows-x64 embed",
            "value": 468.71,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.994"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.61,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
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
          "id": "8fa0d68a24cbfe189082adafbe3ff20534fbe62a",
          "message": "Release 0.4.4 (#104)",
          "timestamp": "2026-11-04T11:36:38Z",
          "tree_id": "d0fa8543f928ab04ae6714dc8af316b49336f12f",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/8fa0d68a24cbfe189082adafbe3ff20534fbe62a"
        },
        "date": 1793824590308,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 440.91,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 24.39,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.003"
          },
          {
            "name": "macos-arm64 embed",
            "value": 415.67,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.007"
          },
          {
            "name": "macos-arm64 decode",
            "value": 24.71,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.047"
          },
          {
            "name": "windows-x64 embed",
            "value": 945.12,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.010"
          },
          {
            "name": "windows-x64 decode",
            "value": 37.33,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.006"
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
          "id": "4ab39abd7f100f04cc9158a2498676ba9e126162",
          "message": "Fix/qwen test prompts (#105)",
          "timestamp": "2026-11-04T22:34:43Z",
          "tree_id": "2c3404189f37412a18a344e79f3dbca021bd1363",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/4ab39abd7f100f04cc9158a2498676ba9e126162"
        },
        "date": 1793833041575,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 482.34,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.59,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 592.92,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.968"
          },
          {
            "name": "macos-arm64 decode",
            "value": 46.15,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.087"
          },
          {
            "name": "windows-x64 embed",
            "value": 474.55,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.74,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.002"
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
          "id": "df0230e2b86a9ec607b93d748f9dbb852fee2785",
          "message": "Add named-entity recognition and load models without tokenizer.json (#106)",
          "timestamp": "2026-11-05T08:13:41Z",
          "tree_id": "58f4b610d6526664546a2b67dd3c2dd6d65b63ec",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/df0230e2b86a9ec607b93d748f9dbb852fee2785"
        },
        "date": 1793867449609,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 485.32,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.007"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.62,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.004"
          },
          {
            "name": "macos-arm64 embed",
            "value": 429.58,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.026"
          },
          {
            "name": "macos-arm64 decode",
            "value": 25.06,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.763"
          },
          {
            "name": "windows-x64 embed",
            "value": 553.53,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.987"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.14,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.981"
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
          "id": "b60b812ba8bcab94fc9802da38b8b45cf63af64b",
          "message": "Chat with model folders and load SentencePiece tokenizers (#107)",
          "timestamp": "2026-11-05T08:53:14Z",
          "tree_id": "1b656fec6125a0716115d1a5a941dfbe22cac99e",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/b60b812ba8bcab94fc9802da38b8b45cf63af64b"
        },
        "date": 1793870249297,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "macos-arm64 embed",
            "value": 496.7,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.918"
          },
          {
            "name": "macos-arm64 decode",
            "value": 35.72,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.980"
          },
          {
            "name": "windows-x64 embed",
            "value": 468.58,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.45,
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
          "id": "def987750c99792f7be7734f417457a0f20d42e8",
          "message": "Benchmark: count a 99-word decode as a full run (#109)",
          "timestamp": "2026-11-05T17:05:59Z",
          "tree_id": "08b3da1ea21ee8230f4e2ec990d310b02b91eb73",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/def987750c99792f7be7734f417457a0f20d42e8"
        },
        "date": 1793899630748,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 436.32,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.002"
          },
          {
            "name": "linux-x64 decode",
            "value": 24.7,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 586.74,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.002"
          },
          {
            "name": "macos-arm64 decode",
            "value": 37.8,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.013"
          },
          {
            "name": "windows-x64 embed",
            "value": 475.01,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.93,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.004"
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
          "id": "08f21f683d4d6cc4e52ba262d90ac33d92f26883",
          "message": "Load summarize, translate and transcribe models from folders; 0.5.0 (#108)",
          "timestamp": "2026-11-08T10:51:17Z",
          "tree_id": "8353a82923d0640f9bf522f0a70f218c43114553",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/08f21f683d4d6cc4e52ba262d90ac33d92f26883"
        },
        "date": 1794135845901,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 493.82,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.9,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "macos-arm64 embed",
            "value": 605.66,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.012"
          },
          {
            "name": "macos-arm64 decode",
            "value": 38.73,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.965"
          },
          {
            "name": "windows-x64 embed",
            "value": 967.27,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 0.976"
          },
          {
            "name": "windows-x64 decode",
            "value": 37.96,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.015"
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
          "id": "6dc3dd5f2727418ab2b23681d9b771396e150d31",
          "message": "Split the ner module into a folder like the other tasks (#110)",
          "timestamp": "2026-11-08T14:59:42Z",
          "tree_id": "61b1d9c88d019b6913b478f69ae89bf44b059fa6",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/6dc3dd5f2727418ab2b23681d9b771396e150d31"
        },
        "date": 1794151006829,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 492.1,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.003"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.74,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "macos-arm64 embed",
            "value": 638.27,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.067"
          },
          {
            "name": "macos-arm64 decode",
            "value": 44.29,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.170"
          },
          {
            "name": "windows-x64 embed",
            "value": 557.1,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.997"
          },
          {
            "name": "windows-x64 decode",
            "value": 28.66,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.007"
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
          "id": "6dc3dd5f2727418ab2b23681d9b771396e150d31",
          "message": "Split the ner module into a folder like the other tasks (#110)",
          "timestamp": "2026-11-08T14:59:42Z",
          "tree_id": "61b1d9c88d019b6913b478f69ae89bf44b059fa6",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/6dc3dd5f2727418ab2b23681d9b771396e150d31"
        },
        "date": 1794707483434,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 434.37,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.997"
          },
          {
            "name": "linux-x64 decode",
            "value": 30.54,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.085"
          },
          {
            "name": "macos-arm64 embed",
            "value": 544.95,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.039"
          },
          {
            "name": "macos-arm64 decode",
            "value": 40.68,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.256"
          },
          {
            "name": "windows-x64 embed",
            "value": 556.98,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.006"
          },
          {
            "name": "windows-x64 decode",
            "value": 28.4,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
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
          "id": "973363112a1bcb0bfe88a93ec20295105d8e892d",
          "message": "Add zero-shot classification (#111)\n\nClassifier::zero_shot, zero_shot_with and zero_shot_batch pick among\nlabels given at run time with a natural-language-inference model, as\ntransformers' pipeline(\"zero-shot-classification\") does: each label goes\ninto a hypothesis (\"This example is {}.\"), and the scores are a softmax\nof the entailment logits across labels, or entailment against\ncontradiction per label with multi_label. SequenceClassifier gains\npredict_logits_pairs for the (text, hypothesis) pairs.\n\nThe registry gets minilm-zeroshot, cross-encoder/nli-MiniLM2-L6-H768\n(82M). On twelve product-review sentences it picked the right aspect for\neleven; xtremedistil-l6-h256-zeroshot (13M) got five.\n\nkjarni classify takes the candidates with --labels when the model is\nzero-shot, a registry zero-shot model or a folder with an entailment\nlabel, plus --hypothesis and --multi-label.\n\nScores are pinned to transformers 4.57 in float32 on the CPU.",
          "timestamp": "2026-11-15T13:12:21Z",
          "tree_id": "f024445dd7b5b804b5e36bf221c0d910abfbcdcd",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/973363112a1bcb0bfe88a93ec20295105d8e892d"
        },
        "date": 1794749656455,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 493.04,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.002"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.85,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.009"
          },
          {
            "name": "macos-arm64 embed",
            "value": 463.64,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.006"
          },
          {
            "name": "macos-arm64 decode",
            "value": 27.95,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.081"
          },
          {
            "name": "windows-x64 embed",
            "value": 466.94,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.004"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.37,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.997"
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
          "id": "26c464ce3867917261c1c5c1f851a4ff720c4657",
          "message": "Add kjarni serve, an OpenAI-compatible HTTP server; 0.5.1 (#112)\n\nChat, completions, embeddings, speech-to-text and the model list follow\nOpenAI's API, so its client libraries work by changing the base URL.\nReranking follows Cohere's and Jina's format; classification (including\nzero-shot), translation, summarization and entity recognition use Kjarni's\nown. Models load on their first request and stay loaded.\n\nThe Rust API gains what the server needed: Chat::render_messages and\nreply_prefix, Generator::count_tokens and max_new_tokens,\nEmbedder::count_tokens and Transcriber::with_settings.\n\nVersions move to 0.5.1, which also ships zero-shot classification.",
          "timestamp": "2026-11-15T16:12:59Z",
          "tree_id": "097a13aa8a9105c78779cf7acc836e08318ab04d",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/26c464ce3867917261c1c5c1f851a4ff720c4657"
        },
        "date": 1794760600202,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 493.02,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "linux-x64 decode",
            "value": 25.65,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 370.22,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.906"
          },
          {
            "name": "macos-arm64 decode",
            "value": 21.8,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.724"
          },
          {
            "name": "windows-x64 embed",
            "value": 470.15,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.994"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.84,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
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
          "id": "26c464ce3867917261c1c5c1f851a4ff720c4657",
          "message": "Add kjarni serve, an OpenAI-compatible HTTP server; 0.5.1 (#112)\n\nChat, completions, embeddings, speech-to-text and the model list follow\nOpenAI's API, so its client libraries work by changing the base URL.\nReranking follows Cohere's and Jina's format; classification (including\nzero-shot), translation, summarization and entity recognition use Kjarni's\nown. Models load on their first request and stay loaded.\n\nThe Rust API gains what the server needed: Chat::render_messages and\nreply_prefix, Generator::count_tokens and max_new_tokens,\nEmbedder::count_tokens and Transcriber::with_settings.\n\nVersions move to 0.5.1, which also ships zero-shot classification.",
          "timestamp": "2026-11-15T16:12:59Z",
          "tree_id": "097a13aa8a9105c78779cf7acc836e08318ab04d",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/26c464ce3867917261c1c5c1f851a4ff720c4657"
        },
        "date": 1794761792421,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 657.02,
            "unit": "sentences/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 0.982"
          },
          {
            "name": "linux-x64 decode",
            "value": 24.13,
            "unit": "tokens/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 0.993"
          },
          {
            "name": "macos-arm64 embed",
            "value": 398.61,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.959"
          },
          {
            "name": "macos-arm64 decode",
            "value": 27.26,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.267"
          },
          {
            "name": "windows-x64 embed",
            "value": 464.46,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.992"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.84,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.002"
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
          "id": "62fc5241e8e0fa1ac88d68f8c9a329311f3d3972",
          "message": "Add Qwen 3.5 (hybrid Gated DeltaNet) on the CPU (#114)\n\nQwen 3.5 interleaves three Gated DeltaNet layers with each gated\nfull-attention layer. This adds the architecture as qwen3_5, text only:\na multimodal checkpoint loads its decoder and skips the vision tower and\nMTP head.\n\n- Gated attention: sigmoid output gate, per-head q/k norm, partial\n  rotary (64 of 256 dims).\n- Gated DeltaNet: causal depthwise conv1d, l2-normalized q/k, gated\n  delta-rule recurrence, gated RMSNorm.\n- HybridCache: KV cache for attention layers plus conv window and\n  recurrent state per DeltaNet layer, with snapshots at each prefill end\n  and 16 tokens before it so the prefix cache can reuse them.\n- Cache::reusable_prefix lets a cache report how much of a prefix it can\n  keep; speculation refuses caches that cannot rewind.\n- Chat template matching transformers' rendering (non-thinking mode).\n- Registry: Qwen3.5-0.8B, 2B and 4B; folder detection for\n  Qwen3_5ForConditionalGeneration and Qwen3_5ForCausalLM; CLI and chat.\n\nTests: a tiny committed checkpoint checked against transformers (grouped\nvalue heads, untied head, text-only prefix), plus ignored parity tests\nfor 0.8B and 4B pinned to transformers 5.16.1 logits and greedy output.",
          "timestamp": "2026-11-15T20:04:52Z",
          "tree_id": "612ad7d8a235ae94b3ee66c9804fd3bc55032ad2",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/62fc5241e8e0fa1ac88d68f8c9a329311f3d3972"
        },
        "date": 1794774031423,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 437.54,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.997"
          },
          {
            "name": "linux-x64 decode",
            "value": 30.38,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.982"
          },
          {
            "name": "macos-arm64 embed",
            "value": 501.5,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.142"
          },
          {
            "name": "macos-arm64 decode",
            "value": 33.56,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.972"
          },
          {
            "name": "windows-x64 embed",
            "value": 991.57,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 0.997"
          },
          {
            "name": "windows-x64 decode",
            "value": 37.45,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 0.978"
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
          "id": "7f4514fcbdd75336ab3f6df1ae57a62c2194ef7b",
          "message": "Load Qwen 3.5 from GGUF; add Q5_K; commit the tiny test checkpoint (#115)\n\n* Load Qwen 3.5 from GGUF; add Q5_K; commit the tiny test checkpoint\n\nQwen 3.5 GGUFs written by llama.cpp's converter now load, from the\nregistry with --gguf (unsloth's Q4_K_M for 0.8B, 2B and 4B) or with\n--model-path.\n\n- Name mapping for the Gated DeltaNet tensors (attn_qkv, attn_gate,\n  ssm_alpha, ssm_beta, ssm_conv1d, ssm_a, ssm_dt, ssm_norm, ssm_out) and\n  Qwen 3.5's post_attention_norm and q/k norms.\n- The converter's changes are undone where they matter: norms already\n  carry their +1, ssm_a is -exp(A_log), conv1d is 2-D, and value heads\n  are tiled, so head h reads key head h % num_key_heads.\n- Newer files that merge attn_qkv with attn_gate and ssm_beta with\n  ssm_alpha load through row slices of the merged tensor.\n- The config comes from qwen35.* metadata, including recurrent_layers,\n  and leaves out the multi-token-prediction block.\n\nQ5_K: block type, dequantization matching ggml, an integer dot product\nagainst Q8_K activations, and the matmul. Every Q4_K_M Qwen 3.5 file\nstores some tensors as Q5_K. A Q5_K tensor on the GPU is an error.\n\nGGUF loading: the tensor data start is taken from the end of the last\ntensor rather than the sum of sizes, which fell short whenever tensors\nwere padded.\n\nThe tiny Qwen 3.5 checkpoint's model.safetensors was ignored by\n*.safetensors, so CI could not load it; test fixtures are now excluded\nfrom that rule.\n\nTests: Q5_K dequantization pinned to gguf-py on a real block; tiny.gguf\n(make_gguf.py) against the transformers values, covering tiled heads,\nmerged tensors and recurrent_layers; ignored Q4_K_M tests on 0.8B and 4B\nagainst the f32 transformers greedy output.\n\n* Build Qwen 3.5's GGUF config on wasm, where GgufLoader is compiled out",
          "timestamp": "2026-11-15T22:23:21Z",
          "tree_id": "a62cbcdd9b5fbe745492ecb07cf8ab1ad5cee93d",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/7f4514fcbdd75336ab3f6df1ae57a62c2194ef7b"
        },
        "date": 1794782774267,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 634.26,
            "unit": "sentences/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 0.994"
          },
          {
            "name": "linux-x64 decode",
            "value": 26.06,
            "unit": "tokens/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 1.007"
          },
          {
            "name": "macos-arm64 embed",
            "value": 376.57,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 decode",
            "value": 20.32,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.946"
          },
          {
            "name": "windows-x64 embed",
            "value": 466.68,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.004"
          },
          {
            "name": "windows-x64 decode",
            "value": 20.44,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.991"
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
          "id": "9204e947962c00d7ce71ee230906c1534a113901",
          "message": "Add kjarni agent, tool calling, and faster K-quant kernels (#116)\n\n- kjarni agent: a terminal assistant with file, shell, web and memory\n  tools, approvals for writes/commands/network, saved sessions\n  (-c, --resume), one-shot -p, and summary folding past 16k tokens.\n  The agent is also a library module, kjarni::agent, with a Tool trait.\n- Tool calling: Qwen 3.5's template renders tools, calls and results as\n  transformers does; parse_tool_calls reads Qwen 3.5 XML and Qwen 2.5/3\n  JSON calls. Jinja templates that read `tools` receive them.\n- kjarni serve: /v1/chat/completions takes tools, tool_calls and tool\n  messages and returns tool_calls, streamed or not; tool_choice auto/none.\n- AVX2 Q5_K and Q6_K dot products and integer Q4_K accumulation, bit-exact\n  with the scalar code; a faster Gated DeltaNet recurrence.\n- GGUF K-quant token tables stay quantized: the 4B peaks at 4.0 GB, not 6.0.",
          "timestamp": "2026-11-16T08:21:29Z",
          "tree_id": "796af0b07f3565db3443de0aab7de7fd51a64ff6",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/9204e947962c00d7ce71ee230906c1534a113901"
        },
        "date": 1794818671576,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 443.92,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.997"
          },
          {
            "name": "linux-x64 decode",
            "value": 35.77,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.454"
          },
          {
            "name": "macos-arm64 embed",
            "value": 659.69,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.056"
          },
          {
            "name": "macos-arm64 decode",
            "value": 44.54,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.976"
          },
          {
            "name": "windows-x64 embed",
            "value": 687.84,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 0.994"
          },
          {
            "name": "windows-x64 decode",
            "value": 28.94,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 1.338"
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
          "id": "0d4e5a7ddf49a8584f258769ad3f48035528b5c3",
          "message": "Sort the Embeddings import in loaders/embeddings.rs (cargo fmt) (#117)",
          "timestamp": "2026-11-16T08:41:14Z",
          "tree_id": "5c2e37ed06e5a9e6f1f07a5474af1d75b1f14702",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/0d4e5a7ddf49a8584f258769ad3f48035528b5c3"
        },
        "date": 1794819747363,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 657.93,
            "unit": "sentences/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 29.88,
            "unit": "tokens/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 1.009"
          },
          {
            "name": "macos-arm64 embed",
            "value": 559.86,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.927"
          },
          {
            "name": "macos-arm64 decode",
            "value": 38.14,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.930"
          },
          {
            "name": "windows-x64 embed",
            "value": 608.86,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 0.995"
          },
          {
            "name": "windows-x64 decode",
            "value": 26.75,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 0.999"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "wsj040303@gmail.com",
            "name": "ledd",
            "username": "wsj040303"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d36729167163b0243831b770c144326e2eb96eea",
          "message": "Document index write-lock limits on network drives (#113)\n\n* docs: explain index write locks on network drives\n\n* docs: clarify cross-machine index locking\n\n---------\n\nCo-authored-by: wsj <279978307+wsj040303@users.noreply.github.com>",
          "timestamp": "2026-11-16T10:26:49Z",
          "tree_id": "0f49925f1a3c386ab545518b455062e6f2936ba3",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/d36729167163b0243831b770c144326e2eb96eea"
        },
        "date": 1794826387274,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 432.98,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.997"
          },
          {
            "name": "linux-x64 decode",
            "value": 44.31,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.994"
          },
          {
            "name": "macos-arm64 embed",
            "value": 354.23,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.887"
          },
          {
            "name": "macos-arm64 decode",
            "value": 23.29,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.985"
          },
          {
            "name": "windows-x64 embed",
            "value": 469.13,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.41,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.979"
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
          "id": "b3328a13d71a6219c11aab1076e7f39b204c295b",
          "message": "Tests/coverage (#118)\n\n* Add tests where coverage was thin; fix config overrides and a serve error\n\nCoverage over every test, ignored ones included, run single-threaded:\n84.2% of lines, 81.9% of production lines. Tests added where it was thin:\n\n- kjarni config: defaults, task settings, per-model overrides, load\n  settings, dtype parsing, loading files and the cache dir (was 0%).\n- agent tool helpers: default summary, argument parsing, access names.\n- agent on a real model: qwen3.5-0.8b reads a file and answers from it,\n  through Chat's AgentModel implementation (was 0%).\n- agent CLI slash commands: quit, clear, forget, sessions (was 0%).\n- serve: streamed /v1/completions (never tested), context overflow,\n  tool-call argument forms, tool and tool_choice validation.\n- ffi: cancel tokens and the reranker, with results matching the library\n  (both were 0%).\n\nFixes found while reading uncovered code:\n\n- [models.\"qwen3.5-4b\"] in kjarni.toml never matched; only the dashed\n  key did. Both forms now match.\n- A model's quantize_embeddings setting was parsed and ignored.\n- /v1/completions named `messages` instead of `prompt` on a context\n  overflow.\n\nRemoves streaming_prefill_matches_and_is_faster: its env switch is read\nonce per process, so it compared the streaming path with itself.\ndecoder_attention's own test compares the two paths.\n\n* Test Phi-3 and Mistral against transformers; fix LongRoPE past the trained context\n\n- Tiny random Phi-3 and Mistral checkpoints (about 460 KB each) with\n  what transformers computes on them, and the make.py that writes both.\n  A shared check compares every logit of a short prompt, the last logits\n  of a long one, and greedy continuations. Both model files were at 0%.\n- Phi-3.5 past 4,096 tokens: LongRoPE now uses its long factors for a\n  forward pass that reaches past the trained context, as transformers\n  decides per pass; Kjarni used the short factors at every length. Phi-3\n  keeps original_max_position_embeddings at the top level of config.json\n  and it was not passed to RoPE. The CPU switches; the GPU path still\n  uses the short factors.\n- FFI indexer and searcher through the C interface: create with progress\n  and cancellation, add, info, semantic, hybrid and keyword search,\n  delete, and null handling. Both files were at 0%.\n- CLI tests that run the kjarni binary: help, version, error messages,\n  model search and info, inspect on GGUF and safetensors, and generate on\n  the tiny Mistral, token for token what transformers gives.",
          "timestamp": "2026-11-16T12:36:48Z",
          "tree_id": "c4906194d9dbb34a7b6dd3f8def4c5c791fcfc23",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/b3328a13d71a6219c11aab1076e7f39b204c295b"
        },
        "date": 1794833518554,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 488.02,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.997"
          },
          {
            "name": "linux-x64 decode",
            "value": 35.87,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 525.41,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.956"
          },
          {
            "name": "macos-arm64 decode",
            "value": 33.83,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.986"
          },
          {
            "name": "windows-x64 embed",
            "value": 1002.56,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.003"
          },
          {
            "name": "windows-x64 decode",
            "value": 55.57,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.039"
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
          "id": "64af1b57868697a89a88d12bdb44b05e98b229d5",
          "message": "Qwen 3.5 mixture of experts, thinking mode, 9B and 27B (#119)\n\n- Qwen 3.5 MoE (qwen3.5-35b-a3b): top-k softmax router over the experts plus\n  a sigmoid-gated shared expert in every layer. Experts load from\n  transformers' per-expert or fused tensors and from llama.cpp's stacked\n  ffn_*_exps; a tiny Qwen3_5Moe checkpoint matches transformers to 1e-4 from\n  safetensors and GGUF, greedy output included.\n- One-token MoE decode hands every chosen expert's rows to the threads in one\n  dispatch, through LinearLayer::rows_into and the existing per-row kernels.\n- ModelWeights reads name[start:end] row ranges of stacked tensors.\n- Thinking mode: chat --think, ChatBuilder::thinking, Conversation::set_thinking;\n  serve takes enable_thinking / reasoning_effort and returns reasoning_content.\n- The DeltaNet convolution has a one-token path: Qwen3.5-0.8B decodes 28.8 ->\n  31.8 tok/s. qwen2.5-0.5b and llama3.2-1b unchanged (A/B against main).\n- qwen3.5-9b and qwen3.5-27b registry entries.\n- generate --gguf no longer downloads the safetensors.",
          "timestamp": "2026-11-16T16:09:07Z",
          "tree_id": "a88e3cfd0e88772b11c13f70252666de88e7ba6c",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/64af1b57868697a89a88d12bdb44b05e98b229d5"
        },
        "date": 1794846805487,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 437.04,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "linux-x64 decode",
            "value": 45.14,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.003"
          },
          {
            "name": "macos-arm64 embed",
            "value": 691.99,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.064"
          },
          {
            "name": "macos-arm64 decode",
            "value": 35.85,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 0.979"
          },
          {
            "name": "windows-x64 embed",
            "value": 463.74,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.997"
          },
          {
            "name": "windows-x64 decode",
            "value": 26.25,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.977"
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
          "id": "1a59b6f36a30f845a3b1d6cf4e43fa0a10d5cc15",
          "message": "Load models straight from the Hugging Face Hub (#120)",
          "timestamp": "2026-11-16T19:28:58Z",
          "tree_id": "fdbad28652471caa7dd593f0589ccc40c8f0acdb",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/1a59b6f36a30f845a3b1d6cf4e43fa0a10d5cc15"
        },
        "date": 1794858503627,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 663.94,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.009"
          },
          {
            "name": "linux-x64 decode",
            "value": 60.01,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.988"
          },
          {
            "name": "macos-arm64 embed",
            "value": 379.85,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.944"
          },
          {
            "name": "macos-arm64 decode",
            "value": 21.76,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.979"
          },
          {
            "name": "windows-x64 embed",
            "value": 558.68,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 0.999"
          },
          {
            "name": "windows-x64 decode",
            "value": 39.27,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.009"
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
          "id": "89b05dea668f2cce858ed21bbbd1203e5960d27f",
          "message": "The agent searches documents by meaning; 0.6.0 (#121)\n\n* The agent searches documents by meaning; 0.6.0\n\nsearch_documents finds the passages in the working folder, or a folder\ninside it, that answer a question: hybrid search, reranked by\nminilm-l6-v2-cross-encoder. A folder is indexed on its first search and\nonly changed files are embedded after that; a folder inside an indexed\none is searched through that index. A folder with more than 10,000\ndocuments is refused with its largest subfolders, so the model narrows\nthe search instead of indexing a whole home folder.\n\nThe indexing moves from the MCP server into kjarni::folder_search, which\nsearch_folder and the agent now share; indexes live in folder-indexes in\nthe model cache.\n\nVersions move to 0.6.0.\n\n* Keep the agent's prompt cache after remember\n\nThe system prompt listed the newest memory facts at every step, so a\nremember call changed the top of the prompt and the next step prefilled\nthe whole conversation again. The facts are now read when a conversation\nstarts or is summarized; facts saved in between are already in the\nconversation. A forgotten fact still leaves the prompt at once.\n\nA list, remember and answer run with qwen3.5-4b on four cores went from\n271 s to 112 s. The prefix cache logs how many prompt tokens it reused,\nat debug level.\n\n* Print paths with display() in the folder_search example",
          "timestamp": "2026-11-16T21:18:54Z",
          "tree_id": "ca88dd2b8fa5af1cb117f2c0e8ca0b851d25e496",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/89b05dea668f2cce858ed21bbbd1203e5960d27f"
        },
        "date": 1794865115261,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 440.32,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 45.17,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.996"
          },
          {
            "name": "macos-arm64 embed",
            "value": 652.22,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.997"
          },
          {
            "name": "macos-arm64 decode",
            "value": 42.85,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.878"
          },
          {
            "name": "windows-x64 embed",
            "value": 560.25,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.017"
          },
          {
            "name": "windows-x64 decode",
            "value": 38.65,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.014"
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
          "id": "89b05dea668f2cce858ed21bbbd1203e5960d27f",
          "message": "The agent searches documents by meaning; 0.6.0 (#121)\n\n* The agent searches documents by meaning; 0.6.0\n\nsearch_documents finds the passages in the working folder, or a folder\ninside it, that answer a question: hybrid search, reranked by\nminilm-l6-v2-cross-encoder. A folder is indexed on its first search and\nonly changed files are embedded after that; a folder inside an indexed\none is searched through that index. A folder with more than 10,000\ndocuments is refused with its largest subfolders, so the model narrows\nthe search instead of indexing a whole home folder.\n\nThe indexing moves from the MCP server into kjarni::folder_search, which\nsearch_folder and the agent now share; indexes live in folder-indexes in\nthe model cache.\n\nVersions move to 0.6.0.\n\n* Keep the agent's prompt cache after remember\n\nThe system prompt listed the newest memory facts at every step, so a\nremember call changed the top of the prompt and the next step prefilled\nthe whole conversation again. The facts are now read when a conversation\nstarts or is summarized; facts saved in between are already in the\nconversation. A forgotten fact still leaves the prompt at once.\n\nA list, remember and answer run with qwen3.5-4b on four cores went from\n271 s to 112 s. The prefix cache logs how many prompt tokens it reused,\nat debug level.\n\n* Print paths with display() in the folder_search example",
          "timestamp": "2026-11-16T21:18:54Z",
          "tree_id": "ca88dd2b8fa5af1cb117f2c0e8ca0b851d25e496",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/89b05dea668f2cce858ed21bbbd1203e5960d27f"
        },
        "date": 1794867561775,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 440.96,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.994"
          },
          {
            "name": "linux-x64 decode",
            "value": 35.17,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.437"
          },
          {
            "name": "macos-arm64 embed",
            "value": 674.26,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 0.961"
          },
          {
            "name": "macos-arm64 decode",
            "value": 39.43,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.100"
          },
          {
            "name": "windows-x64 embed",
            "value": 526.95,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.003"
          },
          {
            "name": "windows-x64 decode",
            "value": 24.51,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.348"
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
          "id": "c6ae2f4ec2524b74325d98308fb0f5cbc7e1b0c9",
          "message": "Q5_K stays packed on the GPU (#122)\n\n* The agent reuses its prompt cache across messages\n\nQwen 3.5's template drops think blocks from turns before the last user\nmessage, so each new message changed the prompt from the previous reply on\nand the model re-read it. Conversation::set_preserve_thinking keeps those\nblocks; the agent sets it, chat keeps the model's own rendering.\n\n* The agent reads project notes from KJARNI.md or AGENTS.md\n\nFrom the working folder and each folder above it up to the repository\nroot, outermost first, at most 16 KB, into the system prompt.\n\n* Qwen 3.5 on the GPU\n\nThe dense models run on wgpu: the shared projections, norms and SwiGLU,\nand kernels for what only Qwen 3.5 has (per-head norm and partial rotary\non q and k, the output gate, the causal convolution, the delta rule and\nthe gated norm). GpuKVCache can carry recurrent states with snapshots,\nso prefix reuse works as on the CPU. The attention's scores, softmax and\noutput are now callable on their own as GpuRoPEAttention::attend, and\nits kernel helper is the public ComputeKernel. kjarni agent takes --gpu.\n\n* Q5_K stays packed on the GPU\n\nThe linear layer reads Q5_K blocks in the shader: a GEMV that does eight\nrows per workgroup like the Q4_K one, and tile loads for both GEMMs. The\n176-byte block is word-aligned, so it uploads as stored. Before, Q5_K\ntensors were expanded to BF16 on upload.",
          "timestamp": "2026-11-17T07:50:15Z",
          "tree_id": "6c51db7453f913d2d6e6eacf79863e6022af1c71",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/c6ae2f4ec2524b74325d98308fb0f5cbc7e1b0c9"
        },
        "date": 1794903172849,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 490.79,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.13,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "macos-arm64 embed",
            "value": 319.6,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.044"
          },
          {
            "name": "macos-arm64 decode",
            "value": 18.51,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.006"
          },
          {
            "name": "windows-x64 embed",
            "value": 475.76,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.005"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.94,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
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
          "id": "dd7b484570158d8f94510046861f301be8b076de",
          "message": "Feature/daemon (#123)",
          "timestamp": "2026-11-17T11:14:48Z",
          "tree_id": "718bc36c59d89cc6dd7c57d3b93541679ef584c3",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/dd7b484570158d8f94510046861f301be8b076de"
        },
        "date": 1794915505511,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 673.17,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.008"
          },
          {
            "name": "linux-x64 decode",
            "value": 63.32,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.007"
          },
          {
            "name": "macos-arm64 embed",
            "value": 710.51,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.036"
          },
          {
            "name": "macos-arm64 decode",
            "value": 40.69,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.073"
          },
          {
            "name": "windows-x64 embed",
            "value": 470.81,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.002"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.35,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.999"
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
          "id": "fac06530138eb0ae1262b0373857b5a396e426dd",
          "message": "Feature/web UI (#124)",
          "timestamp": "2026-11-17T12:44:35Z",
          "tree_id": "5777cf5c1d647db2cd5c26fb98721fd859f75c86",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/fac06530138eb0ae1262b0373857b5a396e426dd"
        },
        "date": 1794920890079,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 495.19,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.09,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.987"
          },
          {
            "name": "macos-arm64 embed",
            "value": 710.53,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 0.860"
          },
          {
            "name": "macos-arm64 decode",
            "value": 41.28,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 0.854"
          },
          {
            "name": "windows-x64 embed",
            "value": 463.53,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.989"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.44,
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
          "id": "87c1edad6c8aec4e89c1b6a712e0c0b6e6b5c475",
          "message": "Schedules and ntfy notifications in the daemon (#125)",
          "timestamp": "2026-11-17T18:53:42Z",
          "tree_id": "0046752d0c5eefcf6366029d23f5c3d86c9f9a3b",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/87c1edad6c8aec4e89c1b6a712e0c0b6e6b5c475"
        },
        "date": 1794943028406,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 494.94,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.004"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.18,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.998"
          },
          {
            "name": "macos-arm64 embed",
            "value": 480.36,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.110"
          },
          {
            "name": "macos-arm64 decode",
            "value": 23.41,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.659"
          },
          {
            "name": "windows-x64 embed",
            "value": 913.97,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.024"
          },
          {
            "name": "windows-x64 decode",
            "value": 50.66,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.010"
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
          "id": "748159850042e2db1053e1fbafd70d6e69f60e93",
          "message": "Switch the daemon's model from the web UI, and a Memory page (#126)\n\n* Schedules and ntfy notifications in the daemon\n\nA schedule is a saved message the daemon sends itself at set times, daily\nat a time (optionally on chosen weekdays) or every so many minutes or\nhours. Each run is a new conversation. With nobody there to approve calls,\na schedule lists the tools that may run; any other call that needs\napproval is declined with a notice, and read-only tools always run. A run\nmissed while the computer was off happens once on the next check.\n\nWhen a run ends, the daemon can publish its reply to an ntfy topic the\nuser set, with an optional access token, linking back to the run's\nconversation. The token stays in the daemon; clients only see that one is\nset. Schedules and settings live in schedules.json in the daemon's folder,\nreadable by the user only.\n\nThe web UI gets a Schedules page: the list with next and last runs, a form\nfor making and editing schedules, and the notification settings with a\ntest button.\n\n* Declare rust-version 1.90; compare translation on the same model\n\nwgpu pulls in ordered-float 5.5, which needs Rust 1.90, so an older\ncompiler failed inside a dependency with no word about kjarni's own\nrequirement. Every crate now inherits rust-version = \"1.90\" from the\nworkspace; the workspace checks clean on 1.90.0.\n\nThe seq2seq benchmark gave PyTorch flan-t5-base while Kjarni's Translator,\nsince it began picking Opus-MT for English to German, ran a different\nmodel, so the timing and the same-output check compared two models.\nKjarni now names flan-t5-base; both engines give the same German again.\n\nThe README's Python example prints POSITIVE, as the label comes back.\n\n* Version 0.7.0\n\n* daemon run downloads its models with progress before loading\n\n`daemon install` fetched a missing model first, where the progress shows,\nbut `daemon run` left it to the loader, which downloads quietly, so a first\nrun sat at \"Loading qwen3.5-4b...\" for the whole download. Both now go\nthrough one step that downloads the chat model with progress and the memory\nmodel quietly, whichever the cache lacks.\n\n* Switch the daemon's model from the web UI, and a Memory page\n\nThe Settings page lists the Qwen 3.5 models the agent can run and switches\nbetween them while the daemon runs: it downloads the new model if needed,\nwaits for replies in progress, holds new messages and schedules, unloads the\nold model and loads the new one, falling back to the old one if the load\nfails. The choice is saved in settings.json in the daemon's folder and loaded\nat the next start unless --model is given; install saves its model there\ninstead of passing --model in the startup entry.\n\nThe Memory page lists, searches, adds and forgets memories through the\nexisting /api/memory routes.",
          "timestamp": "2026-11-17T20:02:12Z",
          "tree_id": "b3a07e1361ef23aa0323fb0a57122c5b5567f37c",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/748159850042e2db1053e1fbafd70d6e69f60e93"
        },
        "date": 1794947189609,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 493.93,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.31,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "macos-arm64 embed",
            "value": 491.62,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.184"
          },
          {
            "name": "macos-arm64 decode",
            "value": 36.25,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.416"
          },
          {
            "name": "windows-x64 embed",
            "value": 468.94,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.006"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.07,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.995"
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
          "id": "748159850042e2db1053e1fbafd70d6e69f60e93",
          "message": "Switch the daemon's model from the web UI, and a Memory page (#126)\n\n* Schedules and ntfy notifications in the daemon\n\nA schedule is a saved message the daemon sends itself at set times, daily\nat a time (optionally on chosen weekdays) or every so many minutes or\nhours. Each run is a new conversation. With nobody there to approve calls,\na schedule lists the tools that may run; any other call that needs\napproval is declined with a notice, and read-only tools always run. A run\nmissed while the computer was off happens once on the next check.\n\nWhen a run ends, the daemon can publish its reply to an ntfy topic the\nuser set, with an optional access token, linking back to the run's\nconversation. The token stays in the daemon; clients only see that one is\nset. Schedules and settings live in schedules.json in the daemon's folder,\nreadable by the user only.\n\nThe web UI gets a Schedules page: the list with next and last runs, a form\nfor making and editing schedules, and the notification settings with a\ntest button.\n\n* Declare rust-version 1.90; compare translation on the same model\n\nwgpu pulls in ordered-float 5.5, which needs Rust 1.90, so an older\ncompiler failed inside a dependency with no word about kjarni's own\nrequirement. Every crate now inherits rust-version = \"1.90\" from the\nworkspace; the workspace checks clean on 1.90.0.\n\nThe seq2seq benchmark gave PyTorch flan-t5-base while Kjarni's Translator,\nsince it began picking Opus-MT for English to German, ran a different\nmodel, so the timing and the same-output check compared two models.\nKjarni now names flan-t5-base; both engines give the same German again.\n\nThe README's Python example prints POSITIVE, as the label comes back.\n\n* Version 0.7.0\n\n* daemon run downloads its models with progress before loading\n\n`daemon install` fetched a missing model first, where the progress shows,\nbut `daemon run` left it to the loader, which downloads quietly, so a first\nrun sat at \"Loading qwen3.5-4b...\" for the whole download. Both now go\nthrough one step that downloads the chat model with progress and the memory\nmodel quietly, whichever the cache lacks.\n\n* Switch the daemon's model from the web UI, and a Memory page\n\nThe Settings page lists the Qwen 3.5 models the agent can run and switches\nbetween them while the daemon runs: it downloads the new model if needed,\nwaits for replies in progress, holds new messages and schedules, unloads the\nold model and loads the new one, falling back to the old one if the load\nfails. The choice is saved in settings.json in the daemon's folder and loaded\nat the next start unless --model is given; install saves its model there\ninstead of passing --model in the startup entry.\n\nThe Memory page lists, searches, adds and forgets memories through the\nexisting /api/memory routes.",
          "timestamp": "2026-11-17T20:02:12Z",
          "tree_id": "b3a07e1361ef23aa0323fb0a57122c5b5567f37c",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/748159850042e2db1053e1fbafd70d6e69f60e93"
        },
        "date": 1794956673729,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 650.59,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "linux-x64 decode",
            "value": 60.11,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.995"
          },
          {
            "name": "macos-arm64 embed",
            "value": 452.28,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.874"
          },
          {
            "name": "macos-arm64 decode",
            "value": 28.91,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.953"
          },
          {
            "name": "windows-x64 embed",
            "value": 471.86,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.997"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.97,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
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
          "id": "ac9d447bfee0574533177a1df921f7840a0decdd",
          "message": "Decline with a reason, a changed-files panel, and transcribe and photo search for the agent (#127)\n\nDeclining a call can say what to do instead, which the model is told\n(Approval::DenyWith, the approval API's reason, a prompt after no in\nkjarni agent). Tools say which file they write and the agent reports the\ncontent before and after; the daemon keeps each conversation's changes\nbeside it and serves their diffs, shown in a panel in the web UI. The\nagent gains transcribe and search_images, sharing their code with the\nMCP server.",
          "timestamp": "2026-11-18T11:25:52Z",
          "tree_id": "e85f865dcfd82b4dd191841549c818b4e025374a",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/ac9d447bfee0574533177a1df921f7840a0decdd"
        },
        "date": 1795002546139,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 685.68,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 62.79,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.972"
          },
          {
            "name": "macos-arm64 embed",
            "value": 505.04,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.940"
          },
          {
            "name": "macos-arm64 decode",
            "value": 32.35,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.029"
          },
          {
            "name": "windows-x64 embed",
            "value": 575.27,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 1.000"
          },
          {
            "name": "windows-x64 decode",
            "value": 25.73,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 1.002"
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
          "id": "92553ec7500cb07e05e323dba3f396e782252f26",
          "message": "Messages sent mid-turn, device sign-in codes, and a context meter (#128)",
          "timestamp": "2026-11-18T16:59:12Z",
          "tree_id": "143846f20931df6ebb3962cc23f7118c100da5cc",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/92553ec7500cb07e05e323dba3f396e782252f26"
        },
        "date": 1795022613043,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 440.42,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.007"
          },
          {
            "name": "linux-x64 decode",
            "value": 45.29,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "windows-x64 embed",
            "value": 434.7,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
          },
          {
            "name": "windows-x64 decode",
            "value": 29.54,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
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
          "id": "92553ec7500cb07e05e323dba3f396e782252f26",
          "message": "Messages sent mid-turn, device sign-in codes, and a context meter (#128)",
          "timestamp": "2026-11-18T16:59:12Z",
          "tree_id": "143846f20931df6ebb3962cc23f7118c100da5cc",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/92553ec7500cb07e05e323dba3f396e782252f26"
        },
        "date": 1795031579831,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 440.42,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.007"
          },
          {
            "name": "linux-x64 decode",
            "value": 45.29,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.005"
          },
          {
            "name": "macos-arm64 embed",
            "value": 567.91,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.901"
          },
          {
            "name": "macos-arm64 decode",
            "value": 43.06,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.985"
          },
          {
            "name": "windows-x64 embed",
            "value": 434.7,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
          },
          {
            "name": "windows-x64 decode",
            "value": 29.54,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.000"
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
          "id": "5383560be3e90f1e67ed027f7f593a5148c3f3c9",
          "message": "Benchmark install size and cold start; (#129)",
          "timestamp": "2026-11-19T14:04:33Z",
          "tree_id": "f483554cb1dcc6d37cc9fe8e8eae20ab46ab5189",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/5383560be3e90f1e67ed027f7f593a5148c3f3c9"
        },
        "date": 1795098166913,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 526.1,
            "unit": "sentences/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 1.007"
          },
          {
            "name": "linux-x64 decode",
            "value": 30.93,
            "unit": "tokens/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 1.006"
          },
          {
            "name": "macos-arm64 embed",
            "value": 467.14,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.089"
          },
          {
            "name": "macos-arm64 decode",
            "value": 30.95,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.956"
          },
          {
            "name": "windows-x64 embed",
            "value": 974.35,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.027"
          },
          {
            "name": "windows-x64 decode",
            "value": 54.64,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.003"
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
          "id": "5383560be3e90f1e67ed027f7f593a5148c3f3c9",
          "message": "Benchmark install size and cold start; (#129)",
          "timestamp": "2026-11-19T14:04:33Z",
          "tree_id": "f483554cb1dcc6d37cc9fe8e8eae20ab46ab5189",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/5383560be3e90f1e67ed027f7f593a5148c3f3c9"
        },
        "date": 1795115252618,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 491.49,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.998"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.14,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.001"
          },
          {
            "name": "macos-arm64 embed",
            "value": 438.82,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.012"
          },
          {
            "name": "macos-arm64 decode",
            "value": 25.7,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.942"
          },
          {
            "name": "windows-x64 embed",
            "value": 554.84,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.017"
          },
          {
            "name": "windows-x64 decode",
            "value": 24.24,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 0.995"
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
          "id": "54c3e66f09920d4608e7aa5cad29eb1feabc43d5",
          "message": "Attend to long GPU prompts in slices; end a turn whose model panics with an error (#131)",
          "timestamp": "2026-11-19T22:19:15Z",
          "tree_id": "d4edea072e81806f58d8da58640d141cd942510d",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/54c3e66f09920d4608e7aa5cad29eb1feabc43d5"
        },
        "date": 1795128239481,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 655.36,
            "unit": "sentences/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 1.001"
          },
          {
            "name": "linux-x64 decode",
            "value": 30.43,
            "unit": "tokens/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 0.993"
          },
          {
            "name": "macos-arm64 embed",
            "value": 1058.33,
            "unit": "sentences/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 1.016"
          },
          {
            "name": "macos-arm64 decode",
            "value": 58.38,
            "unit": "tokens/s",
            "extra": "CPU: Apple M2 Pro (Virtual); vs previous commit: 0.951"
          },
          {
            "name": "windows-x64 embed",
            "value": 456.96,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.48,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
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
          "id": "fb8574885914c25b4213c7196711e4435c07e907",
          "message": "Prefill on the GPU 512 tokens at a time (#132)",
          "timestamp": "2026-11-20T09:13:28Z",
          "tree_id": "4ac75d323a51e7c665f6379846a4584b85222bb1",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/fb8574885914c25b4213c7196711e4435c07e907"
        },
        "date": 1795167361786,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 436.18,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 44.32,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "macos-arm64 embed",
            "value": 399.56,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.042"
          },
          {
            "name": "macos-arm64 decode",
            "value": 32.68,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.205"
          },
          {
            "name": "windows-x64 embed",
            "value": 553.84,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.005"
          },
          {
            "name": "windows-x64 decode",
            "value": 38.7,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.021"
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
          "id": "79b09dece5517c49788cf4c537b500ba9b5d5822",
          "message": "Perf/gpu decode (#133)",
          "timestamp": "2026-11-22T15:53:25Z",
          "tree_id": "dd72b44078789c857e38121b7fff1d7c3fea9375",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/79b09dece5517c49788cf4c537b500ba9b5d5822"
        },
        "date": 1795364292335,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 672.81,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.002"
          },
          {
            "name": "linux-x64 decode",
            "value": 63.01,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.030"
          },
          {
            "name": "macos-arm64 embed",
            "value": 321.18,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.911"
          },
          {
            "name": "macos-arm64 decode",
            "value": 19.23,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.970"
          },
          {
            "name": "windows-x64 embed",
            "value": 468.97,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 0.993"
          },
          {
            "name": "windows-x64 decode",
            "value": 27.61,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 1 Stepping 1, AuthenticAMD; vs previous commit: 1.006"
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
          "id": "821bbe6f8a5d6616a519e0cf82a3d602e2147cda",
          "message": "Transcribe the speech after a Whisper window that ends early (#134)\n\nWhisper can end a 30-second window before its audio does. On a LibriVox\nreading of the Gettysburg Address, whisper-small closed the first window\nafter the LibriVox notice and the transcript went straight from it to \"a new\nnation, conceived in liberty\", missing the title, the date and \"Four score\nand seven years ago our fathers brought forth on this continent\". Later in\nthe same recording \"in vain; that this nation, under God, shall have a new\nbirth of freedom; and that\" was missing the same way.\n\nopenai-whisper moves on from the last timestamp a window reached, so it\ndecodes that audio again. Kjarni decodes fixed windows in a batch, so a window\nthat stopped early lost its remainder. Now, when a window's transcript ends\nmore than a second before its audio and the rest is not silence, the rest is\ndecoded as a window of its own, up to four times. This applies to transcribe\nand to streaming with timestamps. Without timestamps a window is one segment\nthat spans it, so it never ends early.\n\nTranscripts of the JFK sample, the same with 8 s of trailing or 4 s of leading\nsilence, and the 55 s benchmark clip are unchanged.\n\nAdds a 40-second public-domain fixture and tests that fail without the change.",
          "timestamp": "2026-11-23T08:20:17Z",
          "tree_id": "dd72b44078789c857e38121b7fff1d7c3fea9375",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/821bbe6f8a5d6616a519e0cf82a3d602e2147cda"
        },
        "date": 1795423506732,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 437.7,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 44.7,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V74 80-Core Processor; vs previous commit: 1.006"
          },
          {
            "name": "macos-arm64 embed",
            "value": 567.34,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.011"
          },
          {
            "name": "macos-arm64 decode",
            "value": 38.65,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.028"
          },
          {
            "name": "windows-x64 embed",
            "value": 689.16,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 1.007"
          },
          {
            "name": "windows-x64 decode",
            "value": 28.93,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 207 Stepping 2, GenuineIntel; vs previous commit: 1.003"
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
          "id": "c0af557e9edf3db6aa3752107fef2bb884a4299b",
          "message": "Perf/gpu fuse f16kv (#135)\n\n* GPU: run every kernel on the shared compute pass\n\nThe remaining kernels (add, RoPE, RMSNorm's neighbours in the FFN block,\nlayout ops, softmax, argmax and others) each opened a compute pass of\ntheir own, about 10 µs of GPU time apiece, several times per layer per\ntoken. They now record through gpu_profile!, which reuses the encoder's\nopen pass, or a profiled pass when the profiling feature is on.\n\n* GPU: fuse the residual add into RMSNorm; f16 KV cache over quantised weights\n\nEach decoder layer returns its residual and FFN output unsummed, and the\nnext layer's input norm adds them in the same kernel (RMSNorm's add_main\nentry), as does the post-attention norm. One add runs after the last\nlayer. The sum is computed as the add kernel computes it, so the output\nis bitwise unchanged.\n\nLlama-family and Qwen 3.5 decoders whose attention weights are quantised\nstore keys and values as f16 when the device has SHADER_F16, halving the\ncache and its reads. Float weights and KJARNI_GPU_EXACT=1 keep f32, so\nGPU output stays in step with the CPU there.",
          "timestamp": "2026-11-23T08:21:13Z",
          "tree_id": "80cdd84bba209007a4117d7a878b256dc671b44b",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/c0af557e9edf3db6aa3752107fef2bb884a4299b"
        },
        "date": 1795424972312,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 488.78,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 1.000"
          },
          {
            "name": "linux-x64 decode",
            "value": 35.41,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.992"
          },
          {
            "name": "macos-arm64 embed",
            "value": 579.98,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.945"
          },
          {
            "name": "macos-arm64 decode",
            "value": 38.9,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.941"
          },
          {
            "name": "windows-x64 embed",
            "value": 536.54,
            "unit": "sentences/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 0.993"
          },
          {
            "name": "windows-x64 decode",
            "value": 24.79,
            "unit": "tokens/s",
            "extra": "CPU: Intel64 Family 6 Model 106 Stepping 6, GenuineIntel; vs previous commit: 1.002"
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
          "id": "5ead611a57ffe9cd2ca4d55156b9436f06b98848",
          "message": "Perf/gpu prefill tc (#136)",
          "timestamp": "2026-11-23T09:18:47Z",
          "tree_id": "7aca81466849d13fce5556f9868f305950f2be04",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/5ead611a57ffe9cd2ca4d55156b9436f06b98848"
        },
        "date": 1795426539135,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 658.6,
            "unit": "sentences/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 0.998"
          },
          {
            "name": "linux-x64 decode",
            "value": 29.18,
            "unit": "tokens/s",
            "extra": "CPU: Intel(R) Xeon(R) 6975P-C; vs previous commit: 1.008"
          },
          {
            "name": "macos-arm64 embed",
            "value": 483.43,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.287"
          },
          {
            "name": "macos-arm64 decode",
            "value": 21.46,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 0.793"
          },
          {
            "name": "windows-x64 embed",
            "value": 562.8,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.003"
          },
          {
            "name": "windows-x64 decode",
            "value": 39.2,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.018"
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
          "id": "5ead611a57ffe9cd2ca4d55156b9436f06b98848",
          "message": "Perf/gpu prefill tc (#136)",
          "timestamp": "2026-11-23T09:18:47Z",
          "tree_id": "7aca81466849d13fce5556f9868f305950f2be04",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/5ead611a57ffe9cd2ca4d55156b9436f06b98848"
        },
        "date": 1795441007738,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 571.53,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 1.021"
          },
          {
            "name": "linux-x64 decode",
            "value": 42.81,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 9V45 96-Core Processor; vs previous commit: 0.994"
          },
          {
            "name": "macos-arm64 embed",
            "value": 638.14,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.142"
          },
          {
            "name": "macos-arm64 decode",
            "value": 47.82,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.169"
          },
          {
            "name": "windows-x64 embed",
            "value": 555.54,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.008"
          },
          {
            "name": "windows-x64 decode",
            "value": 39.19,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 25 Model 17 Stepping 1, AuthenticAMD; vs previous commit: 1.001"
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
          "id": "d979c28631eaa9999d8b54184093f82ddeb43815",
          "message": "GPU: allocate in 64 MB blocks so long prompts free their memory (#137)\n\ngpu-allocator frees a memory block only when every buffer in it is gone.\nA prompt's attention temporaries grow with its length, and the cache\nsnapshots created meanwhile outlive them, so with wgpu's default 256 MB\nblocks each snapshot pinned a block of freed temporaries. After an\n8192-token prompt on Qwen3.5-4B, 7.4 GB was reserved for 4.7 GB in use,\nand Qwen3.5-9B ran out of a 12 GB card at 8192 tokens.\n\nMemoryHints::MemoryUsage caps shared blocks at 64 MB and gives larger\nbuffers memory of their own: 5.2 GB reserved for the same 4.7 GB, and 9B\nruns 8192 tokens at an 11.0 GB peak. Prompt and generation speed on\nQwen2.5-0.5B, Llama 3.2 1B and Qwen3.5-4B are unchanged.",
          "timestamp": "2026-11-24T08:37:29Z",
          "tree_id": "7ccba9f2bdb435f922d1d9dea68e9780c8544e63",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/d979c28631eaa9999d8b54184093f82ddeb43815"
        },
        "date": 1795510482268,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 495.32,
            "unit": "sentences/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.999"
          },
          {
            "name": "linux-x64 decode",
            "value": 36.08,
            "unit": "tokens/s",
            "extra": "CPU: AMD EPYC 7763 64-Core Processor; vs previous commit: 0.998"
          },
          {
            "name": "macos-arm64 embed",
            "value": 396.04,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.098"
          },
          {
            "name": "macos-arm64 decode",
            "value": 26.02,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.133"
          },
          {
            "name": "windows-x64 embed",
            "value": 976.04,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.003"
          },
          {
            "name": "windows-x64 decode",
            "value": 53.27,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.035"
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
          "id": "e84beb40fd60fbd8286dfdb537c21c7d5678a3ab",
          "message": "CLIP fixtures with transformers 5; skip CLIP parity tests without fix… (#139)\n\n* CLIP fixtures with transformers 5; skip CLIP parity tests without fixtures\n\ntransformers 5 returns a model output from get_image_features and\nget_text_features, so indexing it took the hidden states instead of the\nprojected embedding and every parity test failed against a 1-wide\nreference. test-everything.sh said the CLIP parity tests would skip\nwithout a torch Python, but exported a fixtures path that did not exist,\nwhich those tests treat as an error.\n\n* 0.8.2\n\n* Benchmarks: every engine at its best thread count and precision\n\nEach engine on the CPU is now run at its default thread count and at 8,\nand PyTorch in bf16 and f32 on the same safetensors, and the faster run is\nreported. Quantised models are compared only with llama.cpp. All results\nre-measured on one fresh boot; charts, BENCHMARKS.md and the README's\nbenchmark section regenerated from them. The README states the one-machine\nscope.\n\n* Benchmarks: report the fastest load of each engine's runs\n\nThe first run can include reading the model from disk, so the load time\nfrom the run with the fastest batch was sometimes a cold one. Loads\nre-timed warm, three times each, keeping the faster figure.",
          "timestamp": "2026-11-26T11:37:57Z",
          "tree_id": "3a18c06a06e6cbea034ea24d0d7d06bba246082a",
          "url": "https://github.com/olafurjohannsson/kjarni/commit/e84beb40fd60fbd8286dfdb537c21c7d5678a3ab"
        },
        "date": 1795694226042,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "linux-x64 embed",
            "value": 603.99,
            "unit": "sentences/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 1.000"
          },
          {
            "name": "linux-x64 decode",
            "value": 34.57,
            "unit": "tokens/s",
            "extra": "CPU: INTEL(R) XEON(R) PLATINUM 8573C; vs previous commit: 1.013"
          },
          {
            "name": "macos-arm64 embed",
            "value": 366.88,
            "unit": "sentences/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.009"
          },
          {
            "name": "macos-arm64 decode",
            "value": 20.04,
            "unit": "tokens/s",
            "extra": "CPU: Apple M1 (Virtual); vs previous commit: 1.021"
          },
          {
            "name": "windows-x64 embed",
            "value": 903.96,
            "unit": "sentences/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.020"
          },
          {
            "name": "windows-x64 decode",
            "value": 50.09,
            "unit": "tokens/s",
            "extra": "CPU: AMD64 Family 26 Model 2 Stepping 1, AuthenticAMD; vs previous commit: 1.008"
          }
        ]
      }
    ]
  }
}