window.BENCHMARK_DATA = {
  "lastUpdate": 1794826387581,
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
      }
    ]
  }
}