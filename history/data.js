window.BENCHMARK_DATA = {
  "lastUpdate": 1793870250095,
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
      }
    ]
  }
}