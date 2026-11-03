window.BENCHMARK_DATA = {
  "lastUpdate": 1793698720926,
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
      }
    ]
  }
}