#!/usr/bin/env python3
"""Run the shipped CLI binary and check its answers against the Linux build.

The macOS and Windows jobs used to build the release binaries and upload them
without ever executing them, so a binary that crashed on load, or that computed
different numbers on NEON than on AVX2, would have reached the release archive,
the NuGet package and the wheel unnoticed. This runs the exact binary that ships.

The expected values were recorded from a release build of v0.2.8 on x86_64 Linux.
Tolerances allow for a different summation order across SIMD instruction sets,
not for a different answer.

Usage: python3 scripts/smoke_test.py path/to/kjarni[.exe]
"""

import json
import math
import subprocess
import sys

EMBED_TEXT = "Kjarni runs transformer models locally."
EMBED_DIMS = 384
EMBED_FIRST = [
    -0.48449725, 0.050634537, -0.076398276, -0.26228786, 0.2099083, -0.3151892,
    -1.1667886, 0.02973278, 0.000987611, 0.005218636, 0.03976811, -0.18868078,
]
EMBED_NORM = 5.944001007953482

SIMILARITY = 0.859813
RERANK_SCORES = [9.58177, -11.100061]
CLASSIFY_SCORE = 0.9998789

TOL = 1e-4

failures = []


def run(cli, *args):
    result = subprocess.run(
        [cli, *args], capture_output=True, text=True, encoding="utf-8", timeout=600
    )
    if result.returncode != 0:
        sys.exit(f"`kjarni {' '.join(args)}` exited {result.returncode}:\n{result.stderr}")
    return result.stdout


def close(name, got, want, tol=TOL):
    ok = abs(got - want) <= tol * max(1.0, abs(want))
    print(f"{'ok  ' if ok else 'FAIL'} {name}: {got} (expected {want})")
    if not ok:
        failures.append(name)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cli = sys.argv[1]

    print(run(cli, "--version").strip())
    for model in ("minilm-l6-v2", "minilm-l6-v2-cross-encoder", "distilbert-sentiment"):
        run(cli, "model", "download", model)

    vector = [float(x) for x in run(cli, "embed", "-q", EMBED_TEXT).split()]
    if len(vector) != EMBED_DIMS:
        sys.exit(f"embed returned {len(vector)} dimensions, expected {EMBED_DIMS}")
    for i, want in enumerate(EMBED_FIRST):
        close(f"embed[{i}]", vector[i], want)
    close("embed norm", math.sqrt(sum(x * x for x in vector)), EMBED_NORM)

    close("similarity", float(run(cli, "similarity", "-q", "doctor", "physician")), SIMILARITY)

    ranked = json.loads(run(
        cli, "rerank", "-q", "--format", "json", "what is machine learning?",
        "Machine learning is a subset of AI.", "The weather today is sunny.",
    ))
    if [r["index"] for r in ranked] != [0, 1]:
        failures.append("rerank order")
        print(f"FAIL rerank order: {[r['index'] for r in ranked]}")
    for r, want in zip(ranked, RERANK_SCORES):
        close(f"rerank[{r['index']}]", r["score"], want)

    label = json.loads(run(cli, "classify", "-q", "--format", "json", "I love this product"))
    if label["label"] != "POSITIVE":
        failures.append("classify label")
        print(f"FAIL classify label: {label['label']}")
    close("classify score", label["score"], CLASSIFY_SCORE)

    if failures:
        sys.exit(f"\n{len(failures)} check(s) differ from the Linux build: {', '.join(failures)}")
    print("\nAll checks match the Linux build.")


if __name__ == "__main__":
    main()
