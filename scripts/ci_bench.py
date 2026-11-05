#!/usr/bin/env python3
"""Estimate embedding and decode speed on a CI runner, and compare two builds.

Runners are shared, so an absolute number here is only a trend for that platform.
Alternating two builds on one runner cancels most of that noise, which makes their
ratio the figure to trust. Published comparisons stay in bench/BENCHMARKS.md.

Only the CLI is used, so any older commit can serve as the baseline. Each rate comes
from the difference between a long and a one-item run, which removes model load.

Usage: ci_bench.py --platform NAME --head PATH [--base PATH] [--rounds N] --out FILE
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time

DECODE_MODEL = "qwen2.5-0.5b-instruct"
# A prompt the model does not finish within DECODE_TOKENS. Stopping early would leave
# fewer tokens than the rate assumes and report a speed that is too low.
DECODE_PROMPT = "Write a very long essay about the history of Iceland, in many chapters."
DECODE_TOKENS = 129
# Words stand in for tokens, which the CLI does not report. English runs about 0.75
# words a token, so a full run is near 97 words and its exact count moves with the
# text; on linux-x64 the full 129 tokens came to 99. An early stop leaves far fewer.
DECODE_MIN_WORDS = 80
EMBED_MODEL = "minilm-l6-v2"
EMBED_LINES = 2000

# A drop smaller than this is within what alternating runs on a shared runner can show.
WARN_RATIO = 0.85


def rate(items, long_seconds, short_seconds):
    """Items a second, given a run of `items` and a run of one item."""
    spent = long_seconds - short_seconds
    if spent <= 0:
        raise ValueError(f"long run ({long_seconds}s) was not slower than short ({short_seconds}s)")
    return (items - 1) / spent


def check_ran_to_length(text):
    """Fail on decode output too short to have used all of DECODE_TOKENS."""
    words = len(text.split())
    if words < DECODE_MIN_WORDS:
        raise ValueError(f"decode stopped early: {words} words, expected {DECODE_MIN_WORDS}+")


def verdict(head, base):
    """The head/base ratio, and whether it is low enough to warn about."""
    ratio = head / base
    return ratio, ratio < WARN_RATIO


def timed(cli, args, stdin=None):
    start = time.perf_counter()
    result = subprocess.run(
        [cli, *args], input=stdin, capture_output=True, text=True, encoding="utf-8", timeout=1800
    )
    seconds = time.perf_counter() - start
    if result.returncode != 0:
        raise RuntimeError(f"{cli} {' '.join(args)} exited {result.returncode}:\n{result.stderr}")
    return seconds, result.stdout


def decode_rate(cli):
    args = ["generate", "-q", "-m", DECODE_MODEL, "--gguf", "--greedy"]
    long_seconds, text = timed(cli, [*args, "-n", str(DECODE_TOKENS), DECODE_PROMPT])
    check_ran_to_length(text)
    short_seconds, _ = timed(cli, [*args, "-n", "1", DECODE_PROMPT])
    return rate(DECODE_TOKENS, long_seconds, short_seconds), text.strip()


def embed_rate(cli):
    lines = "\n".join(f"Sentence number {i} about a quick brown fox." for i in range(EMBED_LINES))
    long_seconds, _ = timed(cli, ["embed", "-q", "-m", EMBED_MODEL], stdin=lines)
    short_seconds, _ = timed(cli, ["embed", "-q", "-m", EMBED_MODEL, "One sentence."])
    return rate(EMBED_LINES, long_seconds, short_seconds)


def cpu_name():
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                return next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
    except (OSError, StopIteration, subprocess.SubprocessError):
        pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or platform.processor() or "unknown"


def measure(builds, rounds):
    """Alternate the builds. The first round warms the page cache and is dropped."""
    embed = {name: [] for name in builds}
    decode = {name: [] for name in builds}
    text = {}
    for round_number in range(rounds + 1):
        for name, cli in builds.items():
            embed_value = embed_rate(cli)
            decode_value, text[name] = decode_rate(cli)
            if round_number > 0:
                embed[name].append(embed_value)
                decode[name].append(decode_value)
    return embed, decode, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--base")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    builds = {"head": args.head}
    if args.base:
        builds["base"] = args.base
    embed, decode, text = measure(builds, args.rounds)
    cpu = cpu_name()

    entries, rows, warnings = [], [], []
    for task, unit, runs in (("embed", "sentences/s", embed), ("decode", "tokens/s", decode)):
        head = statistics.median(runs["head"])
        extra = f"CPU: {cpu}"
        row = f"| {args.platform} | {task} | {head:.1f} {unit} |"
        if args.base:
            ratio, low = verdict(head, statistics.median(runs["base"]))
            extra += f"; vs previous commit: {ratio:.3f}"
            row += f" {ratio:.3f} |"
            if low:
                warnings.append(f"{args.platform} {task} is at {ratio:.0%} of the previous commit")
        else:
            row += " n/a |"
        entries.append({
            "name": f"{args.platform} {task}", "unit": unit, "value": round(head, 2), "extra": extra,
        })
        rows.append(row)

    if args.base and text["head"] != text["base"]:
        warnings.append(f"{args.platform} decode output changed, so its speed ratio may not compare")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    report = "\n".join([f"CPU: {cpu}", "", "| Platform | Task | Speed | vs previous |",
                        "|---|---|--:|--:|", *rows, ""])
    print(report)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(report + "\n")
    for warning in warnings:
        print(f"::warning title=Possible speed regression::{warning}")


if __name__ == "__main__":
    main()
