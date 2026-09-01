#!/usr/bin/env bash
#
# Runs everything CI cannot: GPU tests, the 317 #[ignore]d tests, debug
# assertions, and the C++/WASM/C#/Python suites that need a real toolchain.
#
#   ./test-everything.sh              # every stage, in order
#   ./test-everything.sh fmt clippy   # just those
#   ./test-everything.sh --list       # what stages exist
#   ./test-everything.sh -x           # stop at the first failure
#
# Everything runs with --test-threads=1 on purpose. These tests each want the
# whole machine: a parallel run oversubscribes every core and the GPU at once.
#
# Expect the debug stage to take hours. It is the slowest thing here by a wide
# margin, because an unoptimised decode loop is roughly twenty times slower and
# nothing overlaps. Run `test-release` first if you want a fast answer, and this
# one when you can leave it.

set -uo pipefail
cd "$(dirname "$0")"

STOP_ON_FAIL=0
VERBOSE=0
LOGDIR="${KJARNI_TEST_LOGS:-$PWD/.test-logs}"
KJQ_DIR="${KJARNI_KJQ_DIR:-/tmp/kjq}"
export RUST_BACKTRACE="${RUST_BACKTRACE:-1}"
export KJARNI_KJQ_DIR="$KJQ_DIR"

ALL_STAGES=(fmt clippy wasm-check fixtures test-debug test-release csharp cpp wasm-browser python typescript)

declare -A RESULT ELAPSED
FAILED=0

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
dim()  { printf '\033[2m%s\033[0m\n' "$*"; }

# Lines worth seeing while a stage runs. Everything else is in the log: compiler
# warnings especially, which run to dozens per stage and drown the result.
FILTER='^test result|^ *[0-9]+ (checks|tests)|^ok |^  ok |^  FAIL|FAILED|^error|^Passed!|^Failed!|panicked|all checks passed|^Ran [0-9]+ test|^OK$|tests passed'

run_stage() {
    local name="$1"; shift
    local log="$LOGDIR/$name.log"
    bold "── $name ─────────────────────────────────────────────"
    local start; start=$(date +%s)

    if [ "$VERBOSE" = 1 ]; then
        "$@" 2>&1 | tee "$log"
    else
        "$@" 2>&1 | tee "$log" | grep --line-buffered -E "$FILTER" | sed 's/^/  /'
    fi
    local status=${PIPESTATUS[0]}

    if [ "$status" = 0 ]; then
        RESULT[$name]=pass
    else
        RESULT[$name]=FAIL
        FAILED=$((FAILED + 1))
        # The filter hides the reason a stage failed, so show the tail verbatim.
        echo
        dim "  last 25 lines of $log:"
        tail -25 "$log" | sed 's/^/  | /'
        [ "$STOP_ON_FAIL" = 1 ] && { summary; exit 1; }
    fi
    ELAPSED[$name]=$(( $(date +%s) - start ))
    echo
}

# The quantizer needs numpy and safetensors, which are not Kjarni dependencies.
# A throwaway venv keeps them out of the system python, and out of your way.
VENV="${KJARNI_TEST_VENV:-$PWD/.test-venv}"
ensure_venv() {
    if [ ! -x "$VENV/bin/python" ]; then
        echo "  creating $VENV"
        python3 -m venv "$VENV" >/dev/null || return 1
        "$VENV/bin/pip" install -q -r crates/kjarni-wasm/scripts/requirements.txt || return 1
    fi
}

# ── Stages ───────────────────────────────────────────────────────

stage_fmt()        { cargo fmt --all --check; }
stage_clippy()     { cargo clippy --workspace --all-targets -- -D warnings; }
stage_wasm_check() { cargo check -p kjarni-wasm --target wasm32-unknown-unknown --no-default-features; }

# The .kjq fixtures several suites need. Built from whatever the model cache
# already holds, so this is cheap after the first run.
stage_fixtures() {
    mkdir -p "$KJQ_DIR"
    local cache="${HOME}/.cache/kjarni"
    # source : output : format. Both encodings are built, because the container
    # has to keep reading KJQ1 (the encoders, which are published that way) while
    # KJQ8 is what the browser needs for a decoder: block-quantised weights that
    # are never expanded to f32. Qwen is built only as KJQ8, which is what is
    # published and the only encoding that fits a browser.
    local triples=(
        "sentence-transformers_all-MiniLM-L6-v2:all-MiniLM-L6-v2-q8.kjq:kjq1"
        "cross-encoder_ms-marco-MiniLM-L-6-v2:ms-marco-MiniLM-L-6-v2-q8.kjq:kjq1"
        "distilbert_distilbert-base-uncased-finetuned-sst-2-english:distilbert-sentiment-q8.kjq:kjq1"
        "sentence-transformers_all-MiniLM-L6-v2:all-MiniLM-L6-v2-kjq8.kjq:kjq8"
        "Qwen_Qwen2.5-0.5B-Instruct:qwen05b-kjq8.kjq:kjq8"
    )
    ensure_venv || return 1
    local py="$VENV/bin/python"

    for triple in "${triples[@]}"; do
        local src out fmt magic
        IFS=: read -r src out fmt <<<"$triple"
        [ "$fmt" = "kjq8" ] && magic=KJQ8 || magic=KJQ1

        # Existence is not enough. An interrupted run leaves a partial file that
        # looks present forever after, and the failure surfaces much later as
        # "truncated .kjq: expected a u32 at offset ...". Parse the container.
        #
        # The magic is checked against the one this fixture is supposed to carry,
        # so a KJQ1 file sitting where a KJQ8 one belongs is rebuilt rather than
        # silently used.
        if [ -f "$KJQ_DIR/$out" ]; then
            if "$py" - "$KJQ_DIR/$out" "$magic" <<'CHECK'
# Walk the whole container. A header-only check passes on a file truncated
# mid-tensor, which is exactly the shape an interrupted quantize run leaves.
import struct, sys

path, want = sys.argv[1], sys.argv[2].encode()
BLOCK, BYTES_PER_BLOCK = 32, 34   # f16 scale + 32 int8, per BlockQ8_0

def u32(f):
    b = f.read(4)
    if len(b) != 4:
        raise EOFError
    return struct.unpack("<I", b)[0]

def skip(f, n):
    if len(f.read(n)) != n:
        raise EOFError

try:
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != want:
            sys.exit(1)
        skip(f, u32(f))                       # config json
        skip(f, u32(f))                       # tokenizer json
        for _ in range(u32(f)):               # num_tensors
            skip(f, u32(f))                   # name
            dims = [u32(f) for _ in range(u32(f))]
            numel = 1
            for d in dims:
                numel *= d
            quantized = f.read(1)
            if len(quantized) != 1:
                raise EOFError
            if not quantized[0]:
                skip(f, numel * 4)            # f32 data
            elif magic == b"KJQ8":
                # Blocks carry their own f16 scale, so there is no tensor scale.
                skip(f, dims[0] * (dims[1] // BLOCK) * BYTES_PER_BLOCK)
            else:
                skip(f, 4)                    # scale f32
                skip(f, numel)                # int8 data
except (EOFError, struct.error, IndexError):
    sys.exit(1)
CHECK
            then
                dim "  have $out"; continue
            fi
            echo "  rebuilding $out (wrong encoding, truncated or malformed)"
            rm -f "$KJQ_DIR/$out"
        fi

        [ -d "$cache/$src" ] || { echo "  skip $out (weights not cached: $src)"; continue; }
        echo "  building $out ($fmt)"
        # Write to a temp name and rename, so an interrupted run cannot leave a
        # partial file that the check above then has to catch.
        "$py" crates/kjarni-wasm/scripts/quantize_model.py \
            --model-dir "$cache/$src" --format "$fmt" \
            --output "$KJQ_DIR/$out.partial" >/dev/null || return 1
        mv "$KJQ_DIR/$out.partial" "$KJQ_DIR/$out"
    done
}

# Debug: `debug_assert!` is live here and nowhere else, which is the whole point
# of paying for it. --include-ignored picks up the GPU suites.
stage_test_debug() {
    cargo test --workspace -- --include-ignored --test-threads=1
}

# Release: the same tests at usable speed, and the only place release-only
# behaviour (overflow wrapping, elided bounds checks) can show itself.
stage_test_release() {
    cargo test --workspace --release -- --include-ignored --test-threads=1
}

stage_csharp() {
    dotnet test crates/kjarni-ffi/bindings/csharp/Kjarni.sln \
        --configuration Release -- xUnit.MaxParallelThreads=1
}

# Sanitizers are what make the ownership assertions in the C++ wrapper mean
# anything, so this builds with them rather than testing the release path.
stage_cpp() {
    cargo build --release --package kjarni-ffi || return 1
    local dir=crates/kjarni-ffi/tests/cpp
    cmake -S "$dir" -B "$dir/build" \
        -DCMAKE_BUILD_TYPE=Debug -DKJARNI_SANITIZE=ON \
        -DKJARNI_LIB_DIR="$PWD/target/release" >/dev/null || return 1
    cmake --build "$dir/build" >/dev/null || return 1
    ASAN_OPTIONS=detect_leaks=1 LD_LIBRARY_PATH="$PWD/target/release" \
        ctest --test-dir "$dir/build" --output-on-failure
}

# The one check that loads the artefact people actually download.
stage_wasm_browser() {
    command -v wasm-pack >/dev/null || { echo "wasm-pack not installed"; return 1; }
    ( cd crates/kjarni-wasm \
      && RUSTFLAGS='-C target-feature=+simd128' \
         wasm-pack build --release --target web -- --no-default-features ) || return 1
    ( cd crates/kjarni-wasm/tests/browser \
      && npm install --no-audit --no-fund >/dev/null \
      && npx playwright install chromium >/dev/null 2>&1
      node run.mjs )
}

stage_python() {
    ensure_venv || return 1
    # Run from the scripts directory: `discover -s <nested/path>` fails with
    # "Start directory is not importable" unless the tree is a package.
    ( cd crates/kjarni-wasm/scripts && "$VENV/bin/python" -m unittest discover -p "test_*.py" )
}

stage_typescript() {
    ( cd crates/kjarni-wasm/ts \
      && npm install --no-audit --no-fund >/dev/null \
      && npm run build )
}

# ── Summary ──────────────────────────────────────────────────────

summary() {
    echo
    bold "── Summary ──────────────────────────────────────────"
    for s in "${ALL_STAGES[@]}"; do
        [ -n "${RESULT[$s]:-}" ] || continue
        printf '  %-14s %-5s %4ss\n' "$s" "${RESULT[$s]}" "${ELAPSED[$s]:-0}"
    done
    echo
    dim "  logs: $LOGDIR"
    echo
    if [ "$FAILED" = 0 ]; then
        bold "everything passed"
    else
        bold "$FAILED stage(s) failed"
    fi
}

# ── Dispatch ─────────────────────────────────────────────────────

STAGES=()
for arg in "$@"; do
    case "$arg" in
        -x|--stop) STOP_ON_FAIL=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        --list) printf '%s\n' "${ALL_STAGES[@]}"; exit 0 ;;
        -h|--help) sed -n '2,20p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) STAGES+=("$arg") ;;
    esac
done
[ ${#STAGES[@]} -eq 0 ] && STAGES=("${ALL_STAGES[@]}")

mkdir -p "$LOGDIR"

for s in "${STAGES[@]}"; do
    fn="stage_${s//-/_}"
    if ! declare -F "$fn" >/dev/null; then
        echo "unknown stage: $s (see --list)"; exit 2
    fi
    run_stage "$s" "$fn"
done

summary
exit $(( FAILED > 0 ))
