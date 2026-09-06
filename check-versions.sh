#!/usr/bin/env bash
#
# Every version string in the repo, checked against one source of truth:
# [workspace.package] version in Cargo.toml.
#
# Two of these matter for different reasons. crates.io takes the version from
# Cargo.toml and nothing else, so a tag that disagrees with the manifest aborts
# the publish. npm, NuGet and PyPI are stamped from the tag at publish time, so
# their committed versions do not decide what ships, but a stale one still
# misleads anyone who builds locally, and `kjarni_version()` in the FFI crate is
# compiled from CARGO_PKG_VERSION and reported to every language binding at
# runtime.
#
#   ./check-versions.sh              verify, non-zero exit on any mismatch
#   ./check-versions.sh --set 0.1.9  rewrite every source to that version
#
set -uo pipefail
cd "$(dirname "$0")"

MODE=check
NEW=""
if [ "${1:-}" = "--set" ]; then
  MODE=set
  NEW="${2:-}"
  if ! [[ "$NEW" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo "usage: $0 --set X.Y.Z" >&2
    exit 2
  fi
fi

WORKSPACE_TOML=Cargo.toml
PYPROJECT=crates/kjarni-ffi/bindings/python/pyproject.toml
TS_PACKAGE=crates/kjarni-wasm/ts/package.json
CSPROJ=(
  crates/kjarni-ffi/bindings/csharp/Kjarni/Kjarni.csproj
  crates/kjarni-ffi/bindings/csharp/Kjarni.Extensions.AI/Kjarni.Extensions.AI.csproj
)

fail=0
report() {  # report <name> <found> <want>
  if [ "$2" = "$3" ]; then
    printf '  ok    %-46s %s\n' "$1" "$2"
  else
    printf '  FAIL  %-46s %s (want %s)\n' "$1" "${2:-<missing>}" "$3"
    fail=1
  fi
}

# ── the source of truth ────────────────────────────────────────────
workspace_version() {
  python3 - "$WORKSPACE_TOML" <<'PY'
import re, sys
s = open(sys.argv[1]).read()
m = re.search(r'^\[workspace\.package\](.*?)(?=^\[)', s, re.S | re.M)
v = re.search(r'^version\s*=\s*"([^"]+)"', m.group(1), re.M)
print(v.group(1))
PY
}

if [ "$MODE" = set ]; then
  python3 - "$NEW" "$WORKSPACE_TOML" "$PYPROJECT" "$TS_PACKAGE" "${CSPROJ[@]}" <<'PY'
import re, sys
new, wtoml, pyproject, tspkg, *csprojs = sys.argv[1:]

s = open(wtoml).read()
# [workspace.package] version
m = re.search(r'^\[workspace\.package\](.*?)(?=^\[)', s, re.S | re.M)
blk = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{new}"', m.group(1), count=1, flags=re.M)
s = s[:m.start(1)] + blk + s[m.end(1):]
# the internal path dependencies carry a version alongside the path; cargo
# publish requires it, and a stale one resolves to an older release silently.
s = re.sub(r'(^kjarni[a-z-]* = \{ path = "[^"]+", version = ")[^"]+(" \})',
           rf'\g<1>{new}\g<2>', s, flags=re.M)
open(wtoml, "w").write(s)

s = open(pyproject).read()
s = re.sub(r'^version\s*=\s*"[^"]+"', f'version = "{new}"', s, count=1, flags=re.M)
open(pyproject, "w").write(s)

s = open(tspkg).read()
s = re.sub(r'("version"\s*:\s*")[^"]+(")', rf'\g<1>{new}\g<2>', s, count=1)
open(tspkg, "w").write(s)

for c in csprojs:
    s = open(c).read()
    s = re.sub(r'<Version>[^<]+</Version>', f'<Version>{new}</Version>', s, count=1)
    open(c, "w").write(s)
PY
  echo "set every version to $NEW"
fi

WANT=$(workspace_version)
echo "workspace version: $WANT"
echo

# ── every crate must inherit, or its published version silently diverges ──
echo "crates:"
while read -r name inherits value; do
  if [ "$inherits" = "yes" ]; then
    printf '  ok    %-46s version.workspace = true\n' "$name"
  else
    printf '  FAIL  %-46s version = %s (want version.workspace = true)\n' "$name" "$value"
    fail=1
  fi
done < <(python3 - <<'PY'
import pathlib, re
for f in sorted(pathlib.Path("crates").glob("*/Cargo.toml")):
    s = f.read_text()
    m = re.search(r'^\[package\](.*?)(?=^\[)', s, re.S | re.M)
    if not m:
        continue
    v = re.search(r'^version(\.workspace)?\s*=\s*(.+)$', m.group(1), re.M)
    inherits = "yes" if (v and v.group(1)) else "no"
    print(f.parent.name, inherits, (v.group(2).strip() if v else "<missing>"))
PY
)

# ── the internal dependency pins ──────────────────────────────────
echo
echo "internal dependency pins:"
while read -r crate version; do
  report "$crate" "$version" "$WANT"
done < <(grep -oE '^kjarni[a-z-]* = \{ path = "[^"]+", version = "[^"]+"' "$WORKSPACE_TOML" \
         | sed -E 's/^([a-z-]+) = .*version = "(.*)"$/\1 \2/')

# ── the language packages ─────────────────────────────────────────
echo
echo "language packages:"
report "$PYPROJECT" \
  "$(grep -m1 -E '^version *= *"' "$PYPROJECT" | cut -d'"' -f2)" "$WANT"
report "$TS_PACKAGE" \
  "$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["version"])' "$TS_PACKAGE")" "$WANT"
for c in "${CSPROJ[@]}"; do
  report "$c" "$(grep -m1 -oE '<Version>[^<]+</Version>' "$c" | sed -E 's/<\/?Version>//g')" "$WANT"
done

# ── metadata crates.io requires ───────────────────────────────────
#
# crates.io rejects a publish with 400 for a missing license, and it does so per
# crate, at the end of a loop that has already published the others. kjarni-ffi
# failed exactly this way at v0.1.9 after the six before it had gone out, which
# cannot be undone: those versions are immutable. Check it here instead.
echo
echo "crates.io metadata:"
while read -r name field; do
  if [ "$field" = "ok" ]; then
    printf '  ok    %-46s license, description, repository\n' "$name"
  else
    printf '  FAIL  %-46s missing: %s\n' "$name" "$field"
    fail=1
  fi
done < <(cargo metadata --no-deps --format-version 1 2>/dev/null | python3 -c '
import json, sys
# Only crates the publish loop actually pushes.
published = {"kjarni", "kjarni-cli", "kjarni-ffi", "kjarni-models",
             "kjarni-rag", "kjarni-search", "kjarni-transformers"}
for p in sorted(json.load(sys.stdin)["packages"], key=lambda x: x["name"]):
    if p["name"] not in published:
        continue
    missing = [f for f in ("license", "description", "repository") if not p.get(f)]
    print(p["name"], ",".join(missing) if missing else "ok")
')

# ── the tag, when CI is running on one ────────────────────────────
if [ -n "${GITHUB_REF_NAME:-}" ] && [[ "${GITHUB_REF_NAME}" == v* ]]; then
  echo
  echo "git tag:"
  report "${GITHUB_REF_NAME}" "${GITHUB_REF_NAME#v}" "$WANT"
fi

echo
if [ "$fail" -ne 0 ]; then
  echo "version drift found. Run: $0 --set $WANT" >&2
  exit 1
fi
echo "all versions agree on $WANT"
