package kjarni

import (
	"os"
	"regexp"
	"runtime"
	"sort"
	"testing"
)

// findHeader locates the cbindgen-generated header in either layout: the module
// as pushed to kjarni-go, where CI copies kjarni.h alongside the sources, and
// this repo, where it lives under crates/kjarni-ffi/include.
func findHeader(t *testing.T) string {
	t.Helper()
	for _, p := range []string{"kjarni.h", "../../include/kjarni.h"} {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Fatal("kjarni.h not found in either ./ or ../../include/; " +
		"the module cannot be checked against the C ABI without it")
	return ""
}

// symbolsResolvedByGo scrapes every native symbol name this package looks up at
// init time, so the test cannot drift out of date as bindings are added.
func symbolsResolvedByGo(t *testing.T) []string {
	t.Helper()
	src, err := os.ReadFile("ffi.go")
	if err != nil {
		t.Fatalf("reading ffi.go: %v", err)
	}

	pattern := regexp.MustCompile(`(?:Dlsym|RegisterLibFunc)\([^)]*?"(kjarni_[a-z0-9_]+)"`)
	seen := map[string]bool{}
	for _, m := range pattern.FindAllSubmatch(src, -1) {
		seen[string(m[1])] = true
	}

	if len(seen) == 0 {
		t.Fatal("scraped no symbol names from ffi.go; the pattern is out of date")
	}

	names := make([]string, 0, len(seen))
	for n := range seen {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// The Rust side owns these names. Renaming one there compiles fine on both
// sides and fails at the user's first call, so check the Go bindings against
// the generated header rather than waiting for a runtime dlsym failure.
func TestEveryResolvedSymbolExistsInHeader(t *testing.T) {
	header, err := os.ReadFile(findHeader(t))
	if err != nil {
		t.Fatalf("reading header: %v", err)
	}

	for _, name := range symbolsResolvedByGo(t) {
		// Word boundary: kjarni_chat_send must not be satisfied by
		// kjarni_chat_send_something_else.
		re := regexp.MustCompile(`\b` + regexp.QuoteMeta(name) + `\b`)
		if !re.Match(header) {
			t.Errorf("ffi.go resolves %q, which the C header does not declare", name)
		}
	}
}

// The same check against the binary that actually ships. This catches a header
// and a library that disagree, which the source-level test above cannot see.
func TestInitFFIResolvesEverySymbol(t *testing.T) {
	if runtime.GOOS != "linux" && runtime.GOOS != "windows" {
		t.Skipf("no embedded library for %s", runtime.GOOS)
	}
	if err := initFFI(); err != nil {
		t.Fatalf("resolving symbols against the embedded library: %v", err)
	}
}

func TestCStringGoStringRoundTrip(t *testing.T) {
	cases := []string{
		"",
		"hello",
		"a string with spaces",
		"unicode: sæll heimur",
		"emoji: \U0001F600",
	}

	for _, want := range cases {
		ptr, keepAlive := cString(want)
		got := goString(ptr)
		keepAlive()
		if got != want {
			t.Errorf("round trip of %q produced %q", want, got)
		}
	}
}

func TestGoStringOnNullPointer(t *testing.T) {
	if got := goString(0); got != "" {
		t.Errorf("goString(0) = %q, want empty string", got)
	}
}
