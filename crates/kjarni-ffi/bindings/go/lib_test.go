package kjarni

import (
	"io/fs"
	"runtime"
	"testing"
)

// The module ships the native library inside the binary, so a mistake in how the
// release job lays out lib/ is invisible until a user's first call. CI assembles
// those directories by hand and once wrote hyphens where doLoadLibrary reads
// underscores, which produced a module that compiled and failed at runtime.
func TestEmbeddedLibrariesPresent(t *testing.T) {
	// Every path doLoadLibrary can ask libFS for. Keep in sync with its switch.
	paths := []string{
		"lib/linux_amd64/libkjarni_ffi.so",
		"lib/windows_amd64/kjarni_ffi.dll",
	}

	for _, p := range paths {
		data, err := libFS.ReadFile(p)
		if err != nil {
			t.Errorf("embedded library %s is missing: %v", p, err)
			continue
		}
		if len(data) == 0 {
			t.Errorf("embedded library %s is empty", p)
		}
	}
}

// Guards against a library being added to lib/ that nothing loads, or a
// directory being renamed so the embed silently picks up a stale layout.
func TestEmbeddedLibrariesAreExactlyTheExpectedSet(t *testing.T) {
	want := map[string]bool{
		"lib/linux_amd64/libkjarni_ffi.so": true,
		"lib/windows_amd64/kjarni_ffi.dll": true,
	}

	got := map[string]bool{}
	err := fs.WalkDir(libFS, "lib", func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() {
			got[path] = true
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking embedded lib/: %v", err)
	}

	for p := range want {
		if !got[p] {
			t.Errorf("expected embedded library %s, not found", p)
		}
	}
	for p := range got {
		if !want[p] {
			t.Errorf("unexpected file embedded in the module: %s", p)
		}
	}
}

// doLoadLibrary returns a clear error rather than dlopening something wrong on
// a platform whose library the module does not carry.
func TestLoadLibraryOnCurrentPlatform(t *testing.T) {
	handle, err := loadLibrary()

	switch runtime.GOOS {
	case "linux", "windows":
		if err != nil {
			t.Fatalf("loading the embedded library on %s failed: %v", runtime.GOOS, err)
		}
		if handle == 0 {
			t.Fatal("loadLibrary returned a nil handle with no error")
		}
	default:
		if err == nil {
			t.Fatalf("expected an unsupported-OS error on %s, got a handle", runtime.GOOS)
		}
	}
}
