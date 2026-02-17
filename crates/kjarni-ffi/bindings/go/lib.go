package kjarni

import (
	"embed"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"

	"github.com/ebitengine/purego"
)

var libFS embed.FS

var (
	libOnce sync.Once
	libErr  error
	lib     uintptr
)

func loadLibrary() (uintptr, error) {
	libOnce.Do(func() {
		lib, libErr = doLoadLibrary()
	})
	return lib, libErr
}

func doLoadLibrary() (uintptr, error) {
	var embeddedPath, fileName string

	switch runtime.GOOS {
	case "linux":
		embeddedPath = "lib/linux_amd64/libkjarni_ffi.so"
		fileName = "libkjarni_ffi.so"
	case "windows":
		embeddedPath = "lib/windows_amd64/kjarni_ffi.dll"
		fileName = "kjarni_ffi.dll"
	default:
		return 0, fmt.Errorf("unsupported OS: %s", runtime.GOOS)
	}

	data, err := libFS.ReadFile(embeddedPath)
	if err != nil {
		return 0, fmt.Errorf("reading embedded library: %w", err)
	}

	tmpDir, err := os.MkdirTemp("", "kjarni-*")
	if err != nil {
		return 0, fmt.Errorf("creating temp dir: %w", err)
	}

	libPath := filepath.Join(tmpDir, fileName)
	if err := os.WriteFile(libPath, data, 0755); err != nil {
		return 0, fmt.Errorf("writing library: %w", err)
	}

	handle, err := purego.Dlopen(libPath, purego.RTLD_NOW|purego.RTLD_GLOBAL)
	if err != nil {
		return 0, fmt.Errorf("loading library: %w", err)
	}

	return handle, nil
}