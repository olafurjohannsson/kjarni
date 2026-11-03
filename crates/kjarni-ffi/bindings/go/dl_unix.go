//go:build !windows

package kjarni

import "github.com/ebitengine/purego"

func openLibrary(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_GLOBAL)
}

func symbol(handle uintptr, name string) (uintptr, error) {
	return purego.Dlsym(handle, name)
}
