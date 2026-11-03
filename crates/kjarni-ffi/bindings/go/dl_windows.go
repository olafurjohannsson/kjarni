//go:build windows

package kjarni

import "syscall"

// purego has no Dlopen or Dlsym on Windows; LoadLibrary and GetProcAddress do the
// same job, and purego.RegisterLibFunc accepts the handle LoadLibrary returns.
func openLibrary(path string) (uintptr, error) {
	handle, err := syscall.LoadLibrary(path)
	return uintptr(handle), err
}

func symbol(handle uintptr, name string) (uintptr, error) {
	return syscall.GetProcAddress(syscall.Handle(handle), name)
}
