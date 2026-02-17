package kjarni

import "fmt"

type ErrorCode int32

const (
	ErrOk             ErrorCode = 0
	ErrNullPointer    ErrorCode = 1
	ErrInvalidUtf8    ErrorCode = 2
	ErrModelNotFound  ErrorCode = 3
	ErrLoadFailed     ErrorCode = 4
	ErrInferenceFailed ErrorCode = 5
	ErrGpuUnavailable ErrorCode = 6
	ErrInvalidConfig  ErrorCode = 7
	ErrCancelled      ErrorCode = 8
	ErrTimeout        ErrorCode = 9
	ErrStreamEnded    ErrorCode = 10
	ErrUnknown        ErrorCode = 255
)

type KjarniError struct {
	Code    ErrorCode
	Message string
}

func (e *KjarniError) Error() string {
	return fmt.Sprintf("kjarni: %s (code %d)", e.Message, e.Code)
}