package kjarni

type options struct {
	quiet  bool
	device string // "cpu" || "gpu"
}

// Option configures a classifier, embedder, or other kjarni component.
type Option func(*options)

// WithQuiet suppresses log output during model loading and inference.
func WithQuiet(quiet bool) Option {
	return func(o *options) {
		o.quiet = quiet
	}
}

// WithDevice sets the compute device. Supported values: "cpu", "gpu".
func WithDevice(device string) Option {
	return func(o *options) {
		o.device = device
	}
}

func applyOptions(opts []Option) options {
	o := options{
		device: "cpu",
	}
	for _, opt := range opts {
		opt(&o)
	}
	return o
}

func deviceCode(device string) int32 {
	if device == "gpu" {
		return 1
	}
	return 0
}

func boolToInt(b bool) int32 {
	if b {
		return 1
	}
	return 0
}