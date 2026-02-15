package kjarni

type options struct {
	quiet  bool
	device string // "cpu" || "gpu"
}

type Option func(*options)

func WithQuiet(quiet bool) Option {
	return func(o *options) {
		o.quiet = quiet
	}
}

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