package kjarni

import (
	"os"
	"regexp"
	"strconv"
	"testing"
)

// The default has to stay "cpu": a caller that passes no options must not end up
// dispatching to a GPU that may not exist.
func TestApplyOptionsDefaults(t *testing.T) {
	o := applyOptions(nil)
	if o.device != "cpu" {
		t.Errorf("default device = %q, want \"cpu\"", o.device)
	}
	if o.quiet {
		t.Error("default quiet = true, want false")
	}
}

func TestApplyOptions(t *testing.T) {
	tests := []struct {
		name      string
		opts      []Option
		wantDev   string
		wantQuiet bool
	}{
		{"device only", []Option{WithDevice("gpu")}, "gpu", false},
		{"quiet only", []Option{WithQuiet(true)}, "cpu", true},
		{"both", []Option{WithDevice("gpu"), WithQuiet(true)}, "gpu", true},
		{"quiet false is explicit", []Option{WithQuiet(false)}, "cpu", false},
		{"last device wins", []Option{WithDevice("gpu"), WithDevice("cpu")}, "cpu", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			o := applyOptions(tt.opts)
			if o.device != tt.wantDev {
				t.Errorf("device = %q, want %q", o.device, tt.wantDev)
			}
			if o.quiet != tt.wantQuiet {
				t.Errorf("quiet = %v, want %v", o.quiet, tt.wantQuiet)
			}
		})
	}
}

// deviceCode feeds the Device field of every FFI config struct, so it has to
// agree with the C enum rather than with a remembered pair of numbers.
func TestDeviceCodeMatchesHeader(t *testing.T) {
	header, err := os.ReadFile(findHeader(t))
	if err != nil {
		t.Fatalf("reading header: %v", err)
	}

	pattern := regexp.MustCompile(`KJARNI_DEVICE_([A-Z]+) = (\d+)`)
	matches := pattern.FindAllStringSubmatch(string(header), -1)
	if len(matches) == 0 {
		t.Fatal("found no device enumerators in the header")
	}

	for _, m := range matches {
		name, raw := m[1], m[2]
		value, err := strconv.Atoi(raw)
		if err != nil {
			t.Fatalf("parsing KJARNI_DEVICE_%s = %s: %v", name, raw, err)
		}

		var device string
		switch name {
		case "CPU":
			device = "cpu"
		case "GPU":
			device = "gpu"
		default:
			t.Errorf("the header declares KJARNI_DEVICE_%s, which WithDevice does not accept", name)
			continue
		}

		if got := deviceCode(device); int(got) != value {
			t.Errorf("deviceCode(%q) = %d, header says %d", device, got, value)
		}
	}
}

// An unrecognised device name falls back to CPU rather than to whatever integer
// happens to be next, so a typo degrades instead of dispatching somewhere wrong.
func TestDeviceCodeUnknownFallsBackToCPU(t *testing.T) {
	for _, d := range []string{"", "GPU", "cuda", "metal", "nonsense"} {
		if got := deviceCode(d); got != 0 {
			t.Errorf("deviceCode(%q) = %d, want 0 (cpu)", d, got)
		}
	}
}

func TestBoolToInt(t *testing.T) {
	if got := boolToInt(true); got != 1 {
		t.Errorf("boolToInt(true) = %d, want 1", got)
	}
	if got := boolToInt(false); got != 0 {
		t.Errorf("boolToInt(false) = %d, want 0", got)
	}
}
