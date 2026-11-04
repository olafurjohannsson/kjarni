package kjarni

import "testing"

func TestRuneToByteOffsets(t *testing.T) {
	got := runeToByteOffsets("Þór a")
	want := []int{0, 2, 4, 5, 6, 7}
	if len(got) != len(want) {
		t.Fatalf("got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}
