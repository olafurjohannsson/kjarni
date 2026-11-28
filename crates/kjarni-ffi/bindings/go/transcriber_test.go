package kjarni

import (
	"math"
	"testing"
)

func requireSubtitleSymbols(t *testing.T) {
	t.Helper()
	if err := ensureInit(); err != nil {
		t.Fatalf("ensureInit: %v", err)
	}
	if _transcriptionToSRTSym == 0 || _transcriptionToVTTSym == 0 {
		t.Skip("embedded library lacks kjarni_transcription_to_srt/_vtt; stage freshly built kjarni_ffi to run")
	}
}

func TestSubtitleEmptySegments(t *testing.T) {
	requireSubtitleSymbols(t)
	if got, err := SegmentsToSRT(nil); err != nil || got != "" {
		t.Errorf("SegmentsToSRT(nil) = (%q, %v), want (\"\", nil)", got, err)
	}
	if got, err := SegmentsToVTT(nil); err != nil || got != "WEBVTT\n\n" {
		t.Errorf("SegmentsToVTT(nil) = (%q, %v), want (\"WEBVTT\\n\\n\", nil)", got, err)
	}
}

func TestSubtitleShortAndEmptySegments(t *testing.T) {
	requireSubtitleSymbols(t)
	segs := []Segment{
		{Start: 0.0, End: 1.5, Text: "  Hello there. "},
		{Start: 1.5, End: 2.0, Text: "  \n\t  "},
		{Start: 61.25, End: 3725.004, Text: " General\n\tKenobi. "},
	}
	wantSRT := "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n" +
		"2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != wantSRT {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, wantSRT)
	}
	wantVTT := "WEBVTT\n\n" +
		"00:00:00.000 --> 00:00:01.500\nHello there.\n\n" +
		"00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n"
	if got, err := SegmentsToVTT(segs); err != nil || got != wantVTT {
		t.Errorf("SegmentsToVTT = (%q, %v), want (%q, nil)", got, err, wantVTT)
	}
}

func TestSubtitleWrapsOver42CharsIntoTwoLines(t *testing.T) {
	requireSubtitleSymbols(t)
	segs := []Segment{
		{Start: 0.0, End: 4.0, Text: "This is a moderately long subtitle line that should wrap onto two lines."},
	}
	want := "1\n00:00:00,000 --> 00:00:04,000\n" +
		"This is a moderately long subtitle line\n" +
		"that should wrap onto two lines.\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != want {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleSplitsVeryLongSegmentProportionally(t *testing.T) {
	requireSubtitleSymbols(t)
	segs := []Segment{
		{
			Start: 10.0,
			End:   20.0,
			Text: "Alpha bravo charlie delta echo foxtrot golf " +
				"hotel india juliet kilo lima mike november " +
				"oscar papa quebec.",
		},
	}
	want := "1\n00:00:10,000 --> 00:00:17,404\n" +
		"Alpha bravo charlie delta echo foxtrot\n" +
		"golf hotel india juliet kilo lima mike\n\n" +
		"2\n00:00:17,404 --> 00:00:20,000\n" +
		"november oscar papa quebec.\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != want {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleSingleWordOver42CharsIntact(t *testing.T) {
	requireSubtitleSymbols(t)
	const longWord = "supercalifragilisticexpialidociousandmorewords"
	segs := []Segment{{Start: 0.0, End: 6.0, Text: "Prefix " + longWord + " suffix"}}
	want := "1\n00:00:00,000 --> 00:00:05,390\nPrefix\n" + longWord + "\n\n" +
		"2\n00:00:05,390 --> 00:00:06,000\nsuffix\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != want {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleUnicodeCharacterCount(t *testing.T) {
	requireSubtitleSymbols(t)
	const line = "Þórður sagði halló við alla í Reykjavík í"
	segs := []Segment{{Start: 0.0, End: 3.0, Text: line + " dag."}}
	want := "1\n00:00:00,000 --> 00:00:03,000\n" + line + "\ndag.\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != want {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleTimestampRoundingAndBounds(t *testing.T) {
	requireSubtitleSymbols(t)
	segs := []Segment{
		{Start: 0.0, End: 0.9996, Text: "Rounding up."},
		{Start: float32(math.NaN()), End: float32(math.Inf(1)), Text: "Non-finite."},
		{Start: -2.5, End: -1.0, Text: "Negative."},
		{Start: 5.0, End: 3.0, Text: "Reversed."},
	}
	want := "1\n00:00:00,000 --> 00:00:01,000\nRounding up.\n\n" +
		"2\n00:00:00,000 --> 00:00:00,000\nNon-finite.\n\n" +
		"3\n00:00:00,000 --> 00:00:00,000\nNegative.\n\n" +
		"4\n00:00:05,000 --> 00:00:05,000\nReversed.\n\n"
	if got, err := SegmentsToSRT(segs); err != nil || got != want {
		t.Errorf("SegmentsToSRT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleVTTEscapesArrow(t *testing.T) {
	requireSubtitleSymbols(t)
	segs := []Segment{{Start: 0.0, End: 2.0, Text: "From A --> B and x-->y"}}
	want := "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFrom A -- > B and x-- >y\n\n"
	if got, err := SegmentsToVTT(segs); err != nil || got != want {
		t.Errorf("SegmentsToVTT = (%q, %v), want (%q, nil)", got, err, want)
	}
}

func TestSubtitleTranscriptFallback(t *testing.T) {
	requireSubtitleSymbols(t)
	tr := &Transcript{Text: "  Just full text. ", Language: "en", DurationSecs: 4.5}
	if got, err := tr.ToSRT(); err != nil || got != "1\n00:00:00,000 --> 00:00:04,500\nJust full text.\n\n" {
		t.Errorf("ToSRT = (%q, %v)", got, err)
	}
	if got, err := tr.ToVTT(); err != nil || got != "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust full text.\n\n" {
		t.Errorf("ToVTT = (%q, %v)", got, err)
	}

	empty := &Transcript{Text: "   ", Language: "en", DurationSecs: 1.0}
	if got, err := empty.ToSRT(); err != nil || got != "" {
		t.Errorf("empty ToSRT = (%q, %v), want (\"\", nil)", got, err)
	}
	if got, err := empty.ToVTT(); err != nil || got != "WEBVTT\n\n" {
		t.Errorf("empty ToVTT = (%q, %v), want (\"WEBVTT\\n\\n\", nil)", got, err)
	}

	var nilTr *Transcript
	if got, err := nilTr.ToSRT(); err != nil || got != "" {
		t.Errorf("nil ToSRT = (%q, %v), want (\"\", nil)", got, err)
	}
	if got, err := nilTr.ToVTT(); err != nil || got != "WEBVTT\n\n" {
		t.Errorf("nil ToVTT = (%q, %v), want (\"WEBVTT\\n\\n\", nil)", got, err)
	}
}
