package kjarni

import (
	"math"
	"testing"
)

func TestSubtitleEmptySegments(t *testing.T) {
	if got := SegmentsToSRT(nil); got != "" {
		t.Errorf("SegmentsToSRT(nil) = %q, want empty", got)
	}
	if got := SegmentsToVTT(nil); got != "WEBVTT\n\n" {
		t.Errorf("SegmentsToVTT(nil) = %q, want WEBVTT header", got)
	}
}

func TestSubtitleShortAndEmptySegments(t *testing.T) {
	segs := []Segment{
		{Start: 0.0, End: 1.5, Text: "  Hello there. "},
		{Start: 1.5, End: 2.0, Text: "  \n\t  "},
		{Start: 61.25, End: 3725.004, Text: " General\n\tKenobi. "},
	}
	wantSRT := "1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n" +
		"2\n00:01:01,250 --> 01:02:05,004\nGeneral Kenobi.\n\n"
	if got := SegmentsToSRT(segs); got != wantSRT {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, wantSRT)
	}
	wantVTT := "WEBVTT\n\n" +
		"00:00:00.000 --> 00:00:01.500\nHello there.\n\n" +
		"00:01:01.250 --> 01:02:05.004\nGeneral Kenobi.\n\n"
	if got := SegmentsToVTT(segs); got != wantVTT {
		t.Errorf("SegmentsToVTT =\n%q\nwant:\n%q", got, wantVTT)
	}
}

func TestSubtitleWrapsOver42CharsIntoTwoLines(t *testing.T) {
	segs := []Segment{
		{Start: 0.0, End: 4.0, Text: "This is a moderately long subtitle line that should wrap onto two lines."},
	}
	want := "1\n00:00:00,000 --> 00:00:04,000\n" +
		"This is a moderately long subtitle line\n" +
		"that should wrap onto two lines.\n\n"
	if got := SegmentsToSRT(segs); got != want {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleSplitsVeryLongSegmentProportionally(t *testing.T) {
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
	if got := SegmentsToSRT(segs); got != want {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleSingleWordOver42CharsIntact(t *testing.T) {
	const longWord = "supercalifragilisticexpialidociousandmorewords"
	segs := []Segment{{Start: 0.0, End: 6.0, Text: "Prefix " + longWord + " suffix"}}
	want := "1\n00:00:00,000 --> 00:00:05,390\nPrefix\n" + longWord + "\n\n" +
		"2\n00:00:05,390 --> 00:00:06,000\nsuffix\n\n"
	if got := SegmentsToSRT(segs); got != want {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleUnicodeCharacterCount(t *testing.T) {
	const line = "Þórður sagði halló við alla í Reykjavík í"
	segs := []Segment{{Start: 0.0, End: 3.0, Text: line + " dag."}}
	want := "1\n00:00:00,000 --> 00:00:03,000\n" + line + "\ndag.\n\n"
	if got := SegmentsToSRT(segs); got != want {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleTimestampRoundingAndBounds(t *testing.T) {
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
	if got := SegmentsToSRT(segs); got != want {
		t.Errorf("SegmentsToSRT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleVTTEscapesArrow(t *testing.T) {
	segs := []Segment{{Start: 0.0, End: 2.0, Text: "From A --> B and x-->y"}}
	want := "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nFrom A -- > B and x-- >y\n\n"
	if got := SegmentsToVTT(segs); got != want {
		t.Errorf("SegmentsToVTT =\n%q\nwant:\n%q", got, want)
	}
}

func TestSubtitleTranscriptFallback(t *testing.T) {
	tr := &Transcript{Text: "  Just full text. ", Language: "en", DurationSecs: 4.5}
	if got, want := tr.ToSRT(), "1\n00:00:00,000 --> 00:00:04,500\nJust full text.\n\n"; got != want {
		t.Errorf("ToSRT =\n%q\nwant:\n%q", got, want)
	}
	if got, want := tr.ToVTT(), "WEBVTT\n\n00:00:00.000 --> 00:00:04.500\nJust full text.\n\n"; got != want {
		t.Errorf("ToVTT =\n%q\nwant:\n%q", got, want)
	}

	empty := &Transcript{Text: "   ", Language: "en", DurationSecs: 1.0}
	if got := empty.ToSRT(); got != "" {
		t.Errorf("empty ToSRT = %q, want empty", got)
	}
	if got := empty.ToVTT(); got != "WEBVTT\n\n" {
		t.Errorf("empty ToVTT = %q, want header", got)
	}
}
