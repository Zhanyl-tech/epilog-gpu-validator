package slurm

import (
	"reflect"
	"testing"
)

func TestGPUIndicesParsesEveryFormatSlurmEmits(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want []int
	}{
		{"comma list", "0,1,2,3", []int{0, 1, 2, 3}},
		{"single", "2", []int{2}},
		{"range", "0-3", []int{0, 1, 2, 3}},
		{"mixed", "0,2-4,7", []int{0, 2, 3, 4, 7}},
		{"spaces", " 1, 3 ", []int{1, 3}},
		{"empty", "", nil},
		// Slurm sets this literal when the job had no GPU device files.
		{"NoDevFiles", "NoDevFiles", nil},
		// Depending on version and GresTypes, entries can be UUIDs. Those are
		// unusable as -i arguments, so they are skipped rather than guessed at.
		{"uuids skipped", "GPU-abc123,GPU-def456", nil},
		{"mixed uuid and index", "GPU-abc123,2", []int{2}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Env{RawGPUs: tc.raw}.GPUIndices()
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("GPUIndices(%q) = %v, want %v", tc.raw, got, tc.want)
			}
		})
	}
}

func TestGPUIndicesBoundsRunawayRanges(t *testing.T) {
	// A malformed range must not allocate unboundedly.
	got := Env{RawGPUs: "0-100000"}.GPUIndices()
	if len(got) > 64 {
		t.Fatalf("range expansion should be bounded, got %d entries", len(got))
	}
}

func TestSanitizeReasonStripsCharactersSlurmRejects(t *testing.T) {
	got := sanitizeReason(`epilog "gpu0" ecc; rm -rf /$(x)`)
	for _, bad := range []string{`"`, ";", "$", "(", ")"} {
		if contains(got, bad) {
			t.Errorf("reason still contains %q: %q", bad, got)
		}
	}
}

func TestSanitizeReasonIsBounded(t *testing.T) {
	long := ""
	for i := 0; i < 400; i++ {
		long += "x"
	}
	if got := sanitizeReason(long); len(got) > 200 {
		t.Fatalf("reason not truncated: %d chars", len(got))
	}
}

func TestSanitizeReasonCollapsesWhitespace(t *testing.T) {
	if got := sanitizeReason("a   b\n\nc"); got != "a b c" {
		t.Errorf("got %q", got)
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}
