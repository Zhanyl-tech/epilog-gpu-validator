package gpu

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

// Fixture provenance: every row below is SYNTHETIC. It follows the column
// order of QueryFields and the value formats NVIDIA documents (hex clock
// bitmask, "[N/A]", "Enabled"/"Disabled"), but none of it was captured from a
// real GPU; no GPU was available when these were written. Rows captured from
// real hardware, labelled with GPU model and driver version, would make these
// tests stronger and belong next to them.

// An idle H100-shaped row: link power-managed to gen1, GpuIdle clock bit.
const idleCSV = "0, GPU-5117a000-0000-4000-8000-000000000000, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 16, 16, 1, 5, 0, 0, 0, No, 0, No, 0x0000000000000001, 34, 1, 81559, Enabled"

func mustParse(t *testing.T, csv string) Result {
	t.Helper()
	res, err := ParseCSV([]byte(csv))
	if err != nil {
		t.Fatalf("ParseCSV: %v", err)
	}
	return res
}

func TestParseIdleRowReadsEveryField(t *testing.T) {
	res := mustParse(t, idleCSV+"\n")
	if len(res.GPUs) != 1 || len(res.Malformed) != 0 {
		t.Fatalf("want 1 GPU, 0 malformed; got %+v", res)
	}
	h := res.GPUs[0]
	want := Health{
		Index: 0, UUID: "GPU-5117a000-0000-4000-8000-000000000000", Name: "NVIDIA H100 80GB HBM3",
		PCIBusID:         "00000000:18:00.0",
		PCIeWidthCurrent: 16, PCIeWidthMax: 16, PCIeGenCurrent: 1, PCIeGenMax: 5,
		ClockEventReasons: ClockReasonGPUIdle,
		TemperatureC:      34, MemUsedMiB: 1, MemTotalMiB: 81559, PersistenceMode: true,
	}
	if !reflect.DeepEqual(h, want) {
		t.Fatalf("parsed\n %+v\nwant\n %+v", h, want)
	}
}

func TestClockReasonsAreABitmaskNotNames(t *testing.T) {
	// The field is documented as a "Bitmask of active clock event reasons".
	// The old parser split it on commas and looked for names like
	// "hw_slowdown", which never appear in real output.
	row := strings.Replace(idleCSV, "0x0000000000000001", "0x0000000000000049", 1)
	h := mustParse(t, row).GPUs[0]
	if h.ClockEventReasons != ClockReasonGPUIdle|ClockReasonHWSlowdown|ClockReasonHWThermalSlowdown {
		t.Fatalf("mask = %#x", h.ClockEventReasons)
	}
	if got := ClockReasonNames(h.ClockEventReasons); !reflect.DeepEqual(got, []string{"gpu_idle", "hw_slowdown", "hw_thermal_slowdown"}) {
		t.Fatalf("names = %v", got)
	}
}

func TestParseClockReasons(t *testing.T) {
	cases := []struct {
		in   string
		want uint64
		ok   bool
	}{
		{"0x0000000000000000", 0, true},
		{"0x0000000000000001", 1, true},
		{"0X00000000000000C0", 0xc0, true},
		{"Not Active", 0, true},
		{"[N/A]", 0, false},
		{"[Not Supported]", 0, false},
		{"hw_slowdown", 0, false}, // names are not a mask
		{"0x", 0, false},
		{"0xZZ", 0, false},
		{"", 0, false},
	}
	for _, c := range cases {
		got, ok := parseClockReasons(c.in)
		if got != c.want || ok != c.ok {
			t.Errorf("parseClockReasons(%q) = %#x,%v want %#x,%v", c.in, got, ok, c.want, c.ok)
		}
	}
}

func TestUnknownClockBitsAreNamedNotDropped(t *testing.T) {
	got := ClockReasonNames(ClockReasonSWPowerCap | 1<<40)
	if len(got) != 2 || got[0] != "sw_power_cap" || !strings.HasPrefix(got[1], "unknown_bits_0x") {
		t.Fatalf("got %v", got)
	}
}

func TestNAFieldsAreUnreadableNotZero(t *testing.T) {
	// A card with ECC disabled, or a pre-Ampere card without row remapping,
	// reports these as [N/A]. Parsing them as 0 made such a card look exactly
	// like a verified-healthy one.
	row := "0, GPU-5117a000-0000-4000-8000-000000000000, Some GPU, 00000000:18:00.0, 16, 16, 5, 5, [N/A], [N/A], [N/A], [N/A], [N/A], [Not Supported], 0x0000000000000000, 34, 1, 81559, Enabled"
	h := mustParse(t, row).GPUs[0]
	want := []string{
		FieldECCUncorrVolatile, FieldECCUncorrAggregate, FieldECCCorrVolatile,
		FieldRemapPending, FieldRemapUncorrectable, FieldRemapFailure,
	}
	if !reflect.DeepEqual(h.Unreadable, want) {
		t.Fatalf("Unreadable = %v, want %v", h.Unreadable, want)
	}
	for _, f := range want {
		if h.Readable(f) {
			t.Errorf("%s reported readable", f)
		}
	}
	if !h.Readable(FieldTemperature) {
		t.Error("temperature was readable")
	}
}

func TestGarbageAndNegativeNumbersAreUnreadable(t *testing.T) {
	row := strings.Replace(idleCSV, ", 34, 1, 81559,", ", hot, -1, 81559,", 1)
	h := mustParse(t, row).GPUs[0]
	if h.Readable(FieldTemperature) || h.Readable(FieldMemUsed) {
		t.Fatalf("Unreadable = %v", h.Unreadable)
	}
}

func TestPersistenceMode(t *testing.T) {
	for in, want := range map[string]struct{ on, readable bool }{
		"Enabled":  {true, true},
		"Disabled": {false, true},
		"[N/A]":    {false, false},
		"maybe":    {false, false},
	} {
		row := strings.Replace(idleCSV, ", Enabled", ", "+in, 1)
		h := mustParse(t, row).GPUs[0]
		if h.PersistenceMode != want.on || h.Readable(FieldPersistence) != want.readable {
			t.Errorf("%q: on=%v readable=%v", in, h.PersistenceMode, h.Readable(FieldPersistence))
		}
	}
}

func TestRemapPendingAndFailureAcceptYesNoOrCount(t *testing.T) {
	// The CSV form of these two fields is not documented where this repo
	// could check; the -q form prints Yes/No. Both shapes are accepted.
	cases := []struct {
		pending, failure string
		wantPending      int64
		wantFailure      bool
	}{
		{"No", "No", 0, false},
		{"Yes", "Yes", 1, true},
		{"0", "0", 0, false},
		{"2", "1", 2, true},
	}
	for _, c := range cases {
		row := strings.Replace(idleCSV, "0, 0, 0, No, 0, No,", "0, 0, 0, "+c.pending+", 0, "+c.failure+",", 1)
		h := mustParse(t, row).GPUs[0]
		if h.RemappedRowsPending != c.wantPending || h.RemappedRowsFailure != c.wantFailure || len(h.Unreadable) != 0 {
			t.Errorf("%s/%s: pending=%d failure=%v unreadable=%v", c.pending, c.failure, h.RemappedRowsPending, h.RemappedRowsFailure, h.Unreadable)
		}
	}
}

func TestShortAndUnidentifiableRowsAreMalformedNotDropped(t *testing.T) {
	// The old parser silently skipped rows with fewer than 18 columns.
	csv := strings.Join([]string{
		idleCSV,
		"1, GPU-x, short row",
		strings.Replace(idleCSV, "0, GPU-5117", "[N/A], GPU-5117", 1),
		strings.Replace(idleCSV, "00000000:18:00.0", "[N/A]", 1),
	}, "\n")
	res := mustParse(t, csv)
	if len(res.GPUs) != 1 {
		t.Fatalf("want the one good row, got %d", len(res.GPUs))
	}
	if len(res.Malformed) != 3 {
		t.Fatalf("want 3 malformed rows, got %v", res.Malformed)
	}
}

func TestNonCSVOutputIsAParseError(t *testing.T) {
	_, err := ParseCSV([]byte("a,\"unterminated\n"))
	var pe *ParseError
	if !errors.As(err, &pe) {
		t.Fatalf("want *ParseError, got %v", err)
	}
}

// ── exit codes ─────────────────────────────────────────────────────────────

type fakeExit int

func (e fakeExit) Error() string { return "exit status " + strconv.Itoa(int(e)) }
func (e fakeExit) ExitCode() int { return int(e) }

func sourceReturning(out string, err error) *SMISource {
	return &SMISource{Binary: "/fake/nvidia-smi", Run: func(context.Context, string, ...string) ([]byte, error) {
		return []byte(out), err
	}}
}

func TestQueryReportsTheExitCode(t *testing.T) {
	for _, code := range []int{2, 6, 8, 9, 10, 12, 14, 15, 255} {
		_, err := sourceReturning("", fakeExit(code)).Query(context.Background())
		var qe *QueryError
		if !errors.As(err, &qe) {
			t.Fatalf("code %d: want *QueryError, got %T %v", code, err, err)
		}
		if qe.ExitCode != code || qe.Timeout {
			t.Errorf("code %d: got ExitCode=%d Timeout=%v", code, qe.ExitCode, qe.Timeout)
		}
		if !strings.Contains(qe.Error(), ExitCodeMeaning[code]) {
			t.Errorf("code %d: message %q lacks the manual's meaning", code, qe.Error())
		}
	}
}

func TestOnlyDocumentedHardwareExitCodesAreFaults(t *testing.T) {
	// From the nvidia-smi manual: 8 power cables, 10 interrupt issue, 14
	// infoROM corrupted, 15 fallen off the bus. 9 (driver not loaded) and 12
	// (NVML missing) are monitoring failures.
	for code := -1; code <= 255; code++ {
		want := code == 8 || code == 10 || code == 14 || code == 15
		if HardwareFaultExitCode(code) != want {
			t.Errorf("HardwareFaultExitCode(%d) = %v", code, !want)
		}
	}
}

func TestQueryArgsDoNotUseDashI(t *testing.T) {
	// -i is documented for "a single specified GPU"; the whole node is
	// queried and filtered instead.
	for _, a := range QueryArgs() {
		if a == "-i" || strings.HasPrefix(a, "--id") {
			t.Fatalf("args contain %q: %v", a, QueryArgs())
		}
	}
	if got := strings.Count(QueryArgs()[0], ",") + 1; got != len(QueryFields) {
		t.Fatalf("query asks for %d fields, parser expects %d", got, len(QueryFields))
	}
}

// ── the real exec path, with fake binaries ─────────────────────────────────

func writeScript(t *testing.T, body string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "nvidia-smi")
	if err := os.WriteFile(p, []byte("#!/bin/sh\n"+body+"\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestRealExecParsesStdout(t *testing.T) {
	bin := writeScript(t, "echo '"+idleCSV+"'")
	res, err := NewSMISource(bin, 5*time.Second).Query(context.Background())
	if err != nil || len(res.GPUs) != 1 {
		t.Fatalf("res=%+v err=%v", res, err)
	}
}

func TestRealExecCarriesExitCodeAndStderr(t *testing.T) {
	bin := writeScript(t, "echo 'Unable to determine the device handle for GPU 0000:3B:00.0' >&2; exit 15")
	_, err := NewSMISource(bin, 5*time.Second).Query(context.Background())
	var qe *QueryError
	if !errors.As(err, &qe) || qe.ExitCode != 15 || !strings.Contains(qe.Stderr, "device handle") {
		t.Fatalf("got %#v", err)
	}
}

func TestMissingBinaryIsNotAnExitCode(t *testing.T) {
	_, err := NewSMISource(filepath.Join(t.TempDir(), "absent"), time.Second).Query(context.Background())
	var qe *QueryError
	if !errors.As(err, &qe) || qe.ExitCode != -1 || HardwareFaultExitCode(qe.ExitCode) {
		t.Fatalf("got %#v", err)
	}
}

func TestTimeoutIsBoundedEvenWhenAChildHoldsStdout(t *testing.T) {
	// The shell is killed at the deadline, but its background child keeps
	// the stdout pipe open. Without Cmd.WaitDelay and the abandon timer,
	// Output() waited for the child: the audit measured 12s against a 2s
	// budget. Slurm drains a node whose Epilog times out.
	bin := writeScript(t, "sleep 5 &\nwait")
	start := time.Now()
	_, err := NewSMISource(bin, 200*time.Millisecond).Query(context.Background())
	elapsed := time.Since(start)
	var qe *QueryError
	if !errors.As(err, &qe) || !qe.Timeout {
		t.Fatalf("want a timeout QueryError, got %#v", err)
	}
	if limit := 200*time.Millisecond + abandonAfter + time.Second; elapsed > limit {
		t.Fatalf("query took %s, limit %s", elapsed, limit)
	}
}

// ── identity ───────────────────────────────────────────────────────────────

func TestNormalizeBusID(t *testing.T) {
	same := []string{"00000000:3B:00.0", "0000:3b:00.0", "0:3b:0.0", "0000:3b:00:0"}
	for _, s := range same {
		got, ok := NormalizeBusID(s)
		if !ok || got != "00000000:3B:00.0" {
			t.Errorf("NormalizeBusID(%q) = %q,%v", s, got, ok)
		}
	}
	for _, s := range []string{"", "[N/A]", "3b:00.0", "0000:3b:00.8", "GPU-abc", "0000:3g:00.0"} {
		if _, ok := NormalizeBusID(s); ok {
			t.Errorf("NormalizeBusID(%q) accepted", s)
		}
	}
}

func TestIsGPUUUID(t *testing.T) {
	if !IsGPUUUID("GPU-5117a000-0000-4000-8000-000000000000") {
		t.Error("valid UUID rejected")
	}
	for _, s := range []string{"GPU-abc123", "MIG-5117a000-0000-4000-8000-000000000000", "5117a000-0000-4000-8000-000000000000"} {
		if IsGPUUUID(s) {
			t.Errorf("%q accepted", s)
		}
	}
}

func TestMatchPicksOnlyTheJobsGPUs(t *testing.T) {
	gpus := []Health{
		{Index: 0, UUID: "GPU-aaaaaaaa-0000-4000-8000-000000000000", PCIBusID: "00000000:18:00.0"},
		{Index: 1, UUID: "GPU-bbbbbbbb-0000-4000-8000-000000000000", PCIBusID: "00000000:28:00.0"},
		{Index: 2, UUID: "GPU-cccccccc-0000-4000-8000-000000000000", PCIBusID: "00000000:38:00.0"},
	}
	targets := []Target{
		{SlurmID: "1", BusID: "00000000:38:00.0"},
		{SlurmID: "GPU-AAAAAAAA-0000-4000-8000-000000000000", UUID: "GPU-AAAAAAAA-0000-4000-8000-000000000000"},
		{SlurmID: "7", BusID: "00000000:99:00.0"},
	}
	matched, missing := Match(targets, gpus)
	if len(matched) != 2 || matched[0].Index != 2 || matched[0].SlurmID != "1" || matched[1].Index != 0 {
		t.Fatalf("matched = %+v", matched)
	}
	if len(missing) != 1 || missing[0].SlurmID != "7" {
		t.Fatalf("missing = %+v", missing)
	}
}

// ── /proc resolver ─────────────────────────────────────────────────────────

func writeProc(t *testing.T, entries map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for bdf, info := range entries {
		if err := os.MkdirAll(filepath.Join(dir, bdf), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, bdf, "information"), []byte(info), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return dir
}

func TestProcResolverMapsMinorsToBusIDs(t *testing.T) {
	// Only the "Device Minor" line is read; the others are illustrative.
	dir := writeProc(t, map[string]string{
		"0000:3b:00.0": "Model: \t\t NVIDIA H100\nDevice Minor: \t 1\n",
		"0000:18:00.0": "Model: \t\t NVIDIA H100\nDevice Minor: \t 0\n",
		"not-a-gpu":    "Device Minor: 9\n",
	})
	got, err := ProcResolver{Dir: dir}.DeviceMinors()
	if err != nil {
		t.Fatal(err)
	}
	want := map[int]string{0: "00000000:18:00.0", 1: "00000000:3B:00.0"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestProcResolverRefusesAnUntrustworthyMap(t *testing.T) {
	cases := map[string]map[string]string{
		"duplicate minor": {"0000:18:00.0": "Device Minor: 0\n", "0000:28:00.0": "Device Minor: 0\n"},
		"no minor line":   {"0000:18:00.0": "Model: x\n"},
		"garbage minor":   {"0000:18:00.0": "Device Minor: zero\n"},
		"no GPUs":         {},
	}
	for name, entries := range cases {
		if _, err := (ProcResolver{Dir: writeProc(t, entries)}).DeviceMinors(); err == nil {
			t.Errorf("%s: want error", name)
		}
	}
	if _, err := (ProcResolver{Dir: filepath.Join(t.TempDir(), "absent")}).DeviceMinors(); err == nil {
		t.Error("absent dir: want error")
	}
}

// ── simulator ──────────────────────────────────────────────────────────────

func TestSimulatorPrintsWhatTheParserReads(t *testing.T) {
	sim := NewSim(ScenarioHealthy, 4)
	res, err := sim.Source().Query(context.Background())
	if err != nil || len(res.GPUs) != 4 || len(res.Malformed) != 0 {
		t.Fatalf("res=%+v err=%v", res, err)
	}
	for _, h := range res.GPUs {
		// The idle model: gen1 of gen5, GpuIdle only, nothing unreadable.
		if h.PCIeGenCurrent != 1 || h.ClockEventReasons != ClockReasonGPUIdle || len(h.Unreadable) != 0 {
			t.Errorf("gpu%d: %+v", h.Index, h)
		}
	}
}

func TestSimulatorDeviceMinorLayouts(t *testing.T) {
	for _, tc := range []struct {
		reversed bool
		want     int // nvidia-smi index of /dev/nvidia0
	}{{false, 0}, {true, 3}} {
		sim := NewSim(ScenarioHealthy, 4)
		sim.MinorsReversed = tc.reversed
		minors, _ := sim.DeviceMinors()
		res, _ := sim.Source().Query(context.Background())
		matched, _ := Match([]Target{{SlurmID: "0", BusID: minors[0]}}, res.GPUs)
		if len(matched) != 1 || matched[0].Index != tc.want {
			t.Fatalf("reversed=%v: /dev/nvidia0 should be nvidia-smi gpu%d, got %+v", tc.reversed, tc.want, matched)
		}
	}
}

func TestSimulatorFaultIsOnTheFirstGPUInPCIOrder(t *testing.T) {
	// Whatever the minor layout, the fault is on NVML index 0.
	for _, reversed := range []bool{false, true} {
		sim := NewSim(ScenarioECC, 4)
		sim.MinorsReversed = reversed
		res, _ := sim.Source().Query(context.Background())
		for _, h := range res.GPUs {
			if faulty := h.ECCUncorrectableVolatile > 0; faulty != (h.Index == 0) {
				t.Fatalf("reversed=%v: gpu%d volatile ECC %d", reversed, h.Index, h.ECCUncorrectableVolatile)
			}
		}
	}
}

func TestEveryScenarioRuns(t *testing.T) {
	for _, s := range Scenarios {
		if !ValidScenario(string(s)) {
			t.Errorf("%s not valid", s)
		}
		res, err := NewSim(s, 4).Source().Query(context.Background())
		if err == nil && len(res.Malformed) != 0 {
			t.Errorf("%s: malformed rows %v", s, res.Malformed)
		}
	}
	if ValidScenario("ecc-uncorrectable") {
		t.Error("an unknown scenario must be rejected, not run as healthy")
	}
}
