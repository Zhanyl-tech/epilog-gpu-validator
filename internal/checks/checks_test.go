package checks

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
)

// healthy is a GPU with every field readable and every value clean,
// including a link at max generation (as under load).
func healthy() gpu.Health {
	return gpu.Health{
		Index: 0, UUID: "GPU-abc", Name: "NVIDIA H100 80GB HBM3", PCIBusID: "00000000:18:00.0",
		PCIeWidthCurrent: 16, PCIeWidthMax: 16,
		PCIeGenCurrent: 5, PCIeGenMax: 5,
		TemperatureC: 42, MemUsedMiB: 0, MemTotalMiB: 81559,
		PersistenceMode: true,
	}
}

// idle is what a healthy GPU is documented to look like at Epilog time: link
// generation power-managed down, GpuIdle clock bit set.
func idle() gpu.Health {
	h := healthy()
	h.PCIeGenCurrent = 1
	h.ClockEventReasons = gpu.ClockReasonGPUIdle
	return h
}

func worst(fs []Finding) Severity { return Summarize(fs).Worst }

func hasCheck(fs []Finding, name string) *Finding {
	for i := range fs {
		if fs[i].Check == name {
			return &fs[i]
		}
	}
	return nil
}

// ── The property that matters most ─────────────────────────────────────────

func TestHealthyGPUProducesNoFindings(t *testing.T) {
	// A false positive drains a working node. On a busy cluster Epilog runs
	// thousands of times a day, so anything but silence here is an outage
	// generator.
	if fs := Evaluate(healthy(), DefaultConfig()); len(fs) != 0 {
		t.Fatalf("healthy GPU produced findings: %v", fs)
	}
}

func TestIdleHealthyGPUNeverDrains(t *testing.T) {
	// Regression for the fleet-wide false positive: pcie.link.gen.current
	// "may be reduced when the GPU is not in use", and the Epilog only runs
	// when it is not. This used to be Degraded, which drained every node
	// after every job under --enforce.
	r := Summarize(Evaluate(idle(), DefaultConfig()))
	if r.Drain {
		t.Fatalf("an idle healthy GPU must not drain: %s", r.Reason)
	}
	if r.Worst != Unknown {
		t.Fatalf("worst = %s, want unknown (link state not determinable at idle)", r.Worst)
	}
	for _, f := range r.Findings {
		if f.Check != "pcie-gen" {
			t.Errorf("unexpected finding %v (the GpuIdle bit must be ignored)", f)
		}
	}
}

func TestTransientConditionsNeverDrain(t *testing.T) {
	cases := []struct {
		name   string
		mutate func(*gpu.Health)
	}{
		{"software thermal slowdown", func(h *gpu.Health) {
			h.ClockEventReasons = gpu.ClockReasonSWThermalSlowdown
			h.TemperatureC = 86
		}},
		{"software power cap", func(h *gpu.Health) { h.ClockEventReasons = gpu.ClockReasonSWPowerCap }},
		{"board limit and reliability policy", func(h *gpu.Health) {
			h.ClockEventReasons = gpu.ClockReasonBoardLimit | gpu.ClockReasonReliability
		}},
		{"hw_slowdown bit alone", func(h *gpu.Health) { h.ClockEventReasons = gpu.ClockReasonHWSlowdown }},
		{"unknown clock bits", func(h *gpu.Health) { h.ClockEventReasons = 1 << 50 }},
		{"hot but not throttling", func(h *gpu.Health) { h.TemperatureC = 91 }},
		{"historic ECC, none since driver load", func(h *gpu.Health) { h.ECCUncorrectableAggregate = 42 }},
		{"pending row remap", func(h *gpu.Health) { h.RemappedRowsPending = 1 }},
		{"rows remapped in the past", func(h *gpu.Health) { h.RemappedRowsUncorrectable = 3 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := healthy()
			tc.mutate(&h)
			r := Summarize(Evaluate(h, DefaultConfig()))
			if r.Drain {
				t.Fatalf("%s must not drain the node (severity %s)", tc.name, r.Worst)
			}
			if r.Worst != Transient {
				t.Errorf("%s: want transient, got %s", tc.name, r.Worst)
			}
		})
	}
}

func TestPersistentFaultsDrain(t *testing.T) {
	cases := []struct {
		name   string
		want   Severity
		mutate func(*gpu.Health)
	}{
		{"uncorrectable ECC since driver load", Fatal, func(h *gpu.Health) { h.ECCUncorrectableVolatile = 1 }},
		{"row remap failure", Fatal, func(h *gpu.Health) { h.RemappedRowsFailure = true }},
		{"hw thermal slowdown at idle", Degraded, func(h *gpu.Health) {
			h.ClockEventReasons = gpu.ClockReasonGPUIdle | gpu.ClockReasonHWSlowdown | gpu.ClockReasonHWThermalSlowdown
		}},
		{"hw power brake at idle", Degraded, func(h *gpu.Health) {
			h.ClockEventReasons = gpu.ClockReasonHWPowerBrakeSlowdown
		}},
		{"memory left allocated", Degraded, func(h *gpu.Health) { h.MemUsedMiB = 40000 }},
		{"correctable ECC above threshold", Degraded, func(h *gpu.Health) { h.ECCCorrectableVolatile = 5000 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := idle()
			tc.mutate(&h)
			r := Summarize(Evaluate(h, DefaultConfig()))
			if !r.Drain {
				t.Fatalf("%s should drain, got severity %s", tc.name, r.Worst)
			}
			if r.Worst != tc.want {
				t.Errorf("expected %s, got %s", tc.want, r.Worst)
			}
		})
	}
}

// ── Never drain on ignorance ───────────────────────────────────────────────

func allFields() []string { return append([]string(nil), gpu.QueryFields[4:]...) }

func TestUnreadableFieldsAreNeverUsedAsEvidence(t *testing.T) {
	// Every value below would drain if it were believed. Every field is
	// marked unreadable, so none of it may be.
	h := gpu.Health{
		Index: 0, UUID: "GPU-abc",
		PCIeWidthCurrent: 1, PCIeWidthMax: 16, PCIeGenCurrent: 1, PCIeGenMax: 5,
		ECCUncorrectableVolatile: 9, ECCUncorrectableAggregate: 9, ECCCorrectableVolatile: 1e6,
		RemappedRowsPending: 1, RemappedRowsUncorrectable: 9, RemappedRowsFailure: true,
		ClockEventReasons: gpu.ClockReasonHWThermalSlowdown, TemperatureC: 120,
		MemUsedMiB: 80000, MemTotalMiB: 81559,
		Unreadable: allFields(),
	}
	cfg := DefaultConfig()
	cfg.DrainOnPCIeWidth, cfg.DrainOnPendingRemap = true, true
	r := Summarize(Evaluate(h, cfg))
	if r.Drain || r.Worst != Unknown || !r.Incomplete {
		t.Fatalf("unreadable fields drove a decision: %+v", r)
	}
	if f := hasCheck(r.Findings, "unreadable"); f == nil || !strings.Contains(f.Detail, gpu.FieldECCUncorrVolatile) {
		t.Fatalf("the unreadable fields must be named: %v", r.Findings)
	}
}

func TestNAECCIsUnknownNotHealthy(t *testing.T) {
	// A card with ECC off reports [N/A]; that must not read as "0 errors".
	h := healthy()
	h.Unreadable = []string{gpu.FieldECCUncorrVolatile, gpu.FieldECCUncorrAggregate, gpu.FieldECCCorrVolatile}
	r := Summarize(Evaluate(h, DefaultConfig()))
	if r.Worst != Unknown || r.Drain {
		t.Fatalf("got %+v", r)
	}
}

func TestLifetimeECCWithUnreadableVolatileIsUnknown(t *testing.T) {
	h := healthy()
	h.ECCUncorrectableAggregate = 5
	h.Unreadable = []string{gpu.FieldECCUncorrVolatile}
	f := hasCheck(Evaluate(h, DefaultConfig()), "ecc-uncorrectable-history")
	if f == nil || f.Severity != Unknown {
		t.Fatalf("cannot claim 'none since driver load' without the volatile counter: %v", f)
	}
}

func TestPersistenceOffMakesVolatileECCUnreliable(t *testing.T) {
	// nvidia-smi: volatile counters count "since the last driver load", and
	// "On Linux the driver unloads when no active clients exist" unless
	// persistence mode is on.
	h := healthy()
	h.PersistenceMode = false
	r := Summarize(Evaluate(h, DefaultConfig()))
	if f := hasCheck(r.Findings, "ecc-volatile-unreliable"); f == nil || f.Severity != Unknown {
		t.Fatalf("want an unknown ecc-volatile-unreliable finding, got %v", r.Findings)
	}
	if r.Drain {
		t.Fatal("persistence mode off is not a hardware fault")
	}

	h.ECCUncorrectableAggregate = 4
	if f := hasCheck(Evaluate(h, DefaultConfig()), "ecc-uncorrectable-history"); f == nil || f.Severity != Unknown {
		t.Fatalf("lifetime errors with a possibly-reset volatile counter are unknown: %v", f)
	}

	h.ECCUncorrectableVolatile = 1
	if worst(Evaluate(h, DefaultConfig())) != Fatal {
		t.Fatal("a non-zero volatile count is still positive evidence")
	}
}

// ── Distinctions that are easy to get wrong ────────────────────────────────

func TestVolatileAndAggregateECCAreDifferentFacts(t *testing.T) {
	// Errors since driver load mean the card is failing now. Lifetime errors
	// on a card clean since then do not justify draining.
	now := healthy()
	now.ECCUncorrectableVolatile = 2
	now.ECCUncorrectableAggregate = 2
	if !Summarize(Evaluate(now, DefaultConfig())).Drain {
		t.Error("uncorrectable ECC since driver load must drain")
	}

	historic := healthy()
	historic.ECCUncorrectableAggregate = 99
	r := Summarize(Evaluate(historic, DefaultConfig()))
	if r.Drain {
		t.Error("lifetime-only ECC must not drain")
	}
	if f := hasCheck(r.Findings, "ecc-uncorrectable-history"); f == nil || f.Severity != Transient {
		t.Error("but it should still be reported")
	}
}

func TestSoftwareAndHardwareThrottleAreDifferentFacts(t *testing.T) {
	sw := idle()
	sw.ClockEventReasons |= gpu.ClockReasonSWThermalSlowdown
	if Summarize(Evaluate(sw, DefaultConfig())).Drain {
		t.Error("software throttling is the card protecting itself; must not drain")
	}

	hw := idle()
	hw.ClockEventReasons |= gpu.ClockReasonHWThermalSlowdown
	if !Summarize(Evaluate(hw, DefaultConfig())).Drain {
		t.Error("hardware thermal slowdown on an idle GPU means a hard limit is being hit; must drain")
	}
}

func TestIgnoredClockBitsProduceNothing(t *testing.T) {
	h := healthy()
	h.ClockEventReasons = gpu.ClockReasonGPUIdle | gpu.ClockReasonApplicationsClocksSetting | gpu.ClockReasonSyncBoost
	if fs := Evaluate(h, DefaultConfig()); len(fs) != 0 {
		t.Fatalf("idle/app-clocks/sync-boost are not findings: %v", fs)
	}
}

func TestCorrectableECCOnlyDrainsAboveThreshold(t *testing.T) {
	h := healthy()
	h.ECCCorrectableVolatile = 12
	if Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("a handful of correctable errors is normal")
	}
	h.ECCCorrectableVolatile = 5000
	if !Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("above the site threshold it drains")
	}
	cfg := DefaultConfig()
	cfg.MaxCorrectableECC = 0
	if Summarize(Evaluate(h, cfg)).Drain {
		t.Error("0 disables the check")
	}
}

// ── PCIe ───────────────────────────────────────────────────────────────────

func TestPCIeBelowMaxIsNotEvidenceAtIdle(t *testing.T) {
	h := idle()
	h.PCIeWidthCurrent, h.PCIeGenCurrent = 8, 3

	r := Summarize(Evaluate(h, DefaultConfig()))
	if r.Drain {
		t.Fatal("NVIDIA documents link width and gen may be reduced when not in use; must not drain by default")
	}
	for _, c := range []string{"pcie-width", "pcie-gen"} {
		if f := hasCheck(r.Findings, c); f == nil || f.Severity != Unknown {
			t.Errorf("%s should be reported as unknown, got %v", c, f)
		}
	}
}

func TestSitesCanOptIntoDrainingOnWidthButNeverOnGen(t *testing.T) {
	cfg := DefaultConfig()
	cfg.DrainOnPCIeWidth = true

	narrow := idle()
	narrow.PCIeWidthCurrent = 8
	if r := Summarize(Evaluate(narrow, cfg)); !r.Drain || r.Worst != Degraded {
		t.Fatalf("--drain-on-pcie-width should drain a narrow link, got %+v", r)
	}
	if Summarize(Evaluate(idle(), cfg)).Drain {
		t.Fatal("link generation is power-managed; it must never drain, even with the opt-in")
	}
}

func TestSitesWithWiredDownSlotsCanSilencePCIe(t *testing.T) {
	h := idle()
	h.PCIeWidthCurrent = 8
	cfg := DefaultConfig()
	cfg.AllowPCIeDowngrade = true
	cfg.DrainOnPCIeWidth = true // AllowPCIeDowngrade wins: nothing is reported
	if fs := Evaluate(h, cfg); hasCheck(fs, "pcie-width") != nil || hasCheck(fs, "pcie-gen") != nil {
		t.Fatalf("AllowPCIeDowngrade should suppress PCIe findings: %v", fs)
	}
}

// ── Remap, sharing ─────────────────────────────────────────────────────────

func TestPendingRemapIsOptionallyDrainable(t *testing.T) {
	h := healthy()
	h.RemappedRowsPending = 1
	if Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("default: a pending remap should not drain")
	}
	cfg := DefaultConfig()
	cfg.DrainOnPendingRemap = true
	if !Summarize(Evaluate(h, cfg)).Drain {
		t.Error("DrainOnPendingRemap should make it drain")
	}
}

func TestRemapFailureOutranksTheRemapCount(t *testing.T) {
	h := healthy()
	h.RemappedRowsFailure = true
	h.RemappedRowsUncorrectable = 9
	fs := Evaluate(h, DefaultConfig())
	if hasCheck(fs, "row-remap-failure") == nil || hasCheck(fs, "row-remap-uncorrectable") != nil {
		t.Fatalf("got %v", fs)
	}
}

func TestSharedGPUsDoNotDrainOnHeldMemory(t *testing.T) {
	// With gres/mps or shards, another job may still be using the GPU.
	h := healthy()
	h.MemUsedMiB = 40000
	cfg := DefaultConfig()
	cfg.SharedGPUs = true
	r := Summarize(Evaluate(h, cfg))
	if r.Drain || r.Worst != Unknown {
		t.Fatalf("shared GPU memory is not evidence: %+v", r)
	}
}

// ── Query failures ─────────────────────────────────────────────────────────

func TestQueryFailureClassification(t *testing.T) {
	cases := []struct {
		name  string
		err   error
		want  Severity
		check string
	}{
		{"fell off the bus", &gpu.QueryError{ExitCode: 15}, Fatal, "nvidia-smi-exit-15"},
		{"infoROM corrupted", &gpu.QueryError{ExitCode: 14}, Fatal, "nvidia-smi-exit-14"},
		{"interrupt issue", &gpu.QueryError{ExitCode: 10}, Fatal, "nvidia-smi-exit-10"},
		{"power cables", &gpu.QueryError{ExitCode: 8}, Fatal, "nvidia-smi-exit-8"},
		{"driver not loaded", &gpu.QueryError{ExitCode: 9}, Unknown, "query-failed"},
		{"NVML missing", &gpu.QueryError{ExitCode: 12}, Unknown, "query-failed"},
		{"bad field", &gpu.QueryError{ExitCode: 2}, Unknown, "query-failed"},
		{"internal error", &gpu.QueryError{ExitCode: 255}, Unknown, "query-failed"},
		{"not found", &gpu.QueryError{ExitCode: -1, Err: errors.New("no such file")}, Unknown, "query-failed"},
		// A killed process can report anything; a timeout is never evidence.
		{"timeout", &gpu.QueryError{ExitCode: 15, Timeout: true, Err: context.DeadlineExceeded}, Unknown, "query-failed"},
		{"unparseable", &gpu.ParseError{Err: errors.New("bad csv")}, Unknown, "query-failed"},
		{"anything else", errors.New("boom"), Unknown, "query-failed"},
	}
	for _, c := range cases {
		f := QueryFailure(c.err)
		if f.Severity != c.want || f.Check != c.check || f.GPULabel != NodeLabel {
			t.Errorf("%s: got %s/%s/%s", c.name, f.Check, f.Severity, f.GPULabel)
		}
	}
}

// ── The drain reason ───────────────────────────────────────────────────────

func TestDrainReasonNamesTheGPUAndTheCheck(t *testing.T) {
	h := healthy()
	h.Index = 3
	h.ECCUncorrectableVolatile = 1
	r := Summarize(Evaluate(h, DefaultConfig()))
	if r.Reason != "epilog-gpu-validator fatal: ecc-uncorrectable:gpu3" {
		t.Errorf("reason = %q", r.Reason)
	}
}

func TestDrainReasonNamesEveryFaultyGPU(t *testing.T) {
	// The reason used to be deduplicated by check alone, so a second GPU
	// with the same fault vanished from `sinfo -R`.
	var fs []Finding
	for _, i := range []int{3, 0} {
		h := healthy()
		h.Index = i
		h.ECCUncorrectableVolatile = 2
		fs = append(fs, Evaluate(h, DefaultConfig())...)
	}
	if got := Summarize(fs).Reason; got != "epilog-gpu-validator fatal: ecc-uncorrectable:gpu0,gpu3" {
		t.Fatalf("reason = %q", got)
	}
}

func TestDrainReasonSortsGPUsNumerically(t *testing.T) {
	var fs []Finding
	for _, i := range []int{10, 2, 1} {
		fs = append(fs, Finding{GPUIndex: i, GPULabel: fmt.Sprintf("gpu%d", i), Check: "ecc-uncorrectable", Severity: Fatal})
	}
	fs = append(fs, Finding{GPULabel: NodeLabel, Check: "nvidia-smi-exit-15", Severity: Fatal})
	want := "epilog-gpu-validator fatal: ecc-uncorrectable:gpu1,gpu2,gpu10 nvidia-smi-exit-15:node"
	if got := Summarize(fs).Reason; got != want {
		t.Fatalf("reason = %q, want %q", got, want)
	}
}

func TestDrainReasonIsBoundedAndSaysSo(t *testing.T) {
	var fs []Finding
	for i := 0; i < 64; i++ {
		fs = append(fs, Finding{GPUIndex: i, GPULabel: fmt.Sprintf("gpu%d", i), Check: "ecc-uncorrectable", Severity: Fatal})
	}
	full := "epilog-gpu-validator fatal: " + Describe(fs, Fatal)
	if len(full) <= MaxReasonLen {
		t.Fatalf("setup: the untruncated reason (%d chars) must exceed the bound to test it", len(full))
	}
	r := Summarize(fs)
	if len(r.Reason) != MaxReasonLen || !strings.HasSuffix(r.Reason, "...") {
		t.Fatalf("reason is %d chars (%q); want exactly %d ending in ...", len(r.Reason), r.Reason, MaxReasonLen)
	}
}

func TestReasonOnlyIncludesWorstSeverity(t *testing.T) {
	h := idle()
	h.ECCUncorrectableVolatile = 1                          // fatal
	h.ClockEventReasons |= gpu.ClockReasonSWThermalSlowdown // transient
	r := Summarize(Evaluate(h, DefaultConfig()))
	if r.Worst != Fatal {
		t.Fatalf("worst should be fatal, got %s", r.Worst)
	}
	if strings.Contains(r.Reason, "throttle") || strings.Contains(r.Reason, "pcie") {
		t.Errorf("reason should not dilute the fatal finding: %q", r.Reason)
	}
}

func TestNoFindingsMeansEmptyReason(t *testing.T) {
	r := Summarize(nil)
	if r.Drain || r.Reason != "" || r.Worst != OK || r.Incomplete {
		t.Fatalf("clean result should be silent, got %+v", r)
	}
}

// ── Severity ordering ──────────────────────────────────────────────────────

func TestOnlyDegradedAndAboveDrain(t *testing.T) {
	for _, s := range []Severity{OK, Unknown, Transient} {
		if s.ShouldDrain() {
			t.Errorf("%s must not drain", s)
		}
	}
	for _, s := range []Severity{Degraded, Fatal} {
		if !s.ShouldDrain() {
			t.Errorf("%s must drain", s)
		}
	}
}

func TestWorstSeverityWins(t *testing.T) {
	fs := []Finding{
		{Check: "a", Severity: Transient},
		{Check: "b", Severity: Fatal},
		{Check: "c", Severity: Degraded},
	}
	if got := worst(fs); got != Fatal {
		t.Fatalf("expected fatal, got %s", got)
	}
}
