package checks

import (
	"strings"
	"testing"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
)

func healthy() gpu.Health {
	return gpu.Health{
		Present: true, Index: 0, UUID: "GPU-abc", Name: "NVIDIA H100 80GB HBM3",
		PCIeWidthCurrent: 16, PCIeWidthMax: 16,
		PCIeGenCurrent: 5, PCIeGenMax: 5,
		TemperatureC: 42, MemUsedMiB: 0, MemTotalMiB: 81559,
		PersistenceM: true,
	}
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

func TestTransientConditionsNeverDrain(t *testing.T) {
	cases := []struct {
		name  string
		mutate func(*gpu.Health)
	}{
		{"software thermal throttle", func(h *gpu.Health) {
			h.ThrottleReasons = []string{"sw_thermal_slowdown"}
			h.TemperatureC = 86
		}},
		{"software power cap", func(h *gpu.Health) {
			h.ThrottleReasons = []string{"sw_power_cap"}
		}},
		{"hot but not throttling", func(h *gpu.Health) { h.TemperatureC = 91 }},
		{"historic ECC, none this boot", func(h *gpu.Health) {
			h.ECCUncorrectableAggregate = 42
		}},
		{"pending row remap", func(h *gpu.Health) { h.RemappedRowsPending = 2 }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := healthy()
			tc.mutate(&h)
			r := Summarize(Evaluate(h, DefaultConfig()))
			if r.Drain {
				t.Fatalf("%s must not drain the node (severity %s)", tc.name, r.Worst)
			}
		})
	}
}

func TestPersistentFaultsDrain(t *testing.T) {
	cases := []struct {
		name  string
		want  Severity
		mutate func(*gpu.Health)
	}{
		{"uncorrectable ECC this boot", Fatal, func(h *gpu.Health) {
			h.ECCUncorrectableVolatile = 1
		}},
		{"row remap failure", Fatal, func(h *gpu.Health) {
			h.RemappedRowsFailure = true
		}},
		{"device absent", Fatal, func(h *gpu.Health) { h.Present = false }},
		{"pcie width downgrade", Degraded, func(h *gpu.Health) {
			h.PCIeWidthCurrent = 8
		}},
		{"pcie gen downgrade", Degraded, func(h *gpu.Health) {
			h.PCIeGenCurrent = 3
		}},
		{"hardware slowdown", Degraded, func(h *gpu.Health) {
			h.ThrottleReasons = []string{"hw_thermal_slowdown"}
		}},
		{"memory left allocated", Degraded, func(h *gpu.Health) {
			h.MemUsedMiB = 40000
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := healthy()
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

// ── Distinctions that are easy to get wrong ────────────────────────────────

func TestVolatileAndAggregateECCAreDifferentFacts(t *testing.T) {
	// Errors since boot mean the card is failing now. Lifetime errors on a card
	// that has been clean since its last reset do not justify draining.
	now := healthy()
	now.ECCUncorrectableVolatile = 2
	now.ECCUncorrectableAggregate = 2
	if !Summarize(Evaluate(now, DefaultConfig())).Drain {
		t.Error("uncorrectable ECC this power cycle must drain")
	}

	historic := healthy()
	historic.ECCUncorrectableAggregate = 99
	r := Summarize(Evaluate(historic, DefaultConfig()))
	if r.Drain {
		t.Error("lifetime-only ECC must not drain")
	}
	if f := hasCheck(r.Findings, "ecc-uncorrectable-history"); f == nil {
		t.Error("but it should still be reported")
	}
}

func TestSoftwareAndHardwareThrottleAreDifferentFacts(t *testing.T) {
	sw := healthy()
	sw.ThrottleReasons = []string{"sw_thermal_slowdown"}
	if Summarize(Evaluate(sw, DefaultConfig())).Drain {
		t.Error("software throttling is the card protecting itself; must not drain")
	}

	hw := healthy()
	hw.ThrottleReasons = []string{"hw_thermal_slowdown"}
	if !Summarize(Evaluate(hw, DefaultConfig())).Drain {
		t.Error("hardware slowdown means a hard limit was exceeded; must drain")
	}
}

func TestCorrectableECCOnlyDrainsAboveThreshold(t *testing.T) {
	// Correctable errors are normal in small numbers. Draining on the first one
	// would drain the fleet.
	h := healthy()
	h.ECCCorrectableVolatile = 12
	if Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("a handful of correctable errors is normal")
	}

	h.ECCCorrectableVolatile = 5000
	if !Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("a spike in correctable errors precedes an uncorrectable one")
	}
}

// ── Configurability ────────────────────────────────────────────────────────

func TestSitesWithWiredDownSlotsCanOptOut(t *testing.T) {
	// Some chassis genuinely wire cards below full width. Such a site would
	// otherwise drain every node on the first job completion.
	h := healthy()
	h.PCIeWidthCurrent = 8

	if !Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Fatal("setup: should drain by default")
	}

	cfg := DefaultConfig()
	cfg.AllowPCIeDowngrade = true
	if Summarize(Evaluate(h, cfg)).Drain {
		t.Error("AllowPCIeDowngrade should suppress the drain")
	}
}

func TestPendingRemapIsOptionallyDrainable(t *testing.T) {
	h := healthy()
	h.RemappedRowsPending = 3

	if Summarize(Evaluate(h, DefaultConfig())).Drain {
		t.Error("default: a pending remap should not drain")
	}

	cfg := DefaultConfig()
	cfg.DrainOnPendingRemap = true
	if !Summarize(Evaluate(h, cfg)).Drain {
		t.Error("DrainOnPendingRemap should make it drain")
	}
}

// ── The drain reason ───────────────────────────────────────────────────────

func TestDrainReasonNamesTheGPUAndTheCheck(t *testing.T) {
	// Whoever runs `sinfo -R` at 3am gets this string and nothing else.
	h := healthy()
	h.Index = 3
	h.ECCUncorrectableVolatile = 1

	r := Summarize(Evaluate(h, DefaultConfig()))
	if !strings.Contains(r.Reason, "gpu3") {
		t.Errorf("reason should name the GPU: %q", r.Reason)
	}
	if !strings.Contains(r.Reason, "ecc-uncorrectable") {
		t.Errorf("reason should name the check: %q", r.Reason)
	}
	if !strings.Contains(r.Reason, "fatal") {
		t.Errorf("reason should carry the severity: %q", r.Reason)
	}
}

func TestDrainReasonIsBounded(t *testing.T) {
	// Slurm truncates long reasons; a reason nobody can read is no reason.
	var findings []Finding
	for i := 0; i < 64; i++ {
		findings = append(findings, Finding{
			GPUIndex: i, Check: "ecc-uncorrectable", Severity: Fatal,
			Detail: "a very long detail string that should never reach the reason field",
		})
	}
	r := Summarize(findings)
	if len(r.Reason) > 200 {
		t.Fatalf("reason is %d chars; Slurm will truncate it", len(r.Reason))
	}
}

func TestReasonOnlyIncludesWorstSeverity(t *testing.T) {
	// A fatal fault and a transient one in the same pass: the reason should
	// read "fatal", not bury it among warnings.
	h := healthy()
	h.ECCUncorrectableVolatile = 1     // fatal
	h.ThrottleReasons = []string{"sw_thermal_slowdown"} // transient

	r := Summarize(Evaluate(h, DefaultConfig()))
	if r.Worst != Fatal {
		t.Fatalf("worst should be fatal, got %s", r.Worst)
	}
	if strings.Contains(r.Reason, "throttle") {
		t.Errorf("reason should not dilute the fatal finding: %q", r.Reason)
	}
}

func TestNoFindingsMeansEmptyReason(t *testing.T) {
	r := Summarize(nil)
	if r.Drain || r.Reason != "" || r.Worst != OK {
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
