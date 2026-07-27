// Package checks decides whether a GPU is healthy enough for the next job.
//
// # Why the severity split matters more than the checks
//
// Slurm drains a node when Epilog exits non-zero. That makes this the most
// dangerous kind of tool: a false positive removes a working node from the
// cluster, and if the cause is fleet-wide (a driver bug, a monitoring gap, a
// thermal event during a heatwave) it removes *every* node, one job completion
// at a time, faster than anyone can react.
//
// So every signal is classified by whether it is persistent hardware
// degradation or a transient condition that will clear on its own:
//
//	Fatal      The device already corrupted data or has run out of spare
//	           capacity. The next job will hit it too. Drain.
//	Degraded   Persistently underperforming — works, quietly slower. Drain,
//	           because a silently half-speed node is worse than a missing one:
//	           it poisons every collective it joins.
//	Transient  Real, but self-clearing. Thermal throttling on a hot node is
//	           the machine protecting itself, not a fault. Report, never drain.
//	Unknown    Could not be determined. Never drain — see below.
//
// # Never drain on ignorance
//
// If nvidia-smi is missing, times out, or returns something unparseable, that
// is a *monitoring* failure. Draining on it converts a broken health check into
// a cluster-wide outage. The same reasoning as gap detection in gpu-reaper:
// absence of evidence is not evidence of a fault.
package checks

import (
	"fmt"
	"sort"
	"strings"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
)

type Severity int

const (
	OK Severity = iota
	Unknown
	Transient
	Degraded
	Fatal
)

func (s Severity) String() string {
	switch s {
	case OK:
		return "ok"
	case Unknown:
		return "unknown"
	case Transient:
		return "transient"
	case Degraded:
		return "degraded"
	case Fatal:
		return "fatal"
	}
	return "?"
}

// ShouldDrain reports whether a severity justifies removing the node.
func (s Severity) ShouldDrain() bool { return s >= Degraded }

// Finding is one problem on one GPU.
type Finding struct {
	GPUIndex int
	GPUUUID  string
	Check    string
	Severity Severity
	Detail   string
}

func (f Finding) String() string {
	return fmt.Sprintf("gpu%d %s/%s: %s", f.GPUIndex, f.Check, f.Severity, f.Detail)
}

// Config tunes the thresholds a site cares about.
type Config struct {
	// MaxCorrectableECC is the volatile correctable-error count above which a
	// card is considered to be degrading. Correctable errors are normal in
	// small numbers; a spike is the leading indicator of an uncorrectable one.
	MaxCorrectableECC int64
	// AllowPCIeDowngrade skips the link-width check. Some chassis genuinely
	// wire cards at lower widths, and a site that knows that should be able to
	// say so rather than drain its whole fleet.
	AllowPCIeDowngrade bool
	// MaxTemperatureC flags a card running hot even when not yet throttling.
	MaxTemperatureC int
	// RequirePersistenceMode treats persistence mode being off as degraded;
	// it costs real latency on job start but is a config issue, not hardware.
	RequirePersistenceMode bool
	// DrainOnPendingRemap decides whether a pending row remap — which needs a
	// GPU reset to apply — drains the node. Default false: the card still
	// works, and most sites would rather batch the reset.
	DrainOnPendingRemap bool
}

func DefaultConfig() Config {
	return Config{
		MaxCorrectableECC:      1000,
		AllowPCIeDowngrade:     false,
		MaxTemperatureC:        90,
		RequirePersistenceMode: false,
		DrainOnPendingRemap:    false,
	}
}

// Evaluate classifies one GPU.
func Evaluate(h gpu.Health, cfg Config) []Finding {
	var out []Finding
	add := func(check string, sev Severity, format string, args ...any) {
		out = append(out, Finding{
			GPUIndex: h.Index, GPUUUID: h.UUID, Check: check,
			Severity: sev, Detail: fmt.Sprintf(format, args...),
		})
	}

	if !h.Present {
		add("presence", Fatal, "device did not respond to query")
		return out
	}

	// ── Memory integrity ────────────────────────────────────────────────
	// Uncorrectable ECC means corruption already reached a computation. The
	// job that just finished may well have produced wrong numbers.
	if h.ECCUncorrectableVolatile > 0 {
		add("ecc-uncorrectable", Fatal,
			"%d uncorrectable ECC error(s) this power cycle; results from the last job are suspect",
			h.ECCUncorrectableVolatile)
	} else if h.ECCUncorrectableAggregate > 0 {
		// Historic but not since boot: worth surfacing, not worth draining.
		add("ecc-uncorrectable-history", Transient,
			"%d lifetime uncorrectable ECC error(s), none this power cycle",
			h.ECCUncorrectableAggregate)
	}

	if cfg.MaxCorrectableECC > 0 && h.ECCCorrectableVolatile > cfg.MaxCorrectableECC {
		add("ecc-correctable", Degraded,
			"%d correctable ECC errors exceeds threshold %d; memory is degrading",
			h.ECCCorrectableVolatile, cfg.MaxCorrectableECC)
	}

	// ── Row remapping ───────────────────────────────────────────────────
	// A failure means no spare rows remain: the next fault is uncorrectable.
	if h.RemappedRowsFailure {
		add("row-remap-failure", Fatal,
			"row remapping has failed; the device is out of spare rows")
	} else if h.RemappedRowsUncorrectable > 0 {
		add("row-remap-uncorrectable", Degraded,
			"%d row(s) remapped due to uncorrectable errors", h.RemappedRowsUncorrectable)
	}

	if h.RemappedRowsPending > 0 {
		sev := Transient
		if cfg.DrainOnPendingRemap {
			sev = Degraded
		}
		add("row-remap-pending", sev,
			"%d row remap(s) pending a GPU reset", h.RemappedRowsPending)
	}

	// ── PCIe link ───────────────────────────────────────────────────────
	// The silent one. A card at x8 on an x16 slot passes every functional
	// test and halves host-to-device bandwidth for every job that lands on it.
	if !cfg.AllowPCIeDowngrade {
		if h.PCIeWidthMax > 0 && h.PCIeWidthCurrent > 0 && h.PCIeWidthCurrent < h.PCIeWidthMax {
			add("pcie-width", Degraded,
				"link trained at x%d, device supports x%d",
				h.PCIeWidthCurrent, h.PCIeWidthMax)
		}
		if h.PCIeGenMax > 0 && h.PCIeGenCurrent > 0 && h.PCIeGenCurrent < h.PCIeGenMax {
			add("pcie-gen", Degraded,
				"link trained at gen%d, device supports gen%d",
				h.PCIeGenCurrent, h.PCIeGenMax)
		}
	}

	// ── Throttling ──────────────────────────────────────────────────────
	// Software thermal throttling is the card protecting itself and clears
	// when it cools. Hardware slowdown means it already exceeded a hard limit.
	for _, reason := range h.ThrottleReasons {
		r := strings.ToLower(reason)
		switch {
		case strings.Contains(r, "hw_thermal_slowdown"),
			strings.Contains(r, "hw_power_brake"):
			add("throttle", Degraded, "hardware slowdown active: %s", reason)
		case strings.Contains(r, "hw_slowdown"):
			add("throttle", Degraded, "hardware slowdown active: %s", reason)
		case strings.Contains(r, "sw_thermal"), strings.Contains(r, "sw_power_cap"):
			add("throttle", Transient, "software throttling: %s (self-clearing)", reason)
		default:
			add("throttle", Transient, "throttling: %s", reason)
		}
	}

	if cfg.MaxTemperatureC > 0 && h.TemperatureC >= cfg.MaxTemperatureC {
		add("temperature", Transient,
			"%d°C at or above %d°C; likely airflow, not the card",
			h.TemperatureC, cfg.MaxTemperatureC)
	}

	// ── Leftover state ──────────────────────────────────────────────────
	// After Epilog the job is gone, so held memory means a process survived
	// teardown. The next job gets less memory than it asked for.
	if h.MemTotalMiB > 0 {
		if frac := float64(h.MemUsedMiB) / float64(h.MemTotalMiB); frac > 0.05 {
			add("leaked-memory", Degraded,
				"%d MiB (%.0f%%) still allocated after job teardown",
				h.MemUsedMiB, frac*100)
		}
	}

	if cfg.RequirePersistenceMode && !h.PersistenceM {
		add("persistence-mode", Transient, "persistence mode is disabled")
	}

	return out
}

// Result aggregates findings across a node's GPUs.
type Result struct {
	Findings []Finding
	// Worst is the highest severity seen.
	Worst Severity
	// Drain is the decision Epilog acts on.
	Drain bool
	// Reason is written into the Slurm drain reason, so it has to be short and
	// mean something to whoever runs `sinfo -R` at 3am.
	Reason string
}

// Summarize turns findings into a decision.
func Summarize(findings []Finding) Result {
	r := Result{Findings: findings, Worst: OK}
	for _, f := range findings {
		if f.Severity > r.Worst {
			r.Worst = f.Severity
		}
	}
	r.Drain = r.Worst.ShouldDrain()

	if !r.Drain {
		return r
	}

	// Build a reason from the worst findings only. Slurm truncates long drain
	// reasons, and a reason nobody can read is as good as no reason.
	var parts []string
	seen := map[string]bool{}
	for _, f := range findings {
		if f.Severity < r.Worst || seen[f.Check] {
			continue
		}
		seen[f.Check] = true
		parts = append(parts, fmt.Sprintf("gpu%d:%s", f.GPUIndex, f.Check))
	}
	sort.Strings(parts)

	reason := fmt.Sprintf("epilog-gpu-validator %s: %s", r.Worst, strings.Join(parts, ","))
	if len(reason) > 200 {
		reason = reason[:197] + "..."
	}
	r.Reason = reason
	return r
}
