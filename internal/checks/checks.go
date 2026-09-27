// Package checks decides whether a GPU is healthy enough for the next job.
//
// # Why the severity split matters more than the checks
//
// Slurm drains a node when Epilog exits non-zero ("If the Epilog fails
// (returns a non-zero exit code), this will result in the node being set to a
// DRAIN state", https://slurm.schedmd.com/prolog_epilog.html). That makes
// this the most dangerous kind of tool: a false positive removes a working
// node from the cluster, and if the cause is fleet-wide (a driver bug, a
// monitoring gap, a power-management behaviour everyone's GPUs share) it
// removes *every* node, one job completion at a time, faster than anyone can
// react.
//
// So every signal is classified by whether it is evidence of persistent
// hardware degradation or a condition that will clear on its own:
//
//	Fatal      The device already corrupted data or reported a hardware fault
//	           outright. The next job will hit it too. Drain.
//	Degraded   Evidence of persistent underperformance. Drain.
//	Transient  Real, but self-clearing or historic. Report, never drain.
//	Unknown    Could not be determined. Never drain; always say what was
//	           not checked.
//
// "Persistent" is a judgement made per signal type from one reading. The tool
// keeps no history, so it does not observe persistence over time.
//
// # Never drain on ignorance
//
// If nvidia-smi is missing, times out, or returns something unparseable, that
// is a *monitoring* failure. Draining on it converts a broken health check into
// a cluster-wide outage. The same goes for a field nvidia-smi reports as
// "[N/A]": it is an Unknown finding, not a healthy zero. Absence of evidence
// is not evidence of a fault, and it is not evidence of health either.
package checks

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
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

// NodeLabel marks findings that are about the node, not one GPU.
const NodeLabel = "node"

// Finding is one problem on one GPU (or, with GPULabel "node", the node).
type Finding struct {
	// GPUIndex is the NVML index (-1 for node-level or unmatched GPUs).
	GPUIndex int
	GPUUUID  string
	// GPULabel names the GPU in the drain reason: "gpu3" is NVML index 3.
	GPULabel string
	// SlurmID is the Epilog-environment entry for this GPU, if any.
	SlurmID  string
	Check    string
	Severity Severity
	Detail   string
}

func (f Finding) String() string {
	return fmt.Sprintf("%s %s/%s: %s", f.GPULabel, f.Check, f.Severity, f.Detail)
}

// Config tunes the thresholds a site cares about. The numeric defaults are
// site-policy starting points, not values validated against fleet data.
type Config struct {
	// MaxCorrectableECC is the volatile correctable-error count (since the
	// last driver load) above which a card is considered to be degrading.
	// Because the counter only grows until the driver reloads, a long-lived
	// node with a steady trickle will eventually cross any fixed number.
	MaxCorrectableECC int64
	// AllowPCIeDowngrade suppresses PCIe link findings entirely, for sites
	// that know their links run below max and do not want them reported.
	AllowPCIeDowngrade bool
	// DrainOnPCIeWidth treats a link narrower than max as Degraded. Off by
	// default: NVIDIA documents that current width "may be reduced when the
	// GPU is not in use", which is always the case at Epilog time. Turn it on
	// only after confirming, on this hardware, that idle healthy GPUs report
	// full width.
	DrainOnPCIeWidth bool
	// MaxTemperatureC flags a card running hot even when not throttling.
	MaxTemperatureC int
	// DrainOnPendingRemap decides whether a pending row remap, which needs a
	// GPU reset to apply, drains the node. Default false: the card still
	// works, and most sites would rather batch the reset.
	DrainOnPendingRemap bool
	// SharedGPUs means a GPU may be allocated to more than one job at once
	// (Slurm gres/mps or shards). Memory still held after this job may then
	// belong to another running job, so it is Unknown, not Degraded.
	SharedGPUs bool
	// LeakedMemoryFraction is the share of memory.total still used after
	// teardown that counts as leaked.
	LeakedMemoryFraction float64
}

func DefaultConfig() Config {
	return Config{
		MaxCorrectableECC:    1000,
		MaxTemperatureC:      90,
		LeakedMemoryFraction: 0.05,
	}
}

// Evaluate classifies one GPU.
func Evaluate(h gpu.Health, cfg Config) []Finding {
	var out []Finding
	add := func(check string, sev Severity, format string, args ...any) {
		out = append(out, Finding{
			GPUIndex: h.Index, GPUUUID: h.UUID, GPULabel: h.Label(), SlurmID: h.SlurmID,
			Check: check, Severity: sev, Detail: fmt.Sprintf(format, args...),
		})
	}
	readable := h.Readable

	// ── What could not be read ──────────────────────────────────────────
	// "[N/A]" on a pre-Ampere card's remap fields, or on a card with ECC
	// disabled, must not look the same as a verified-clean card.
	if len(h.Unreadable) > 0 {
		add("unreadable", Unknown, "not checked, nvidia-smi could not report: %s",
			strings.Join(h.Unreadable, ", "))
	}

	// ── Memory integrity ────────────────────────────────────────────────
	// Uncorrectable ECC means corruption already reached a computation. The
	// job that just finished may well have produced wrong numbers.
	//
	// The volatile counter counts "since the last driver load", and "On Linux
	// the driver unloads when no active clients exist" unless persistence
	// mode is on (nvidia-smi manual). At Epilog time the job's clients are
	// gone, so with persistence off a zero volatile count proves nothing.
	volatileReadable := readable(gpu.FieldECCUncorrVolatile)
	persistenceKnown := readable(gpu.FieldPersistence)
	volatileTrustworthy := volatileReadable && persistenceKnown && h.PersistenceMode

	switch {
	case volatileReadable && h.ECCUncorrectableVolatile > 0:
		add("ecc-uncorrectable", Fatal,
			"%d uncorrectable ECC error(s) since the last driver load; results from the last job are suspect",
			h.ECCUncorrectableVolatile)
	case readable(gpu.FieldECCUncorrAggregate) && h.ECCUncorrectableAggregate > 0:
		if volatileTrustworthy {
			add("ecc-uncorrectable-history", Transient,
				"%d lifetime uncorrectable ECC error(s), none since the last driver load",
				h.ECCUncorrectableAggregate)
		} else {
			add("ecc-uncorrectable-history", Unknown,
				"%d lifetime uncorrectable ECC error(s); whether any came from this job cannot be told because the volatile counter is unreadable or may have reset",
				h.ECCUncorrectableAggregate)
		}
	}
	if volatileReadable && persistenceKnown && !h.PersistenceMode {
		add("ecc-volatile-unreliable", Unknown,
			"persistence mode is disabled, so the driver may have unloaded after the job and reset the volatile ECC counters; a zero volatile count is not evidence of clean memory")
	}

	if cfg.MaxCorrectableECC > 0 && readable(gpu.FieldECCCorrVolatile) &&
		h.ECCCorrectableVolatile > cfg.MaxCorrectableECC {
		add("ecc-correctable", Degraded,
			"%d correctable ECC errors since the last driver load exceeds the site threshold %d",
			h.ECCCorrectableVolatile, cfg.MaxCorrectableECC)
	}

	// ── Row remapping ───────────────────────────────────────────────────
	// A failure flag means a remap could not be applied. NVIDIA's RMA policy
	// keys on it: "the RMA criteria is met when the row-remapping failure
	// flag is set and validated by the field diagnostic"
	// (docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html).
	// A successful remap is the designed repair, and the uncorrectable remap
	// count is a lifetime number stored on the card: draining on it would
	// drain a repaired GPU after every job for the rest of its life.
	failed := readable(gpu.FieldRemapFailure) && h.RemappedRowsFailure
	if failed {
		add("row-remap-failure", Fatal,
			"a row remapping has failed; under NVIDIA's RMA policy this flag, validated by NVIDIA's field diagnostic, meets the RMA criteria")
	}
	if !failed && readable(gpu.FieldRemapUncorrectable) && h.RemappedRowsUncorrectable > 0 {
		add("row-remap-uncorrectable", Transient,
			"%d row(s) remapped after uncorrectable errors (lifetime count; remapping is the designed repair)",
			h.RemappedRowsUncorrectable)
	}
	if readable(gpu.FieldRemapPending) && h.RemappedRowsPending > 0 {
		sev := Transient
		if cfg.DrainOnPendingRemap {
			sev = Degraded
		}
		add("row-remap-pending", sev,
			"a row remap is pending; nvidia-smi: \"The GPU must be reset for the remapping to go into effect\"")
	}

	// ── PCIe link ───────────────────────────────────────────────────────
	// nvidia-smi's manual on the current link generation and width: "These
	// may be reduced when the GPU is not in use." The Epilog only runs when
	// the GPU is not in use, so a reading below max is what a healthy idle
	// GPU reports. Draining on it would drain every node after every job.
	// A genuinely downtrained link has to be caught under load (DCGM
	// diagnostics, or a periodic drain-and-test job), not here.
	if !cfg.AllowPCIeDowngrade {
		if readable(gpu.FieldPCIeWidthCurrent) && readable(gpu.FieldPCIeWidthMax) &&
			h.PCIeWidthMax > 0 && h.PCIeWidthCurrent > 0 && h.PCIeWidthCurrent < h.PCIeWidthMax {
			if cfg.DrainOnPCIeWidth {
				add("pcie-width", Degraded,
					"link at x%d, device supports x%d (--drain-on-pcie-width: this site treats a narrow idle link as a fault)",
					h.PCIeWidthCurrent, h.PCIeWidthMax)
			} else {
				add("pcie-width", Unknown,
					"link at x%d, device supports x%d; width may be reduced when the GPU is not in use, so an idle reading is not evidence",
					h.PCIeWidthCurrent, h.PCIeWidthMax)
			}
		}
		if readable(gpu.FieldPCIeGenCurrent) && readable(gpu.FieldPCIeGenMax) &&
			h.PCIeGenMax > 0 && h.PCIeGenCurrent > 0 && h.PCIeGenCurrent < h.PCIeGenMax {
			add("pcie-gen", Unknown,
				"link at gen%d, device supports gen%d; link generation is power-managed when the GPU is not in use, so an idle reading is not evidence",
				h.PCIeGenCurrent, h.PCIeGenMax)
		}
	}

	// ── Clock event reasons ─────────────────────────────────────────────
	// A hex bitmask (see gpu/clock.go for the nvml.h constants). HW thermal
	// slowdown and HW power brake still engaged on an idle GPU mean a hard
	// limit is being hit with no load at all. The bare HW slowdown bit is
	// weaker: nvml.h says it "May be also reported during PState or clock
	// change", which is exactly what a GPU dropping to idle is doing.
	if readable(gpu.FieldClockReasons) {
		m := h.ClockEventReasons
		hard := m & (gpu.ClockReasonHWThermalSlowdown | gpu.ClockReasonHWPowerBrakeSlowdown)
		switch {
		case hard != 0:
			add("throttle", Degraded, "hardware slowdown engaged with the GPU idle: %s",
				strings.Join(gpu.ClockReasonNames(hard|m&gpu.ClockReasonHWSlowdown), "+"))
		case m&gpu.ClockReasonHWSlowdown != 0:
			add("throttle", Transient,
				"hw_slowdown alone; nvml.h notes it may also be reported during a P-state or clock change")
		}
		ignored := gpu.ClockReasonGPUIdle | gpu.ClockReasonApplicationsClocksSetting | gpu.ClockReasonSyncBoost
		hw := gpu.ClockReasonHWSlowdown | gpu.ClockReasonHWThermalSlowdown | gpu.ClockReasonHWPowerBrakeSlowdown
		if rest := m &^ (ignored | hw); rest != 0 {
			add("throttle", Transient, "clock limiters active: %s (self-clearing or policy)",
				strings.Join(gpu.ClockReasonNames(rest), "+"))
		}
	}

	if cfg.MaxTemperatureC > 0 && readable(gpu.FieldTemperature) && h.TemperatureC >= cfg.MaxTemperatureC {
		add("temperature", Transient,
			"%d°C at or above the site threshold %d°C; likely airflow, not the card",
			h.TemperatureC, cfg.MaxTemperatureC)
	}

	// ── Leftover state ──────────────────────────────────────────────────
	// After Epilog the job is gone, so held memory means a process survived
	// teardown and the next job gets less memory than it asked for. Unless the
	// GPU is shared (MPS, shards): then another job may still be using it.
	if readable(gpu.FieldMemUsed) && readable(gpu.FieldMemTotal) && h.MemTotalMiB > 0 {
		frac := float64(h.MemUsedMiB) / float64(h.MemTotalMiB)
		if frac > cfg.LeakedMemoryFraction {
			if cfg.SharedGPUs {
				add("leaked-memory", Unknown,
					"%d MiB (%.0f%%) still allocated, but this GPU may be shared with a running job (MPS or shards)",
					h.MemUsedMiB, frac*100)
			} else {
				add("leaked-memory", Degraded,
					"%d MiB (%.0f%%) still allocated after job teardown",
					h.MemUsedMiB, frac*100)
			}
		}
	}

	return out
}

// QueryFailure turns a failed nvidia-smi run into a node-level finding.
//
// Most failures are monitoring failures (driver not loaded, NVML missing,
// bad arguments, timeout) and are Unknown. Four exit codes are themselves
// documented hardware faults in the nvidia-smi manual (8, 10, 14, 15) and are
// Fatal. nvidia-smi does not say which GPU, so the finding is for the node.
func QueryFailure(err error) Finding {
	f := Finding{GPUIndex: -1, GPULabel: NodeLabel}
	var qe *gpu.QueryError
	if errors.As(err, &qe) && !qe.Timeout && gpu.HardwareFaultExitCode(qe.ExitCode) {
		f.Check = fmt.Sprintf("nvidia-smi-exit-%d", qe.ExitCode)
		f.Severity = Fatal
		f.Detail = fmt.Sprintf("nvidia-smi exited %d, documented as %q", qe.ExitCode, gpu.ExitCodeMeaning[qe.ExitCode])
		if qe.Stderr != "" {
			f.Detail += ": " + qe.Stderr
		}
		return f
	}
	f.Check = "query-failed"
	f.Severity = Unknown
	f.Detail = "nothing checked (" + err.Error() + "); a monitoring failure is not evidence of a hardware fault"
	return f
}

// NotChecked is an Unknown finding for something this run could not look at.
func NotChecked(label, check, detail string) Finding {
	if label == "" {
		label = NodeLabel
	}
	return Finding{GPUIndex: -1, GPULabel: label, Check: check, Severity: Unknown, Detail: detail}
}

// Result aggregates findings across a node's GPUs.
type Result struct {
	Findings []Finding
	// Worst is the highest severity seen.
	Worst Severity
	// Drain is the decision Epilog acts on.
	Drain bool
	// Incomplete is true when something was not checked (any Unknown).
	Incomplete bool
	// Reason is written into the Slurm drain reason, so it has to be short and
	// mean something to whoever runs `sinfo -R` at 3am.
	Reason string
}

// MaxReasonLen bounds the drain reason. This is this tool's own limit, to
// keep `sinfo -R` readable; it is not a documented Slurm limit.
const MaxReasonLen = 200

// Summarize turns findings into a decision.
func Summarize(findings []Finding) Result {
	r := Result{Findings: findings, Worst: OK}
	for _, f := range findings {
		if f.Severity > r.Worst {
			r.Worst = f.Severity
		}
		if f.Severity == Unknown {
			r.Incomplete = true
		}
	}
	r.Drain = r.Worst.ShouldDrain()
	if !r.Drain {
		return r
	}
	reason := fmt.Sprintf("epilog-gpu-validator %s: %s", r.Worst, Describe(findings, r.Worst))
	if len(reason) > MaxReasonLen {
		reason = reason[:MaxReasonLen-3] + "..."
	}
	r.Reason = reason
	return r
}

// Describe names the findings at one severity, grouped by check:
// "ecc-uncorrectable:gpu0,gpu3 row-remap-failure:gpu1". Every GPU is named,
// in numeric order (gpu2 before gpu10).
func Describe(findings []Finding, sev Severity) string {
	byCheck := map[string][]string{}
	for _, f := range findings {
		if f.Severity != sev {
			continue
		}
		labels := byCheck[f.Check]
		dup := false
		for _, l := range labels {
			if l == f.GPULabel {
				dup = true
				break
			}
		}
		if !dup {
			byCheck[f.Check] = append(labels, f.GPULabel)
		}
	}
	names := make([]string, 0, len(byCheck))
	for c := range byCheck {
		names = append(names, c)
	}
	sort.Strings(names)
	parts := make([]string, 0, len(names))
	for _, c := range names {
		labels := byCheck[c]
		sort.Slice(labels, func(i, j int) bool { return labelLess(labels[i], labels[j]) })
		parts = append(parts, c+":"+strings.Join(labels, ","))
	}
	return strings.Join(parts, " ")
}

// labelLess orders "gpu2" before "gpu10": prefix first, then the trailing
// number numerically.
func labelLess(a, b string) bool {
	ap, an := splitTrailingNumber(a)
	bp, bn := splitTrailingNumber(b)
	if ap != bp {
		return ap < bp
	}
	if an != bn {
		return an < bn
	}
	return a < b
}

func splitTrailingNumber(s string) (string, int) {
	i := len(s)
	for i > 0 && s[i-1] >= '0' && s[i-1] <= '9' {
		i--
	}
	if i == len(s) {
		return s, -1
	}
	n, err := strconv.Atoi(s[i:])
	if err != nil {
		return s, -1
	}
	return s[:i], n
}
