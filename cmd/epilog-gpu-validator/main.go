// Command epilog-gpu-validator checks the GPUs a finished job used and drains
// the node if they are persistently faulty.
//
// Designed for Slurm's Epilog. Two constraints shape everything:
//
//   - It runs on every job completion, and Slurm kills a slow Epilog. The whole
//     run is bounded by --budget and it exits clean if it runs out of time.
//   - Slurm drains the node when Epilog exits non-zero. That makes a false
//     positive an outage, so the exit code is 0 unless a *persistent* fault was
//     positively identified. Anything unknown exits 0.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/checks"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

var version = "dev"

// Exit codes. Slurm only distinguishes zero from non-zero, but a distinct code
// makes the intent legible in logs and lets a wrapper script tell the cases
// apart.
const (
	exitOK            = 0
	exitDrainRequested = 1
)

func main() {
	os.Exit(run())
}

func run() int {
	var (
		budget    = flag.Duration("budget", 20*time.Second, "hard time limit for the whole check")
		enforce   = flag.Bool("enforce", false, "actually drain the node (default: report only)")
		allGPUs   = flag.Bool("all-gpus", false, "check every GPU, not just the job's")
		simulate  = flag.String("simulate", "", "run against a synthetic GPU: healthy|pcie-degraded|ecc|remap-failure|remap-pending|thermal|hw-slowdown|missing")
		jsonOut   = flag.Bool("json", false, "emit findings as JSON")
		maxCorrECC = flag.Int64("max-correctable-ecc", 1000, "correctable ECC threshold")
		allowPCIe = flag.Bool("allow-pcie-downgrade", false, "do not drain on a narrower-than-max PCIe link")
		drainRemap = flag.Bool("drain-on-pending-remap", false, "drain when a row remap is pending a reset")
		showVer   = flag.Bool("version", false, "print version and exit")
	)
	flag.Parse()

	if *showVer {
		fmt.Println(version)
		return exitOK
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, nil))
	env := slurm.FromEnvironment()

	// The budget covers everything. Slurm killing this mid-run is worse than
	// it finishing early with nothing to say.
	ctx, cancel := context.WithTimeout(context.Background(), *budget)
	defer cancel()

	var src gpu.Source
	if *simulate != "" {
		src = gpu.NewSimSource(gpu.Scenario(*simulate), 4)
	} else {
		src = gpu.NewSMISource()
	}

	// Only the GPUs this job held. On a shared node, checking everything means
	// another job's faulty card drains the node for a job that never touched it.
	indices := env.GPUIndices()
	if *allGPUs {
		indices = nil
	} else if len(indices) == 0 && *simulate == "" {
		// Cannot tell what the job used. Checking everything would risk
		// attributing a neighbour's fault; checking nothing is the safe read.
		logger.Info("no GPU set in the Epilog environment; nothing to check",
			"job_id", env.JobID, "raw", env.RawGPUs)
		return exitOK
	}

	health, err := src.Query(ctx, indices)
	if err != nil {
		// A failed query is a monitoring failure, not a hardware fault.
		// Draining here would turn a broken nvidia-smi into a cluster outage,
		// one job completion at a time.
		logger.Warn("GPU query failed; not draining",
			"source", src.Name(), "err", err, "job_id", env.JobID)
		return exitOK
	}
	if len(health) == 0 {
		logger.Warn("GPU query returned nothing; not draining", "job_id", env.JobID)
		return exitOK
	}

	cfg := checks.DefaultConfig()
	cfg.MaxCorrectableECC = *maxCorrECC
	cfg.AllowPCIeDowngrade = *allowPCIe
	cfg.DrainOnPendingRemap = *drainRemap

	var findings []checks.Finding
	for _, h := range health {
		findings = append(findings, checks.Evaluate(h, cfg)...)
	}
	result := checks.Summarize(findings)

	if *jsonOut {
		emitJSON(env, src.Name(), result)
	} else {
		for _, f := range result.Findings {
			lvl := slog.LevelInfo
			if f.Severity.ShouldDrain() {
				lvl = slog.LevelError
			}
			logger.Log(ctx, lvl, "finding",
				"job_id", env.JobID, "node", env.NodeName,
				"gpu", f.GPUIndex, "check", f.Check,
				"severity", f.Severity.String(), "detail", f.Detail)
		}
	}

	if !result.Drain {
		return exitOK
	}

	// Report-only is the default. A health check that starts draining nodes on
	// the day it is installed does not survive to a second day.
	if !*enforce {
		logger.Warn("would drain node (report-only; pass --enforce to act)",
			"node", env.NodeName, "reason", result.Reason)
		return exitOK
	}

	ctl := slurm.NewSControl()
	if err := ctl.Drain(ctx, env.NodeName, result.Reason); err != nil {
		logger.Error("drain failed", "node", env.NodeName, "err", err)
		// Still signal the fault: a non-zero exit makes Slurm drain the node
		// itself, which is the outcome we wanted.
		return exitDrainRequested
	}
	logger.Error("node drained", "node", env.NodeName, "reason", result.Reason)
	return exitDrainRequested
}

func emitJSON(env slurm.Env, source string, r checks.Result) {
	type findingJSON struct {
		GPU      int    `json:"gpu"`
		UUID     string `json:"uuid,omitempty"`
		Check    string `json:"check"`
		Severity string `json:"severity"`
		Detail   string `json:"detail"`
	}
	out := struct {
		JobID    string        `json:"job_id,omitempty"`
		Node     string        `json:"node"`
		Source   string        `json:"source"`
		Worst    string        `json:"worst_severity"`
		Drain    bool          `json:"drain"`
		Reason   string        `json:"reason,omitempty"`
		Findings []findingJSON `json:"findings"`
	}{
		JobID: env.JobID, Node: env.NodeName, Source: source,
		Worst: r.Worst.String(), Drain: r.Drain, Reason: r.Reason,
		Findings: []findingJSON{},
	}
	for _, f := range r.Findings {
		out.Findings = append(out.Findings, findingJSON{
			GPU: f.GPUIndex, UUID: f.GPUUUID, Check: f.Check,
			Severity: f.Severity.String(), Detail: f.Detail,
		})
	}
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	_ = enc.Encode(out)
}
