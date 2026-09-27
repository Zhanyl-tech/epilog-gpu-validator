package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/checks"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

// tableCase is one row of the README table. Each row runs the real decision
// code (run, not --simulate) with --enforce, against a simulated nvidia-smi
// and a stand-in scontrol that only records calls.
type tableCase struct {
	name     string
	sim      gpu.Scenario
	gpus     int  // GPUs on the simulated node; 0 means 4
	reversed bool // device minors in reverse PCI order
	args     []string
	env      map[string]string
	noSMI    bool // --nvidia-smi does not exist
	badSCtl  bool // --scontrol fails the root-safety check
	comment  string
}

var jobEnv = map[string]string{"SLURM_JOB_ID": "42", "SLURMD_NODENAME": "gpu001", "CUDA_VISIBLE_DEVICES": "0,1"}

// uuidEnv is gres.conf Flags=env_uuid: UUIDs in CUDA_VISIBLE_DEVICES, GRES
// indices in SLURM_JOB_GPUS. The UUIDs are those of the simulator's gpu0 and
// gpu1.
var uuidEnv = map[string]string{
	"SLURM_JOB_ID": "42", "SLURMD_NODENAME": "gpu001", "SLURM_JOB_GPUS": "0,1",
	"CUDA_VISIBLE_DEVICES": "GPU-5117a000-0000-4000-8000-000000000000,GPU-5117a000-0000-4000-8000-000000000001",
}

func tableCases() []tableCase {
	var cs []tableCase
	for _, s := range gpu.Scenarios {
		cs = append(cs, tableCase{name: string(s), sim: s, env: jobEnv})
	}
	return append(cs,
		tableCase{name: "leaked-memory", sim: gpu.ScenarioLeakedMemory, env: jobEnv, args: []string{"--shared-gpus"}, comment: "--shared-gpus"},
		tableCase{name: "pcie-degraded", sim: gpu.ScenarioPCIeDegraded, env: jobEnv, args: []string{"--drain-on-pcie-width"}, comment: "--drain-on-pcie-width"},
		tableCase{name: "ecc, minors reversed", sim: gpu.ScenarioECC, env: jobEnv, reversed: true},
		tableCase{name: "ecc, minors reversed", sim: gpu.ScenarioECC, env: jobEnv, reversed: true, args: []string{"--gpu-numbering", "nvml"}, comment: "--gpu-numbering nvml"},
		tableCase{name: "ecc, minors reversed", sim: gpu.ScenarioECC, env: jobEnv, reversed: true, args: []string{"--gpu-numbering", "minor"}, comment: "--gpu-numbering minor"},
		tableCase{name: "ecc, minors reversed, env_uuid", sim: gpu.ScenarioECC, env: uuidEnv, reversed: true},
		tableCase{name: "job GPU missing from output", sim: gpu.ScenarioHealthy, env: jobEnv, gpus: 1},
		tableCase{name: "ecc, scontrol group-writable", sim: gpu.ScenarioECC, env: jobEnv, badSCtl: true},
		tableCase{name: "no GPUs in env", sim: gpu.ScenarioHealthy, env: map[string]string{"SLURM_JOB_ID": "42", "SLURMD_NODENAME": "gpu001"}},
		tableCase{name: "GPU vars disagree", sim: gpu.ScenarioHealthy, env: map[string]string{"SLURM_JOB_ID": "42", "SLURMD_NODENAME": "gpu001", "CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "1"}},
		tableCase{name: "nvidia-smi absent", sim: gpu.ScenarioECC, env: jobEnv, noSMI: true},
		tableCase{name: "flag typo", sim: gpu.ScenarioECC, env: jobEnv, args: []string{"--budget", "20"}, comment: "--budget 20"},
		tableCase{name: "unknown flag", sim: gpu.ScenarioECC, env: jobEnv, args: []string{"--no-such-flag"}},
	)
}

// recordingController stands in for scontrol.
type recordingController struct {
	mu    sync.Mutex
	calls []string
	err   error
}

func (c *recordingController) Name() string { return "recording" }
func (c *recordingController) Drain(_ context.Context, node, reason string) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.calls = append(c.calls, node+" "+reason)
	return c.err
}

type nopSyslog struct{}

func (nopSyslog) Err(string) error     { return nil }
func (nopSyslog) Warning(string) error { return nil }
func (nopSyslog) Info(string) error    { return nil }
func (nopSyslog) Close() error         { return nil }

// tableDeps wires run to a simulated node, with a stand-in scontrol.
func tableDeps(sim *gpu.Sim, env map[string]string, smiMissing, sctlUnsafe bool, ctl slurm.Controller, stdout io.Writer) deps {
	return deps{
		getenv:   func(k string) string { return env[k] },
		hostname: func() (string, error) { return "host-from-os", nil },
		stdout:   stdout,
		stderr:   io.Discard,
		now:      func() time.Time { return time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC) },
		newSource: func(string, time.Duration) gpu.Source {
			return sim.Source()
		},
		newResolver:   func(string) gpu.Resolver { return sim },
		newController: func(string) slurm.Controller { return ctl },
		checkBinary: func(p string) error {
			if smiMissing && p == defaultNvidiaSMI {
				return errors.New(p + ": no such file or directory")
			}
			if sctlUnsafe && p == defaultSControl {
				return errors.New(p + ": writable by group or others (mode -rwxrwxr-x); refusing to run it as root")
			}
			return nil
		},
		openSyslog:  func() (syslogSink, error) { return nopSyslog{}, nil },
		openLogFile: func(string) (io.WriteCloser, error) { return nil, errors.New("no log file in the table") },
	}
}

// scenarioTable renders the README table. It touches no hardware and no
// Slurm, and its output is deterministic, so a test can compare it with the
// README byte for byte.
func scenarioTable() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%-42s %-17s %-9s %-4s %s\n", "CASE", "STATUS", "WORST", "EXIT", "DETAIL")
	fmt.Fprintln(&b, strings.Repeat("-", 118))
	for _, c := range tableCases() {
		var out bytes.Buffer
		ctl := &recordingController{}
		args := append([]string{"--enforce", "--json"}, c.args...)
		sim := gpu.NewSim(c.sim, c.gpus)
		sim.MinorsReversed = c.reversed
		code := run(args, tableDeps(sim, c.env, c.noSMI, c.badSCtl, ctl, &out))

		var rec record
		detail := ""
		if err := json.Unmarshal(out.Bytes(), &rec); err != nil {
			detail = "no JSON record: " + err.Error()
		} else {
			detail = tableDetail(rec)
		}
		name := c.name
		if c.comment != "" && !strings.Contains(name, c.comment) {
			name += " " + c.comment
		}
		fmt.Fprintf(&b, "%-42s %-17s %-9s %-4d %s\n", name, rec.Status, rec.Worst, code, orDash(detail))
	}
	return b.String()
}

func tableDetail(rec record) string {
	switch {
	case rec.Reason != "":
		return rec.Reason
	case len(rec.Errors) > 0:
		return clip(rec.Errors[0], 72)
	}
	// The findings at the worst severity, in the drain reason's format.
	var fs []checks.Finding
	for _, f := range rec.Findings {
		fs = append(fs, checks.Finding{GPULabel: f.Label, Check: f.Check, Severity: parseSeverity(f.Severity)})
	}
	return checks.Describe(fs, parseSeverity(rec.Worst))
}

func parseSeverity(s string) checks.Severity {
	for sev := checks.OK; sev <= checks.Fatal; sev++ {
		if sev.String() == s {
			return sev
		}
	}
	return checks.OK
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-3] + "..."
}
