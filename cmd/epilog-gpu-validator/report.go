package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/checks"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
)

// Run statuses, as they appear in the JSON record, the log file and syslog.
const (
	statusOK               = "ok"                // every one of the job's GPUs was checked; no drain
	statusPartiallyChecked = "partially-checked" // some of the job's GPUs were checked, some not; no drain; exit 0
	statusWouldDrain       = "would-drain"       // drain decided, but report-only or simulated
	statusDrained          = "drained"           // --enforce: scontrol drained the node; exit 1
	statusDrainFailed      = "drain-failed"      // --enforce: scontrol failed or was refused; exit 1 anyway
	statusNoGPUs           = "no-gpus"           // the job held no GPUs
	statusNotChecked       = "not-checked"       // none of the GPUs could be checked; exit 0
	statusConfigError      = "config-error"      // this tool is misconfigured; exit 0
)

type gpuEnvJSON struct {
	Var string `json:"var"`
	Raw string `json:"raw"`
}

type gpuJSON struct {
	SlurmID  string `json:"slurm_id,omitempty"`
	Index    int    `json:"index"`
	UUID     string `json:"uuid"`
	PCIBusID string `json:"pci_bus_id"`
}

type findingJSON struct {
	GPU      int    `json:"gpu"`
	Label    string `json:"label"`
	UUID     string `json:"uuid,omitempty"`
	SlurmID  string `json:"slurm_id,omitempty"`
	Check    string `json:"check"`
	Severity string `json:"severity"`
	Detail   string `json:"detail"`
}

// record is one run, as written to stdout (--json) and --log-file.
type record struct {
	Time      string        `json:"time"`
	Version   string        `json:"version"`
	JobID     string        `json:"job_id,omitempty"`
	Node      string        `json:"node,omitempty"`
	Source    string        `json:"source,omitempty"`
	Simulated bool          `json:"simulated,omitempty"`
	Enforce   bool          `json:"enforce"`
	Status    string        `json:"status"`
	Worst     string        `json:"worst_severity"`
	Drain     bool          `json:"drain"`
	Reason    string        `json:"reason,omitempty"`
	ExitCode  int           `json:"exit_code"`
	GPUEnv    *gpuEnvJSON   `json:"gpu_env,omitempty"`
	GPUs      []gpuJSON     `json:"gpus_checked,omitempty"`
	Findings  []findingJSON `json:"findings"`
	Errors    []string      `json:"errors,omitempty"`
	Warnings  []string      `json:"warnings,omitempty"`
}

// reporter collects one run and writes it out once, in flush.
type reporter struct {
	d        deps
	o        options
	rec      record
	findings []checks.Finding
	// gaps are the findings that name a GPU this run should have checked
	// and did not. They are also in findings.
	gaps  []checks.Finding
	errs  []string
	warns []string
	// silent is set for --help, --version, --check-config and
	// --scenario-table, which print their own output.
	silent bool
	log    *slog.Logger
}

func newReporter(d deps) *reporter {
	return &reporter{
		d:   d,
		log: slog.New(slog.NewTextHandler(d.stderr, nil)),
		rec: record{
			Time:    d.now().UTC().Format(time.RFC3339),
			Version: version,
			Status:  statusConfigError,
			Worst:   checks.OK.String(),
		},
	}
}

func (r *reporter) setStatus(s string) { r.rec.Status = s }

func (r *reporter) add(fs ...checks.Finding) { r.findings = append(r.findings, fs...) }

// gap records findings for GPUs of the job that were not checked.
func (r *reporter) gap(fs ...checks.Finding) {
	r.gaps = append(r.gaps, fs...)
	r.add(fs...)
}

func (r *reporter) warn(msg string) {
	r.warns = append(r.warns, msg)
	r.log.Warn(msg)
}

func (r *reporter) configError(err error) {
	r.errs = append(r.errs, err.Error())
	r.rec.Status = statusConfigError
}

func (r *reporter) checked(h gpu.Health) {
	r.rec.GPUs = append(r.rec.GPUs, gpuJSON{SlurmID: h.SlurmID, Index: h.Index, UUID: h.UUID, PCIBusID: h.PCIBusID})
}

// summary is the one line an operator reads: in syslog, on stderr, at 3am.
func (r *reporter) summary() string {
	rec := r.rec
	var what string
	switch rec.Status {
	case statusConfigError:
		what = "GPUs NOT CHECKED, configuration error (exiting 0 so Slurm does not drain the node): " + strings.Join(r.errs, "; ")
	case statusNotChecked:
		what = "GPUs NOT CHECKED (exiting 0): " + checks.Describe(r.findings, checks.Unknown)
		// The first reason in full: "driver not loaded" is what an operator
		// needs, and the check name alone does not say it.
		for _, f := range r.findings {
			if f.Severity == checks.Unknown {
				what += ": " + clip(f.Detail, 240)
				break
			}
		}
	case statusNoGPUs:
		what = "no GPUs in the Epilog environment; nothing to check"
	case statusOK:
		what = "ok"
		if s := checks.Summarize(r.findings); s.Incomplete {
			what += "; not determined: " + checks.Describe(r.findings, checks.Unknown)
		}
	case statusPartiallyChecked:
		what = "SOME GPUs NOT CHECKED (exiting 0): " + checks.Describe(r.gaps, checks.Unknown)
		if len(r.gaps) > 0 {
			what += ": " + clip(r.gaps[0].Detail, 240)
		}
		what += "; the GPUs that were checked showed nothing drain-worthy"
	case statusWouldDrain:
		if rec.Simulated {
			what = "WOULD DRAIN (simulated; the simulator never drains): " + rec.Reason
		} else {
			what = "WOULD DRAIN (report-only; --enforce not set): " + rec.Reason
		}
	case statusDrained:
		what = "DRAINED node: " + rec.Reason
	case statusDrainFailed:
		what = "DRAIN FAILED (exiting 1 so Slurm drains the node itself): " + rec.Reason + " (" + strings.Join(r.errs, "; ") + ")"
	default:
		what = rec.Status
	}
	return fmt.Sprintf("job=%s node=%s status=%s: %s", orDash(rec.JobID), orDash(rec.Node), rec.Status, what)
}

func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

func (r *reporter) level() slog.Level {
	switch r.rec.Status {
	case statusConfigError, statusNotChecked, statusPartiallyChecked, statusDrained, statusDrainFailed:
		return slog.LevelError
	case statusWouldDrain:
		return slog.LevelWarn
	}
	return slog.LevelInfo
}

func (r *reporter) flush(code int) {
	if r.silent {
		return
	}
	r.rec.ExitCode = code
	r.rec.Errors, r.rec.Warnings = r.errs, r.warns
	r.rec.Findings = make([]findingJSON, 0, len(r.findings))
	for _, f := range r.findings {
		r.rec.Findings = append(r.rec.Findings, findingJSON{
			GPU: f.GPUIndex, Label: f.GPULabel, UUID: f.GPUUUID, SlurmID: f.SlurmID,
			Check: f.Check, Severity: f.Severity.String(), Detail: f.Detail,
		})
	}
	switch r.rec.Status {
	case statusConfigError:
		// Nothing was determined, which is what Unknown means.
		r.rec.Worst = checks.Unknown.String()
	case statusNotChecked, statusNoGPUs:
		r.rec.Worst = checks.Summarize(r.findings).Worst.String()
	}

	if r.o.jsonOut {
		enc := json.NewEncoder(r.d.stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(r.rec)
	} else {
		for _, f := range r.findings {
			lvl := slog.LevelInfo
			if f.Severity.ShouldDrain() {
				lvl = slog.LevelError
			}
			r.log.Log(context.Background(), lvl, "finding",
				"gpu", f.GPULabel, "check", f.Check,
				"severity", f.Severity.String(), "detail", f.Detail)
		}
	}
	// The summary always goes to stderr, JSON or not: a misconfigured or
	// blind run must not be quiet.
	summary := r.summary()
	r.log.Log(context.Background(), r.level(), summary)

	if r.o.logFile != "" {
		if err := r.appendLogFile(); err != nil {
			r.log.Error("cannot write --log-file; this run is only on stderr", "path", r.o.logFile, "err", err)
		}
	}
	if r.o.syslog && r.rec.Status != statusNoGPUs {
		if err := r.sendSyslog(summary); err != nil {
			r.log.Error("cannot write to syslog", "err", err)
		}
	}
}

func (r *reporter) appendLogFile() error {
	line, err := json.Marshal(r.rec)
	if err != nil {
		return err
	}
	w, err := r.d.openLogFile(r.o.logFile)
	if err != nil {
		return err
	}
	// One write call per record on an O_APPEND file: each record lands at
	// the end of the file even when several Epilogs on one node finish at
	// once. (POSIX does not promise that very large appends never interleave;
	// these records are small.)
	_, werr := w.Write(append(line, '\n'))
	cerr := w.Close()
	if werr != nil {
		return werr
	}
	return cerr
}

func (r *reporter) sendSyslog(summary string) error {
	s, err := r.d.openSyslog()
	if err != nil {
		return err
	}
	defer s.Close()
	switch r.level() {
	case slog.LevelError:
		return s.Err(summary)
	case slog.LevelWarn:
		return s.Warning(summary)
	}
	return s.Info(summary)
}
