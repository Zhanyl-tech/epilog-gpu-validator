package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

// ── harness ────────────────────────────────────────────────────────────────

type fakeSyslog struct {
	mu    sync.Mutex
	lines []string
}

func (s *fakeSyslog) add(p, m string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.lines = append(s.lines, p+" "+m)
	return nil
}
func (s *fakeSyslog) Err(m string) error     { return s.add("err", m) }
func (s *fakeSyslog) Warning(m string) error { return s.add("warning", m) }
func (s *fakeSyslog) Info(m string) error    { return s.add("info", m) }
func (s *fakeSyslog) Close() error           { return nil }

type countingSource struct {
	gpu.Source
	calls int
}

func (c *countingSource) Query(ctx context.Context) (gpu.Result, error) {
	c.calls++
	return c.Source.Query(ctx)
}

type panicSource struct{}

func (panicSource) Name() string                              { return "panics" }
func (panicSource) Query(context.Context) (gpu.Result, error) { panic("boom") }

type harness struct {
	t        *testing.T
	env      map[string]string
	sim      *gpu.Sim
	src      gpu.Source   // overrides sim's source when set
	resolver gpu.Resolver // overrides sim's device map when set
	ctl      *recordingController
	syslog   *fakeSyslog
	stdout   bytes.Buffer
	stderr   bytes.Buffer
	missing  map[string]bool // paths checkBinary reports as absent
	unsafe   map[string]bool // paths checkBinary refuses as group-writable
	queries  *countingSource
	ctlMade  int
}

type failingResolver struct{}

func (failingResolver) DeviceMinors() (map[int]string, error) {
	return nil, errors.New("read /proc/driver/nvidia/gpus: no such file or directory")
}

func newHarness(t *testing.T, s gpu.Scenario) *harness {
	return &harness{
		t:       t,
		env:     map[string]string{"SLURM_JOB_ID": "7", "SLURMD_NODENAME": "gpu001", "CUDA_VISIBLE_DEVICES": "0,1"},
		sim:     gpu.NewSim(s, 4),
		ctl:     &recordingController{},
		syslog:  &fakeSyslog{},
		missing: map[string]bool{},
		unsafe:  map[string]bool{},
	}
}

func (h *harness) deps() deps {
	src := h.src
	if src == nil {
		src = h.sim.Source()
	}
	h.queries = &countingSource{Source: src}
	resolver := h.resolver
	if resolver == nil {
		resolver = h.sim
	}
	return deps{
		getenv:        func(k string) string { return h.env[k] },
		hostname:      func() (string, error) { return "host-from-os", nil },
		stdout:        &h.stdout,
		stderr:        &h.stderr,
		now:           func() time.Time { return time.Date(2026, 9, 26, 0, 0, 0, 0, time.UTC) },
		newSource:     func(string, time.Duration) gpu.Source { return h.queries },
		newResolver:   func(string) gpu.Resolver { return resolver },
		newController: func(string) slurm.Controller { h.ctlMade++; return h.ctl },
		checkBinary: func(p string) error {
			if h.missing[p] {
				return errors.New(p + ": no such file or directory")
			}
			if h.unsafe[p] {
				return errors.New(p + ": writable by group or others (mode -rwxrwxr-x); refusing to run it as root")
			}
			return nil
		},
		openSyslog: func() (syslogSink, error) { return h.syslog, nil },
		openLogFile: func(p string) (io.WriteCloser, error) {
			return os.OpenFile(p, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o600)
		},
	}
}

func (h *harness) run(args ...string) (int, record) {
	h.t.Helper()
	h.stdout.Reset()
	h.stderr.Reset()
	code := run(append([]string{"--json"}, args...), h.deps())
	var rec record
	if err := json.Unmarshal(h.stdout.Bytes(), &rec); err != nil {
		h.t.Fatalf("no JSON record on stdout (%v): %q\nstderr: %s", err, h.stdout.String(), h.stderr.String())
	}
	return code, rec
}

func mustExit(t *testing.T, got, want int, rec record) {
	t.Helper()
	if got != want {
		t.Fatalf("exit %d, want %d; record %+v", got, want, rec)
	}
	if rec.ExitCode != got {
		t.Fatalf("record says exit %d, process exits %d", rec.ExitCode, got)
	}
}

// ── the contract: misconfiguration never drains ────────────────────────────

func TestBadCommandLinesExitZero(t *testing.T) {
	// flag.Parse's default exits 2, and Slurm drains on any non-zero exit.
	for _, args := range [][]string{
		{"--budget", "20"},                  // unit forgotten
		{"--no-such-flag"},                  // wrapper newer than the binary
		{"--enforce", "stray"},              // positional argument
		{"--budget", "0s"},                  // parses, but invalid
		{"--simulate", "ecc-uncorrectable"}, // unknown scenario
		{"--max-temperature", "-1"},
		{"--gpu-numbering", "pci"}, // not a mode
	} {
		h := newHarness(t, gpu.ScenarioECC)
		code, rec := h.run(append([]string{"--enforce"}, args...)...)
		mustExit(t, code, exitOK, rec)
		if rec.Status != statusConfigError || len(rec.Errors) == 0 {
			t.Errorf("%v: status %q errors %v", args, rec.Status, rec.Errors)
		}
		if h.queries.calls != 0 || len(h.ctl.calls) != 0 {
			t.Errorf("%v: a misconfigured run must not query or drain", args)
		}
		if !strings.Contains(h.stderr.String(), "NOT CHECKED") {
			t.Errorf("%v: a misconfigured run must say so on stderr: %s", args, h.stderr.String())
		}
	}
}

func TestHelpAndVersionExitZero(t *testing.T) {
	h := newHarness(t, gpu.ScenarioHealthy)
	if code := run([]string{"--help"}, h.deps()); code != exitOK {
		t.Fatalf("--help exited %d", code)
	}
	h.stdout.Reset()
	if code := run([]string{"--version"}, h.deps()); code != exitOK || strings.TrimSpace(h.stdout.String()) != version {
		t.Fatalf("--version: exit %d, out %q", code, h.stdout.String())
	}
}

func TestMissingNvidiaSMIExitsZeroLoudly(t *testing.T) {
	// The Epilog has no PATH; before absolute paths the tool quietly did
	// nothing. Now the absence is a config error that is hard to miss.
	h := newHarness(t, gpu.ScenarioECC)
	h.missing[defaultNvidiaSMI] = true
	logFile := filepath.Join(t.TempDir(), "egv.jsonl")
	code, rec := h.run("--log-file", logFile, "--syslog", "--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusConfigError || h.queries.calls != 0 || len(h.ctl.calls) != 0 {
		t.Fatalf("got %+v, queries %d", rec, h.queries.calls)
	}
	if len(h.syslog.lines) != 1 || !strings.HasPrefix(h.syslog.lines[0], "err ") || !strings.Contains(h.syslog.lines[0], "NOT CHECKED") {
		t.Fatalf("syslog: %v", h.syslog.lines)
	}
	data, _ := os.ReadFile(logFile)
	if !strings.Contains(string(data), `"status":"config-error"`) {
		t.Fatalf("log file: %s", data)
	}
}

func TestTypoAfterLoggingFlagsIsStillLogged(t *testing.T) {
	// Flags before a bad one are applied, which is why the wrapper passes
	// --log-file and --syslog first.
	h := newHarness(t, gpu.ScenarioHealthy)
	logFile := filepath.Join(t.TempDir(), "egv.jsonl")
	code, rec := h.run("--log-file", logFile, "--syslog", "--budget", "20")
	mustExit(t, code, exitOK, rec)
	if data, _ := os.ReadFile(logFile); !strings.Contains(string(data), "invalid value") {
		t.Fatalf("log file: %q", data)
	}
	if len(h.syslog.lines) != 1 {
		t.Fatalf("syslog: %v", h.syslog.lines)
	}
}

func TestAPanicExitsZero(t *testing.T) {
	h := newHarness(t, gpu.ScenarioHealthy)
	h.src = panicSource{}
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusConfigError || !strings.Contains(strings.Join(rec.Errors, " "), "panic") {
		t.Fatalf("got %+v", rec)
	}
}

// ── the contract: no evidence, no drain ────────────────────────────────────

func TestJobWithoutGPUsIsNotQueried(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	delete(h.env, "CUDA_VISIBLE_DEVICES")
	code, rec := h.run("--enforce", "--syslog")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNoGPUs || h.queries.calls != 0 {
		t.Fatalf("got %+v, queries %d", rec, h.queries.calls)
	}
	if len(h.syslog.lines) != 0 {
		t.Fatalf("CPU-only jobs should not add syslog noise: %v", h.syslog.lines)
	}
}

func TestAmbiguousGPUSetChecksNothing(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	h.env["SLURM_JOB_GPUS"] = "2"
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked || h.queries.calls != 0 {
		t.Fatalf("got %+v", rec)
	}
}

func TestUnmappableDeviceNumbersCheckNothing(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	h.env["CUDA_VISIBLE_DEVICES"] = "9" // no GPU 9 on a 4-GPU node, in any numbering
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked || h.ctlMade != 0 || len(rec.Findings) != 1 || rec.Findings[0].Check != checkNotInOutput {
		t.Fatalf("got %+v", rec)
	}
	h.env["CUDA_VISIBLE_DEVICES"] = "9"
	code, rec = h.run("--enforce", "--gpu-numbering", "minor")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked || rec.Findings[0].Check != checkDeviceMapNoGPU {
		t.Fatalf("minor: got %+v", rec)
	}
}

// ── which GPU a Slurm number means ─────────────────────────────────────────

// Slurm writes its GRES index into CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS
// (gres_common.c, gres_common_prep_set_env). On a node whose device minors
// run in reverse PCI order, "0" is gpu0 if gres.conf numbers by PCI bus ID
// (AutoDetect=nvml) and gpu3 (/dev/nvidia0) if it numbers by device file.
// The simulator's fault is on gpu0.
func TestGPUNumberingOnANodeWhoseMinorsAreNotInPCIOrder(t *testing.T) {
	for _, tc := range []struct {
		mode       string
		wantCode   int
		wantStatus string
		wantGPUs   []int // nvidia-smi indices checked
	}{
		// auto cannot tell which GPU "0" is here, so it checks nothing
		// rather than risk blaming the job for a neighbour's fault (or
		// missing its own). Either guess is wrong for some gres.conf.
		{"auto", exitOK, statusNotChecked, nil},
		{"nvml", exitDrainRequested, statusDrained, []int{0, 1}},
		{"minor", exitOK, statusOK, []int{3, 2}},
	} {
		h := newHarness(t, gpu.ScenarioECC)
		h.sim.MinorsReversed = true
		code, rec := h.run("--enforce", "--gpu-numbering", tc.mode)
		mustExit(t, code, tc.wantCode, rec)
		var got []int
		for _, g := range rec.GPUs {
			got = append(got, g.Index)
		}
		if rec.Status != tc.wantStatus || !equalInts(got, tc.wantGPUs) {
			t.Fatalf("%s: status %s, checked %v; record %+v", tc.mode, rec.Status, got, rec)
		}
		if tc.mode == "auto" {
			for _, f := range rec.Findings {
				if f.Check != checkNumberAmbiguous || !strings.Contains(f.Detail, "--gpu-numbering") {
					t.Fatalf("auto: finding %+v should name the ambiguity and the fix", f)
				}
			}
		}
		if tc.mode == "nvml" && rec.Reason != "epilog-gpu-validator fatal: ecc-uncorrectable:gpu0" {
			t.Fatalf("nvml: reason %q", rec.Reason)
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestEnvUUIDJobsAreChecked(t *testing.T) {
	// gres.conf Flags=env_uuid: UUIDs in CUDA_VISIBLE_DEVICES, indices in
	// SLURM_JOB_GPUS. A set comparison between the two used to make every
	// such job "ambiguous". The UUID names the device whatever the numbering,
	// even on a node where auto would refuse the numbers.
	h := newHarness(t, gpu.ScenarioECC)
	h.sim.MinorsReversed = true
	h.env["CUDA_VISIBLE_DEVICES"] = "GPU-5117a000-0000-4000-8000-000000000000"
	h.env["SLURM_JOB_GPUS"] = "0"
	code, rec := h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
	if rec.Reason != "epilog-gpu-validator fatal: ecc-uncorrectable:gpu0" || rec.GPUEnv.Var != "CUDA_VISIBLE_DEVICES" {
		t.Fatalf("got %+v", rec)
	}
}

func TestUUIDNumberingNeverMapsNumbers(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	code, rec := h.run("--enforce", "--gpu-numbering", "uuid")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked || len(rec.Findings) != 2 || rec.Findings[0].Check != checkNumberNotUsed {
		t.Fatalf("got %+v", rec)
	}
}

func TestUnreadableDeviceMap(t *testing.T) {
	// auto and minor need /proc/driver/nvidia/gpus; nvml does not.
	h := newHarness(t, gpu.ScenarioECC)
	h.resolver = failingResolver{}
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked || rec.Findings[0].Check != checkDeviceMapNone {
		t.Fatalf("auto: got %+v", rec)
	}
	code, rec = h.run("--enforce", "--gpu-numbering", "nvml")
	mustExit(t, code, exitDrainRequested, rec)
	if rec.Status != statusDrained {
		t.Fatalf("nvml: got %+v", rec)
	}
}

func TestMonitoringFailuresNeverDrain(t *testing.T) {
	for _, s := range []gpu.Scenario{gpu.ScenarioNoDriver} {
		h := newHarness(t, s)
		code, rec := h.run("--enforce")
		mustExit(t, code, exitOK, rec)
		if rec.Status != statusNotChecked || len(h.ctl.calls) != 0 {
			t.Fatalf("%s: got %+v", s, rec)
		}
	}
	for name, err := range map[string]error{
		"timeout":  &gpu.QueryError{ExitCode: -1, Timeout: true, Err: context.DeadlineExceeded},
		"exit 12":  &gpu.QueryError{ExitCode: 12},
		"exit 255": &gpu.QueryError{ExitCode: 255},
		"parse":    &gpu.ParseError{Err: errors.New("bad")},
	} {
		h := newHarness(t, gpu.ScenarioHealthy)
		h.src = errSource{err}
		code, rec := h.run("--enforce")
		mustExit(t, code, exitOK, rec)
		if rec.Status != statusNotChecked || len(h.ctl.calls) != 0 {
			t.Fatalf("%s: got %+v", name, rec)
		}
	}
}

func TestNotCheckedSyslogLineSaysWhy(t *testing.T) {
	h := newHarness(t, gpu.ScenarioNoDriver)
	code, rec := h.run("--enforce", "--syslog")
	mustExit(t, code, exitOK, rec)
	if len(h.syslog.lines) != 1 {
		t.Fatalf("syslog: %v", h.syslog.lines)
	}
	line := h.syslog.lines[0]
	if !strings.HasPrefix(line, "err ") || !strings.Contains(line, "NOT CHECKED") || !strings.Contains(line, "NVIDIA driver is not loaded") {
		t.Fatalf("an operator reading syslog should learn why: %q", line)
	}
}

type errSource struct{ err error }

func (e errSource) Name() string                              { return "err" }
func (e errSource) Query(context.Context) (gpu.Result, error) { return gpu.Result{}, e.err }

type rowsSource struct{ csv string }

func (r rowsSource) Name() string { return "rows" }
func (r rowsSource) Query(context.Context) (gpu.Result, error) {
	return gpu.ParseCSV([]byte(r.csv))
}

func TestEmptyOrForeignOutputChecksNothing(t *testing.T) {
	h := newHarness(t, gpu.ScenarioHealthy)
	h.src = rowsSource{""}
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusNotChecked {
		t.Fatalf("got %+v", rec)
	}
	// The job's GPUs are missing from otherwise good output: reported, not
	// guessed. (nvidia-smi exit 15, documented for a GPU that has "fallen off
	// the bus", drains; that a lost GPU produces it is not observed here.)
	for _, f := range rec.Findings {
		if f.Check != checkNotInOutput || f.Severity != "unknown" {
			t.Errorf("unexpected finding %+v", f)
		}
	}
}

func TestAJobGPUMissingFromOutputIsPartiallyCheckedNotOK(t *testing.T) {
	// The job held Slurm GPUs 0 and 1; nvidia-smi lists only one GPU. "ok"
	// would hide that GPU 1 was never looked at.
	h := newHarness(t, gpu.ScenarioHealthy)
	h.sim = gpu.NewSim(gpu.ScenarioHealthy, 1)
	logFile := filepath.Join(t.TempDir(), "egv.jsonl")
	code, rec := h.run("--enforce", "--syslog", "--log-file", logFile)
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusPartiallyChecked || len(rec.GPUs) != 1 || h.ctlMade != 0 {
		t.Fatalf("got %+v", rec)
	}
	if len(h.syslog.lines) != 1 || !strings.HasPrefix(h.syslog.lines[0], "err ") || !strings.Contains(h.syslog.lines[0], "SOME GPUs NOT CHECKED") || !strings.Contains(h.syslog.lines[0], "not-in-output:slurm-gpu1") {
		t.Fatalf("syslog: %v", h.syslog.lines)
	}
	if data, _ := os.ReadFile(logFile); !strings.Contains(string(data), `"status":"partially-checked"`) {
		t.Fatalf("log file: %s", data)
	}
	// A fault on the GPU that was checked still drains.
	h.sim = gpu.NewSim(gpu.ScenarioECC, 1)
	code, rec = h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
}

func TestMalformedRowWithAllGPUsIsPartiallyChecked(t *testing.T) {
	// With --all-gpus every GPU on the node is a target, so a row that could
	// not be parsed is a GPU that was not checked. (SYNTHETIC rows.)
	h := newHarness(t, gpu.ScenarioHealthy)
	h.src = rowsSource{"0, GPU-5117a000-0000-4000-8000-000000000000, NVIDIA H100 80GB HBM3, 00000000:18:00.0, 16, 16, 1, 5, 0, 0, 0, No, 0, No, 0x0000000000000001, 34, 1, 81559, Enabled\n1, short row\n"}
	code, rec := h.run("--enforce", "--all-gpus")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusPartiallyChecked {
		t.Fatalf("got %+v", rec)
	}
}

func TestIdleHealthyNodeNeverExitsNonZero(t *testing.T) {
	// The fleet-wide false positive, end to end: an idle GPU's link at gen1
	// of gen5 used to exit 1 under --enforce.
	h := newHarness(t, gpu.ScenarioHealthy)
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusOK || rec.Drain || h.ctlMade != 0 {
		t.Fatalf("got %+v", rec)
	}
}

func TestOnlyTheJobsOwnGPUsAreJudged(t *testing.T) {
	// The simulator's fault is on gpu0. A job that held Slurm GPU 1 must not
	// be blamed for it.
	h := newHarness(t, gpu.ScenarioECC)
	h.env["CUDA_VISIBLE_DEVICES"] = "1"
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Drain || rec.Status != statusOK || len(rec.GPUs) != 1 || rec.GPUs[0].Index != 1 || rec.GPUs[0].SlurmID != "1" {
		t.Fatalf("got %+v", rec)
	}
}

func TestUUIDsFromSlurmAreUsedDirectly(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	h.env["CUDA_VISIBLE_DEVICES"] = "GPU-5117a000-0000-4000-8000-000000000000"
	code, rec := h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
	if rec.Reason != "epilog-gpu-validator fatal: ecc-uncorrectable:gpu0" {
		t.Fatalf("got %+v", rec)
	}
	// A healthy neighbour named by UUID is checked, and does not drain.
	h.env["CUDA_VISIBLE_DEVICES"] = "GPU-5117a000-0000-4000-8000-000000000003"
	code, rec = h.run("--enforce")
	mustExit(t, code, exitOK, rec)
}

// ── the contract: evidence drains only when asked ──────────────────────────

func TestReportOnlyNeverCallsScontrol(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	code, rec := h.run("--syslog")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusWouldDrain || !rec.Drain || h.ctlMade != 0 {
		t.Fatalf("got %+v", rec)
	}
	if len(h.syslog.lines) != 1 || !strings.HasPrefix(h.syslog.lines[0], "warning ") {
		t.Fatalf("syslog: %v", h.syslog.lines)
	}
}

func TestEnforceDrainsThenExitsOne(t *testing.T) {
	h := newHarness(t, gpu.ScenarioECC)
	code, rec := h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
	want := []string{"gpu001 epilog-gpu-validator fatal: ecc-uncorrectable:gpu0"}
	if rec.Status != statusDrained || strings.Join(h.ctl.calls, "|") != want[0] {
		t.Fatalf("status %s, drain calls %v", rec.Status, h.ctl.calls)
	}
}

func TestFailedDrainStillExitsOne(t *testing.T) {
	// Slurm drains on the non-zero exit itself, which is the outcome wanted.
	h := newHarness(t, gpu.ScenarioECC)
	h.ctl.err = errors.New("scontrol: permission denied")
	code, rec := h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
	if rec.Status != statusDrainFailed {
		t.Fatalf("got %+v", rec)
	}
}

func TestMissingScontrolIsLoudButNotFatalToTheCheck(t *testing.T) {
	h := newHarness(t, gpu.ScenarioHealthy)
	h.missing[defaultSControl] = true
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusOK || len(rec.Warnings) == 0 || !strings.Contains(rec.Warnings[0], "--scontrol") {
		t.Fatalf("got %+v", rec)
	}
}

func TestAnScontrolThatFailsTheSafetyCheckIsNeverRun(t *testing.T) {
	// A group-writable (or foreign-owned, or missing) scontrol used to be
	// warned about and then executed as root at the first fault. Now it is
	// never run; exit 1 still gets the node drained, by Slurm itself.
	for _, bad := range []map[string]bool{
		{defaultSControl: true}, // unsafe
		nil,                     // missing (set below)
	} {
		h := newHarness(t, gpu.ScenarioECC)
		if bad != nil {
			h.unsafe = bad
		} else {
			h.missing[defaultSControl] = true
		}
		code, rec := h.run("--enforce")
		mustExit(t, code, exitDrainRequested, rec)
		if rec.Status != statusDrainFailed || h.ctlMade != 0 || len(h.ctl.calls) != 0 {
			t.Fatalf("status %s, controller made %d, calls %v", rec.Status, h.ctlMade, h.ctl.calls)
		}
		if !strings.Contains(strings.Join(rec.Errors, " "), "scontrol not run") || !strings.Contains(rec.Warnings[0], "will not be run") {
			t.Fatalf("errors %v warnings %v", rec.Errors, rec.Warnings)
		}
	}
}

func TestDocumentedHardwareExitCodeDrainsUnderEnforce(t *testing.T) {
	h := newHarness(t, gpu.ScenarioOffBus)
	code, rec := h.run()
	mustExit(t, code, exitOK, rec)
	if rec.Status != statusWouldDrain {
		t.Fatalf("report-only: got %+v", rec)
	}
	code, rec = h.run("--enforce")
	mustExit(t, code, exitDrainRequested, rec)
	if rec.Reason != "epilog-gpu-validator fatal: nvidia-smi-exit-15:node" {
		t.Fatalf("got %+v", rec)
	}
}

func TestSimulateNeverDrainsAndNeverExitsNonZero(t *testing.T) {
	// `make scenarios` used to run --simulate --enforce, and --simulate only
	// swapped the GPU source: on a host with scontrol and admin rights it
	// really drained the node.
	for _, s := range gpu.Scenarios {
		h := newHarness(t, gpu.ScenarioHealthy)
		code, rec := h.run("--simulate", string(s), "--enforce")
		mustExit(t, code, exitOK, rec)
		if h.ctlMade != 0 || len(h.ctl.calls) != 0 {
			t.Fatalf("%s: the simulator called the controller", s)
		}
		if !rec.Simulated {
			t.Fatalf("%s: record not marked simulated", s)
		}
	}
}

func TestSharedGPUHintDowngradesLeakedMemory(t *testing.T) {
	h := newHarness(t, gpu.ScenarioLeakedMemory)
	h.env["SLURM_SHARDS_ON_NODE"] = "1"
	code, rec := h.run("--enforce")
	mustExit(t, code, exitOK, rec)
	if rec.Drain || len(rec.Warnings) == 0 {
		t.Fatalf("got %+v", rec)
	}
}

// ── --check-config ─────────────────────────────────────────────────────────

func TestCheckConfig(t *testing.T) {
	h := newHarness(t, gpu.ScenarioHealthy)
	delete(h.env, "SLURM_JOB_ID")
	if code := run([]string{"--check-config"}, h.deps()); code != exitOK || !strings.Contains(h.stdout.String(), "config ok") {
		t.Fatalf("exit %d:\n%s", code, h.stdout.String())
	}

	h.stdout.Reset()
	h.missing[defaultNvidiaSMI] = true
	if code := run([]string{"--check-config"}, h.deps()); code != exitConfigInvalid {
		t.Fatalf("missing nvidia-smi at a shell: exit %d, want %d\n%s", code, exitConfigInvalid, h.stdout.String())
	}

	h.stdout.Reset()
	if code := run([]string{"--check-config", "--budget", "20"}, h.deps()); code != exitConfigInvalid {
		t.Fatalf("typo at a shell: exit %d", code)
	}
}

func TestCheckConfigFailsWhenJobsWouldNotBeChecked(t *testing.T) {
	checkConfigAt := func(h *harness, args ...string) (int, string) {
		h.stdout.Reset()
		code := run(append([]string{"--check-config"}, args...), h.deps())
		return code, h.stdout.String()
	}

	// An unreadable device map used to be a warning followed by "config ok",
	// on exactly the setup where every numbered job is not-checked.
	h := newHarness(t, gpu.ScenarioHealthy)
	delete(h.env, "SLURM_JOB_ID")
	h.resolver = failingResolver{}
	if code, out := checkConfigAt(h); code != exitConfigInvalid || !strings.Contains(out, "FAIL device map") {
		t.Fatalf("auto, no device map: exit %d\n%s", code, out)
	}
	// nvml numbering does not read the map.
	if code, out := checkConfigAt(h, "--gpu-numbering", "nvml"); code != exitOK || !strings.Contains(out, "warn device map") {
		t.Fatalf("nvml, no device map: exit %d\n%s", code, out)
	}

	// Minors not in PCI order: auto would leave every numbered job unchecked.
	h = newHarness(t, gpu.ScenarioHealthy)
	delete(h.env, "SLURM_JOB_ID")
	h.sim.MinorsReversed = true
	code, out := checkConfigAt(h)
	if code != exitConfigInvalid || !strings.Contains(out, "FAIL gpu numbering") || !strings.Contains(out, "nvidia-smi index 0 is 00000000:18:00.0, /dev/nvidia0 is 00000000:48:00.0") {
		t.Fatalf("auto, reversed minors: exit %d\n%s", code, out)
	}
	if code, out := checkConfigAt(h, "--gpu-numbering", "nvml"); code != exitOK || !strings.Contains(out, "ok   gpu numbering") {
		t.Fatalf("nvml, reversed minors: exit %d\n%s", code, out)
	}

	// Minors in PCI order: auto is fine and says why.
	h.sim.MinorsReversed = false
	if code, out := checkConfigAt(h); code != exitOK || !strings.Contains(out, "ok   gpu numbering: auto") {
		t.Fatalf("auto, aligned minors: exit %d\n%s", code, out)
	}
}

func TestCheckConfigInsideAnEpilogNeverExitsNonZero(t *testing.T) {
	// Someone pasted --check-config into the Epilog wrapper.
	h := newHarness(t, gpu.ScenarioHealthy)
	h.missing[defaultNvidiaSMI] = true
	if code := run([]string{"--check-config"}, h.deps()); code != exitOK {
		t.Fatalf("exit %d inside a job environment", code)
	}
	if code := run([]string{"--check-config", "--budget", "20"}, h.deps()); code != exitOK {
		t.Fatalf("typo: exit %d inside a job environment", code)
	}
}

// ── the README table ───────────────────────────────────────────────────────

const (
	tableStart = "<!-- scenario-table:start -->\n```\n"
	tableEnd   = "```\n<!-- scenario-table:end -->"
)

func TestREADMEScenarioTableIsCurrent(t *testing.T) {
	// The table is generated, not typed. If this fails, paste the output of
	// `go run ./cmd/epilog-gpu-validator --scenario-table` into README.md.
	data, err := os.ReadFile("../../README.md")
	if err != nil {
		t.Fatal(err)
	}
	s := string(data)
	i, j := strings.Index(s, tableStart), strings.Index(s, tableEnd)
	if i < 0 || j < i {
		t.Fatal("README.md has no scenario-table markers")
	}
	got := s[i+len(tableStart) : j]
	if want := scenarioTable(); got != want {
		t.Fatalf("README table is stale.\n--- README ---\n%s--- generated ---\n%s", got, want)
	}
}
