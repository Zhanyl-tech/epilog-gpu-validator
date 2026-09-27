package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"strings"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/checks"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

// Default tool locations. Slurm gives the Epilog no search path, so these
// have to be absolute. /usr/bin is a common location, not a universal one (a
// from-source build installs wherever its --prefix pointed), so check yours
// with `command -v` and set the flag.
const (
	defaultNvidiaSMI = "/usr/bin/nvidia-smi"
	defaultSControl  = "/usr/bin/scontrol"
)

type options struct {
	budget         time.Duration
	queryTimeout   time.Duration
	enforce        bool
	allGPUs        bool
	simulate       string
	jsonOut        bool
	maxCorrECC     int64
	maxTemp        int
	allowPCIe      bool
	drainPCIeWidth bool
	drainRemap     bool
	sharedGPUs     bool
	numbering      string
	nvidiaSMI      string
	procDir        string
	scontrol       string
	logFile        string
	syslog         bool
	checkConfig    bool
	scenarioTable  bool
	showVersion    bool
}

func newFlagSet(o *options, stderr io.Writer) *flag.FlagSet {
	fs := flag.NewFlagSet("epilog-gpu-validator", flag.ContinueOnError)
	fs.SetOutput(stderr)
	fs.DurationVar(&o.budget, "budget", 20*time.Second, "hard time limit for the whole run; keep it well under Slurm's EpilogTimeout")
	fs.DurationVar(&o.queryTimeout, "query-timeout", 10*time.Second, "time limit for the nvidia-smi call (capped at --budget); the rest of the budget is left for scontrol")
	fs.BoolVar(&o.enforce, "enforce", false, "actually drain the node and exit 1 on a Degraded/Fatal finding (default: report only, always exit 0)")
	fs.BoolVar(&o.allGPUs, "all-gpus", false, "check every GPU on the node, not just the job's")
	fs.StringVar(&o.simulate, "simulate", "", "run against a simulated node; never drains, never exits non-zero. One of: "+scenarioList())
	fs.BoolVar(&o.jsonOut, "json", false, "print the run record as JSON on stdout")
	fs.Int64Var(&o.maxCorrECC, "max-correctable-ecc", 1000, "correctable ECC errors since driver load above which a GPU is Degraded (site policy; 0 disables)")
	fs.IntVar(&o.maxTemp, "max-temperature", 90, "temperature in °C at or above which a Transient finding is reported (site policy; 0 disables)")
	fs.BoolVar(&o.allowPCIe, "allow-pcie-downgrade", false, "do not report PCIe links below max at all")
	fs.BoolVar(&o.drainPCIeWidth, "drain-on-pcie-width", false, "treat an idle link narrower than max as Degraded; only after confirming idle healthy GPUs on this hardware report full width")
	fs.BoolVar(&o.drainRemap, "drain-on-pending-remap", false, "drain when a row remap is pending a GPU reset")
	fs.BoolVar(&o.sharedGPUs, "shared-gpus", false, "GPUs may be shared between jobs (gres/mps, shards): memory still in use is Unknown, not Degraded")
	fs.StringVar(&o.nvidiaSMI, "nvidia-smi", defaultNvidiaSMI, "absolute path to nvidia-smi (the Epilog has no PATH)")
	fs.StringVar(&o.scontrol, "scontrol", defaultSControl, "absolute path to scontrol, used only with --enforce")
	fs.StringVar(&o.numbering, "gpu-numbering", string(numberingAuto), "what a number in the Epilog's GPU variables means on this node: "+numberingList()+" (see README, Only the job's own GPUs)")
	fs.StringVar(&o.procDir, "driver-proc-dir", gpu.DefaultProcDir, "where the NVIDIA driver lists GPUs by PCI address; read by --gpu-numbering auto and minor (change only for tests or unusual container layouts)")
	fs.StringVar(&o.logFile, "log-file", "", "append one JSON line per run to this file")
	fs.BoolVar(&o.syslog, "syslog", false, "send a one-line summary of each run to syslog (tag epilog-gpu-validator)")
	fs.BoolVar(&o.checkConfig, "check-config", false, "install-time check: verify paths and run one query, print the result, exit 0 or 78")
	fs.BoolVar(&o.scenarioTable, "scenario-table", false, "print the README scenario table (simulated; touches no hardware and no Slurm)")
	fs.BoolVar(&o.showVersion, "version", false, "print version and exit")
	return fs
}

// parseFlags never exits the process. flag.Parse's default ExitOnError exits
// 2 on a typo, and Slurm drains the node on any non-zero Epilog exit, so one
// wrapper typo would drain the fleet one job at a time.
//
// Flags that precede a bad one are still set when the error comes back, which
// is why the wrapper passes --log-file and --syslog first: a later typo is
// then still reported where the operator looks.
func parseFlags(args []string, stderr io.Writer) (options, error) {
	var o options
	fs := newFlagSet(&o, stderr)
	err := fs.Parse(args)
	if err == nil && fs.NArg() > 0 {
		err = fmt.Errorf("unexpected arguments %q", fs.Args())
	}
	return o, err
}

func (o options) validate() error {
	var errs []string
	if o.budget <= 0 {
		errs = append(errs, "--budget must be positive")
	}
	if o.queryTimeout <= 0 {
		errs = append(errs, "--query-timeout must be positive")
	}
	if o.maxCorrECC < 0 {
		errs = append(errs, "--max-correctable-ecc must not be negative")
	}
	if o.maxTemp < 0 {
		errs = append(errs, "--max-temperature must not be negative")
	}
	if o.simulate != "" && !gpu.ValidScenario(o.simulate) {
		errs = append(errs, fmt.Sprintf("--simulate %q is not a scenario (%s)", o.simulate, scenarioList()))
	}
	if !validNumbering(o.numbering) {
		errs = append(errs, fmt.Sprintf("--gpu-numbering %q is not one of %s", o.numbering, numberingList()))
	}
	if len(errs) > 0 {
		return errors.New(strings.Join(errs, "; "))
	}
	return nil
}

func (o options) queryLimit() time.Duration {
	if o.queryTimeout < o.budget {
		return o.queryTimeout
	}
	return o.budget
}

func (o options) config(getenv func(string) string) (checks.Config, string) {
	cfg := checks.DefaultConfig()
	cfg.MaxCorrectableECC = o.maxCorrECC
	cfg.MaxTemperatureC = o.maxTemp
	cfg.AllowPCIeDowngrade = o.allowPCIe
	cfg.DrainOnPCIeWidth = o.drainPCIeWidth
	cfg.DrainOnPendingRemap = o.drainRemap
	cfg.SharedGPUs = o.sharedGPUs
	hint := ""
	if !cfg.SharedGPUs {
		if hint = slurm.SharedGPUHint(getenv); hint != "" {
			cfg.SharedGPUs = true
		}
	}
	return cfg, hint
}

func scenarioList() string {
	names := make([]string, len(gpu.Scenarios))
	for i, s := range gpu.Scenarios {
		names[i] = string(s)
	}
	return strings.Join(names, "|")
}

// inJobEnvironment reports whether this looks like a Slurm Prolog/Epilog
// (or job) environment, where any non-zero exit can drain the node.
func inJobEnvironment(getenv func(string) string) bool {
	return getenv("SLURM_JOB_ID") != "" || getenv("SLURM_JOBID") != ""
}

// run is the whole program. It returns the exit code and never panics.
func run(args []string, d deps) int {
	r := newReporter(d)
	code := exitOK
	func() {
		defer func() {
			if p := recover(); p != nil {
				// A Go panic exits 2, and Slurm drains on any non-zero exit.
				// Only the goroutine running execute is covered; nothing
				// else here starts one that can panic.
				r.configError(fmt.Errorf("internal error (panic): %v", p))
				code = exitOK
			}
		}()
		code = execute(args, d, r)
	}()
	func() {
		// Reporting must never change the exit code.
		defer func() { _ = recover() }()
		r.flush(code)
	}()
	return code
}

func execute(args []string, d deps, r *reporter) int {
	o, err := parseFlags(args, d.stderr)
	r.o = o
	if errors.Is(err, flag.ErrHelp) {
		r.silent = true
		return exitOK
	}
	if err != nil {
		r.configError(fmt.Errorf("bad command line: %w", err))
		if wantsCheckConfig(args) && !inJobEnvironment(d.getenv) {
			r.silent = true
			fmt.Fprintf(d.stdout, "FAIL command line: %v\n", err)
			return exitConfigInvalid
		}
		return exitOK
	}
	if o.showVersion {
		r.silent = true
		fmt.Fprintln(d.stdout, version)
		return exitOK
	}
	if o.scenarioTable {
		r.silent = true
		fmt.Fprint(d.stdout, scenarioTable())
		return exitOK
	}
	if o.checkConfig {
		r.silent = true
		return checkConfig(o, d)
	}

	env := slurm.FromLookup(d.getenv, d.hostname)
	r.rec.JobID, r.rec.Node, r.rec.Enforce = env.JobID, env.NodeName, o.enforce

	if err := o.validate(); err != nil {
		r.configError(err)
		return exitOK
	}

	var (
		src      gpu.Source
		resolver gpu.Resolver
		sim      = o.simulate != ""
	)
	if sim {
		s := gpu.NewSim(gpu.Scenario(o.simulate), 4)
		src, resolver = s.Source(), s
		r.rec.Simulated = true
		if o.enforce {
			r.warn("--enforce is ignored with --simulate: the simulator never drains and never exits non-zero")
		}
	} else {
		src, resolver = d.newSource(o.nvidiaSMI, o.queryLimit()), d.newResolver(o.procDir)
	}
	r.rec.Source = src.Name()

	// ── Which GPUs did the job hold? ──────────────────────────────────────
	// Only those. On a shared node, checking everything means another job's
	// faulty card drains the node for a job that never touched it.
	selectAll := o.allGPUs
	var spec slurm.GPUSpec
	if !selectAll {
		var err error
		spec, err = env.JobGPUs()
		switch {
		case errors.Is(err, slurm.ErrNoGPUs) && sim:
			selectAll = true // no job around the simulator: check its whole node
		case errors.Is(err, slurm.ErrNoGPUs):
			r.setStatus(statusNoGPUs)
			return exitOK
		case err != nil:
			r.add(checks.NotChecked("", "gpu-set-ambiguous", err.Error()+"; checking nothing rather than guessing"))
			r.setStatus(statusNotChecked)
			return exitOK
		default:
			r.rec.GPUEnv = &gpuEnvJSON{Var: spec.Var, Raw: spec.Raw}
			if len(spec.Numbers) == 0 && len(spec.UUIDs) == 0 {
				// Only entries this tool cannot use (MIG, garbage).
				_, unresolved := resolveTargets(spec, gpuNumbering(o.numbering), resolver, nil)
				r.gap(unresolved...)
				r.setStatus(statusNotChecked)
				return exitOK
			}
		}
	}

	// ── Preflight, once there is something to check ──────────────────────
	var scontrolErr error
	if !sim {
		if err := d.checkBinary(o.nvidiaSMI); err != nil {
			r.configError(fmt.Errorf("--nvidia-smi: %w", err))
			return exitOK
		}
		if o.enforce {
			if err := d.checkBinary(o.scontrol); err != nil {
				// Not a reason to skip the check, but scontrol is then never
				// run: a binary that fails these checks could be replaced
				// by someone other than root, and the Epilog runs as root.
				// A drain still happens, through exit 1 and Slurm's own
				// Epilog-failure drain, with Slurm's generic reason.
				scontrolErr = fmt.Errorf("--scontrol: %w", err)
				r.warn(fmt.Sprintf("%v; scontrol will not be run, so a drain would rely on exit 1 and Slurm's own Epilog-failure drain", scontrolErr))
			}
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), o.budget)
	defer cancel()

	cfg, hint := o.config(d.getenv)
	if hint != "" {
		r.warn(fmt.Sprintf("%s is set, so the GPU may be shared: leftover memory is reported as unknown", hint))
	}

	res, err := src.Query(ctx)
	if err != nil {
		f := checks.QueryFailure(err)
		r.add(f)
		if !f.Severity.ShouldDrain() {
			r.setStatus(statusNotChecked)
			return exitOK
		}
		// A documented hardware-fault exit code: positive evidence, decided
		// below like any other Fatal finding.
	} else {
		for _, m := range res.Malformed {
			f := checks.NotChecked("", checkMalformedRow, "nvidia-smi row not understood, not checked: "+m)
			if selectAll {
				// Every GPU on the node is a target: this one was missed.
				r.gap(f)
			} else {
				// A job GPU in this row shows up below as not matched.
				r.add(f)
			}
		}
		gpus := res.GPUs
		if !selectAll {
			targets, unresolved := resolveTargets(spec, gpuNumbering(o.numbering), resolver, res.GPUs)
			r.gap(unresolved...)
			var missing []gpu.Target
			gpus, missing = gpu.Match(targets, gpus)
			for _, t := range missing {
				// Not evidence either way: nvidia-smi exited 0 and simply
				// did not list it. The manual documents exit 15 for a GPU
				// that "has fallen off the bus", handled above; that a lost
				// GPU produces exit 15 rather than a missing or unreadable
				// row has not been observed here.
				r.gap(checks.NotChecked(targetLabel(t), checkNotInOutput,
					fmt.Sprintf("%s was not in nvidia-smi's output; not checked", t)))
			}
		}
		for _, h := range gpus {
			r.checked(h)
			r.add(checks.Evaluate(h, cfg)...)
		}
	}

	sum := checks.Summarize(r.findings)
	r.rec.Worst, r.rec.Drain, r.rec.Reason = sum.Worst.String(), sum.Drain, sum.Reason

	if !sum.Drain {
		switch {
		case len(r.rec.GPUs) == 0:
			r.setStatus(statusNotChecked)
		case len(r.gaps) > 0:
			// Some of the job's GPUs were checked and some were not. "ok"
			// would hide the ones that were not.
			r.setStatus(statusPartiallyChecked)
		default:
			r.setStatus(statusOK)
		}
		return exitOK
	}

	// Report-only is the default. A health check that starts draining nodes
	// on the day it is installed does not survive to a second day.
	if sim || !o.enforce {
		r.setStatus(statusWouldDrain)
		return exitOK
	}

	if scontrolErr != nil {
		// Never run an scontrol that failed the root-safety check. Exit 1
		// still makes Slurm drain the node, with its own generic reason.
		r.errs = append(r.errs, scontrolErr.Error()+" (scontrol not run)")
		r.setStatus(statusDrainFailed)
		return exitDrainRequested
	}
	ctl := d.newController(o.scontrol)
	if err := ctl.Drain(ctx, env.NodeName, sum.Reason); err != nil {
		r.errs = append(r.errs, err.Error())
		r.setStatus(statusDrainFailed)
		// Still signal the fault: a non-zero exit makes Slurm drain the node
		// itself, which is the outcome wanted.
		return exitDrainRequested
	}
	r.setStatus(statusDrained)
	return exitDrainRequested
}

func wantsCheckConfig(args []string) bool {
	for _, a := range args {
		if a == "-check-config" || a == "--check-config" || strings.HasPrefix(a, "-check-config=") || strings.HasPrefix(a, "--check-config=") {
			return true
		}
	}
	return false
}
