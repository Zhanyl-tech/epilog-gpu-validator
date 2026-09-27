// Command epilog-gpu-validator checks the GPUs a finished job used and drains
// the node if they show evidence of a persistent fault.
//
// Designed for Slurm's Epilog. Two constraints shape everything:
//
//   - It runs on every job completion, and slurm.conf says an Epilog that
//     times out gets the node drained. The whole run is bounded by --budget,
//     and it returns even if nvidia-smi cannot be reaped.
//
//   - Slurm drains the node when Epilog exits non-zero. That makes a false
//     positive an outage, so the exit contract is:
//
//     1  only with --enforce, only for a finding classified Degraded or
//     Fatal, and only on the real (not simulated) path;
//     0  for everything else, including a bad flag, an unknown flag, a
//     missing nvidia-smi, an unreadable environment, a failed query, a
//     timeout and a panic. Each of those is reported loudly (stderr, and
//     --log-file / --syslog when set) because a silent exit 0 would look
//     exactly like a healthy node.
//
// The contract is unit-tested in main_test.go.
package main

import (
	"io"
	"os"
	"time"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/binpath"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

var version = "dev"

// Exit codes. Slurm only distinguishes zero from non-zero.
const (
	exitOK             = 0
	exitDrainRequested = 1
	// exitConfigInvalid is used ONLY by --check-config run outside a Slurm
	// job environment (an admin at a shell). See checkConfig.
	exitConfigInvalid = 78 // EX_CONFIG in sysexits.h
)

// deps is everything run touches outside its own memory, so tests can
// replace it.
type deps struct {
	getenv   func(string) string
	hostname func() (string, error)
	stdout   io.Writer
	stderr   io.Writer
	now      func() time.Time

	newSource     func(binary string, timeout time.Duration) gpu.Source
	newResolver   func(procDir string) gpu.Resolver
	newController func(binary string) slurm.Controller
	checkBinary   func(path string) error
	openSyslog    func() (syslogSink, error)
	openLogFile   func(path string) (io.WriteCloser, error)
}

func realDeps() deps {
	return deps{
		getenv:   os.Getenv,
		hostname: os.Hostname,
		stdout:   os.Stdout,
		stderr:   os.Stderr,
		now:      time.Now,
		newSource: func(binary string, timeout time.Duration) gpu.Source {
			return gpu.NewSMISource(binary, timeout)
		},
		newResolver:   func(dir string) gpu.Resolver { return gpu.ProcResolver{Dir: dir} },
		newController: func(binary string) slurm.Controller { return slurm.NewSControl(binary) },
		checkBinary:   binpath.Check,
		openSyslog:    openSyslog,
		openLogFile: func(path string) (io.WriteCloser, error) {
			return os.OpenFile(path, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0o640)
		},
	}
}

func main() {
	os.Exit(run(os.Args[1:], realDeps()))
}
