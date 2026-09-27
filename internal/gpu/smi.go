package gpu

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Runner executes a command and returns its stdout. It exists so tests and
// the simulator can stand in for a real nvidia-smi. An error that has an
// ExitCode() int method (as *exec.ExitError does) is classified by code.
type Runner func(ctx context.Context, name string, args ...string) ([]byte, error)

// SMISource queries nvidia-smi.
type SMISource struct {
	// Binary is an absolute path. Slurm runs the Epilog with no search path
	// ("for security reasons, these programs do not have a search path set",
	// https://slurm.schedmd.com/prolog_epilog.html), so a bare "nvidia-smi"
	// is never found there, and a PATH lookup as root is a hijack risk.
	Binary string
	// Timeout caps one nvidia-smi invocation; zero means the caller's
	// context alone decides.
	Timeout time.Duration
	// Run defaults to RunCommand.
	Run Runner
	// name overrides Name(), for the simulator.
	name string
}

// NewSMISource returns a source for the nvidia-smi at binary.
func NewSMISource(binary string, timeout time.Duration) *SMISource {
	return &SMISource{Binary: binary, Timeout: timeout, Run: RunCommand}
}

func (s *SMISource) Name() string {
	if s.name != "" {
		return s.name
	}
	return "nvidia-smi"
}

// QueryArgs is the argument list for the one query this tool makes. There
// is no -i: the manual documents -i as selecting "a single specified GPU", so
// the whole node is queried once and the job's GPUs are picked out after.
func QueryArgs() []string {
	return []string{"--query-gpu=" + strings.Join(QueryFields, ","), "--format=csv,noheader,nounits"}
}

// Query runs nvidia-smi once for every GPU on the node.
func (s *SMISource) Query(ctx context.Context) (Result, error) {
	if s.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.Timeout)
		defer cancel()
	}
	run := s.Run
	if run == nil {
		run = RunCommand
	}
	out, err := run(ctx, s.Binary, QueryArgs()...)
	if err != nil {
		return Result{}, classifyRunError(ctx, err)
	}
	return ParseCSV(out)
}

// ── errors ─────────────────────────────────────────────────────────────────

// ExitCodeMeaning is the RETURN VALUE list from the nvidia-smi manual
// (https://docs.nvidia.com/deploy/nvidia-smi/index.html), verbatim.
var ExitCodeMeaning = map[int]string{
	2:   "A supplied argument or flag is invalid",
	3:   "The requested operation is not available on target device",
	4:   "The current user does not have permission to access this device or perform this operation",
	6:   "A query to find an object was unsuccessful",
	8:   "A device's external power cables are not properly attached",
	9:   "NVIDIA driver is not loaded",
	10:  "NVIDIA Kernel detected an interrupt issue with a GPU",
	12:  "NVML Shared Library couldn't be found or loaded",
	13:  "Local version of NVML doesn't implement this function",
	14:  "infoROM is corrupted",
	15:  "The GPU has fallen off the bus or has otherwise become inaccessible",
	255: "Other error or internal driver error occurred",
}

// HardwareFaultExitCode reports whether an nvidia-smi exit code is itself
// documented evidence of a hardware fault, as opposed to a monitoring
// failure (driver not loaded, NVML missing, bad arguments, internal error).
func HardwareFaultExitCode(code int) bool {
	switch code {
	case 8, 10, 14, 15:
		return true
	}
	return false
}

// QueryError is a failed nvidia-smi run.
type QueryError struct {
	// ExitCode is the process exit code, or -1 when it did not exit on its
	// own (not found, killed on timeout).
	ExitCode int
	// Timeout is true when the budget ran out.
	Timeout bool
	// Stderr is the first part of what nvidia-smi printed, for the log.
	Stderr string
	Err    error
}

func (e *QueryError) Error() string {
	var b strings.Builder
	b.WriteString("nvidia-smi")
	switch {
	case e.Timeout:
		b.WriteString(": timed out")
	case e.ExitCode >= 0:
		fmt.Fprintf(&b, ": exit %d", e.ExitCode)
		if m, ok := ExitCodeMeaning[e.ExitCode]; ok {
			fmt.Fprintf(&b, " (%s)", m)
		}
	}
	if e.Err != nil {
		fmt.Fprintf(&b, ": %v", e.Err)
	}
	if e.Stderr != "" {
		fmt.Fprintf(&b, ": %s", e.Stderr)
	}
	return b.String()
}

func (e *QueryError) Unwrap() error { return e.Err }

// ParseError means nvidia-smi exited 0 but its output was not CSV.
type ParseError struct{ Err error }

func (e *ParseError) Error() string { return "parse nvidia-smi output: " + e.Err.Error() }
func (e *ParseError) Unwrap() error { return e.Err }

// exitCoder is satisfied by *exec.ExitError and by test fakes.
type exitCoder interface{ ExitCode() int }

// stderrer is satisfied by *RunError, which carries the child's stderr.
type stderrer interface{ StderrText() string }

func classifyRunError(ctx context.Context, err error) *QueryError {
	qe := &QueryError{ExitCode: -1, Err: err}
	var ec exitCoder
	if errors.As(err, &ec) {
		qe.ExitCode = ec.ExitCode()
	}
	var se stderrer
	if errors.As(err, &se) {
		qe.Stderr = se.StderrText()
	}
	// A deadline beats an exit code: a process killed on timeout reports -1
	// or a signal, and neither says anything about the hardware.
	if errors.Is(err, context.DeadlineExceeded) || ctx.Err() != nil {
		qe.Timeout = true
		qe.ExitCode = -1
	}
	return qe
}

// ── running ────────────────────────────────────────────────────────────────

// waitDelay bounds how long Wait may block after the context is done: on a
// child that ignores the kill, or on a grandchild still holding stdout. See
// os/exec Cmd.WaitDelay (Go 1.20+).
const waitDelay = 500 * time.Millisecond

// abandonAfter is how long past the deadline RunCommand keeps waiting before
// it gives up on reaping the child at all. A process stuck in uninterruptible
// sleep on a wedged driver cannot be killed, and waiting for it would carry
// the Epilog past EpilogTimeout, which drains the node.
const abandonAfter = 2 * waitDelay

// RunError is a non-zero exit with the child's stderr attached.
type RunError struct {
	Err    error
	Stderr string
}

func (e *RunError) Error() string      { return e.Err.Error() }
func (e *RunError) Unwrap() error      { return e.Err }
func (e *RunError) StderrText() string { return e.Stderr }

// RunCommand runs name with args and returns stdout. It returns when the
// command finishes or shortly after ctx is done, whichever is first, even if
// the child cannot be reaped (it is then left behind).
func RunCommand(ctx context.Context, name string, args ...string) ([]byte, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	var stdout bytes.Buffer
	stderr := &capBuffer{max: 512}
	cmd.Stdout = &stdout
	cmd.Stderr = stderr
	cmd.WaitDelay = waitDelay
	if err := cmd.Start(); err != nil {
		return nil, err
	}

	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		if err != nil {
			return nil, &RunError{Err: err, Stderr: strings.TrimSpace(stderr.String())}
		}
		return stdout.Bytes(), nil
	case <-ctx.Done():
	}
	// The context is done and CommandContext has sent the kill. Give Wait a
	// bounded chance to finish, then abandon it. The buffers are not read on
	// this path, so the goroutine still writing to them is not a race.
	t := time.NewTimer(abandonAfter)
	defer t.Stop()
	select {
	case <-done:
		return nil, fmt.Errorf("%s: %w", name, ctx.Err())
	case <-t.C:
		return nil, fmt.Errorf("%s: %w (child not reaped; abandoned)", name, ctx.Err())
	}
}

// capBuffer keeps the first max bytes written and discards the rest.
type capBuffer struct {
	buf bytes.Buffer
	max int
}

func (c *capBuffer) Write(p []byte) (int, error) {
	if room := c.max - c.buf.Len(); room > 0 {
		if len(p) > room {
			c.buf.Write(p[:room])
		} else {
			c.buf.Write(p)
		}
	}
	return len(p), nil
}

func (c *capBuffer) String() string { return c.buf.String() }

// ── parsing ────────────────────────────────────────────────────────────────

// ParseCSV parses `--format=csv,noheader,nounits` output for QueryFields.
func ParseCSV(out []byte) (Result, error) {
	r := csv.NewReader(bytes.NewReader(out))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return Result{}, &ParseError{Err: err}
	}
	var res Result
	for _, row := range rows {
		h, err := parseRow(row)
		if err != nil {
			res.Malformed = append(res.Malformed, boundRow(strings.Join(row, ", "))+": "+err.Error())
			continue
		}
		res.GPUs = append(res.GPUs, h)
	}
	return res, nil
}

func boundRow(s string) string {
	if len(s) > 160 {
		return s[:157] + "..."
	}
	return s
}

// parseRow reads one GPU. The identity columns (index, uuid, bus id) must
// parse or the row is rejected: a row that cannot be tied to a device cannot
// be attributed to the job. Every other column that does not parse is
// recorded in Unreadable rather than defaulted to a healthy-looking zero.
func parseRow(row []string) (Health, error) {
	if len(row) != len(QueryFields) {
		return Health{}, fmt.Errorf("%d columns, want %d", len(row), len(QueryFields))
	}
	col := func(field string) string {
		for i, f := range QueryFields {
			if f == field {
				return strings.TrimSpace(row[i])
			}
		}
		panic("unknown field " + field) // programming error, caught by tests
	}

	h := Health{UUID: col(FieldUUID), Name: col(FieldName)}
	idx, err := strconv.Atoi(col(FieldIndex))
	if err != nil || idx < 0 {
		return Health{}, fmt.Errorf("unreadable index %q", col(FieldIndex))
	}
	h.Index = idx
	bus, ok := NormalizeBusID(col(FieldBusID))
	if !ok {
		return Health{}, fmt.Errorf("unreadable pci.bus_id %q", col(FieldBusID))
	}
	h.PCIBusID = bus

	bad := func(field string) { h.Unreadable = append(h.Unreadable, field) }
	i := func(field string, dst *int) {
		v, ok := parseInt(col(field))
		if !ok {
			bad(field)
			return
		}
		*dst = int(v)
	}
	i64 := func(field string, dst *int64) {
		v, ok := parseInt(col(field))
		if !ok {
			bad(field)
			return
		}
		*dst = v
	}

	i(FieldPCIeWidthCurrent, &h.PCIeWidthCurrent)
	i(FieldPCIeWidthMax, &h.PCIeWidthMax)
	i(FieldPCIeGenCurrent, &h.PCIeGenCurrent)
	i(FieldPCIeGenMax, &h.PCIeGenMax)
	i64(FieldECCUncorrVolatile, &h.ECCUncorrectableVolatile)
	i64(FieldECCUncorrAggregate, &h.ECCUncorrectableAggregate)
	i64(FieldECCCorrVolatile, &h.ECCCorrectableVolatile)

	// Pending and failure: the CSV form is not documented here; the -q form
	// prints "Yes"/"No" (nvidia-smi manual, Remapped Rows). Accept a count or
	// Yes/No, and nothing else.
	if v, ok := parseCountOrYesNo(col(FieldRemapPending)); ok {
		h.RemappedRowsPending = v
	} else {
		bad(FieldRemapPending)
	}
	i64(FieldRemapUncorrectable, &h.RemappedRowsUncorrectable)
	if v, ok := parseCountOrYesNo(col(FieldRemapFailure)); ok {
		h.RemappedRowsFailure = v > 0
	} else {
		bad(FieldRemapFailure)
	}

	if v, ok := parseClockReasons(col(FieldClockReasons)); ok {
		h.ClockEventReasons = v
	} else {
		bad(FieldClockReasons)
	}
	i(FieldTemperature, &h.TemperatureC)
	i64(FieldMemUsed, &h.MemUsedMiB)
	i64(FieldMemTotal, &h.MemTotalMiB)

	switch p := col(FieldPersistence); {
	case strings.EqualFold(p, "Enabled"):
		h.PersistenceMode = true
	case strings.EqualFold(p, "Disabled"):
		h.PersistenceMode = false
	default:
		bad(FieldPersistence)
	}
	return h, nil
}

// parseInt reads a non-negative decimal. "[N/A]", "[Not Supported]" and
// anything else non-numeric are unreadable, never zero.
func parseInt(s string) (int64, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.HasPrefix(s, "[") {
		return 0, false
	}
	v, err := strconv.ParseInt(s, 10, 64)
	if err != nil || v < 0 {
		return 0, false
	}
	return v, true
}

func parseCountOrYesNo(s string) (int64, bool) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "yes":
		return 1, true
	case "no":
		return 0, true
	}
	return parseInt(s)
}
