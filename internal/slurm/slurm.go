// Package slurm reads the Epilog environment and drains nodes.
package slurm

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/binpath"
)

// GPU variables the Epilog may carry, in order of preference.
//
// What the numbers in them mean: Slurm writes its own GRES device index,
// which is not necessarily the /dev/nvidiaN minor or the nvidia-smi index.
// gres_common_prep_set_env()
// in src/plugins/gres/common/gres_common.c, which builds the Prolog/Epilog
// environment, formats gres_device->index into both CUDA_VISIBLE_DEVICES and
// SLURM_JOB_GPUS (read at SchedMD/slurm master 9f9da53 and at tags
// slurm-23-02-7-1, slurm-24-05-8-1 and slurm-25-05-3-1). Which physical GPU
// an index names depends on gres.conf; see gpuNumbering in
// cmd/epilog-gpu-validator/targets.go.
//
// With gres.conf Flags=env_uuid (Slurm 26.05 and later: "Add option to use
// UUID strings with CUDA_VISIBLE_DEVICES", CHANGELOG/slurm-26.05.md),
// CUDA_VISIBLE_DEVICES holds GPU UUIDs instead, while SLURM_JOB_GPUS keeps the
// numeric indices. The same function shows both.
//
// The prolog/epilog guide (https://slurm.schedmd.com/prolog_epilog.html)
// documents SLURM_JOB_GPUS as "The GPU IDs of GPUs in the job allocation",
// and says of GPU_DEVICE_ORDINAL: "The considerations for
// CUDA_VISIBLE_DEVICES also apply to GPU_DEVICE_ORDINAL." In the source,
// GPU_DEVICE_ORDINAL gets the numeric indices even with env_uuid.
var gpuVars = []string{"CUDA_VISIBLE_DEVICES", "SLURM_JOB_GPUS", "GPU_DEVICE_ORDINAL"}

// Env is the subset of the Epilog environment this tool uses.
type Env struct {
	JobID    string
	User     string
	NodeName string
	// GPUVars holds whichever of the GPU variables were set, verbatim.
	GPUVars map[string]string
	// Cluster and partition are only used to make log lines searchable.
	Partition string
	Cluster   string
}

// FromEnvironment reads the process environment.
func FromEnvironment() Env { return FromLookup(os.Getenv, os.Hostname) }

// FromLookup reads the environment through getenv, so tests need not touch
// the process environment.
func FromLookup(getenv func(string) string, hostname func() (string, error)) Env {
	node := getenv("SLURMD_NODENAME")
	if node == "" && hostname != nil {
		node, _ = hostname()
	}
	e := Env{
		JobID:     firstNonEmpty(getenv("SLURM_JOB_ID"), getenv("SLURM_JOBID")),
		User:      firstNonEmpty(getenv("SLURM_JOB_USER"), getenv("SLURM_JOB_UID")),
		NodeName:  node,
		GPUVars:   map[string]string{},
		Partition: getenv("SLURM_JOB_PARTITION"),
		Cluster:   getenv("SLURM_CLUSTER_NAME"),
	}
	for _, v := range gpuVars {
		if s := strings.TrimSpace(getenv(v)); s != "" {
			e.GPUVars[v] = s
		}
	}
	return e
}

// GPUSpec is the finished job's GPU set as the Epilog environment states it.
type GPUSpec struct {
	// Var is the variable the set was taken from; Raw is its value.
	Var string
	Raw string
	// Numbers are Slurm's GRES device indices. They are not necessarily
	// /dev/nvidiaN minors or nvidia-smi indices; which GPU a number names
	// depends on gres.conf (see gpuVars).
	Numbers []int
	// UUIDs are entries already in GPU-UUID form (gres.conf
	// Flags=env_uuid). They name one device each, whatever the numbering.
	UUIDs []string
	// Skipped are entries that are neither: MIG device UUIDs (MIG is not
	// supported) or anything unrecognised. Never guessed at.
	Skipped []string
}

var (
	// ErrNoGPUs means the job held no GPUs, as far as the Epilog can tell.
	ErrNoGPUs = errors.New("no GPU variable in the Epilog environment")
	// ErrAmbiguous means the GPU variables disagree, so which GPUs the job
	// held cannot be determined.
	ErrAmbiguous = errors.New("GPU variables in the Epilog environment disagree")
)

var (
	indexRe = regexp.MustCompile(`^\d+$`)
	uuidRe  = regexp.MustCompile(`^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$`)
)

// maxRange bounds range expansion ("0-100000" must not allocate a map).
const maxRange = 64

// JobGPUs returns the devices the finished job held.
//
// Checking every GPU on the node would be wrong on a shared node: another
// job's card could drain the node for a fault the finishing job never touched.
// ErrNoGPUs and ErrAmbiguous both mean "check nothing", never "check
// everything".
//
// When more than one variable is set they must agree. Variables in the same
// form (numbers and numbers, UUIDs and UUIDs) must name the same set. A UUID
// list and a number list cannot be compared entry by entry (with env_uuid,
// CUDA_VISIBLE_DEVICES holds UUIDs and SLURM_JOB_GPUS holds indices for the
// same GPUs), so they only have to name the same number of GPUs. The UUID
// list is then the one used, because a UUID names one device whatever
// gres.conf's numbering is.
func (e Env) JobGPUs() (GPUSpec, error) {
	var specs []GPUSpec
	for _, v := range gpuVars {
		raw, ok := e.GPUVars[v]
		if !ok || raw == "NoDevFiles" {
			// Slurm sets NoDevFiles when the job had no GPU device files.
			continue
		}
		spec := parseGPUList(raw)
		spec.Var = v
		specs = append(specs, spec)
	}
	if len(specs) == 0 {
		return GPUSpec{}, ErrNoGPUs
	}
	for i := range specs {
		for j := i + 1; j < len(specs); j++ {
			if err := agree(specs[i], specs[j]); err != nil {
				return GPUSpec{}, err
			}
		}
	}
	for _, s := range specs {
		if s.uuidsOnly() {
			return s, nil
		}
	}
	return specs[0], nil
}

// uuidsOnly and numbersOnly say which identifier form a variable uses.
// Unrecognised entries count as neither, so a list with one is compared
// entry by entry.
func (s GPUSpec) uuidsOnly() bool {
	return len(s.UUIDs) > 0 && len(s.Numbers) == 0 && len(s.Skipped) == 0
}

func (s GPUSpec) numbersOnly() bool {
	return len(s.Numbers) > 0 && len(s.UUIDs) == 0 && len(s.Skipped) == 0
}

func (s GPUSpec) count() int { return len(s.Numbers) + len(s.UUIDs) + len(s.Skipped) }

func agree(a, b GPUSpec) error {
	if (a.uuidsOnly() && b.numbersOnly()) || (a.numbersOnly() && b.uuidsOnly()) {
		if a.count() != b.count() {
			return fmt.Errorf("%w: %s=%q names %d GPU(s), %s=%q names %d", ErrAmbiguous, a.Var, a.Raw, a.count(), b.Var, b.Raw, b.count())
		}
		return nil
	}
	if !sameSet(a, b) {
		return fmt.Errorf("%w: %s=%q, %s=%q", ErrAmbiguous, a.Var, a.Raw, b.Var, b.Raw)
	}
	return nil
}

func parseGPUList(raw string) GPUSpec {
	spec := GPUSpec{Raw: raw}
	seen := map[int]bool{}
	seenUUID := map[string]bool{}
	addNumber := func(n int) {
		if !seen[n] {
			seen[n] = true
			spec.Numbers = append(spec.Numbers, n)
		}
	}
	for _, part := range strings.Split(raw, ",") {
		p := strings.TrimSpace(part)
		if p == "" {
			continue
		}
		// Ranges appear as "0-3" in some versions.
		if lo, hi, found := strings.Cut(p, "-"); found && indexRe.MatchString(lo) && indexRe.MatchString(hi) {
			l, errL := strconv.Atoi(lo)
			h, errH := strconv.Atoi(hi)
			if errL != nil || errH != nil || h < l {
				spec.Skipped = append(spec.Skipped, p)
				continue
			}
			for n := l; n <= h && n-l < maxRange; n++ {
				addNumber(n)
			}
			continue
		}
		if indexRe.MatchString(p) {
			if n, err := strconv.Atoi(p); err == nil {
				addNumber(n)
				continue
			}
		}
		if uuidRe.MatchString(p) {
			u := strings.ToUpper(p[:3]) + strings.ToLower(p[3:])
			if !seenUUID[u] {
				seenUUID[u] = true
				spec.UUIDs = append(spec.UUIDs, u)
			}
			continue
		}
		spec.Skipped = append(spec.Skipped, p)
	}
	sort.Ints(spec.Numbers)
	sort.Strings(spec.UUIDs)
	sort.Strings(spec.Skipped)
	return spec
}

func sameSet(a, b GPUSpec) bool {
	if len(a.Numbers) != len(b.Numbers) || len(a.UUIDs) != len(b.UUIDs) || len(a.Skipped) != len(b.Skipped) {
		return false
	}
	for i := range a.Numbers {
		if a.Numbers[i] != b.Numbers[i] {
			return false
		}
	}
	for i := range a.UUIDs {
		if a.UUIDs[i] != b.UUIDs[i] {
			return false
		}
	}
	for i := range a.Skipped {
		if a.Skipped[i] != b.Skipped[i] {
			return false
		}
	}
	return true
}

// SharedGPUHint reports an environment variable suggesting the job's GPU may
// be shared with other jobs (gres/shard or gres/mps), or "" if none is set.
// The prolog/epilog guide documents CUDA_MPS_ACTIVE_THREAD_PERCENTAGE as
// "Available in Prolog and Epilog only" when gres/mps is configured and
// requested. SLURM_SHARDS_ON_NODE is documented only for steps (gres guide);
// that it reaches the Epilog is not verified. Either way this can only make
// the tool MORE cautious.
func SharedGPUHint(getenv func(string) string) string {
	for _, v := range []string{"SLURM_SHARDS_ON_NODE", "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"} {
		if strings.TrimSpace(getenv(v)) != "" {
			return v
		}
	}
	return ""
}

// Controller performs the drain.
type Controller interface {
	Drain(ctx context.Context, node, reason string) error
	Name() string
}

// SControl drains via `scontrol update`.
//
// Slurm's prolog/epilog guide says Epilog scripts "should not call Slurm
// commands (e.g. squeue, scontrol, sacctmgr, etc)". This does, but only on the
// fault path: the healthy path makes no Slurm call. The drain is set before
// the non-zero exit so the descriptive reason is the one `sinfo -R` shows.
type SControl struct{ Binary string }

// NewSControl returns a controller for the scontrol at binary, which must be
// an absolute path (the Epilog has no PATH).
func NewSControl(binary string) *SControl { return &SControl{Binary: binary} }

func (s *SControl) Name() string { return "scontrol" }

// Drain runs scontrol, after checking the binary again with binpath.Check
// right before exec. The caller checks it at startup and does not call Drain
// when that fails; this second check narrows the window in which the file
// could be swapped, and protects any other caller. A refused binary is never
// run: the caller still exits 1, so Slurm drains the node with its own
// generic reason.
func (s *SControl) Drain(ctx context.Context, node, reason string) error {
	if node == "" {
		return errors.New("scontrol drain: empty node name")
	}
	if err := binpath.Check(s.Binary); err != nil {
		// Covers a bare name too: never a PATH lookup as root.
		return fmt.Errorf("scontrol drain: not run: %w", err)
	}
	// Slurm rejects a reason containing certain characters; keep it plain.
	reason = sanitizeReason(reason)
	out, err := exec.CommandContext(ctx, s.Binary, "update",
		"NodeName="+node, "State=DRAIN", "Reason="+reason).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scontrol drain %s: %w: %s", node, err, strings.TrimSpace(string(out)))
	}
	return nil
}

var reasonUnsafe = regexp.MustCompile(`[^\w\s:.,\-/=]`)

// sanitizeReason strips characters that have no business in a drain reason
// and bounds its length. Commands run from argv, never a shell, so this is
// about legibility and Slurm's parser, not injection.
func sanitizeReason(r string) string {
	r = reasonUnsafe.ReplaceAllString(r, "")
	r = strings.Join(strings.Fields(r), " ")
	if len(r) > 200 {
		r = r[:197] + "..."
	}
	return r
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}
