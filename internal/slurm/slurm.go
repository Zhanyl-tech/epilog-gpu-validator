// Package slurm reads the Epilog environment and drains nodes.
package slurm

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Env is the subset of the Epilog environment this tool uses.
//
// Slurm exports these into Prolog/Epilog. Notably absent is anything about
// which GPUs the job held in a portable form — see GPUIndices.
type Env struct {
	JobID    string
	User     string
	NodeName string
	// GPUs as reported by Slurm, in whatever form this version uses.
	RawGPUs string
	// Cluster and partition are only used to make log lines searchable.
	Partition string
	Cluster   string
}

func FromEnvironment() Env {
	node := os.Getenv("SLURMD_NODENAME")
	if node == "" {
		node, _ = os.Hostname()
	}
	return Env{
		JobID:     firstNonEmpty(os.Getenv("SLURM_JOB_ID"), os.Getenv("SLURM_JOBID")),
		User:      firstNonEmpty(os.Getenv("SLURM_JOB_USER"), os.Getenv("SLURM_JOB_UID")),
		NodeName:  node,
		RawGPUs:   firstNonEmpty(os.Getenv("SLURM_JOB_GPUS"), os.Getenv("GPU_DEVICE_ORDINAL"), os.Getenv("CUDA_VISIBLE_DEVICES")),
		Partition: os.Getenv("SLURM_JOB_PARTITION"),
		Cluster:   os.Getenv("SLURM_CLUSTER_NAME"),
	}
}

var indexRe = regexp.MustCompile(`^\d+$`)

// GPUIndices returns the device indices the finished job held.
//
// Checking every GPU on the node would be wrong on a shared node: another
// job's card could drain the node for a fault the finishing job never touched.
// Returns nil when the set cannot be determined, which the caller must treat as
// "check nothing" rather than "check everything".
//
// Note the values can be UUIDs rather than ordinals depending on Slurm version
// and GresTypes config; only numeric entries are usable as -i arguments.
func (e Env) GPUIndices() []int {
	raw := strings.TrimSpace(e.RawGPUs)
	if raw == "" || raw == "NoDevFiles" {
		return nil
	}

	var out []int
	for _, part := range strings.Split(raw, ",") {
		p := strings.TrimSpace(part)
		// Ranges appear as "0-3" in some versions.
		if lo, hi, found := strings.Cut(p, "-"); found && indexRe.MatchString(lo) && indexRe.MatchString(hi) {
			l, _ := strconv.Atoi(lo)
			h, _ := strconv.Atoi(hi)
			for n := l; n <= h && n-l < 64; n++ {
				out = append(out, n)
			}
			continue
		}
		if indexRe.MatchString(p) {
			n, _ := strconv.Atoi(p)
			out = append(out, n)
		}
		// Non-numeric entries (UUIDs) are skipped; the caller falls back.
	}
	sort.Ints(out)
	return out
}

// Controller performs the drain.
type Controller interface {
	Drain(ctx context.Context, node, reason string) error
	Name() string
}

// SControl drains via `scontrol update`.
type SControl struct{ Binary string }

func NewSControl() *SControl { return &SControl{Binary: "scontrol"} }

func (s *SControl) Name() string { return "scontrol" }

func (s *SControl) Drain(ctx context.Context, node, reason string) error {
	// Slurm rejects a reason containing certain characters; keep it plain.
	reason = sanitizeReason(reason)
	out, err := exec.CommandContext(ctx, s.Binary, "update",
		"NodeName="+node, "State=DRAIN", "Reason="+reason).CombinedOutput()
	if err != nil {
		return fmt.Errorf("scontrol drain %s: %w: %s", node, err, strings.TrimSpace(string(out)))
	}
	return nil
}

// DryRun logs what would happen. The default, because a health check that
// drains nodes the day it is installed does not get installed twice.
type DryRun struct{}

func (DryRun) Name() string                                { return "dry-run" }
func (DryRun) Drain(context.Context, string, string) error { return nil }

var reasonUnsafe = regexp.MustCompile(`[^\w\s:.,\-/=]`)

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
