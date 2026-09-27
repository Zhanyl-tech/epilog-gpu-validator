package slurm

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"testing"
)

func envWith(vars map[string]string) Env {
	return FromLookup(func(k string) string { return vars[k] }, func() (string, error) { return "host-from-os", nil })
}

func TestJobGPUsParsesEveryFormat(t *testing.T) {
	const uuid = "GPU-5117a000-0000-4000-8000-000000000003"
	cases := []struct {
		name        string
		raw         string
		wantNumbers []int
		wantUUIDs   []string
		wantSkipped []string
	}{
		{"comma list", "0,1,2,3", []int{0, 1, 2, 3}, nil, nil},
		{"single", "2", []int{2}, nil, nil},
		{"range", "0-3", []int{0, 1, 2, 3}, nil, nil},
		{"mixed", "0,2-4,7", []int{0, 2, 3, 4, 7}, nil, nil},
		{"spaces", " 1, 3 ", []int{1, 3}, nil, nil},
		{"duplicates", "1,1,0-1", []int{0, 1}, nil, nil},
		// nvidia-smi accepts "the GPU's UUID" for -i, and UUIDs are the only
		// identifier NVIDIA calls stable, so they are used, not skipped.
		{"uuid", uuid, nil, []string{uuid}, nil},
		{"uuid hex case", "GPU-5117A000-0000-4000-8000-00000000000A", nil, []string{"GPU-5117a000-0000-4000-8000-00000000000a"}, nil},
		{"mixed uuid and index", uuid + ",2", []int{2}, []string{uuid}, nil},
		// MIG is not supported; anything unrecognised is reported, not guessed.
		{"mig", "MIG-5117a000-0000-4000-8000-000000000003", nil, nil, []string{"MIG-5117a000-0000-4000-8000-000000000003"}},
		{"garbage", "gpu0,1", []int{1}, nil, []string{"gpu0"}},
		{"reversed range", "3-1", nil, nil, []string{"3-1"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			spec, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": tc.raw}).JobGPUs()
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(spec.Numbers, tc.wantNumbers) || !reflect.DeepEqual(spec.UUIDs, tc.wantUUIDs) || !reflect.DeepEqual(spec.Skipped, tc.wantSkipped) {
				t.Errorf("JobGPUs(%q) = numbers %v uuids %v skipped %v", tc.raw, spec.Numbers, spec.UUIDs, spec.Skipped)
			}
		})
	}
}

func TestNoGPUsMeansCheckNothing(t *testing.T) {
	for _, vars := range []map[string]string{
		{},
		{"CUDA_VISIBLE_DEVICES": "  "},
		// Slurm sets this literal when the job had no GPU device files.
		{"SLURM_JOB_GPUS": "NoDevFiles"},
	} {
		if _, err := envWith(vars).JobGPUs(); !errors.Is(err, ErrNoGPUs) {
			t.Errorf("%v: want ErrNoGPUs, got %v", vars, err)
		}
	}
}

func TestCUDAVisibleDevicesIsPreferred(t *testing.T) {
	spec, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": "1", "GPU_DEVICE_ORDINAL": "1"}).JobGPUs()
	if err != nil || spec.Var != "CUDA_VISIBLE_DEVICES" {
		t.Fatalf("spec=%+v err=%v", spec, err)
	}
}

func TestDisagreeingVariablesMeanCheckNothing(t *testing.T) {
	// Slurm writes the same GRES indices into both variables, so different
	// sets mean something else wrote one of them. Guessing could check a
	// neighbour's GPU and blame this job.
	_, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": "0", "SLURM_JOB_GPUS": "1"}).JobGPUs()
	if !errors.Is(err, ErrAmbiguous) {
		t.Fatalf("want ErrAmbiguous, got %v", err)
	}
	// Same set, different spelling, is agreement.
	if _, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": "0,1", "SLURM_JOB_GPUS": "0-1"}).JobGPUs(); err != nil {
		t.Fatalf("0,1 and 0-1 agree: %v", err)
	}
	// Two number lists that disagree are caught even when a UUID list is
	// also present and matches both in count.
	_, err = envWith(map[string]string{
		"CUDA_VISIBLE_DEVICES": "GPU-5117a000-0000-4000-8000-000000000000,GPU-5117a000-0000-4000-8000-000000000001",
		"SLURM_JOB_GPUS":       "0,1",
		"GPU_DEVICE_ORDINAL":   "0,2",
	}).JobGPUs()
	if !errors.Is(err, ErrAmbiguous) {
		t.Fatalf("SLURM_JOB_GPUS and GPU_DEVICE_ORDINAL disagree: want ErrAmbiguous, got %v", err)
	}
}

func TestEnvUUIDUsesTheUUIDsAndCountChecksTheIndices(t *testing.T) {
	// gres.conf Flags=env_uuid (Slurm 26.05+): CUDA_VISIBLE_DEVICES holds
	// UUIDs, SLURM_JOB_GPUS still holds GRES indices (gres_common.c,
	// gres_common_prep_set_env). A UUID never equals a number, so a set
	// comparison would make every job ambiguous and check nothing.
	const u0, u1 = "GPU-5117a000-0000-4000-8000-000000000000", "GPU-5117a000-0000-4000-8000-000000000001"
	spec, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": u0, "SLURM_JOB_GPUS": "0"}).JobGPUs()
	if err != nil || spec.Var != "CUDA_VISIBLE_DEVICES" || !reflect.DeepEqual(spec.UUIDs, []string{u0}) || len(spec.Numbers) != 0 {
		t.Fatalf("spec=%+v err=%v", spec, err)
	}
	// The UUID list wins even when it is not the first variable read.
	spec, err = envWith(map[string]string{"SLURM_JOB_GPUS": "3,1", "CUDA_VISIBLE_DEVICES": u1 + "," + u0}).JobGPUs()
	if err != nil || spec.Var != "CUDA_VISIBLE_DEVICES" || len(spec.UUIDs) != 2 {
		t.Fatalf("spec=%+v err=%v", spec, err)
	}
	// Different counts cannot be the same GPUs.
	if _, err := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": u0, "SLURM_JOB_GPUS": "0,1"}).JobGPUs(); !errors.Is(err, ErrAmbiguous) {
		t.Fatalf("1 UUID vs 2 indices: want ErrAmbiguous, got %v", err)
	}
}

func TestRangeExpansionIsBounded(t *testing.T) {
	spec, _ := envWith(map[string]string{"CUDA_VISIBLE_DEVICES": "0-100000"}).JobGPUs()
	if len(spec.Numbers) > maxRange {
		t.Fatalf("range expansion should be bounded, got %d entries", len(spec.Numbers))
	}
}

func TestNodeNameFallsBackToHostname(t *testing.T) {
	if got := envWith(map[string]string{}).NodeName; got != "host-from-os" {
		t.Errorf("got %q", got)
	}
	if got := envWith(map[string]string{"SLURMD_NODENAME": "gpu001"}).NodeName; got != "gpu001" {
		t.Errorf("got %q", got)
	}
}

func TestSharedGPUHint(t *testing.T) {
	get := func(m map[string]string) func(string) string { return func(k string) string { return m[k] } }
	if SharedGPUHint(get(nil)) != "" {
		t.Error("no hint expected")
	}
	if SharedGPUHint(get(map[string]string{"SLURM_SHARDS_ON_NODE": "2"})) != "SLURM_SHARDS_ON_NODE" {
		t.Error("shards hint missed")
	}
}

// ── scontrol ───────────────────────────────────────────────────────────────

func fakeScontrol(t *testing.T, exit int) (bin, log string) {
	t.Helper()
	dir := t.TempDir()
	bin, log = filepath.Join(dir, "scontrol"), filepath.Join(dir, "calls")
	// One argument per line, so the test sees argv exactly as exec passed it.
	script := "#!/bin/sh\nfor a in \"$@\"; do printf '%s\\n' \"$a\"; done > '" + log + "'\necho 'scontrol said no' >&2\nexit " + strconv.Itoa(exit) + "\n"
	if err := os.WriteFile(bin, []byte(script), 0o755); err != nil {
		t.Fatal(err)
	}
	return bin, log
}

func TestDrainCallsScontrolWithArgvNotAShell(t *testing.T) {
	bin, log := fakeScontrol(t, 0)
	err := NewSControl(bin).Drain(context.Background(), "gpu001", `epilog-gpu-validator fatal: ecc-uncorrectable:gpu3; $(reboot)`)
	if err != nil {
		t.Fatal(err)
	}
	got, _ := os.ReadFile(log)
	want := "update\nNodeName=gpu001\nState=DRAIN\nReason=epilog-gpu-validator fatal: ecc-uncorrectable:gpu3 reboot\n"
	if string(got) != want {
		t.Fatalf("argv:\n%s\nwant:\n%s", got, want)
	}
}

func TestDrainFailureCarriesScontrolOutput(t *testing.T) {
	bin, _ := fakeScontrol(t, 1)
	err := NewSControl(bin).Drain(context.Background(), "gpu001", "x")
	if err == nil || !strings.Contains(err.Error(), "scontrol said no") {
		t.Fatalf("got %v", err)
	}
}

func TestDrainRefusesAnEmptyNodeName(t *testing.T) {
	if err := NewSControl("/bin/true").Drain(context.Background(), "", "x"); err == nil {
		t.Fatal("an empty NodeName must not reach scontrol")
	}
}

func TestSanitizeReasonStripsCharactersSlurmRejects(t *testing.T) {
	got := sanitizeReason(`epilog "gpu0" ecc; rm -rf /$(x)`)
	for _, bad := range []string{`"`, ";", "$", "(", ")"} {
		if strings.Contains(got, bad) {
			t.Errorf("reason still contains %q: %q", bad, got)
		}
	}
}

func TestSanitizeReasonIsBounded(t *testing.T) {
	if got := sanitizeReason(strings.Repeat("x", 400)); len(got) > 200 {
		t.Fatalf("reason not truncated: %d chars", len(got))
	}
}

func TestSanitizeReasonCollapsesWhitespace(t *testing.T) {
	if got := sanitizeReason("a   b\n\nc"); got != "a b c" {
		t.Errorf("got %q", got)
	}
}

func TestDrainRefusesAPathLookup(t *testing.T) {
	err := NewSControl("scontrol").Drain(context.Background(), "gpu001", "x")
	if err == nil || !strings.Contains(err.Error(), "absolute") {
		t.Fatalf("a bare scontrol name must not be looked up in PATH as root: %v", err)
	}
}

func TestDrainNeverRunsAGroupWritableScontrol(t *testing.T) {
	// Anyone who can write the file would get root on the node at the first
	// fault. The check runs again right before exec, whatever the caller did.
	bin, log := fakeScontrol(t, 0)
	if err := os.Chmod(bin, 0o775); err != nil {
		t.Fatal(err)
	}
	err := NewSControl(bin).Drain(context.Background(), "gpu001", "x")
	if err == nil || !strings.Contains(err.Error(), "writable by group or others") {
		t.Fatalf("want a refusal, got %v", err)
	}
	if _, statErr := os.Stat(log); !errors.Is(statErr, os.ErrNotExist) {
		t.Fatalf("the refused scontrol was executed (call log exists: %v)", statErr)
	}
}
