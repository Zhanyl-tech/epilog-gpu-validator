package main

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/checks"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/slurm"
)

// gpuNumbering is what a number in the Epilog's GPU variables means.
//
// Slurm writes its own GRES device index there. gres_common_prep_set_env()
// (src/plugins/gres/common/gres_common.c) formats gres_device->index into
// CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS for the Prolog and Epilog; it never
// writes the device-file number. Read at SchedMD/slurm master 9f9da53 and at
// tags slurm-23-02-7-1, slurm-24-05-8-1 and slurm-25-05-3-1.
//
// Which GPU index N is depends on gres.conf, and the docs point two ways:
//
//   - gres.conf, under Links: "the minor number assigned by the OS and used in
//     the device file (i.e. the X in /dev/nvidiaX) is not necessarily the same
//     as the device number/index. The device number is created by sorting the
//     GPUs by PCI bus ID and then numbering them starting from the smallest bus
//     ID" (https://slurm.schedmd.com/gres.conf.html). In the source, with
//     AutoDetect Slurm sorts GPUs by device-file name and then by Links, which
//     AutoDetect=nvml fills in from NVML ("a stand-in for PCI bus ID order",
//     src/plugins/gres/gpu/gres_gpu.c). NVML's own numbers "are assigned via
//     PCI bus ID, from lowest to highest" (https://slurm.schedmd.com/gres.html).
//     Then index N is the GPU nvidia-smi calls index N: numberingNVML.
//   - The gres guide's example is a job allocated /dev/nvidia1 that sees
//     CUDA_VISIBLE_DEVICES=1 in the Prolog and Epilog, and it asks for gres.conf
//     File= entries "in the increasing numeric order". With AutoDetect off, the
//     source numbers GPUs in the order of those File= entries. Then index N is
//     /dev/nvidiaN: numberingMinor.
//
// The two agree whenever the node's device minors follow PCI bus order. The
// gres guide says the mapping between the two "is nondeterministic and
// system dependent", so where they differ the tool cannot tell which GPU a
// number means without being told.
//
// Not covered by either: a gres.conf without AutoDetect that lists File=
// entries out of device order, or one that leaves some of the node's GPUs
// out. Under those, no mode here is right; use UUIDs.
type gpuNumbering string

const (
	// numberingAuto checks number N only when /dev/nvidiaN and nvidia-smi
	// index N are the same GPU on this node, so both readings above agree.
	numberingAuto gpuNumbering = "auto"
	// numberingNVML: N is nvidia-smi index N (PCI bus order).
	numberingNVML gpuNumbering = "nvml"
	// numberingMinor: N is /dev/nvidiaN, via the /proc device map.
	numberingMinor gpuNumbering = "minor"
	// numberingUUID: numbers are never mapped; only GPU UUIDs (gres.conf
	// Flags=env_uuid, Slurm 26.05 and later) are checked.
	numberingUUID gpuNumbering = "uuid"
)

var numberings = []gpuNumbering{numberingAuto, numberingNVML, numberingMinor, numberingUUID}

func numberingList() string {
	names := make([]string, len(numberings))
	for i, n := range numberings {
		names[i] = string(n)
	}
	return strings.Join(names, "|")
}

func validNumbering(s string) bool {
	for _, n := range numberings {
		if string(n) == s {
			return true
		}
	}
	return false
}

// needsDeviceMap reports whether a mode reads /proc/driver/nvidia/gpus.
func (n gpuNumbering) needsDeviceMap() bool {
	return n == numberingAuto || n == numberingMinor
}

func (n gpuNumbering) describe() string {
	switch n {
	case numberingNVML:
		return "Slurm GPU N is nvidia-smi index N (PCI bus order; gres.conf AutoDetect=nvml)"
	case numberingMinor:
		return "Slurm GPU N is /dev/nvidiaN (gres.conf File= entries in device order)"
	case numberingUUID:
		return "only GPU UUIDs are checked (gres.conf Flags=env_uuid, Slurm 26.05 and later)"
	}
	return "Slurm GPU N is checked only where /dev/nvidiaN and nvidia-smi index N are the same GPU"
}

// Check names for the job's GPUs that were not checked. Any of these on a run
// that checked at least one GPU makes it partially-checked, not ok.
const (
	checkUnsupportedID   = "gpu-id-unsupported"
	checkDeviceMapNone   = "device-map-unavailable"
	checkDeviceMapNoGPU  = "device-map-missing"
	checkNumberAmbiguous = "gpu-number-ambiguous"
	checkNumberNotUsed   = "gpu-number-not-used"
	checkNotInOutput     = "not-in-output"
	checkMalformedRow    = "malformed-row"
)

func slurmLabel(n int) string { return "slurm-gpu" + strconv.Itoa(n) }

func targetLabel(t gpu.Target) string {
	if t.BusID != "" {
		return "slurm-gpu" + t.SlurmID
	}
	return t.UUID
}

// resolveTargets turns the job's GPU list into devices that can be matched
// against nvidia-smi rows (by bus ID or UUID). rows is the node's parsed
// nvidia-smi output: nvidia-smi's own index is one of the two numberings.
// Anything that cannot be resolved without guessing is an Unknown finding.
func resolveTargets(spec slurm.GPUSpec, mode gpuNumbering, res gpu.Resolver, rows []gpu.Health) ([]gpu.Target, []checks.Finding) {
	var targets []gpu.Target
	var unchecked []checks.Finding
	seen := map[string]bool{}
	addTarget := func(t gpu.Target) {
		key := t.BusID + "|" + strings.ToLower(t.UUID)
		if !seen[key] {
			seen[key] = true
			targets = append(targets, t)
		}
	}
	notChecked := func(label, check, detail string) {
		unchecked = append(unchecked, checks.NotChecked(label, check, detail))
	}

	for _, s := range spec.Skipped {
		notChecked("", checkUnsupportedID,
			fmt.Sprintf("%s entry %q is neither a GPU number nor a GPU UUID (MIG is not supported); not checked", spec.Var, s))
	}
	for _, u := range spec.UUIDs {
		addTarget(gpu.Target{SlurmID: u, UUID: u})
	}
	if len(spec.Numbers) == 0 {
		return targets, unchecked
	}

	if mode == numberingUUID {
		for _, n := range spec.Numbers {
			notChecked(slurmLabel(n), checkNumberNotUsed,
				fmt.Sprintf("%s entry %d is a Slurm GRES index; with --gpu-numbering uuid only GPU UUIDs are checked; not checked", spec.Var, n))
		}
		return targets, unchecked
	}

	byIndex := map[int]string{}
	for _, h := range rows {
		byIndex[h.Index] = h.PCIBusID
	}
	if mode == numberingNVML {
		for _, n := range spec.Numbers {
			bus, ok := byIndex[n]
			if !ok {
				notChecked(slurmLabel(n), checkNotInOutput,
					fmt.Sprintf("nvidia-smi listed no GPU with index %d (Slurm GPU %d under --gpu-numbering nvml); not checked", n, n))
				continue
			}
			addTarget(gpu.Target{SlurmID: strconv.Itoa(n), BusID: bus})
		}
		return targets, unchecked
	}

	minors, err := res.DeviceMinors()
	if err != nil {
		notChecked("", checkDeviceMapNone, fmt.Sprintf(
			"cannot read the device map (%v), which --gpu-numbering %s needs; Slurm GPU numbers not checked rather than guessed", err, mode))
		return targets, unchecked
	}
	for _, n := range spec.Numbers {
		byMinor, okMinor := minors[n]
		if mode == numberingMinor {
			if !okMinor {
				notChecked(slurmLabel(n), checkDeviceMapNoGPU,
					fmt.Sprintf("no NVIDIA GPU with device minor %d is listed by the driver; not checked", n))
				continue
			}
			addTarget(gpu.Target{SlurmID: strconv.Itoa(n), BusID: byMinor})
			continue
		}
		byNVML, okNVML := byIndex[n]
		switch {
		case okMinor && okNVML && byMinor == byNVML:
			addTarget(gpu.Target{SlurmID: strconv.Itoa(n), BusID: byMinor})
		case !okNVML:
			// Not evidence either way, and without the row the two
			// numberings cannot be compared.
			notChecked(slurmLabel(n), checkNotInOutput,
				fmt.Sprintf("nvidia-smi listed no GPU with index %d, so Slurm GPU %d cannot be confirmed; not checked", n, n))
		default:
			notChecked(slurmLabel(n), checkNumberAmbiguous, fmt.Sprintf(
				"Slurm GPU %d is a GRES index: /dev/nvidia%d is %s but nvidia-smi index %d is %s, and which one Slurm means depends on gres.conf; not checked. Set --gpu-numbering to match gres.conf, or use gres.conf Flags=env_uuid",
				n, n, orNone(byMinor), n, orNone(byNVML)))
		}
	}
	return targets, unchecked
}

func orNone(bus string) string {
	if bus == "" {
		return "absent"
	}
	return bus
}

// numberingDisagreements lists the GPUs whose device minor and nvidia-smi
// index differ, for --check-config.
func numberingDisagreements(minors map[int]string, rows []gpu.Health) []string {
	var out []string
	listed := map[int]bool{}
	for _, h := range rows {
		listed[h.Index] = true
		if bus, ok := minors[h.Index]; !ok || bus != h.PCIBusID {
			out = append(out, fmt.Sprintf("nvidia-smi index %d is %s, /dev/nvidia%d is %s", h.Index, h.PCIBusID, h.Index, orNone(bus)))
		}
	}
	for _, m := range gpu.SortedMinors(minors) {
		if !listed[m] {
			out = append(out, fmt.Sprintf("/dev/nvidia%d is %s, nvidia-smi lists no index %d", m, minors[m], m))
		}
	}
	return out
}
