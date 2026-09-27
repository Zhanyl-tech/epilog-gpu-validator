// Package gpu reads the health signals an Epilog check needs.
//
// Deliberately narrow. Epilog runs on every job completion, and slurm.conf
// says "If the Epilog or slurm_spank_job_epilog time out, the node is
// drained" (EpilogTimeout, https://slurm.schedmd.com/slurm.conf.html). So
// this collects the cheap, high-signal fields in one nvidia-smi invocation
// rather than shelling out repeatedly.
//
// Nothing in this package decides severity. It reports what nvidia-smi said,
// including what it could NOT say (Health.Unreadable, QueryError), and leaves
// the judgement to package checks.
package gpu

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// Query fields, in the order they are requested and parsed. The names are
// nvidia-smi --query-gpu field names; see `nvidia-smi --help-query-gpu`.
const (
	FieldIndex              = "index"
	FieldUUID               = "uuid"
	FieldName               = "name"
	FieldBusID              = "pci.bus_id"
	FieldPCIeWidthCurrent   = "pcie.link.width.current"
	FieldPCIeWidthMax       = "pcie.link.width.max"
	FieldPCIeGenCurrent     = "pcie.link.gen.current"
	FieldPCIeGenMax         = "pcie.link.gen.max"
	FieldECCUncorrVolatile  = "ecc.errors.uncorrected.volatile.total"
	FieldECCUncorrAggregate = "ecc.errors.uncorrected.aggregate.total"
	FieldECCCorrVolatile    = "ecc.errors.corrected.volatile.total"
	FieldRemapPending       = "remapped_rows.pending"
	FieldRemapUncorrectable = "remapped_rows.uncorrectable"
	FieldRemapFailure       = "remapped_rows.failure"
	FieldClockReasons       = "clocks_throttle_reasons.active"
	FieldTemperature        = "temperature.gpu"
	FieldMemUsed            = "memory.used"
	FieldMemTotal           = "memory.total"
	FieldPersistence        = "persistence_mode"
)

// QueryFields is the exact --query-gpu list, in column order.
//
// Not verified against a real driver in this repo: that every one of these is
// accepted by --query-gpu on every driver branch (the remapped_rows.* fields
// in particular are documented under --query-remapped-rows). If nvidia-smi
// rejects a field it exits 2, which is classified as a monitoring failure
// (nothing checked, exit 0), and `--check-config` reports it at install time.
var QueryFields = []string{
	FieldIndex, FieldUUID, FieldName, FieldBusID,
	FieldPCIeWidthCurrent, FieldPCIeWidthMax, FieldPCIeGenCurrent, FieldPCIeGenMax,
	FieldECCUncorrVolatile, FieldECCUncorrAggregate, FieldECCCorrVolatile,
	FieldRemapPending, FieldRemapUncorrectable, FieldRemapFailure,
	FieldClockReasons, FieldTemperature,
	FieldMemUsed, FieldMemTotal, FieldPersistence,
}

// Health is one GPU's state at Epilog time.
type Health struct {
	// Index is the NVML index nvidia-smi printed. It is NOT necessarily the
	// number Slurm hands the Epilog, which is Slurm's own GRES index; which
	// GPU that names depends on gres.conf (see --gpu-numbering). Nor is it
	// necessarily the /dev/nvidiaN minor: the Slurm gres guide says that
	// mapping "is nondeterministic and system dependent"
	// (https://slurm.schedmd.com/gres.html). -1 when unknown.
	Index int
	UUID  string
	Name  string
	// PCIBusID is in canonical form (see NormalizeBusID).
	PCIBusID string
	// SlurmID is the Epilog-environment entry that selected this GPU ("1",
	// "GPU-..."). Empty with --all-gpus.
	SlurmID string

	// PCIe link, current versus the maximum this device supports. NVIDIA's
	// manual says the current values "may be reduced when the GPU is not in
	// use", and at Epilog time the GPU is by construction not in use, so an
	// idle reading below max is not evidence of a downtrained link.
	PCIeWidthCurrent int
	PCIeWidthMax     int
	PCIeGenCurrent   int
	PCIeGenMax       int

	// ECC. nvidia-smi: "Volatile error counters track the number of errors
	// detected since the last driver load", and "On Linux the driver unloads
	// when no active clients exist" unless persistence mode is enabled.
	ECCUncorrectableVolatile  int64
	ECCUncorrectableAggregate int64
	ECCCorrectableVolatile    int64

	// Row remapping (Ampere and later). A pending remap needs a GPU reset; a
	// failure means a remap could not be applied.
	RemappedRowsPending       int64
	RemappedRowsUncorrectable int64
	RemappedRowsFailure       bool

	// ClockEventReasons is the clocks_throttle_reasons.active bitmask. Bit
	// meanings are the nvml.h constants in clock.go.
	ClockEventReasons uint64

	TemperatureC    int
	MemUsedMiB      int64
	MemTotalMiB     int64
	PersistenceMode bool

	// Unreadable names the query fields whose value could not be read:
	// "[N/A]", "[Not Supported]", or not a number. Their Go fields above are
	// zero, and zero must NOT be read as healthy; checks skip them and emit an
	// Unknown finding saying what was not checked.
	Unreadable []string
}

// Readable reports whether a field was read successfully.
func (h Health) Readable(field string) bool {
	for _, f := range h.Unreadable {
		if f == field {
			return false
		}
	}
	return true
}

// Label is the short name used in findings and the drain reason: "gpu3" is
// NVML index 3, the number `nvidia-smi -i 3` takes.
func (h Health) Label() string {
	if h.Index >= 0 {
		return fmt.Sprintf("gpu%d", h.Index)
	}
	if h.SlurmID != "" {
		return "slurm-gpu" + h.SlurmID
	}
	return "gpu?"
}

// Result is what one query returned.
type Result struct {
	GPUs []Health
	// Malformed holds rows that could not be parsed at all (wrong column
	// count, unreadable identity). They are reported, never guessed at.
	Malformed []string
}

// Source supplies per-GPU health for every GPU on the node. Selecting the
// job's GPUs happens afterwards (see Match): `nvidia-smi -i` is documented as
// taking "a single specified GPU", so a comma-separated list is not relied on.
type Source interface {
	Query(ctx context.Context) (Result, error)
	Name() string
}

// ── identity ───────────────────────────────────────────────────────────────

// Target is one GPU the finished job held.
type Target struct {
	// SlurmID is the entry as the Epilog environment gave it.
	SlurmID string
	// BusID (canonical) or UUID identifies the device to nvidia-smi. Exactly
	// one is set.
	BusID string
	UUID  string
}

func (t Target) String() string {
	if t.BusID != "" {
		return fmt.Sprintf("slurm-gpu%s(%s)", t.SlurmID, t.BusID)
	}
	return t.UUID
}

// The separator before the function number is normally "." ("0000:3b:00.0");
// ":" is accepted too because NVIDIA's MIG guide writes the /proc path as
// "domain:bus:device:function".
var busIDRe = regexp.MustCompile(`^([0-9A-Fa-f]{1,8}):([0-9A-Fa-f]{1,2}):([0-9A-Fa-f]{1,2})[.:]([0-7])$`)

// NormalizeBusID turns a PCI address into one canonical form so that
// /proc/driver/nvidia/gpus directory names ("0000:3b:00.0") and nvidia-smi's
// pci.bus_id ("00000000:3B:00.0") compare equal.
func NormalizeBusID(s string) (string, bool) {
	m := busIDRe.FindStringSubmatch(strings.TrimSpace(s))
	if m == nil {
		return "", false
	}
	var n [4]uint64
	for i := range n {
		v, err := strconv.ParseUint(m[i+1], 16, 32)
		if err != nil {
			return "", false
		}
		n[i] = v
	}
	return fmt.Sprintf("%08X:%02X:%02X.%X", n[0], n[1], n[2], n[3]), true
}

var uuidRe = regexp.MustCompile(`^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$`)

// IsGPUUUID reports whether s has the shape of an NVIDIA GPU UUID. MIG
// device UUIDs ("MIG-...") deliberately do not match: MIG is not supported.
func IsGPUUUID(s string) bool { return uuidRe.MatchString(s) }

// Match picks the job's GPUs out of a whole-node query. Every target comes
// back either matched (with SlurmID filled in) or in missing; rows for GPUs
// the job did not hold are dropped.
func Match(targets []Target, gpus []Health) (matched []Health, missing []Target) {
	for _, t := range targets {
		found := false
		for _, h := range gpus {
			if (t.BusID != "" && t.BusID == h.PCIBusID) ||
				(t.UUID != "" && strings.EqualFold(t.UUID, h.UUID)) {
				h.SlurmID = t.SlurmID
				matched = append(matched, h)
				found = true
				break
			}
		}
		if !found {
			missing = append(missing, t)
		}
	}
	return matched, missing
}

// Resolver lists the node's device files by PCI bus ID. It is used to tell
// whether device-file numbering and PCI bus order agree on this node, and to
// resolve Slurm numbers under --gpu-numbering=minor.
type Resolver interface {
	// DeviceMinors returns minor number (the N in /dev/nvidiaN) → canonical
	// PCI bus ID for every NVIDIA GPU the driver knows about.
	DeviceMinors() (map[int]string, error)
}
