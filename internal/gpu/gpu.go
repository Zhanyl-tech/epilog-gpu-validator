// Package gpu reads the health signals an Epilog check needs.
//
// Deliberately narrow. Epilog runs on every job completion and Slurm kills it
// at EpilogMsgTime/EpilogTimeout, so this collects the cheap, high-signal
// fields in one nvidia-smi invocation rather than shelling out repeatedly.
package gpu

import (
	"context"
	"encoding/csv"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

// Health is one GPU's state at Epilog time.
type Health struct {
	Index int
	UUID  string
	Name  string

	// PCIe link, current versus the maximum this device supports. A card
	// running x8 on an x16 slot is the classic silent degradation: everything
	// works, everything is half speed, and no job ever reports an error.
	PCIeWidthCurrent int
	PCIeWidthMax     int
	PCIeGenCurrent   int
	PCIeGenMax       int

	// ECC. Uncorrectable (volatile) means memory corruption already happened.
	ECCUncorrectableVolatile  int64
	ECCUncorrectableAggregate int64
	ECCCorrectableVolatile    int64

	// Row remapping replaced retired pages on Ampere and later. A pending
	// remap needs a reset; a failure means the device is out of spare rows.
	RemappedRowsPending      int64
	RemappedRowsUncorrectable int64
	RemappedRowsFailure      bool

	// Throttling. Thermal and power are usually transient; HW slowdown and
	// especially HW thermal slowdown are not.
	ThrottleReasons []string

	TemperatureC int
	MemUsedMiB   int64
	MemTotalMiB  int64
	PersistenceM bool

	// Present is false when the device could not be queried at all — it fell
	// off the bus, or the driver is wedged.
	Present bool
}

// Source supplies per-GPU health.
type Source interface {
	// Query returns health for the given GPU indices. An empty slice means all
	// visible devices.
	Query(ctx context.Context, indices []int) ([]Health, error)
	Name() string
}

// ── nvidia-smi ─────────────────────────────────────────────────────────────

type SMISource struct {
	Binary string
}

func NewSMISource() *SMISource { return &SMISource{Binary: "nvidia-smi"} }

func (s *SMISource) Name() string { return "nvidia-smi" }

// One query for everything. Each extra nvidia-smi invocation costs ~100-300ms
// on a loaded node, and the Epilog budget is measured in seconds.
const smiFields = "index,uuid,name," +
	"pcie.link.width.current,pcie.link.width.max," +
	"pcie.link.gen.current,pcie.link.gen.max," +
	"ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.aggregate.total," +
	"ecc.errors.corrected.volatile.total," +
	"remapped_rows.pending,remapped_rows.uncorrectable,remapped_rows.failure," +
	"clocks_throttle_reasons.active,temperature.gpu," +
	"memory.used,memory.total,persistence_mode"

func (s *SMISource) Query(ctx context.Context, indices []int) ([]Health, error) {
	args := []string{"--query-gpu=" + smiFields, "--format=csv,noheader,nounits"}
	if len(indices) > 0 {
		strs := make([]string, len(indices))
		for i, n := range indices {
			strs[i] = strconv.Itoa(n)
		}
		args = append(args, "-i", strings.Join(strs, ","))
	}

	out, err := exec.CommandContext(ctx, s.Binary, args...).Output()
	if err != nil {
		return nil, fmt.Errorf("nvidia-smi: %w", err)
	}

	r := csv.NewReader(strings.NewReader(string(out)))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	rows, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parse nvidia-smi output: %w", err)
	}

	var out2 []Health
	for _, row := range rows {
		if len(row) < 18 {
			continue
		}
		out2 = append(out2, Health{
			Present:                   true,
			Index:                     atoi(row[0]),
			UUID:                      strings.TrimSpace(row[1]),
			Name:                      strings.TrimSpace(row[2]),
			PCIeWidthCurrent:          atoi(row[3]),
			PCIeWidthMax:              atoi(row[4]),
			PCIeGenCurrent:            atoi(row[5]),
			PCIeGenMax:                atoi(row[6]),
			ECCUncorrectableVolatile:  atoi64(row[7]),
			ECCUncorrectableAggregate: atoi64(row[8]),
			ECCCorrectableVolatile:    atoi64(row[9]),
			RemappedRowsPending:       atoi64(row[10]),
			RemappedRowsUncorrectable: atoi64(row[11]),
			RemappedRowsFailure:       parseYesNo(row[12]),
			ThrottleReasons:           parseThrottle(row[13]),
			TemperatureC:              atoi(row[14]),
			MemUsedMiB:                atoi64(row[15]),
			MemTotalMiB:               atoi64(row[16]),
			PersistenceM:              strings.EqualFold(strings.TrimSpace(row[17]), "Enabled"),
		})
	}
	return out2, nil
}

// ── Simulator ──────────────────────────────────────────────────────────────

// Scenario names a synthetic fault, so every classification branch is
// reachable without a broken GPU to hand.
type Scenario string

const (
	ScenarioHealthy       Scenario = "healthy"
	ScenarioPCIeDegraded  Scenario = "pcie-degraded"  // x8 on an x16 slot
	ScenarioECCUncorrect  Scenario = "ecc"            // uncorrectable ECC
	ScenarioRemapFailure  Scenario = "remap-failure"  // out of spare rows
	ScenarioRemapPending  Scenario = "remap-pending"  // needs a reset
	ScenarioThermal       Scenario = "thermal"        // hot, throttling
	ScenarioHWSlowdown    Scenario = "hw-slowdown"    // hardware slowdown
	ScenarioFellOffBus    Scenario = "missing"        // device not queryable
)

type SimSource struct {
	Scenario Scenario
	GPUs     int
}

func NewSimSource(s Scenario, gpus int) *SimSource {
	if gpus <= 0 {
		gpus = 4
	}
	return &SimSource{Scenario: s, GPUs: gpus}
}

func (s *SimSource) Name() string { return "simulator/" + string(s.Scenario) }

func (s *SimSource) Query(_ context.Context, indices []int) ([]Health, error) {
	if s.Scenario == ScenarioFellOffBus {
		// The device is gone. nvidia-smi would fail outright.
		return nil, fmt.Errorf("nvidia-smi: no devices were found")
	}

	want := indices
	if len(want) == 0 {
		want = make([]int, s.GPUs)
		for i := range want {
			want[i] = i
		}
	}

	out := make([]Health, 0, len(want))
	for _, i := range want {
		h := Health{
			Present: true, Index: i,
			UUID: fmt.Sprintf("GPU-sim%08d", i), Name: "NVIDIA H100 80GB HBM3",
			PCIeWidthCurrent: 16, PCIeWidthMax: 16,
			PCIeGenCurrent: 5, PCIeGenMax: 5,
			TemperatureC: 41, MemUsedMiB: 0, MemTotalMiB: 81559,
			PersistenceM: true,
		}

		// Only the first GPU is faulted — a real degradation is rarely
		// fleet-wide, and this exercises the "one bad card" path.
		if i == want[0] {
			switch s.Scenario {
			case ScenarioPCIeDegraded:
				h.PCIeWidthCurrent, h.PCIeGenCurrent = 8, 4
			case ScenarioECCUncorrect:
				h.ECCUncorrectableVolatile, h.ECCUncorrectableAggregate = 3, 17
			case ScenarioRemapFailure:
				h.RemappedRowsFailure, h.RemappedRowsUncorrectable = true, 9
			case ScenarioRemapPending:
				h.RemappedRowsPending = 4
			case ScenarioThermal:
				h.TemperatureC = 88
				h.ThrottleReasons = []string{"sw_thermal_slowdown"}
			case ScenarioHWSlowdown:
				h.TemperatureC = 94
				h.ThrottleReasons = []string{"hw_thermal_slowdown", "hw_slowdown"}
			}
		}
		out = append(out, h)
	}
	return out, nil
}

// ── parsing ────────────────────────────────────────────────────────────────

func atoi(s string) int {
	n, _ := strconv.Atoi(clean(s))
	return n
}

func atoi64(s string) int64 {
	n, _ := strconv.ParseInt(clean(s), 10, 64)
	return n
}

func clean(s string) string {
	s = strings.TrimSpace(s)
	// nvidia-smi reports unsupported fields as "[N/A]" or "[Not Supported]".
	if strings.HasPrefix(s, "[") {
		return ""
	}
	return s
}

func parseYesNo(s string) bool {
	return strings.EqualFold(clean(s), "yes")
}

func parseThrottle(s string) []string {
	s = clean(s)
	if s == "" || strings.EqualFold(s, "Not Active") || s == "0x0000000000000000" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(s, ",") {
		if p := strings.TrimSpace(part); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// QueryTimeout is the per-invocation cap. Slurm kills a slow Epilog and can
// mark the node down for it, so hanging is worse than not checking.
const QueryTimeout = 10 * time.Second
