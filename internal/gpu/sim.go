package gpu

import (
	"context"
	"fmt"
	"sort"
	"strings"
)

// Scenario names a synthetic condition, so every classification branch is
// reachable without a broken GPU to hand.
//
// The simulator does not build Health values directly. It prints the same
// CSV nvidia-smi would print for QueryFields, and that text goes through the
// real parser. An earlier version built Health values by hand and drifted
// from real output (throttle reasons as names instead of a hex bitmask, idle
// links at max generation); printing CSV keeps the parser under test in every
// scenario. The rows are MODELS written from NVIDIA's documentation, not
// output captured from hardware.
type Scenario string

const (
	// An idle, healthy H100 as documented: PCIe link power-managed down to
	// gen1 ("may be reduced when the GPU is not in use"), GpuIdle clock bit.
	ScenarioHealthy Scenario = "healthy"
	// x8 on an x16 link, also idle. Indistinguishable at idle from power
	// management, which is the point of the scenario.
	ScenarioPCIeDegraded Scenario = "pcie-degraded"
	ScenarioECC          Scenario = "ecc"           // volatile uncorrectable ECC
	ScenarioRemapFailure Scenario = "remap-failure" // remapping failure flag
	ScenarioRemapPending Scenario = "remap-pending" // needs a reset
	ScenarioThermal      Scenario = "thermal"       // hot, SW thermal slowdown
	// HW slowdown + HW thermal slowdown + idle = 0x49.
	ScenarioHWSlowdown   Scenario = "hw-slowdown"
	ScenarioLeakedMemory Scenario = "leaked-memory" // memory held after teardown
	// ECC and row-remap fields "[N/A]", as on GPUs with ECC off or pre-Ampere.
	ScenarioECCUnavailable Scenario = "ecc-na"
	// Persistence mode off: volatile ECC counters may have reset.
	ScenarioNoPersistence Scenario = "no-persistence"
	// nvidia-smi exits 15: "The GPU has fallen off the bus". That a lost GPU
	// makes a --query-gpu run exit 15, rather than print a row of unreadable
	// fields, is the manual's code list applied to this case, not observed.
	ScenarioOffBus Scenario = "off-bus"
	// nvidia-smi exits 9: "NVIDIA driver is not loaded". A monitoring failure.
	ScenarioNoDriver Scenario = "no-driver"
)

// Scenarios lists every scenario in a stable order.
var Scenarios = []Scenario{
	ScenarioHealthy, ScenarioPCIeDegraded, ScenarioRemapPending, ScenarioThermal,
	ScenarioECCUnavailable, ScenarioNoPersistence, ScenarioNoDriver,
	ScenarioHWSlowdown, ScenarioLeakedMemory, ScenarioECC, ScenarioRemapFailure,
	ScenarioOffBus,
}

// ValidScenario reports whether s names a scenario.
func ValidScenario(s string) bool {
	for _, sc := range Scenarios {
		if string(sc) == s {
			return true
		}
	}
	return false
}

// Sim is a simulated node: a Source and a Resolver that agree.
//
// NVML (nvidia-smi) indices follow PCI bus order, as the Slurm gres guide
// says NVML's numbers do. The fault is always on nvidia-smi gpu0, the first
// GPU in PCI order, which is Slurm's GRES index 0 when gres.conf numbers GPUs
// by PCI bus ID.
//
// By default /dev/nvidiaN minors follow the same order, so every Slurm
// numbering names the same GPU. MinorsReversed numbers the device files in
// REVERSE PCI order (/dev/nvidia0 is gpu3 on a 4-GPU node), the layout the
// gres guide warns about ("Mapping between these two is nondeterministic and
// system dependent"). There, Slurm's "0" means gpu0 or gpu3 depending on
// gres.conf, which is what --gpu-numbering exists for.
type Sim struct {
	Scenario       Scenario
	GPUs           int
	MinorsReversed bool
}

// NewSim returns a simulated node with gpus GPUs (4 if gpus <= 0), device
// minors in PCI order.
func NewSim(s Scenario, gpus int) *Sim {
	if gpus <= 0 {
		gpus = 4
	}
	return &Sim{Scenario: s, GPUs: gpus}
}

func (s *Sim) busID(index int) string {
	b, _ := NormalizeBusID(fmt.Sprintf("0000:%02x:00.0", 0x18+0x10*index))
	return b
}

func (s *Sim) minorToIndex(minor int) int {
	if s.MinorsReversed {
		return s.GPUs - 1 - minor
	}
	return minor
}

// DeviceMinors implements Resolver.
func (s *Sim) DeviceMinors() (map[int]string, error) {
	out := map[int]string{}
	for minor := 0; minor < s.GPUs; minor++ {
		out[minor] = s.busID(s.minorToIndex(minor))
	}
	return out, nil
}

// Source returns an SMISource whose runner prints this node's CSV.
func (s *Sim) Source() *SMISource {
	return &SMISource{
		Binary: "simulated-nvidia-smi",
		Run:    s.run,
		name:   "simulator/" + string(s.Scenario),
	}
}

// simExit is a fake process exit, classified like *exec.ExitError.
type simExit struct {
	code   int
	stderr string
}

func (e simExit) Error() string      { return fmt.Sprintf("exit status %d", e.code) }
func (e simExit) ExitCode() int      { return e.code }
func (e simExit) StderrText() string { return e.stderr }

func (s *Sim) run(_ context.Context, _ string, _ ...string) ([]byte, error) {
	const faulty = 0 // nvidia-smi gpu0: first in PCI bus order
	switch s.Scenario {
	case ScenarioOffBus:
		return nil, simExit{15, fmt.Sprintf("Unable to determine the device handle for GPU %s: GPU is lost (simulated)", s.busID(faulty))}
	case ScenarioNoDriver:
		return nil, simExit{9, "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver (simulated)"}
	}

	var b strings.Builder
	for i := 0; i < s.GPUs; i++ {
		r := idleRow(i, s.busID(i))
		if i == faulty {
			s.apply(r)
		}
		b.WriteString(r.csv())
		b.WriteByte('\n')
	}
	return []byte(b.String()), nil
}

type simRow map[string]string

func idleRow(index int, bus string) simRow {
	return simRow{
		FieldIndex:              fmt.Sprint(index),
		FieldUUID:               fmt.Sprintf("GPU-5117a000-0000-4000-8000-%012x", index),
		FieldName:               "NVIDIA H100 80GB HBM3",
		FieldBusID:              bus,
		FieldPCIeWidthCurrent:   "16",
		FieldPCIeWidthMax:       "16",
		FieldPCIeGenCurrent:     "1",
		FieldPCIeGenMax:         "5",
		FieldECCUncorrVolatile:  "0",
		FieldECCUncorrAggregate: "0",
		FieldECCCorrVolatile:    "0",
		FieldRemapPending:       "No",
		FieldRemapUncorrectable: "0",
		FieldRemapFailure:       "No",
		FieldClockReasons:       "0x0000000000000001",
		FieldTemperature:        "34",
		FieldMemUsed:            "1",
		FieldMemTotal:           "81559",
		FieldPersistence:        "Enabled",
	}
}

func (r simRow) csv() string {
	cols := make([]string, len(QueryFields))
	for i, f := range QueryFields {
		cols[i] = r[f]
	}
	return strings.Join(cols, ", ")
}

func (s *Sim) apply(r simRow) {
	switch s.Scenario {
	case ScenarioPCIeDegraded:
		r[FieldPCIeWidthCurrent] = "8"
	case ScenarioECC:
		r[FieldECCUncorrVolatile], r[FieldECCUncorrAggregate] = "3", "17"
	case ScenarioRemapFailure:
		r[FieldRemapFailure], r[FieldRemapUncorrectable] = "Yes", "9"
	case ScenarioRemapPending:
		r[FieldRemapPending], r[FieldRemapUncorrectable] = "Yes", "1"
	case ScenarioThermal:
		r[FieldTemperature] = "88"
		r[FieldClockReasons] = "0x0000000000000021" // SW thermal + idle
	case ScenarioHWSlowdown:
		r[FieldTemperature] = "94"
		r[FieldClockReasons] = "0x0000000000000049" // HW slowdown + HW thermal + idle
	case ScenarioLeakedMemory:
		r[FieldMemUsed] = "40000"
	case ScenarioECCUnavailable:
		for _, f := range []string{
			FieldECCUncorrVolatile, FieldECCUncorrAggregate, FieldECCCorrVolatile,
			FieldRemapPending, FieldRemapUncorrectable, FieldRemapFailure,
		} {
			r[f] = "[N/A]"
		}
	case ScenarioNoPersistence:
		r[FieldPersistence] = "Disabled"
	}
}

// SortedMinors is a helper for callers that print the simulated node.
func SortedMinors(m map[int]string) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}
