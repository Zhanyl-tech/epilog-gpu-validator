package gpu

import (
	"fmt"
	"strconv"
	"strings"
)

// Clock event ("throttle") reason bits, as reported by the
// clocks_throttle_reasons.active field. nvidia-smi describes that field as a
// "Bitmask of active clock event reasons. See nvml.h for more details." The
// values below are the nvmlClocksEventReason* / nvmlClocksThrottleReason*
// constants from nvml.h (NVML 13.4.92, NVIDIA's cuda_nvml_dev redistributable
// archive). They are a hardware ABI and have to match the header exactly.
const (
	// "Nothing is running on the GPU and the clocks are dropping to Idle
	// state." Always expected at Epilog time.
	ClockReasonGPUIdle uint64 = 0x0000000000000001
	// Deprecated in nvml.h ("No longer used").
	ClockReasonApplicationsClocksSetting uint64 = 0x0000000000000002
	// "The clocks have been optimized to ensure not to exceed currently set
	// power limits."
	ClockReasonSWPowerCap uint64 = 0x0000000000000004
	// "HW Slowdown (reducing the core clocks by a factor of 2 or more) is
	// engaged." nvml.h lists as causes high temperature, an external power
	// brake, Fast Trigger power protection, and: "May be also reported during
	// PState or clock change". The nvidia-smi manual adds: "It is active if
	// either HW Thermal Slowdown or HW Power Brake are active."
	ClockReasonHWSlowdown uint64 = 0x0000000000000008
	// Sync boost group membership; a configuration, not a fault.
	ClockReasonSyncBoost uint64 = 0x0000000000000010
	// Software thermal slowdown: clocks kept under max operating temperature.
	ClockReasonSWThermalSlowdown uint64 = 0x0000000000000020
	// "HW Thermal Slowdown (reducing the core clocks by a factor of 2 or
	// more) is engaged. This is an indicator of: temperature being too high".
	ClockReasonHWThermalSlowdown uint64 = 0x0000000000000040
	// "HW Power Brake Slowdown ... External Power Brake Assertion being
	// triggered (e.g. by the system power supply)".
	ClockReasonHWPowerBrakeSlowdown uint64 = 0x0000000000000080
	// "GPU clocks are limited by current setting of Display clocks".
	ClockReasonDisplayClockSetting uint64 = 0x0000000000000100
	// "The board limit (operating) policy is currently limiting the GPU clocks."
	ClockReasonBoardLimit uint64 = 0x0000000000000200
	// "The reliability policy is currently limiting the GPU clocks."
	ClockReasonReliability uint64 = 0x0000000000000400
)

var clockReasonNames = []struct {
	bit  uint64
	name string
}{
	{ClockReasonGPUIdle, "gpu_idle"},
	{ClockReasonApplicationsClocksSetting, "applications_clocks_setting"},
	{ClockReasonSWPowerCap, "sw_power_cap"},
	{ClockReasonHWSlowdown, "hw_slowdown"},
	{ClockReasonSyncBoost, "sync_boost"},
	{ClockReasonSWThermalSlowdown, "sw_thermal_slowdown"},
	{ClockReasonHWThermalSlowdown, "hw_thermal_slowdown"},
	{ClockReasonHWPowerBrakeSlowdown, "hw_power_brake_slowdown"},
	{ClockReasonDisplayClockSetting, "display_clock_setting"},
	{ClockReasonBoardLimit, "board_limit"},
	{ClockReasonReliability, "reliability"},
}

// ClockReasonNames spells out the bits set in mask, lowest first. Bits nvml.h
// does not define come out as hex, so nothing is silently dropped.
func ClockReasonNames(mask uint64) []string {
	var out []string
	known := uint64(0)
	for _, r := range clockReasonNames {
		known |= r.bit
		if mask&r.bit != 0 {
			out = append(out, r.name)
		}
	}
	if rest := mask &^ known; rest != 0 {
		out = append(out, fmt.Sprintf("unknown_bits_0x%x", rest))
	}
	return out
}

// parseClockReasons reads the bitmask. nvidia-smi prints it as hex
// ("0x0000000000000001"); older output and some fixtures say "Not Active".
func parseClockReasons(s string) (uint64, bool) {
	s = strings.TrimSpace(s)
	if strings.EqualFold(s, "Not Active") {
		return 0, true
	}
	hex := strings.TrimPrefix(strings.TrimPrefix(s, "0x"), "0X")
	if hex == s || hex == "" {
		// No 0x prefix: names, "[N/A]", or garbage. Not a mask we can trust.
		return 0, false
	}
	v, err := strconv.ParseUint(hex, 16, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}
