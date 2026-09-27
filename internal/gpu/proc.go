package gpu

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// DefaultProcDir is where the NVIDIA kernel driver lists its GPUs, one
// directory per PCI address.
const DefaultProcDir = "/proc/driver/nvidia/gpus"

// ProcResolver maps /dev/nvidiaN minor numbers to PCI bus IDs by reading
// /proc/driver/nvidia/gpus/<domain:bus:device.function>/information.
//
// Why this exists: the number Slurm hands the Epilog is its GRES index, and
// which GPU that is depends on gres.conf. The gres guide's example (a job
// allocated /dev/nvidia1 sees CUDA_VISIBLE_DEVICES=1 in the Prolog and
// Epilog) reads it as a device-file number; the gres.conf Links note says
// "the minor number ... is not necessarily the same as the device
// number/index. The device number is created by sorting the GPUs by PCI bus
// ID". The two agree only where the minors follow PCI bus order, and the
// gres guide says the mapping between minors and NVML's numbers "is
// nondeterministic and system dependent" (https://slurm.schedmd.com/gres.html).
// This map is what lets the tool see whether they agree on this node, and
// resolve a device-file number when the site says that is what gres.conf
// uses (--gpu-numbering).
//
// The file format: NVIDIA's MIG user guide says this file "contains a
// "Device Minor" field" (https://docs.nvidia.com/datacenter/tesla/mig-user-guide/latest/device-nodes-and-capabilities.html).
// The directory name is the PCI address. Nothing else in the file is used.
type ProcResolver struct{ Dir string }

// DeviceMinors implements Resolver.
func (p ProcResolver) DeviceMinors() (map[int]string, error) {
	dir := p.Dir
	if dir == "" {
		dir = DefaultProcDir
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", dir, err)
	}
	out := map[int]string{}
	for _, e := range entries {
		bus, ok := NormalizeBusID(e.Name())
		if !ok {
			continue
		}
		minor, err := readDeviceMinor(filepath.Join(dir, e.Name(), "information"))
		if err != nil {
			return nil, err
		}
		if prev, dup := out[minor]; dup {
			// Two devices claiming one minor number: the mapping is not
			// trustworthy, so refuse it rather than pick one.
			return nil, fmt.Errorf("device minor %d claimed by both %s and %s", minor, prev, bus)
		}
		out[minor] = bus
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no GPUs listed under %s", dir)
	}
	return out, nil
}

func readDeviceMinor(path string) (int, error) {
	f, err := os.Open(path)
	if err != nil {
		return 0, err
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		k, v, ok := strings.Cut(sc.Text(), ":")
		if !ok || strings.TrimSpace(k) != "Device Minor" {
			continue
		}
		n, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || n < 0 {
			return 0, fmt.Errorf("%s: unreadable Device Minor %q", path, strings.TrimSpace(v))
		}
		return n, nil
	}
	if err := sc.Err(); err != nil {
		return 0, fmt.Errorf("%s: %w", path, err)
	}
	return 0, fmt.Errorf("%s: no Device Minor line", path)
}
