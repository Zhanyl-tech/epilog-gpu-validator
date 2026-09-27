package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/Zhanyl-tech/epilog-gpu-validator/internal/gpu"
)

// checkConfig is the install-time check: run it by hand on a GPU node, with
// the same flags the Epilog wrapper uses, before turning anything on.
//
// It is the one mode that exits non-zero for a configuration problem (78),
// because an admin at a shell or a config-management run needs to see it
// fail. If it finds itself in a Slurm job environment (SLURM_JOB_ID set) it
// exits 0 regardless: someone has put --check-config into the Epilog by
// mistake, and a non-zero exit there would drain the node.
func checkConfig(o options, d deps) int {
	problems := 0
	line := func(status, what, detail string) {
		if status == "FAIL" {
			problems++
		}
		if detail != "" {
			what += ": " + detail
		}
		fmt.Fprintf(d.stdout, "%-4s %s\n", status, what)
	}

	if err := o.validate(); err != nil {
		line("FAIL", "flags", err.Error())
	} else {
		line("ok", "flags", fmt.Sprintf("--budget %s, nvidia-smi limit %s", o.budget, o.queryLimit()))
	}

	smiOK := false
	if err := d.checkBinary(o.nvidiaSMI); err != nil {
		line("FAIL", "--nvidia-smi", err.Error())
	} else {
		smiOK = true
		line("ok", "--nvidia-smi", o.nvidiaSMI)
	}

	if err := d.checkBinary(o.scontrol); err != nil {
		if o.enforce {
			line("FAIL", "--scontrol", err.Error())
		} else {
			line("warn", "--scontrol", err.Error()+" (only needed with --enforce)")
		}
	} else {
		line("ok", "--scontrol", o.scontrol)
	}

	if o.logFile != "" {
		if w, err := d.openLogFile(o.logFile); err != nil {
			line("FAIL", "--log-file", err.Error())
		} else {
			_ = w.Close()
			line("ok", "--log-file", o.logFile+" is writable")
		}
	}
	if o.syslog {
		if s, err := d.openSyslog(); err != nil {
			line("FAIL", "--syslog", err.Error())
		} else {
			_ = s.Close()
			line("ok", "--syslog", "connected")
		}
	}

	mode := gpuNumbering(o.numbering)
	minors, mapErr := d.newResolver(o.procDir).DeviceMinors()
	switch {
	case mapErr != nil && mode.needsDeviceMap():
		// Every job whose GPU variables hold numbers would be not-checked.
		line("FAIL", "device map", fmt.Sprintf("%v; with --gpu-numbering %s, jobs given GPU numbers would not be checked", mapErr, mode))
	case mapErr != nil:
		line("warn", "device map", fmt.Sprintf("%v (not used with --gpu-numbering %s)", mapErr, mode))
	default:
		var pairs []string
		for _, m := range gpu.SortedMinors(minors) {
			pairs = append(pairs, fmt.Sprintf("/dev/nvidia%d=%s", m, minors[m]))
		}
		line("ok", "device map", strings.Join(pairs, " "))
	}

	if smiOK {
		ctx, cancel := context.WithTimeout(context.Background(), o.queryLimit())
		defer cancel()
		res, err := d.newSource(o.nvidiaSMI, o.queryLimit()).Query(ctx)
		switch {
		case err != nil:
			line("FAIL", "nvidia-smi query", err.Error())
		case len(res.GPUs) == 0:
			line("FAIL", "nvidia-smi query", "no GPU rows parsed")
		default:
			line("ok", "nvidia-smi query", fmt.Sprintf("%d GPU row(s) parsed", len(res.GPUs)))
			for _, h := range res.GPUs {
				if len(h.Unreadable) > 0 {
					line("warn", h.Label()+" "+h.PCIBusID, "unreadable, will be reported as unknown: "+strings.Join(h.Unreadable, ", "))
				}
			}
		}
		for _, m := range res.Malformed {
			line("FAIL", "nvidia-smi row", m)
		}
		if mode == numberingAuto && mapErr == nil && err == nil && len(res.GPUs) > 0 {
			// auto only checks a Slurm GPU number where both numberings
			// name the same GPU. Where they do not, say so now rather than
			// on every job.
			if dis := numberingDisagreements(minors, res.GPUs); len(dis) > 0 {
				line("FAIL", "gpu numbering", fmt.Sprintf(
					"device minors do not follow PCI bus order on this node (%s); Slurm's GPU numbers are GRES indices, so --gpu-numbering auto will not check them. Set --gpu-numbering nvml (gres.conf AutoDetect=nvml) or minor (File= entries in device order), or use gres.conf Flags=env_uuid (Slurm 26.05 and later) with --gpu-numbering uuid",
					strings.Join(dis, "; ")))
			} else {
				line("ok", "gpu numbering", "auto: device minors follow PCI bus order here, so Slurm GPU N is /dev/nvidiaN and nvidia-smi index N")
			}
		}
	}
	if mode != numberingAuto {
		line("ok", "gpu numbering", string(mode)+": "+mode.describe())
	}

	if problems == 0 {
		fmt.Fprintln(d.stdout, "config ok")
		return exitOK
	}
	if inJobEnvironment(d.getenv) {
		fmt.Fprintf(d.stdout, "%d problem(s); exiting 0 anyway because SLURM_JOB_ID is set and a non-zero exit here could drain this node\n", problems)
		return exitOK
	}
	fmt.Fprintf(d.stdout, "%d problem(s)\n", problems)
	return exitConfigInvalid
}
