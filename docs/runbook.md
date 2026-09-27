# Runbook: a node this tool drained (or would have)

For whoever runs `sinfo -R` and sees `epilog-gpu-validator ...` as the reason.

This tool never un-drains. Bringing a node back is a human decision, and this
page is meant to help with it. It has **not** been rehearsed on real hardware.
The commands are the documented ones, cited, but the order of steps is a
suggestion, not a tested procedure. Your site's hardware and vendor process
come first.

## 1. Read the record

The drain reason looks like this:

```
epilog-gpu-validator fatal: ecc-uncorrectable:gpu0,gpu3 row-remap-failure:gpu1
```

- The word after the tool name is the worst severity (`fatal` or `degraded`).
- Each `check:gpuN,...` group names a check and every GPU that failed it.
- `gpuN` is the **NVML index**, the number `nvidia-smi -i N` takes. It is not
  necessarily Slurm's number for the GPU (its GRES index), nor the `N` in
  `/dev/nvidiaN`. `node` means the finding is not tied to one GPU (an
  `nvidia-smi` exit code).
- The reason is capped at 200 characters and ends in `...` when cut. The full
  list is in the log.

The full JSON record for the job is in the log file (the wrapper uses
`/var/log/epilog-gpu-validator.jsonl`). Syslog, under the tag
`epilog-gpu-validator`, has only a one-line summary of each run.

```bash
grep '"job_id":"<JOBID>"' /var/log/epilog-gpu-validator.jsonl
```

`gpus_checked` in the record gives each GPU's index, UUID, PCI bus ID and
Slurm GPU number. Use the UUID or bus ID in the commands below: NVIDIA
recommends them because "device enumeration ordering is not guaranteed to be
consistent between reboots" ([nvidia-smi manual][smi]).

## 2. Confirm, per check

`-d` limits `nvidia-smi -q` to one section ([nvidia-smi manual][smi]).

| Check | What it means | Confirm with | Usual next step |
| --- | --- | --- | --- |
| `nvidia-smi-exit-15` | "The GPU has fallen off the bus or has otherwise become inaccessible" | `nvidia-smi` (which GPU is missing or erroring); kernel log for the PCI address | Reboot or power-cycle the node; if it recurs, reseat or RMA |
| `nvidia-smi-exit-14` | "infoROM is corrupted" | `nvidia-smi -q -i <uuid>` | Vendor support |
| `nvidia-smi-exit-10` | "NVIDIA Kernel detected an interrupt issue with a GPU" | kernel log | Reboot; if it recurs, hardware investigation |
| `nvidia-smi-exit-8` | "A device's external power cables are not properly attached" | physical inspection | Fix cabling |
| `ecc-uncorrectable` | Uncorrectable ECC errors since the last driver load. The last job's results are suspect | `nvidia-smi -q -d ECC -i <uuid>` (volatile section) | Tell the job owner. Check `ROW_REMAPPER` below; a reset may apply a pending remap |
| `row-remap-failure` | A row remap failed. NVIDIA's [RMA policy][rma]: "the RMA criteria is met when the row-remapping failure flag is set and validated by the field diagnostic" | `nvidia-smi -q -d ROW_REMAPPER -i <uuid>` ("Remapping Failure Occurred") | Run NVIDIA's field diagnostic ("The NVIDIA Field Diagnostic tool determines whether a GPU qualifies for RMA"); RMA if it validates the flag |
| `row-remap-pending` (only drains with `--drain-on-pending-remap`) | "The GPU must be reset for the remapping to go into effect" | `nvidia-smi -q -d ROW_REMAPPER -i <uuid>` ("Pending") | GPU reset (below), then re-check |
| `throttle` (degraded) | HW thermal slowdown or HW power brake engaged with the GPU idle | `nvidia-smi -q -i <uuid>` ("Clocks Event Reasons"), `nvidia-smi -q -d TEMPERATURE -i <uuid>` | Check airflow, fans, heatsink, and the power-brake signal from the chassis |
| `ecc-correctable` | Correctable errors since driver load above the site threshold | `nvidia-smi -q -d ECC -i <uuid>` | Watch the rate. The counter only grows until the driver reloads, so a long-lived node can cross a fixed threshold slowly |
| `leaked-memory` | Memory still allocated after the job | `nvidia-smi -q -d PIDS,MEMORY -i <uuid>` | Find and kill the surviving process. If the GPU is shared (MPS, shards), set `--shared-gpus` |
| `pcie-width` (only drains with `--drain-on-pcie-width`) | Link narrower than max at idle | `nvidia-smi -q -i <uuid>` (PCI section: link generation and width), under load as well as idle | Reseat the card or riser. Remember that NVIDIA documents width "may be reduced when the GPU is not in use" |

## 3. GPU reset, if the check calls for one

From the [nvidia-smi manual][smi], `nvidia-smi -r` "Requires root", and
"There can't be any applications using these devices (e.g. CUDA application,
graphics application like X server, monitoring application like other
instance of nvidia-smi)". Drain the node and wait for running jobs to finish
first.

```bash
nvidia-smi -r -i <uuid>
nvidia-smi -q -d ROW_REMAPPER,ECC -i <uuid>     # re-check
```

## 4. Resume

Only once the finding is understood and cleared:

```bash
scontrol update NodeName=<node> State=RESUME
```

If the fault comes back after resume, the next job's Epilog will drain the
node again with a fresh reason. That is by design.

## Findings that never drain

`unknown` and `transient` findings do not drain. They are in the log so the
report-only week has something to read. Some are worth acting on anyway:

- `ecc-volatile-unreliable`: persistence mode is off, so this tool cannot
  see ECC errors from the job that just ended. `nvidia-smi -pm 1` enables it.
  The manual notes that it "does not persist across reboots", so set it where
  the node boots.
- `unreadable`: a field was `[N/A]` or unparseable. The finding lists which
  ones. On a card with ECC disabled, or a pre-Ampere card, this is expected.
- `not-checked` / `config-error` runs (`status` in the record): the tool did
  not check this job's GPUs at all. `--check-config` on the node says why.
- `partially-checked` runs: some of the job's GPUs were checked and showed
  nothing drain-worthy, and others were not checked. The findings name them
  (`not-in-output`, `gpu-number-ambiguous`, `gpu-id-unsupported`, ...).
- `gpu-number-ambiguous`: Slurm's GPU number could mean two different GPUs
  on this node, because its device minors are not in PCI bus order. Set
  `--gpu-numbering` to match gres.conf (see the README, "Only the job's own
  GPUs"); `--check-config` shows the mismatch.
- `drain-failed` with "scontrol not run" in `errors`: `scontrol` failed the
  root-safety check (missing, group- or world-writable, or owned by another
  user), so it was not run. Slurm drained the node on exit 1 with its own
  generic reason. Fix the binary's ownership or mode.

[smi]: https://docs.nvidia.com/deploy/nvidia-smi/index.html
[rma]: https://docs.nvidia.com/deploy/a100-gpu-mem-error-mgmt/rma-policy-thresholds-for-row-remapping.html
