# epilog-gpu-validator

Check the GPUs a job just used, and drain the node if they are *persistently*
faulty — before the next job lands on them.

```
make scenarios
```

Runs every fault case against a synthetic GPU. No NVIDIA hardware required.

```
SCENARIO        SEVERITY  DRAIN  EXIT REASON
────────────────────────────────────────────────────────────────────────────
healthy         ok        no     0
remap-pending   transient no     0
thermal         transient no     0
pcie-degraded   degraded  yes    1    epilog-gpu-validator degraded: gpu0:pcie-gen
hw-slowdown     degraded  yes    1    epilog-gpu-validator degraded: gpu0:throttle
ecc             fatal     yes    1    epilog-gpu-validator fatal: gpu0:ecc-uncorrectable
remap-failure   fatal     yes    1    epilog-gpu-validator fatal: gpu0:row-remap-failure
missing         n/a       no     0    query failed — safe no-op
```

---

## The problem

A GPU develops a fault mid-job. The job that hit it fails, or worse, silently
returns wrong numbers. Slurm marks the node idle. The next job lands on the same
card and fails the same way. Then the next.

Nothing in Slurm looks at GPU health between jobs. Epilog is the hook that
could — it runs on every completion, on the node, as root, while nothing else is
using the hardware.

## Why this is a dangerous tool to write

**Slurm drains the node when Epilog exits non-zero.**

So a false positive doesn't produce a bad metric, it removes a working node. And
if the cause is fleet-wide — a driver bug, a monitoring gap, a heatwave — it
removes *every* node, one job completion at a time, faster than anyone can
react.

Two rules follow, and everything else is detail:

### 1. Never drain on ignorance

If `nvidia-smi` is missing, times out, or returns something unparseable, that is
a **monitoring** failure. Draining on it converts a broken health check into a
cluster-wide outage. Exit code is 0.

That is the `missing` row in the table above, and it is the single most
important line in the test suite.

### 2. Separate persistent faults from transient conditions

| Severity | Meaning | Drains? |
| --- | --- | --- |
| **Fatal** | Already corrupted data, or out of spare capacity | yes |
| **Degraded** | Persistently underperforming — works, quietly slower | yes |
| **Transient** | Real but self-clearing | **no** |
| **Unknown** | Could not be determined | **no** |

A card thermal-throttling on a hot afternoon is the hardware protecting itself.
Draining for it means draining the row when the CRAC unit hiccups. A card at
**x8 on an x16 slot** is a different animal entirely: it passes every functional
test, halves host-to-device bandwidth, and poisons every collective it joins.
Nobody notices for months.

## What it checks

| Check | Severity | Why |
| --- | --- | --- |
| Device not responding | fatal | Fell off the bus |
| Uncorrectable ECC *this power cycle* | fatal | Corruption already reached a computation |
| Row remap **failure** | fatal | No spare rows left; next fault is uncorrectable |
| Correctable ECC above threshold | degraded | Leading indicator of an uncorrectable one |
| Row remap due to uncorrectable errors | degraded | Memory is going |
| PCIe width or gen below max | degraded | The silent one |
| `hw_slowdown` / `hw_thermal_slowdown` | degraded | A hard limit was already exceeded |
| Memory still allocated after teardown | degraded | A process survived the job |
| Uncorrectable ECC *lifetime only* | transient | Historic; clean since reset |
| Row remap **pending** | transient | Needs a reset, still works |
| `sw_thermal_slowdown` / `sw_power_cap` | transient | Self-clearing |
| Temperature at threshold | transient | Usually airflow, not the card |

Volatile versus aggregate ECC is a distinction worth labouring: errors *since
boot* mean the card is failing now; *lifetime* errors on a card clean since its
last reset do not justify taking a node out.

## Only the job's own GPUs

On a shared node, checking every GPU means a neighbour's faulty card drains the
node for a job that never touched it. The validator reads `SLURM_JOB_GPUS` (or
`GPU_DEVICE_ORDINAL` / `CUDA_VISIBLE_DEVICES`) and checks only those.

If that set cannot be determined it checks **nothing** and exits 0. Note that
depending on Slurm version and `GresTypes`, these can contain UUIDs rather than
ordinals — those are skipped rather than guessed at.

## Install

```bash
make install                      # /usr/local/bin/epilog-gpu-validator
cp deploy/epilog.sh /etc/slurm/epilog.d/gpu-validate
```

```conf
# slurm.conf
Epilog=/etc/slurm/epilog.d/gpu-validate
EpilogTimeout=60
```

`EpilogTimeout` must exceed `--budget` (default 20s). If Slurm kills the check
mid-run, the node can be marked down for a check that never reached a
conclusion — the opposite of the point.

**Report-only is the default.** A health check that starts draining nodes the
day it is installed does not get installed twice. Run it for a week, read the
findings, then add `--enforce`.

```bash
epilog-gpu-validator --budget 20s --json              # report only
epilog-gpu-validator --budget 20s --json --enforce    # actually drain
epilog-gpu-validator --simulate ecc --json            # no hardware needed
```

Useful knobs:

| Flag | For |
| --- | --- |
| `--allow-pcie-downgrade` | Chassis that genuinely wire cards below full width |
| `--max-correctable-ecc` | Sites with a different tolerance |
| `--drain-on-pending-remap` | Drain rather than batch the reset |
| `--all-gpus` | Check the whole node, not just the job's cards |

## Limitations

- **NVIDIA only**, via `nvidia-smi`. No ROCm, no Habana.
- **No active load test.** `dcgmproftester` would catch faults that only appear
  under load, but it takes tens of seconds and Epilog does not have that budget.
  Deliberately out of scope; a periodic drain-and-test job is the right home
  for it.
- **No XID scraping.** Reading `dmesg` needs privileges Epilog does not always
  have and is noisy to parse reliably.
- **No MIG awareness.**
- **Never un-drains.** Bringing a node back is a human decision.

## Development

```bash
make test        # race detector + coverage
make scenarios   # the table above
```

The GPU source is an interface with a simulator backend, so every classification
branch is reachable without a broken card to hand — and the CI asserts the two
safety properties directly: a healthy GPU never exits non-zero, and a failed
query never drains.

## The set

Part of a set of tools covering the lifecycle of a GPU allocation, each built on
the same rule — never act on absent evidence:

- **epilog-gpu-validator** — this repo. Hardware faults *between* jobs.
- **[gpu-reaper](https://github.com/Zhanyl-tech/gpu-reaper)** — the companion
  that catches wasted GPUs *during* a job; its "no active load test" limitation
  is the gap this tool's between-jobs timing is meant to complement.
- **[ib-slurm-exporter](https://github.com/Zhanyl-tech/ib-slurm-exporter)** —
  fabric problems attributed to the job causing them.
- **[slurm-scheduler-lab](https://github.com/Zhanyl-tech/slurm-scheduler-lab)** —
  the scheduling policy that decides what runs in the first place.

## License

MIT
