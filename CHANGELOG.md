# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Fixes from an external audit of 0.1.0. Its two critical findings mean 0.1.0
should not be run with `--enforce`. Every behaviour below was tested against
fakes, synthetic `nvidia-smi` output and a simulator. **None of it has been
run on a GPU or a Slurm cluster.**

### Fixed: could drain the whole fleet, or silently check nothing

- **Idle GPUs no longer drain.** 0.1.0 marked a PCIe link below its maximum
  generation as Degraded. NVIDIA's manual says the current link generation and
  width "may be reduced when the GPU is not in use", and the Epilog only runs
  when the GPU is not in use. So on hardware that does this, every node would
  drain after its next job once `PATH` and `--enforce` were set up. That was
  not observed on a GPU; the audit reproduced it only with a synthetic idle
  row. Link generation and width below max are now `unknown` findings.
  `--drain-on-pcie-width` opts back into draining on width (never
  generation), for sites that have confirmed their idle GPUs report full
  width.
- **The tool no longer depends on `PATH`.** Slurm runs the Epilog with no
  search path, so 0.1.0 as shipped never found `nvidia-smi` in a real Epilog,
  checked nothing and exited 0 without saying so. (That is also why the PCIe
  drain above could not have fired through the shipped wrapper.) `nvidia-smi`
  and `scontrol` are now absolute-path flags (`--nvidia-smi`, `--scontrol`,
  default `/usr/bin/...`). They are checked at startup: absolute, regular,
  executable, not group- or world-writable, and owned by root or the invoking
  user. An `nvidia-smi` that fails is a loud `config-error`, with exit 0. An
  `scontrol` that fails is never run: under `--enforce` a drain then relies on
  exit 1 and Slurm's own Epilog-failure drain (`drain-failed`). `scontrol` is
  checked again right before it would run, which also refuses a bare name
  rather than searching `PATH` as root.
- **A flag mistake or a panic can no longer drain nodes.** Flag parsing no
  longer exits 2 (it used `flag.ExitOnError`), and a panic no longer escapes.
  A bad flag, an unknown flag, a stray argument, an invalid value or an
  unknown `--simulate` scenario is now a `config-error`, with exit 0.
- **`--simulate` never drains.** It used to swap only the GPU source and still
  call the real `scontrol` under `--enforce`, and `make scenarios` and CI ran
  exactly that. `--simulate` now returns before any controller is created,
  and always exits 0. `make scenarios` no longer passes `--enforce` to
  anything real. (The unused `slurm.DryRun` controller was removed.)

### Fixed: wrong or missing classification

- **Clock-event reasons are parsed as the hex bitmask they are**, using the
  nvml.h bit values. 0.1.0 matched names like `hw_slowdown` that never appear
  in real output: HW slowdown could never drain, and every idle GPU produced a
  spurious finding. The `GpuIdle`, applications-clocks and sync-boost bits are
  ignored. HW thermal slowdown and HW power brake are Degraded. The bare
  `hw_slowdown` bit is Transient, because nvml.h says it "May be also reported
  during PState or clock change".
- **A Slurm GPU number is no longer assumed to be one particular GPU.** In
  the Epilog, `CUDA_VISIBLE_DEVICES` and `SLURM_JOB_GPUS` hold Slurm's GRES
  index (`gres_common_prep_set_env()` in
  `src/plugins/gres/common/gres_common.c` writes `gres_device->index`). Which
  GPU that is depends on gres.conf: gres.conf's Links note says the index is
  "created by sorting the GPUs by PCI bus ID" and is "not necessarily" the
  `/dev/nvidiaN` minor, while the gres guide's Epilog example reads it as a
  device-file number. 0.1.0 passed the number to `nvidia-smi -i` as an NVML
  index, which is right for `AutoDetect=nvml` and wrong for a gres.conf that
  numbers by device file on a node whose minors are not in PCI order. The new
  `--gpu-numbering` says which applies: `nvml`, `minor` (mapped through
  `/proc/driver/nvidia/gpus/*/information`, "Device Minor"), `uuid`, or
  `auto` (the default), which checks a number only where `/dev/nvidiaN` and
  `nvidia-smi` index N are the same GPU on this node, and otherwise reports
  `gpu-number-ambiguous` and checks nothing for it. If the Slurm GPU
  variables disagree, nothing is checked.
- **UUIDs from Slurm are used, not skipped.** `nvidia-smi` accepts a GPU UUID.
  The claim that UUIDs were unusable was wrong. With gres.conf
  `Flags=env_uuid` (Slurm 26.05 and later) `CUDA_VISIBLE_DEVICES` holds UUIDs
  while `SLURM_JOB_GPUS` holds numbers; the two are compared by count, and
  the UUIDs are used. `MIG-` entries are reported and not checked.
- **The node is queried once, without `-i`.** The manual documents `-i` as
  selecting a single GPU. The job's GPUs are picked out of the full output. A
  job GPU missing from otherwise good output is an `unknown` finding, and the
  run is `partially-checked`, logged at error level, not `ok`. 0.1.0 reported
  `ok`.
- **`nvidia-smi` exit codes 8, 10, 14 and 15 are Fatal.** The manual documents
  them as hardware faults (power cables, interrupt issue, infoROM corrupted,
  fallen off the bus). Other failures (driver not loaded, NVML missing, bad
  arguments, timeout, unparseable output) are still `unknown`, with exit 0.
- **The `unknown` severity is actually produced.** `[N/A]`, `[Not Supported]`
  and unparseable fields used to become zeros that looked exactly like a
  healthy card. They are now listed in an `unreadable` finding and never used
  as evidence. Rows with the wrong column count are reported, not dropped.
- **Volatile ECC and persistence mode.** Volatile counters count "since the
  last driver load", and the driver can unload after the job when persistence
  mode is off. That is now an `unknown` finding, and lifetime-only errors are
  not called "none this power cycle" when the volatile counter cannot be
  trusted. "power cycle" wording changed to "since the last driver load".
- **Shared GPUs.** With gres/mps or shards, memory still in use may belong to
  another running job. `--shared-gpus`, or `SLURM_SHARDS_ON_NODE` /
  `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` in the environment, make
  `leaked-memory` `unknown` instead of Degraded.
- **Drain reasons name every faulty GPU**, grouped by check and sorted
  numerically (`ecc-uncorrectable:gpu0,gpu3`, `gpu2` before `gpu10`). 0.1.0
  kept one GPU per check.
- **`--budget` is a real bound.** `nvidia-smi` runs with `Cmd.WaitDelay`, and
  the tool stops waiting shortly after the deadline even if the child cannot
  be reaped. The audit measured 12s against a 2s budget in 0.1.0. With the
  hanging fake in `scripts/integration.sh` (2s budget, 1s query timeout), the
  run returned in 2s and 1s in the two runs made during this change, timed
  with `date +%s` (1-second resolution) on macOS. The unused `QueryTimeout`
  constant became `--query-timeout` (default 10s), which leaves the rest of
  the budget for `scontrol`.

### Changed

- **Rows remapped after uncorrectable errors are Transient, not Degraded.**
  This was found while doing the fixes above; it was not in the audit. The
  count is a lifetime number stored on the card, and remapping is the designed
  repair. NVIDIA's RMA policy keys on remapping failures, which stay Fatal.
  Draining on the count would have drained a repaired GPU after every job for
  the rest of its life.
- **The `RequirePersistenceMode` setting was removed.** No flag reached it.
  Persistence mode off now always produces the `ecc-volatile-unreliable`
  finding described above.
- **Simulator scenarios.** `missing` was removed. It modelled a generic error
  that real `nvidia-smi` does not produce. It is replaced by `off-bus`
  (exit 15, Fatal) and `no-driver` (exit 9, `unknown`). Added `leaked-memory`,
  `ecc-na` and `no-persistence`. The simulator now prints `nvidia-smi` CSV
  into the real parser, and it models a healthy GPU as idle (gen1 link,
  `GpuIdle` bit). Its fault is on `nvidia-smi` `gpu0`, and its device minors
  follow PCI order, with a reversed layout for the `--gpu-numbering` rows. An
  unknown scenario is a `config-error`; it used to run the healthy scenario.
- `go.mod` now declares `go 1.22.0`, the oldest version CI tests, instead of
  `go 1.26.3`. It is a chosen floor, not a measured minimum. CI tests 1.22
  and stable with `GOTOOLCHAIN=local`. Before, the "1.24" job silently
  downloaded and ran 1.26.3.

### Added

- `--log-file` (one JSON record per run) and `--syslog` (one summary line,
  tag `epilog-gpu-validator`). The Epilog's own stdout and stderr have no
  documented destination, so without these, report-only findings could not be
  read later. Every run's record carries a `status`: `ok`,
  `partially-checked`, `would-drain`, `drained`, `drain-failed`, `no-gpus`,
  `not-checked` or `config-error`. `not-checked`, `partially-checked` and
  `config-error` always log at error level.
- `--check-config`, an install-time check. It verifies paths, log
  destinations and the device map, and runs one real query. Under
  `--gpu-numbering auto` it fails when the device map cannot be read or the
  node's device minors are not in PCI bus order, since either way no job
  given GPU numbers would be checked. It exits 78 on a problem, but exits 0
  whenever `SLURM_JOB_ID` is set, so it cannot drain a node if pasted into
  the Epilog.
- `--gpu-numbering`, `--max-temperature`, `--shared-gpus`,
  `--drain-on-pcie-width`, `--query-timeout`, `--driver-proc-dir`, and
  `--scenario-table` (prints the README table).
- Rewritten `deploy/epilog.sh`. It sets `PATH`, passes absolute tool paths,
  and turns enforcement on with `ENFORCE=1`. In 0.1.0, uncommenting
  `# --enforce` silently did nothing. It sources `/etc/default/epilog-gpu-validator`
  if present. It passes through only exit 0 and 1 from the validator, and
  turns anything else, including its own failures, into exit 0 with a
  message.
- Tests for the exit-code contract (`cmd/epilog-gpu-validator/main_test.go`),
  for `nvidia-smi` output parsing, exit codes and timeouts (`internal/gpu`),
  for the `/proc` device map, for binary-path safety (`internal/binpath`) and
  for `scontrol` argv. All `nvidia-smi` rows in the tests are synthetic and
  labelled as such.
- `scripts/integration.sh` runs the shipped wrapper and the real binary under
  `env -i`, with fake `nvidia-smi`, `scontrol` and `/proc`: 18 cases plus a
  timing bound (a hung `nvidia-smi` must return within 4 whole seconds
  against a 2s budget). CI runs it.
- CI: `permissions: contents: read`, actions pinned by commit SHA, a gofmt
  check, staticcheck, govulncheck and shellcheck. The README table is now
  generated and compared byte for byte by a test; 0.1.0's table had drifted
  from `make scenarios` output. There is also a release workflow for static
  linux/amd64 and linux/arm64 binaries with checksums; it has not yet run on a
  real tag.
- `docs/runbook.md`: what each drain reason means, how to confirm it, and how
  to resume the node.
- README: a status line saying the tool is not yet validated on real hardware,
  prior art (NHC, `HealthCheckProgram`, DCGM), the exit contract, a
  tested-with and assumptions section, Slurm-version notes for the timeout,
  running several Epilog scripts (glob or several `Epilog=` lines), and an
  untested section on containerised slurmd and Slinky.

### Corrected claims

- "Nothing in Slurm looks at GPU health between jobs" now reads: Slurm ships
  no GPU check, and its hooks are `HealthCheckProgram` and `Epilog`.
- The Xid limitation used to say reading `dmesg` "needs privileges Epilog does
  not always have". The Epilog runs as root. The real reasons are now given:
  parsing the kernel log is fragile and not job-scoped, and NVML events or
  DCGM are the proper sources.
- Removed an unmeasured "~100-300ms per nvidia-smi call" figure, and an
  uncited "Slurm truncates long drain reasons". The 200-character bound is
  this tool's own.
- The package comment said Slurm kills the Epilog "at EpilogMsgTime".
  `EpilogMsgTime` is a message-processing setting; the timeout is
  `EpilogTimeout`, or `PrologEpilogTimeout` before 25.05.
- The thresholds (1000 correctable errors, 5% memory, 90°C) are now labelled
  as site-policy defaults that have not been validated.
- The README's "Uncorrectable ECC *this power cycle*" and "can be marked down"
  now read "since the last driver load" and "is drained".
- "Uncorrectable ECC *lifetime only*: transient" now says it is transient
  only when persistence mode is on and the volatile counter is readable, and
  `unknown` otherwise, which is what the code does.
- The README's "The set" said gpu-reaper's "no active load test" limitation
  is the gap this tool complements. This tool runs no load test either; the
  bullet now says so and points to DCGM diagnostics.
- The row-remap-failure wording said NVIDIA's RMA policy treats the flag as
  RMA-eligible. The policy says the criteria are met when the flag "is set
  and validated by the field diagnostic"; the README, runbook and finding
  text now say that.

## [0.1.0] - 2026-07-31

First public release. This entry was rewritten after release to describe what
0.1.0 actually shipped. The original entry mentioned a `make demo` target and
metric names, neither of which existed.

- Classifies GPU health from one `nvidia-smi --query-gpu` call into fatal,
  degraded, transient and unknown, and drains the node through
  `scontrol update ... State=DRAIN` under `--enforce`. The default is
  report-only.
- `make scenarios` runs every classification branch against a synthetic GPU.
  It requires `python3` to render the table.
- Tests covered the classifier and the Slurm environment parsing. The
  `nvidia-smi` parser and `main` had none.
- CI ran `go vet` and `go test -race` on pushes to `main` and on pull
  requests, plus two exit-code checks against the simulator.
- Known problems, found later by audit: see [Unreleased]. In particular, do
  not run 0.1.0 with `--enforce`.

[Unreleased]: https://github.com/Zhanyl-tech/epilog-gpu-validator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zhanyl-tech/epilog-gpu-validator/releases/tag/v0.1.0
