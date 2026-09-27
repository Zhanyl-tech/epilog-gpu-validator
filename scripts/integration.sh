#!/bin/bash
# End-to-end check of deploy/epilog.sh plus the real binary, the way slurmd
# runs an Epilog: `env -i` (so no PATH at all) with only Slurm-style variables
# set. nvidia-smi and scontrol are fake scripts, and /proc/driver/nvidia/gpus
# is a fake directory tree. No GPU, no Slurm.
#
# Usage: scripts/integration.sh   (from the repo root; needs go and bash)
set -u

REPO=$(cd "$(dirname "$0")/.." && pwd)
WORK=$(mktemp -d "${TMPDIR:-/tmp}/egv-integration.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
FAKE=$WORK/fake
PROC=$WORK/proc         # device minors in PCI order: /dev/nvidia0 is gpu0
PROC_REV=$WORK/proc-rev # device minors in reverse PCI order: /dev/nvidia0 is gpu1
mkdir -p "$FAKE" "$PROC/0000:18:00.0" "$PROC/0000:28:00.0" "$PROC_REV/0000:18:00.0" "$PROC_REV/0000:28:00.0"
echo "Device Minor: 0" >"$PROC/0000:18:00.0/information"
echo "Device Minor: 1" >"$PROC/0000:28:00.0/information"
echo "Device Minor: 1" >"$PROC_REV/0000:18:00.0/information"
echo "Device Minor: 0" >"$PROC_REV/0000:28:00.0/information"

BIN=$WORK/epilog-gpu-validator
(cd "$REPO" && go build -o "$BIN" ./cmd/epilog-gpu-validator) || exit 1

cat >"$FAKE/scontrol" <<EOF
#!/bin/sh
printf '%s\n' "\$*" >>"$WORK/scontrol.calls"
EOF
chmod 755 "$FAKE/scontrol"
# The same fake, but writable by the group: must never be run as root.
cp "$FAKE/scontrol" "$FAKE/scontrol-group-writable"
chmod 775 "$FAKE/scontrol-group-writable"

# SYNTHETIC rows in QueryFields order (see internal/gpu/gpu.go); not captured
# from hardware. nvidia-smi gpu0 is at 18:00.0, gpu1 at 28:00.0 (PCI order);
# both idle at gen1.
row() { # index bus ecc_volatile
	echo "$1, GPU-5117a000-0000-4000-8000-00000000000$1, NVIDIA H100 80GB HBM3, 00000000:$2:00.0, 16, 16, 1, 5, $3, $3, 0, No, 0, No, 0x0000000000000001, 34, 1, 81559, Enabled"
}
fake_smi() { # mode
	{
		echo "#!/bin/sh"
		case "$1" in
		healthy) echo "echo '$(row 0 18 0)'; echo '$(row 1 28 0)'" ;;
		only-gpu0) echo "echo '$(row 0 18 0)'" ;;
		ecc-gpu0) echo "echo '$(row 0 18 3)'; echo '$(row 1 28 0)'" ;;
		ecc-gpu1) echo "echo '$(row 0 18 0)'; echo '$(row 1 28 3)'" ;;
		exit15) echo "echo 'Unable to determine the device handle for GPU 0000:28:00.0: GPU is lost' >&2; exit 15" ;;
		exit9) echo "echo 'NVIDIA-SMI has failed because it could not communicate with the NVIDIA driver' >&2; exit 9" ;;
		hang) printf 'sleep 8 &\nwait\n' ;;
		esac
	} >"$FAKE/nvidia-smi"
	chmod 755 "$FAKE/nvidia-smi"
}

FAILS=0
fail() {
	echo "FAIL $*"
	FAILS=$((FAILS + 1))
}
check() { # description expected-exit actual-exit [grep-pattern file]
	local ok=1
	[ "$2" = "$3" ] || ok=0
	if [ $ok = 1 ] && [ $# -ge 5 ]; then
		grep -q -- "$4" "$5" 2>/dev/null || ok=0
	fi
	if [ $ok = 1 ]; then
		echo "ok   $1 (exit $3)"
	else
		fail "$1: exit $3, want $2${4:+; want /$4/ in ${5:-}}"
	fi
}
no_scontrol_call() { # description
	[ ! -e "$WORK/scontrol.calls" ] || fail "$1: scontrol was called: $(cat "$WORK/scontrol.calls")"
}

# epilog <enforce> <cuda_visible_devices> [validator] [nvidia-smi] [extra args...]
# Optional environment: SLURM_JOB_GPUS_VALUE (also set SLURM_JOB_GPUS),
# PROC_OVERRIDE, SCONTROL_OVERRIDE, BUDGET_OVERRIDE.
epilog() {
	local enforce=$1 cvd=$2 validator=${3:-$BIN} smi=${4:-$FAKE/nvidia-smi}
	shift 4 2>/dev/null || shift $#
	rm -f "$WORK/log.jsonl" "$WORK/scontrol.calls"
	# The single-quoted script is expanded by the inner bash, on purpose.
	# shellcheck disable=SC2016
	env -i SLURM_JOB_ID=7 SLURMD_NODENAME=gpu001 CUDA_VISIBLE_DEVICES="$cvd" \
		${SLURM_JOB_GPUS_VALUE:+SLURM_JOB_GPUS="$SLURM_JOB_GPUS_VALUE"} \
		/bin/bash -c '
			. "$1"/deploy/epilog.sh
			VALIDATOR=$2 NVIDIA_SMI=$3 SCONTROL=$4 LOG_FILE=$5 ENFORCE=$6
			SYSLOG=0 # a test must not write to the host syslog
			SITE_CONFIG=/nonexistent BUDGET=${7:-20s}
			EXTRA_ARGS=(--driver-proc-dir "$8")
			shift 8; EXTRA_ARGS+=("$@")
			main' _ "$REPO" "$validator" "$smi" "${SCONTROL_OVERRIDE:-$FAKE/scontrol}" "$WORK/log.jsonl" "$enforce" \
		"${BUDGET_OVERRIDE:-20s}" "${PROC_OVERRIDE:-$PROC}" "$@" \
		>"$WORK/stdout" 2>"$WORK/stderr"
}

fake_smi healthy
epilog 1 0
check "idle healthy GPU, --enforce: no drain" 0 $? '"status":"ok"' "$WORK/log.jsonl"
no_scontrol_call "healthy node"

fake_smi ecc-gpu0
epilog 0 0
check "fault on the job's GPU, report-only: exit 0" 0 $? '"status":"would-drain"' "$WORK/log.jsonl"
no_scontrol_call "report-only mode"

epilog 1 0
check "fault on the job's GPU (Slurm GPU 0 = gpu0), --enforce: drain + exit 1" 1 $? \
	'update NodeName=gpu001 State=DRAIN Reason=epilog-gpu-validator fatal: ecc-uncorrectable:gpu0' "$WORK/scontrol.calls"

fake_smi ecc-gpu1
epilog 1 0
check "fault on a neighbour's GPU, --enforce: no drain" 0 $? '"status":"ok"' "$WORK/log.jsonl"

# Device minors in reverse PCI order. Slurm puts its GRES index in
# CUDA_VISIBLE_DEVICES (gres_common.c, gres_common_prep_set_env), so "0" is
# gpu0 if gres.conf numbers GPUs by PCI bus ID (AutoDetect=nvml) and gpu1
# (/dev/nvidia0) if it numbers them by device file.
fake_smi ecc-gpu0
PROC_OVERRIDE=$PROC_REV epilog 1 0
check "minors not in PCI order, --gpu-numbering auto: nothing checked, exit 0" 0 $? '"status":"not-checked"' "$WORK/log.jsonl"
grep -q 'gpu-number-ambiguous' "$WORK/log.jsonl" || fail "minors not in PCI order: no gpu-number-ambiguous finding"

PROC_OVERRIDE=$PROC_REV epilog 1 0 "$BIN" "$FAKE/nvidia-smi" --gpu-numbering nvml
check "minors not in PCI order, --gpu-numbering nvml: Slurm GPU 0 = gpu0 drains" 1 $? \
	'Reason=epilog-gpu-validator fatal: ecc-uncorrectable:gpu0' "$WORK/scontrol.calls"

PROC_OVERRIDE=$PROC_REV epilog 1 0 "$BIN" "$FAKE/nvidia-smi" --gpu-numbering minor
check "minors not in PCI order, --gpu-numbering minor: Slurm GPU 0 = /dev/nvidia0 = gpu1, healthy" 0 $? '"status":"ok"' "$WORK/log.jsonl"

# gres.conf Flags=env_uuid: a UUID in CUDA_VISIBLE_DEVICES, the index in
# SLURM_JOB_GPUS. The UUID decides, whatever the minors.
SLURM_JOB_GPUS_VALUE=0 PROC_OVERRIDE=$PROC_REV epilog 1 GPU-5117a000-0000-4000-8000-000000000000
check "env_uuid (UUID and index for the same GPU): drain + exit 1" 1 $? \
	'Reason=epilog-gpu-validator fatal: ecc-uncorrectable:gpu0' "$WORK/scontrol.calls"

SCONTROL_OVERRIDE=$FAKE/scontrol-group-writable epilog 1 0
check "group-writable scontrol under --enforce: exit 1, scontrol not run" 1 $? '"status":"drain-failed"' "$WORK/log.jsonl"
no_scontrol_call "group-writable scontrol"

fake_smi only-gpu0
epilog 1 0,1
check "one of the job's GPUs missing from nvidia-smi output: exit 0, partially-checked" 0 $? '"status":"partially-checked"' "$WORK/log.jsonl"

fake_smi exit15
epilog 1 0
check "nvidia-smi exit 15 (fell off the bus), --enforce: exit 1" 1 $? 'nvidia-smi-exit-15' "$WORK/scontrol.calls"

fake_smi exit9
epilog 1 0
check "nvidia-smi exit 9 (driver not loaded), --enforce: exit 0" 0 $? '"status":"not-checked"' "$WORK/log.jsonl"

fake_smi healthy
epilog 1 0 "$BIN" /nonexistent/nvidia-smi
check "nvidia-smi not at the configured path: exit 0, config-error logged" 0 $? '"status":"config-error"' "$WORK/log.jsonl"
grep -q "NOT CHECKED" "$WORK/stderr" || fail "config error not on stderr"

epilog 1 0 /nonexistent/epilog-gpu-validator
check "validator binary missing: exit 0, said on stderr" 0 $? "NOT CHECKED" "$WORK/stderr"

printf 'not a program\n' >"$WORK/garbage"
chmod 755 "$WORK/garbage"
epilog 1 0 "$WORK/garbage"
check "validator is not a real executable: exit 0" 0 $? "outside its contract" "$WORK/stderr"

epilog 1 0 "$BIN" "$FAKE/nvidia-smi" --budget 20
check "flag typo in the wrapper: exit 0, logged" 0 $? 'invalid value' "$WORK/log.jsonl"

# The code may wait up to 1s (abandonAfter) past the query deadline to reap
# nvidia-smi, and `date +%s` has 1-second resolution, so the bound is the 2s
# budget + 1s + 1s.
fake_smi hang
start=$(date +%s)
BUDGET_OVERRIDE=2s epilog 1 0 "$BIN" "$FAKE/nvidia-smi" --query-timeout 1s
code=$?
took=$(($(date +%s) - start))
check "nvidia-smi hangs with a child holding stdout: exit 0" 0 "$code" '"status":"not-checked"' "$WORK/log.jsonl"
if [ "$took" -gt 4 ]; then
	fail "hang case took ${took}s; want at most 4s (2s budget + 1s reap allowance + 1s timer resolution)"
else
	echo "ok   hang case returned in ${took}s (at most 4s allowed: 2s budget + 1s reap allowance + 1s timer resolution)"
fi

# The binary alone, with no PATH at all: absolute paths mean nothing is
# looked up. This is the case that used to fail with "executable file not
# found in \$PATH" and exit 0 having checked nothing.
fake_smi healthy
env -i SLURM_JOB_ID=7 SLURMD_NODENAME=gpu001 CUDA_VISIBLE_DEVICES=0 \
	"$BIN" --json --nvidia-smi "$FAKE/nvidia-smi" --driver-proc-dir "$PROC" >"$WORK/stdout" 2>"$WORK/stderr"
check "binary under env -i with no PATH checks the GPU" 0 $? '"status": "ok"' "$WORK/stdout"

if [ "$FAILS" -ne 0 ]; then
	echo "$FAILS integration check(s) failed"
	exit 1
fi
echo "all integration checks passed"
