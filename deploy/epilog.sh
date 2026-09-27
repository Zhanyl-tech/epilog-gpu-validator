#!/bin/bash
# Slurm Epilog wrapper for epilog-gpu-validator. Install on every GPU node:
#
#     install -d -m 0755 /etc/slurm/epilog.d   # install does not create it
#     install -m 0755 deploy/epilog.sh /etc/slurm/epilog.d/50-gpu-validate
#
#     # slurm.conf (or keep the existing Epilog= line and add this script on
#     # an Epilog= line of its own; slurm.conf allows several)
#     Epilog=/etc/slurm/epilog.d/*
#     EpilogTimeout=60          # Slurm >= 25.05; older: PrologEpilogTimeout
#
# Slurm drains the node when the Epilog exits non-zero or times out. This
# wrapper exits non-zero in exactly one case: the validator exited 1, which it
# only does under ENFORCE=1 for a positively identified fault. Every other
# outcome, including this script failing, exits 0 and says why on stderr and
# via logger.
#
# Start with ENFORCE=0 (report only). Read the log for a week. Set ENFORCE=1
# when the findings have earned it.

# ── Settings: edit here, or override in $SITE_CONFIG ─────────────────────────
VALIDATOR=/usr/local/bin/epilog-gpu-validator
# Absolute paths: Slurm runs the Epilog with no search path. /usr/bin is
# common but not universal; check with `command -v nvidia-smi scontrol`.
NVIDIA_SMI=/usr/bin/nvidia-smi
SCONTROL=/usr/bin/scontrol
BUDGET=20s
# One JSON line per run. The Epilog's own stdout/stderr have no documented
# destination, so this file (and syslog) is where report-only findings live.
LOG_FILE=/var/log/epilog-gpu-validator.jsonl
# 1 = also send a one-line summary of each run to syslog (tag epilog-gpu-validator).
SYSLOG=1
# 1 = drain the node on a Degraded/Fatal finding. Anything else = report only.
ENFORCE=0
# Extra validator flags, e.g. (--shared-gpus --drain-on-pending-remap), or
# (--gpu-numbering nvml) when gres.conf uses AutoDetect=nvml on nodes whose
# device minors are not in PCI order (`--check-config` says so).
EXTRA_ARGS=()
# Sourced if readable; must be root-owned like this script.
SITE_CONFIG=/etc/default/epilog-gpu-validator

main() {
	# "for security reasons, these programs do not have a search path set"
	# (https://slurm.schedmd.com/prolog_epilog.html). Set one for anything
	# this script runs by name; the validator itself only uses the absolute
	# paths above.
	PATH=/usr/sbin:/usr/bin:/sbin:/bin
	export PATH

	# Any failure in this script (a broken site config, for instance) must not
	# become a non-zero Epilog exit, which would drain the node.
	trap on_exit EXIT

	if [ -r "$SITE_CONFIG" ]; then
		# shellcheck source=/dev/null
		. "$SITE_CONFIG"
	fi

	if [ ! -x "$VALIDATOR" ]; then
		say "$VALIDATOR is missing or not executable; GPUs NOT CHECKED; exiting 0 so Slurm does not drain the node"
		exit 0
	fi

	# Logging flags go first: if a later flag is mistyped, the validator has
	# already seen these and reports the mistake where operators look.
	args=()
	if [ "$SYSLOG" = 1 ]; then
		args+=(--syslog)
	fi
	if [ -n "$LOG_FILE" ]; then
		args+=(--log-file "$LOG_FILE")
	fi
	args+=(--budget "$BUDGET" --json --nvidia-smi "$NVIDIA_SMI" --scontrol "$SCONTROL")
	if [ "$ENFORCE" = 1 ]; then
		args+=(--enforce)
	fi
	if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
		args+=("${EXTRA_ARGS[@]}")
	fi

	# Run as a child, not exec: then only the two codes in the validator's
	# contract (0 = no drain, 1 = drain requested) reach Slurm. Anything else
	# (126/127 when it cannot be executed, 2 from an older build that still
	# exited 2 on a bad flag, a crash, a kill) is not evidence of a GPU fault.
	"$VALIDATOR" "${args[@]}"
	rc=$?
	trap - EXIT
	case "$rc" in
	0 | 1) exit "$rc" ;;
	esac
	say "$VALIDATOR exited $rc, which is outside its contract (0 or 1); GPUs NOT CHECKED; exiting 0 so Slurm does not drain the node"
	exit 0
}

on_exit() {
	local rc=$?
	if [ "$rc" -ne 0 ]; then
		say "wrapper failed (status $rc); GPUs NOT CHECKED; exiting 0 so Slurm does not drain the node"
	fi
	exit 0
}

say() {
	echo "epilog-gpu-validator: $*" >&2
	if [ "$SYSLOG" = 1 ] && [ -x /usr/bin/logger ]; then
		/usr/bin/logger -p daemon.err -t epilog-gpu-validator -- "$*" || true
	fi
}

# Run only when executed. The integration test sources this file to point the
# settings at fakes, then calls main itself.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
	main
fi
