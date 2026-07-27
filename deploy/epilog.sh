#!/bin/bash
# Slurm Epilog wrapper. Install on every GPU node and reference from slurm.conf:
#
#     Epilog=/etc/slurm/epilog.d/gpu-validate
#     EpilogTimeout=60          # must exceed --budget below
#
# Slurm drains the node when Epilog exits non-zero, so the budget matters: if
# Slurm kills this mid-run the node may be marked down for a check that never
# reached a conclusion. Keep --budget comfortably under EpilogTimeout.
#
# Start with report-only. Watch the logs for a week. Add --enforce when the
# findings have earned it.

exec /usr/local/bin/epilog-gpu-validator \
  --budget 20s \
  --json
  # --enforce
