#!/usr/bin/env bash
set -euo pipefail

USER_NAME="vsastry"

ACCOUNT="datascience"
SELECT="2"
WALLTIME="60:00"
FILESYSTEMS="home:eagle"

DEBUG_QUEUE="debug"
DEBUG_SCRIPT="launch_mamba2_hgt.sh"

DEBUG_SCALING_QUEUE="debug-scaling"
DEBUG_SCALING_SCRIPT="launch_gla_hgt.sh"

SLEEP_SECS=$((30 * 60))
cd /eagle/datascience/vsastry/projects/LinearAttention/new_repo/flame
# Returns 0 if there exists a job for $USER_NAME in queue $1 with state Q or R.
has_q_or_r_job_in_queue() {
  local queue="$1"
  # Common PBSPro/Torque qstat layout: ... queue ... state (R/Q/etc).
  # We filter lines matching the user, then the queue, then state Q or R.
  qstat 2>/dev/null \
    | awk -v u="$USER_NAME" -v q="$queue" '
        $0 ~ u && $0 ~ q {
          # Try to find a token that is exactly Q or R (state column).
          for (i=1; i<=NF; i++) if ($i=="Q" || $i=="R") { found=1 }
        }
        END { exit(found ? 0 : 1) }
      '
}

submit_if_needed() {
  local queue="$1"
  local script="$2"

  if has_q_or_r_job_in_queue "$queue"; then
    echo "[$(date)] Found $USER_NAME job in queue=$queue with state Q/R. Skipping submit."
  else
    echo "[$(date)] No $USER_NAME job in queue=$queue with state Q/R. Submitting $script ..."
    qsub -A "$ACCOUNT" \
         -l "select=$SELECT" \
         -l "walltime=$WALLTIME" \
         -l "filesystems=$FILESYSTEMS" \
         -q "$queue" \
         "$script"
  fi
}

while true; do
  submit_if_needed "$DEBUG_QUEUE" "$DEBUG_SCRIPT"
  submit_if_needed "$DEBUG_SCALING_QUEUE" "$DEBUG_SCALING_SCRIPT"
  echo "[$(date)] Sleeping for ${SLEEP_SECS}s..."
  sleep "$SLEEP_SECS"
done

