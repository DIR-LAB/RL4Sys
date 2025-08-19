#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# open_multiple_client.sh
# -----------------------------------------------------------------------------
# Launch multiple instances of the RL4Sys job-scheduling simulation in parallel.
#
# Usage:
#   ./open_multiple_client.sh <num_clients> [additional job_main.py args]
#
# Example (start 8 clients):
#   ./open_multiple_client.sh 8 --number-of-iterations 50
#
# The script will run <num_clients> background processes of
# `rl4sys/examples/job_schedual_old/job_main.py`, each with a unique
# `--client-id` and `--seed` to avoid collisions.  Standard output and error of
# each client are redirected to `client_<idx>.log` in the current directory.
# -----------------------------------------------------------------------------

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <num_clients> [job_main_args...]" >&2
  exit 1
fi

NUM_CLIENTS=$1
shift  # Remaining arguments passed to job_main.py

if ! [[ "$NUM_CLIENTS" =~ ^[0-9]+$ ]] || [[ "$NUM_CLIENTS" -le 0 ]]; then
  echo "<num_clients> must be a positive integer" >&2
  exit 1
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
JOB_MAIN_PATH="${SCRIPT_DIR}/rl4sys/examples/job_schedual_old/job_main.py"

# -----------------------------------------------------------------------------
# Prepare per-run log directory: multiclient/<timestamp>
# -----------------------------------------------------------------------------
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/multiclient/${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# Optional: tell user where logs will go.
echo "[Launcher] Logs will be stored under: $LOG_DIR"

if [[ ! -f "$JOB_MAIN_PATH" ]]; then
  echo "job_main.py not found at expected path: $JOB_MAIN_PATH" >&2
  exit 1
fi

for ((idx=1; idx<=NUM_CLIENTS; idx++)); do
  LOG_FILE="${LOG_DIR}/client_${idx}.log"
  echo "[Launcher] Starting client ${idx}/${NUM_CLIENTS} -> log: ${LOG_FILE}"
  # Each client gets a distinct --client-id and --seed to avoid name clashes.
  python -u "$JOB_MAIN_PATH" \
    --client-id "client_${idx}" \
    --seed "$idx" \
    "$@" \
    > "$LOG_FILE" 2>&1 &
  echo "[Launcher]   PID $!"
  sleep 0.2  # slight stagger to reduce simultaneous startup spikes
done

echo "[Launcher] All $NUM_CLIENTS clients started. Use 'jobs -l' or 'ps' to monitor."
