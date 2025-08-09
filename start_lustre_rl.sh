#!/bin/bash
# Start RL tuning script in the background, record its PID, run fio workload, then stop the tuner.
# Usage: ./start.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# 1. Launch tuning script in background
# -----------------------------------------------------------------------------
PY_CMD="python rl4sys/examples/lustre_rpc/tuning_rpc_param_rl.py ./rpc_log rl_lustre 0"

echo "[+] Starting tuning script: $PY_CMD"
$PY_CMD >tuning_stdout.log 2>tuning_stderr.log &
TUNING_PID=$!

echo "[+] Tuning script PID: $TUNING_PID"

# Ensure the tuning script is terminated on script exit (e.g., Ctrl+C)
cleanup() {
  echo "[+] Cleaning up – stopping tuning script (PID $TUNING_PID)"
  kill "${TUNING_PID}" 2>/dev/null || true
  wait "${TUNING_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# -----------------------------------------------------------------------------
# 2. Run fio workload (blocking)
# -----------------------------------------------------------------------------
# NOTE: Do not modify the fio command as requested
fio --name=lustre_test --directory=/mnt/hasanfs \
       --time_based --size=2G --rw=randrw --bs=1M --ioengine=libaio --numjobs=4 --runtime=300

echo "[+] fio completed. Tuning script will be stopped by cleanup trap."