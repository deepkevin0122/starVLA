#!/usr/bin/env bash
set -euo pipefail

source /opt/miniforge/etc/profile.d/conda.sh

trap "kill 0" SIGINT SIGTERM EXIT

(
  conda activate starVLA
  bash examples/LIBERO/eval_files/run_policy_server.sh
) &

(
  conda activate libero
  bash examples/LIBERO/eval_files/eval_libero.sh
)

wait
