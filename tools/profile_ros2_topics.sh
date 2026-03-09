#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <topic1> [topic2 ...]"
  echo "Example: $0 /grasp_viz/summary_json /force_control/wrench_measured"
  exit 1
fi

for topic in "$@"; do
  echo "=== Profiling $topic ==="
  timeout 10s ros2 topic hz "$topic" || true
  timeout 10s ros2 topic bw "$topic" || true
  echo
 done
