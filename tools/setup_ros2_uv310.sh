#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

# Thin convenience wrapper around the canonical installer.
# Default here is the combined H1-2 + RH56 ROS workflow.
"$ROOT_DIR/tools/setup_uv_env.sh" \
  --profile real-h12-hand-ros \
  --python 3.10 \
  --env .venv310 \
  --telemetry

echo ""
echo "Environment ready at $ROOT_DIR/.venv310"
echo "Activate with: source .venv310/bin/activate"
echo "Use with ROS: source /opt/ros/humble/setup.bash && source .venv310/bin/activate"
echo "Arm-only variant: tools/setup_uv_env.sh --profile real-h12-ros --python 3.10 --env .venv310 --telemetry"
