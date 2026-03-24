#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

"$ROOT_DIR/tools/setup_uv_env.sh" \
  --profile real-ur5-ros \
  --python 3.10 \
  --env .venv310 \
  --telemetry

# Ensure critical ROS/runtime deps are present in this venv even if lock state
# changes or submodule dependency metadata drifts.
uv pip install --python .venv310/bin/python \
  splines pyserial psutil Pillow scipy \
  spatialmath-python ur-rtde rerun-sdk

echo ""
echo "Environment ready at $ROOT_DIR/.venv310"
echo "Activate with: source .venv310/bin/activate"
echo "Use with ROS: source /opt/ros/humble/setup.bash && source .venv310/bin/activate"
