#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

source /opt/ros/humble/setup.bash
source .venv310/bin/activate

colcon build \
  --base-paths . magpie_force_control/ros2 \
  --packages-select rh56_controller magpie_force_control_ros

echo ""
echo "Build complete. Source with:"
echo "source /opt/ros/humble/setup.bash"
echo "source $ROOT_DIR/.venv310/bin/activate"
echo "source $ROOT_DIR/install/setup.bash"
