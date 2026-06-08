#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  tools/setup_uv_env.sh [--profile NAME] [--python VERSION] [--env DIR] [--telemetry]

Profiles:
  sim-core       Minimal MuJoCo + math stack
  sim-hand       Sim hand planner/viewer (+mink, no real-hand serial deps)
  sim-ur5        Sim UR5 viewer (+mink only)
  sim-h12        Sim + H1-2 stack
  sim-h12-ur5    Sim + H1-2 + UR5 + mink
  real-ur5       Real RH56 + UR5 (no ROS extras)
  real-ur5-ros   Real RH56 + UR5 with ROS workflow (ROS via apt/rosdep)
  real-h12-ros   Real H1-2 arm with ROS workflow (no RH56 serial stack)
  real-h12-hand-ros  Real H1-2 + RH56 hand with ROS workflow
  full           Install all optional dependencies

Examples:
  tools/setup_uv_env.sh --profile sim-core --python 3.12 --env .venv312
  tools/setup_uv_env.sh --profile real-ur5-ros --python 3.10 --env .venv310 --telemetry
EOF
}

PROFILE="sim-core"
PYTHON_VERSION="3.12"
ENV_DIR=""
ADD_TELEMETRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --env)
      ENV_DIR="${2:-}"
      shift 2
      ;;
    --telemetry)
      ADD_TELEMETRY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

if [[ -z "$ENV_DIR" ]]; then
  ENV_DIR=".venv${PYTHON_VERSION//./}"
fi

EXTRAS=()
case "$PROFILE" in
  sim-core)
    ;;
  sim-hand)
    EXTRAS+=("sim-hand")
    ;;
  sim-ur5)
    EXTRAS+=("sim-ur5")
    ;;
  sim-h12)
    EXTRAS+=("sim-h12")
    ;;
  sim-h12-ur5)
    EXTRAS+=("sim-h12-ur5")
    ;;
  real-ur5)
    EXTRAS+=("real-robot")
    ;;
  real-ur5-ros)
    EXTRAS+=("real-ur5-ros" "ros")
    ;;
  real-h12-ros)
    EXTRAS+=("real-h12-ros" "ros")
    ;;
  real-h12-hand-ros)
    EXTRAS+=("real-h12-hand-ros" "ros")
    ;;
  full)
    EXTRAS+=("full" "ros")
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    usage
    exit 1
    ;;
esac

if [[ $ADD_TELEMETRY -eq 1 ]]; then
  EXTRAS+=("telemetry")
fi

if [[ ! -d "$ENV_DIR" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
fi

SYNC_CMD=(uv sync)
for ex in "${EXTRAS[@]}"; do
  SYNC_CMD+=(--extra "$ex")
done

if [[ "$PROFILE" == "sim-h12-ur5" ]] && [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
  # magpie_control declares real-hardware deps that are not needed for sim
  # and are not installable on Apple Silicon without extra native tooling.
  SYNC_CMD+=(--no-install-package pyrealsense2 --no-install-package ur-rtde)
fi

echo "[setup_uv_env] Profile: $PROFILE"
echo "[setup_uv_env] Python : $PYTHON_VERSION"
echo "[setup_uv_env] Env    : $ENV_DIR"
if [[ ${#EXTRAS[@]} -gt 0 ]]; then
  echo "[setup_uv_env] Extras : ${EXTRAS[*]}"
else
  echo "[setup_uv_env] Extras : (none)"
fi

if [[ "$PROFILE" == *"ros"* ]] && [[ "$PYTHON_VERSION" != "3.10" ]]; then
  echo "[setup_uv_env] WARNING: ROS2 Humble workflows should use Python 3.10."
fi

UV_PROJECT_ENVIRONMENT="$ENV_DIR" "${SYNC_CMD[@]}"

echo ""
echo "Environment ready at $ROOT_DIR/$ENV_DIR"
echo "Activate with: source $ENV_DIR/bin/activate"
if [[ "$PROFILE" == *"ros"* ]]; then
  echo "ROS usage: source /opt/ros/humble/setup.bash && source $ENV_DIR/bin/activate"
fi
