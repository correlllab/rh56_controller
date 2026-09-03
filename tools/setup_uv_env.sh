#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  tools/setup_uv_env.sh [--profile NAME] [--python VERSION] [--env DIR] [--telemetry] [--list]

Profiles:
  sim-core           Minimal MuJoCo + math stack
  sim-hand           Floating RH56 hand planner/viewer (+mink comparison)
  sim-ur5            UR5 + RH56 MuJoCo viewer
  sim-ur5-vision     UR5 + RH56 fixed-camera calibration simulation
  sim-h12            H1-2 + RH56 MuJoCo viewer
  sim-h12-ur5        H1-2 and UR5 sim workflows
  real-hand          Real RH56 hand serial workflows
  real-ur5           Real RH56 + UR5, no ROS
  real-ur5-ros       Real RH56 + UR5 with ROS2 bridge
  real-h12-ros       Real H1-2 arm with ROS2, no RH56 serial hand
  real-h12-hand-ros  Real H1-2 + real RH56 hand with ROS2
  dev-full           Maintainer/dev environment

Compatibility aliases:
  real-robot         Alias for real-ur5
  full               Alias for dev-full

Examples:
  tools/setup_uv_env.sh --profile sim-hand --python 3.12 --env .venv312
  tools/setup_uv_env.sh --profile real-ur5-ros --python 3.10 --env .venv310 --telemetry

See docs/INSTALL_PROFILES.md for profile boundaries, required submodules,
and first-run commands.
EOF
}

print_profiles() {
  cat <<'EOF'
Supported profiles:
  sim-core
  sim-hand
  sim-ur5
  sim-ur5-vision
  sim-h12
  sim-h12-ur5
  real-hand
  real-ur5
  real-ur5-ros
  real-h12-ros
  real-h12-hand-ros
  dev-full

Compatibility aliases:
  real-robot
  full
EOF
}

add_extra() {
  local extra="$1"
  local existing
  for existing in "${EXTRAS[@]}"; do
    if [[ "$existing" == "$extra" ]]; then
      return
    fi
  done
  EXTRAS+=("$extra")
}

add_submodule() {
  local submodule="$1"
  local existing
  for existing in "${SUBMODULES[@]}"; do
    if [[ "$existing" == "$submodule" ]]; then
      return
    fi
  done
  SUBMODULES+=("$submodule")
}

warn_missing_submodules() {
  local missing=()
  local submodule
  for submodule in "${SUBMODULES[@]}"; do
    if [[ ! -d "$submodule" ]] || [[ -z "$(find "$submodule" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      missing+=("$submodule")
    fi
  done

  if [[ ${#missing[@]} -eq 0 ]]; then
    return
  fi

  echo "[setup_uv_env] WARNING: required submodule(s) look empty: ${missing[*]}"
  echo "[setup_uv_env] Initialize them with:"
  echo "  git submodule update --init --recursive --depth 1 ${missing[*]}"
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
    --list)
      print_profiles
      exit 0
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
SUBMODULES=()
USES_MAGPIE=0
case "$PROFILE" in
  sim-core)
    add_submodule "h1_mujoco"
    ;;
  sim-hand)
    add_extra "sim-hand"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    ;;
  sim-ur5)
    add_extra "sim-ur5"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    ;;
  sim-ur5-vision)
    add_extra "sim-ur5-vision"
    add_submodule "h1_mujoco"
    ;;
  sim-h12)
    add_extra "sim-h12"
    add_submodule "h1_mujoco"
    ;;
  sim-h12-ur5)
    add_extra "sim-h12-ur5"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    USES_MAGPIE=1
    ;;
  real-hand)
    add_extra "real-hand"
    add_submodule "h1_mujoco"
    ;;
  real-ur5)
    add_extra "real-ur5"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    USES_MAGPIE=1
    ;;
  real-robot)
    add_extra "real-robot"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    USES_MAGPIE=1
    ;;
  real-ur5-ros)
    add_extra "real-ur5-ros"
    add_extra "ros"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    USES_MAGPIE=1
    ;;
  real-h12-ros)
    add_extra "real-h12-ros"
    add_extra "ros"
    add_submodule "h1_mujoco"
    ;;
  real-h12-hand-ros)
    add_extra "real-h12-hand-ros"
    add_extra "ros"
    add_submodule "h1_mujoco"
    ;;
  dev-full)
    add_extra "dev-full"
    add_extra "ros"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    add_submodule "magpie_force_control"
    add_submodule "rerun_rlds_ur5"
    USES_MAGPIE=1
    ;;
  full)
    add_extra "full"
    add_extra "ros"
    add_submodule "h1_mujoco"
    add_submodule "mink"
    add_submodule "magpie_control"
    add_submodule "magpie_force_control"
    add_submodule "rerun_rlds_ur5"
    USES_MAGPIE=1
    ;;
  *)
    echo "Unknown profile: $PROFILE"
    usage
    exit 1
    ;;
esac

if [[ $ADD_TELEMETRY -eq 1 ]]; then
  add_extra "telemetry"
fi

warn_missing_submodules

if [[ ! -d "$ENV_DIR" ]]; then
  uv venv --python "$PYTHON_VERSION" "$ENV_DIR"
fi

SYNC_CMD=(uv sync)
for ex in "${EXTRAS[@]}"; do
  SYNC_CMD+=(--extra "$ex")
done

if [[ "$USES_MAGPIE" -eq 1 ]] && [[ "$(uname -s)" == "Darwin" ]] && [[ "$(uname -m)" == "arm64" ]]; then
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
if [[ ${#SUBMODULES[@]} -gt 0 ]]; then
  echo "[setup_uv_env] Submods: ${SUBMODULES[*]}"
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
