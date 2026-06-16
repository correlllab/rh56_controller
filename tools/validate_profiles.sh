#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  tools/validate_profiles.sh [options]

Default behavior:
  Set up and validate the main sim profiles:
    sim-hand sim-ur5 sim-h12

Options:
  --profile NAME       Validate one profile. Can be repeated.
  --profiles "LIST"   Validate a space- or comma-separated profile list.
  --python VERSION    Python version for created envs (default: 3.12).
  --env-prefix PREFIX Env path prefix; env is PREFIX-PROFILE.
                      Default envs match the README examples for main sim profiles.
  --check-only        Do not run setup_uv_env.sh; validate existing envs only.
  --paths-only        Only check required submodules/assets; no env setup/import checks.
  --telemetry         Pass --telemetry when setting up envs.
  --keep-going        Continue after profile failures and report all failures.
  -h, --help          Show this help.

Examples:
  tools/validate_profiles.sh
  tools/validate_profiles.sh --check-only
  tools/validate_profiles.sh --profiles "sim-hand,sim-ur5,sim-h12" --python 3.12
  tools/validate_profiles.sh --profile all --paths-only
EOF
}

split_profiles() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  PROFILES=($raw)
}

env_for_profile() {
  local profile="$1"
  local py_suffix="${PYTHON_VERSION//./}"

  if [[ -n "$ENV_PREFIX" ]]; then
    echo "${ENV_PREFIX}-${profile}"
    return
  fi

  case "$profile" in
    sim-hand)
      echo ".venv${py_suffix}"
      ;;
    sim-core)
      echo ".venv${py_suffix}-core"
      ;;
    sim-ur5)
      echo ".venv${py_suffix}-ur5"
      ;;
    sim-h12)
      echo ".venv${py_suffix}-h12"
      ;;
    *)
      echo ".venv${py_suffix}-${profile}"
      ;;
  esac
}

PYTHON_VERSION="3.12"
ENV_PREFIX=""
CHECK_ONLY=0
PATHS_ONLY=0
ADD_TELEMETRY=0
KEEP_GOING=0
PROFILES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILES+=("${2:-}")
      shift 2
      ;;
    --profiles)
      split_profiles "${2:-}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:-}"
      shift 2
      ;;
    --env-prefix)
      ENV_PREFIX="${2:-}"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --paths-only)
      PATHS_ONLY=1
      shift
      ;;
    --telemetry)
      ADD_TELEMETRY=1
      shift
      ;;
    --keep-going)
      KEEP_GOING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ${#PROFILES[@]} -eq 0 ]]; then
  PROFILES=(sim-hand sim-ur5 sim-h12)
fi

if [[ "$PATHS_ONLY" -eq 1 ]]; then
  for profile in "${PROFILES[@]}"; do
    python3 tools/check_profile_imports.py --profile "$profile" --paths-only
  done
  exit 0
fi

if ! command -v uv >/dev/null 2>&1 && [[ "$CHECK_ONLY" -eq 0 ]]; then
  echo "uv not found. Install uv first: https://docs.astral.sh/uv/" >&2
  exit 1
fi

failures=()

for profile in "${PROFILES[@]}"; do
  env_dir="$(env_for_profile "$profile")"
  echo ""
  echo "[validate_profiles] profile=$profile env=$env_dir"

  if [[ "$CHECK_ONLY" -eq 0 ]]; then
    setup_cmd=(tools/setup_uv_env.sh --profile "$profile" --python "$PYTHON_VERSION" --env "$env_dir")
    if [[ "$ADD_TELEMETRY" -eq 1 ]]; then
      setup_cmd+=(--telemetry)
    fi
    "${setup_cmd[@]}" || {
      failures+=("$profile:setup")
      if [[ "$KEEP_GOING" -eq 0 ]]; then
        break
      fi
      continue
    }
  fi

  python_bin="$env_dir/bin/python"
  if [[ ! -x "$python_bin" ]]; then
    echo "[validate_profiles] missing env Python: $python_bin" >&2
    failures+=("$profile:env")
    if [[ "$KEEP_GOING" -eq 0 ]]; then
      break
    fi
    continue
  fi

  "$python_bin" tools/check_profile_imports.py --profile "$profile" || {
    failures+=("$profile:check")
    if [[ "$KEEP_GOING" -eq 0 ]]; then
      break
    fi
  }
done

if [[ ${#failures[@]} -gt 0 ]]; then
  echo ""
  echo "[validate_profiles] FAIL: ${failures[*]}" >&2
  exit 1
fi

echo ""
echo "[validate_profiles] PASS: ${PROFILES[*]}"
