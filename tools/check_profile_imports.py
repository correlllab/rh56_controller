#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "rh56_controller_matplotlib"),
)

BASE_MODULES = ("numpy", "scipy", "matplotlib", "mujoco")
H12_MODULES = ("pinocchio", "pink", "meshcat", "meshcat_shapes", "daqp", "qpsolvers")
UR5_MODULES = ("magpie_control", "spatialmath")


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    python: str
    extras: tuple[str, ...]
    modules: tuple[str, ...]
    paths: tuple[str, ...]
    command: str
    note: str = ""


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec(
        name="sim-core",
        python="3.12",
        extras=(),
        modules=BASE_MODULES,
        paths=("h1_mujoco",),
        command="uv run python -m rh56_controller.grasp_viz --no-mink",
        note="Base import/runtime smoke test; sim-hand is the normal user profile.",
    ),
    ProfileSpec(
        name="sim-hand",
        python="3.12",
        extras=("sim-hand",),
        modules=BASE_MODULES + ("mink",),
        paths=("h1_mujoco", "mink"),
        command="uv run python -m rh56_controller.grasp_viz",
    ),
    ProfileSpec(
        name="sim-ur5",
        python="3.12",
        extras=("sim-ur5",),
        modules=BASE_MODULES + ("mink",),
        paths=("h1_mujoco", "mink"),
        command="uv run python -m rh56_controller.grasp_viz --robot",
    ),
    ProfileSpec(
        name="sim-ur5-vision",
        python="3.12",
        extras=("sim-ur5-vision",),
        modules=BASE_MODULES + ("cv2",),
        paths=("h1_mujoco",),
        command=(
            "python tools/calibrate_ur5_external_camera_sim.py "
            "--out artifacts/ur5_external_camera_calibration"
        ),
    ),
    ProfileSpec(
        name="sim-h12",
        python="3.12",
        extras=("sim-h12",),
        modules=BASE_MODULES + H12_MODULES,
        paths=("h1_mujoco",),
        command="uv run python -m rh56_controller.grasp_viz --h12",
    ),
    ProfileSpec(
        name="sim-h12-ur5",
        python="3.12",
        extras=("sim-h12-ur5",),
        modules=BASE_MODULES + ("mink",) + UR5_MODULES + H12_MODULES,
        paths=("h1_mujoco", "mink", "magpie_control"),
        command="uv run python -m rh56_controller.grasp_viz --h12",
    ),
    ProfileSpec(
        name="real-hand",
        python="3.12",
        extras=("real-hand",),
        modules=BASE_MODULES + ("serial",),
        paths=("h1_mujoco",),
        command="uv run python -m rh56_controller.hand_mirror --port /dev/ttyUSB0",
    ),
    ProfileSpec(
        name="real-ur5",
        python="3.12",
        extras=("real-ur5",),
        modules=BASE_MODULES + ("serial", "mink") + UR5_MODULES,
        paths=("h1_mujoco", "mink", "magpie_control"),
        command=(
            "uv run python -m rh56_controller.grasp_viz --robot "
            "--real-robot --ur5-ip 192.168.0.4 --port /dev/ttyUSB0"
        ),
    ),
    ProfileSpec(
        name="real-ur5-vision",
        python="3.12",
        extras=("real-ur5-vision",),
        modules=BASE_MODULES + ("cv2", "pyrealsense2", "rtde_receive"),
        paths=(),
        command=(
            "python tools/calibrate_ur5_external_camera_real.py --check-only"
        ),
        note="Read-only camera/RTDE receive path; never sends motion commands.",
    ),
    ProfileSpec(
        name="real-ur5-ros",
        python="3.10",
        extras=("real-ur5-ros", "ros"),
        modules=BASE_MODULES + ("serial", "mink") + UR5_MODULES + ("rclpy",),
        paths=("h1_mujoco", "mink", "magpie_control"),
        command=(
            "ros2 launch rh56_controller grasp_viz_ros.launch.py "
            "ur5_ip:=192.168.0.4 serial_port:=/dev/ttyUSB0"
        ),
    ),
    ProfileSpec(
        name="real-h12-ros",
        python="3.10",
        extras=("real-h12-ros", "ros"),
        modules=BASE_MODULES + H12_MODULES + ("rclpy",),
        paths=("h1_mujoco",),
        command="uv run python -m rh56_controller.grasp_viz --h12 --real-h12",
        note="Requires the external h12_ros2_controller workspace to be sourced.",
    ),
    ProfileSpec(
        name="real-h12-hand-ros",
        python="3.10",
        extras=("real-h12-hand-ros", "ros"),
        modules=BASE_MODULES + ("serial",) + H12_MODULES + ("rclpy",),
        paths=("h1_mujoco",),
        command=(
            "uv run python -m rh56_controller.grasp_viz "
            "--h12 --real-h12 --port /dev/ttyUSB0"
        ),
        note="Requires the external h12_ros2_controller workspace to be sourced.",
    ),
    ProfileSpec(
        name="dev-full",
        python="3.10 or 3.12",
        extras=("dev-full", "ros"),
        modules=(
            BASE_MODULES
            + ("serial", "mink")
            + UR5_MODULES
            + H12_MODULES
            + ("rclpy", "rerun")
        ),
        paths=(
            "h1_mujoco",
            "mink",
            "magpie_control",
            "magpie_force_control",
            "rerun_rlds_ur5",
        ),
        command="Use the first command for the workflow you are testing.",
    ),
    ProfileSpec(
        name="real-robot",
        python="3.12",
        extras=("real-robot",),
        modules=BASE_MODULES + ("serial", "mink") + UR5_MODULES,
        paths=("h1_mujoco", "mink", "magpie_control"),
        command="Compatibility alias for real-ur5.",
        note="Prefer profile real-ur5 in new docs and scripts.",
    ),
    ProfileSpec(
        name="full",
        python="3.10 or 3.12",
        extras=("full", "ros"),
        modules=(
            BASE_MODULES
            + ("serial", "mink")
            + UR5_MODULES
            + H12_MODULES
            + ("rclpy", "rerun")
        ),
        paths=(
            "h1_mujoco",
            "mink",
            "magpie_control",
            "magpie_force_control",
            "rerun_rlds_ur5",
        ),
        command="Compatibility alias for dev-full.",
        note="Prefer profile dev-full in new docs and scripts.",
    ),
)

PROFILE_MAP = {spec.name: spec for spec in PROFILE_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate import and local-asset availability for install profiles."
    )
    parser.add_argument(
        "--profile",
        default="sim-core",
        help="Profile to validate, or 'all' to validate every profile.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported profiles, extras, and first-run commands.",
    )
    parser.add_argument(
        "--no-paths",
        action="store_true",
        help="Only validate Python imports; skip required local path checks.",
    )
    parser.add_argument(
        "--paths-only",
        action="store_true",
        help="Only validate required local paths; skip Python import checks.",
    )
    return parser.parse_args()


def try_import(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def path_status(rel_path: str) -> tuple[bool, str]:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return False, "missing"
    if path.is_dir():
        try:
            next(path.iterdir())
        except StopIteration:
            return False, "empty"
        except OSError as exc:
            return False, str(exc)
    return True, "ok"


def print_profile_list() -> None:
    print("Supported profiles:")
    for spec in PROFILE_SPECS:
        extras = ", ".join(spec.extras) if spec.extras else "(base)"
        print(f"\n- {spec.name}")
        print(f"  python: {spec.python}")
        print(f"  extras: {extras}")
        print(f"  command: {spec.command}")
        if spec.note:
            print(f"  note: {spec.note}")


def validate_profile(profile_name: str, check_imports: bool, check_paths: bool) -> bool:
    spec = PROFILE_MAP[profile_name]
    print(f"\n[check_profile_imports] profile={spec.name}")
    print(f"  python: {spec.python}")
    print(f"  extras: {', '.join(spec.extras) if spec.extras else '(base)'}")
    print(f"  command: {spec.command}")
    if spec.note:
        print(f"  note: {spec.note}")

    failed_modules: list[str] = []
    if check_imports:
        for module_name in spec.modules:
            ok, error = try_import(module_name)
            if ok:
                print(f"  [ok] import {module_name}")
            else:
                print(f"  [missing] import {module_name} ({error})")
                failed_modules.append(module_name)

    failed_paths: list[str] = []
    if check_paths:
        for rel_path in spec.paths:
            ok, status = path_status(rel_path)
            if ok:
                print(f"  [ok] path {rel_path}")
            else:
                print(f"  [missing] path {rel_path} ({status})")
                failed_paths.append(rel_path)

    if failed_modules or failed_paths:
        if failed_modules:
            print(
                f"[check_profile_imports] missing imports: {', '.join(failed_modules)}"
            )
        if failed_paths:
            print(
                "[check_profile_imports] missing/empty paths: "
                + ", ".join(failed_paths)
            )
            print(
                "[check_profile_imports] initialize submodules with: "
                "git submodule update --init --recursive --depth 1 "
                + " ".join(failed_paths)
            )
        print(f"[check_profile_imports] FAIL ({spec.name})")
        return False

    print(f"[check_profile_imports] PASS ({spec.name})")
    return True


def main() -> int:
    args = parse_args()
    if args.list:
        print_profile_list()
        if args.profile == "sim-core":
            return 0
        print("")

    target = args.profile.strip()
    if args.no_paths and args.paths_only:
        print("--no-paths and --paths-only cannot be used together.", file=sys.stderr)
        return 2

    check_imports = not args.paths_only
    check_paths = not args.no_paths

    if target == "all":
        all_ok = True
        for spec in PROFILE_SPECS:
            all_ok = validate_profile(spec.name, check_imports, check_paths) and all_ok
        return 0 if all_ok else 1

    if target not in PROFILE_MAP:
        print(f"Unknown profile: {target}", file=sys.stderr)
        print("Use --list to show supported profiles.", file=sys.stderr)
        return 2

    ok = validate_profile(target, check_imports, check_paths)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
