#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ProfileSpec:
    name: str
    modules: tuple[str, ...]


PROFILE_SPECS: tuple[ProfileSpec, ...] = (
    ProfileSpec("sim-core", ("numpy", "scipy", "matplotlib", "mujoco")),
    ProfileSpec("sim-hand", ("numpy", "scipy", "matplotlib", "mujoco", "mink")),
    ProfileSpec("sim-ur5", ("numpy", "scipy", "matplotlib", "mujoco", "mink")),
    ProfileSpec(
        "sim-h12",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "pinocchio",
            "pink",
            "meshcat",
            "meshcat_shapes",
        ),
    ),
    ProfileSpec(
        "sim-h12-ur5",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "mink",
            "magpie_control",
            "spatialmath",
            "pinocchio",
            "pink",
            "meshcat",
            "meshcat_shapes",
        ),
    ),
    ProfileSpec(
        "real-ur5",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "serial",
            "mink",
            "magpie_control",
            "spatialmath",
        ),
    ),
    ProfileSpec(
        "real-ur5-ros",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "serial",
            "mink",
            "magpie_control",
            "spatialmath",
            "rclpy",
        ),
    ),
    ProfileSpec(
        "real-h12-ros",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "pinocchio",
            "pink",
            "meshcat",
            "meshcat_shapes",
            "rclpy",
        ),
    ),
    ProfileSpec(
        "real-h12-hand-ros",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "serial",
            "pinocchio",
            "pink",
            "meshcat",
            "meshcat_shapes",
            "rclpy",
        ),
    ),
    ProfileSpec(
        "full",
        (
            "numpy",
            "scipy",
            "matplotlib",
            "mujoco",
            "serial",
            "mink",
            "magpie_control",
            "spatialmath",
            "pinocchio",
            "pink",
            "meshcat",
            "meshcat_shapes",
            "rerun",
        ),
    ),
)

PROFILE_MAP = {spec.name: spec for spec in PROFILE_SPECS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate import availability for uv dependency profiles."
    )
    parser.add_argument(
        "--profile",
        default="sim-core",
        help="Profile to validate, or 'all' to validate every profile (default: sim-core).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List supported profiles and expected imports.",
    )
    return parser.parse_args()


def try_import(module_name: str) -> tuple[bool, str | None]:
    try:
        importlib.import_module(module_name)
        return True, None
    except Exception as exc:
        return False, str(exc)


def print_profile_list() -> None:
    print("Supported profiles:")
    for spec in PROFILE_SPECS:
        print(f"  - {spec.name}: {', '.join(spec.modules)}")


def validate_profile(profile_name: str) -> bool:
    spec = PROFILE_MAP[profile_name]
    print(f"\n[check_profile_imports] profile={spec.name}")
    failed = []
    for module_name in spec.modules:
        ok, error = try_import(module_name)
        if ok:
            print(f"  ✅ {module_name}")
        else:
            print(f"  ❌ {module_name} ({error})")
            failed.append(module_name)

    if failed:
        print(
            f"[check_profile_imports] FAIL ({spec.name}): missing {len(failed)} module(s): {', '.join(failed)}"
        )
        return False

    print(f"[check_profile_imports] PASS ({spec.name})")
    return True


def main() -> int:
    args = parse_args()
    if args.list:
        print_profile_list()
        if args.profile != "sim-core":
            print("")

    target = args.profile.strip()
    if target == "all":
        all_ok = True
        for spec in PROFILE_SPECS:
            all_ok = validate_profile(spec.name) and all_ok
        return 0 if all_ok else 1

    if target not in PROFILE_MAP:
        print(f"Unknown profile: {target}", file=sys.stderr)
        print("Use --list to show supported profiles.", file=sys.stderr)
        return 2

    ok = validate_profile(target)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
