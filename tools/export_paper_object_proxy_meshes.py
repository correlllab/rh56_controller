#!/usr/bin/env python3
"""Export the 15 paper-object primitive proxies as centered OBJ meshes.

The meshes are inputs to the pretrained GraspGen-X baseline. They preserve the
dimensions and box/cylinder/sphere assumptions already used by the RH56
collision-grid experiment; they are not scans of the physical objects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS  # noqa: E402
from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the 15 RH56 paper-object primitive proxies as OBJ meshes."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/paper_15_object_grasp_success/proxy_meshes"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import trimesh
    except ImportError as exc:
        raise SystemExit(
            "Mesh export requires trimesh. Run this script with the GraspGen-X environment."
        ) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for name in PAPER_OBJECTS:
        obj = BUILTIN_OBJECTS[name]
        if obj.collision_shape == "box":
            mesh = trimesh.creation.box(extents=obj.size_m)
        elif obj.collision_shape == "cylinder":
            mesh = trimesh.creation.cylinder(
                radius=min(obj.size_m[0], obj.size_m[1]) / 2.0,
                height=obj.size_m[2],
                sections=64,
            )
        elif obj.collision_shape == "sphere":
            mesh = trimesh.creation.icosphere(
                subdivisions=3,
                radius=min(obj.size_m) / 2.0,
            )
        else:
            raise ValueError(f"Unsupported proxy shape: {obj.collision_shape}")

        path = args.out / f"{name}.obj"
        mesh.export(path)
        rows.append(
            {
                "object": name,
                "label": obj.label,
                "shape": obj.collision_shape,
                "size_m": list(obj.size_m),
                "grasp_width_m": obj.grasp_width_m,
                "mesh": str(path.resolve()),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "vertices": int(len(mesh.vertices)),
                "faces": int(len(mesh.faces)),
            }
        )
        print(f"{name:25s} -> {path}")

    manifest = {
        "simulation_only": True,
        "measured_meshes": False,
        "warning": (
            "These are centered primitive proxies derived from estimated object dimensions, "
            "not measured/scanned geometry from the physical 15-object experiment."
        ),
        "objects": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
