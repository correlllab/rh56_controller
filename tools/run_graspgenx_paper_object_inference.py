#!/usr/bin/env python3
"""Run one loaded GraspGen-X model over the 15 RH56 paper-object proxies.

This script must be executed with the independent GraspGen-X environment. It
loads the official generator/discriminator once, uses deterministic per-object
random seeds, and writes one Isaac-grasp YAML candidate set per object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


RH56_ROOT = Path(__file__).resolve().parents[1]
if str(RH56_ROOT) not in sys.path:
    sys.path.insert(0, str(RH56_ROOT))

from tools.run_strategy_pregrasp_paper_batch import PAPER_OBJECTS  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic GraspGen-X candidates for the 15 RH56 object proxies."
    )
    parser.add_argument(
        "--graspgenx-root",
        type=Path,
        default=Path("/home/tanxuan/workspace/GraspGenX"),
    )
    parser.add_argument(
        "--mesh-dir",
        type=Path,
        default=RH56_ROOT / "artifacts/paper_15_object_grasp_success/proxy_meshes",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RH56_ROOT / "artifacts/paper_15_object_grasp_success/graspgenx_candidates",
    )
    parser.add_argument("--seed", type=int, default=20260309)
    parser.add_argument("--num-grasps", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--gripper", default="inspire_hand")
    parser.add_argument("--checkpoints", type=Path, default=None)
    return parser.parse_args(argv)


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_grasps < 1 or args.topk < 1:
        raise ValueError("--num-grasps and --topk must be positive")
    if not args.graspgenx_root.is_dir():
        raise FileNotFoundError(args.graspgenx_root)

    scripts_dir = args.graspgenx_root / "scripts"
    for entry in (str(args.graspgenx_root), str(scripts_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    import torch
    import trimesh.transformations as tra
    from demo_object_mesh import load_mesh_data
    from demo_object_pc import _resolve_default_checkpoints, load_model_cfg
    from graspgenx.dataset.eval_utils import save_to_isaac_grasp_format
    from graspgenx.grasp_server import GraspGenXSampler

    if not torch.cuda.is_available():
        raise RuntimeError("GraspGen-X inference requires a CUDA GPU")

    os.environ.setdefault("GRASPGENX_CHECKPOINT_DIR", str(args.graspgenx_root / "ext/graspgenx_checkpoints"))
    checkpoint_root = str(args.checkpoints) if args.checkpoints else _resolve_default_checkpoints()
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
        None,
        None,
    )
    assets_dir = args.graspgenx_root / "assets"
    sampler = GraspGenXSampler(model_cfg, args.gripper, assets_dir=str(assets_dir))

    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, object_name in enumerate(PAPER_OBJECTS):
        object_seed = args.seed + index
        np.random.seed(object_seed)
        torch.manual_seed(object_seed)
        torch.cuda.manual_seed_all(object_seed)
        mesh_path = args.mesh_dir / f"{object_name}.obj"
        if not mesh_path.is_file():
            raise FileNotFoundError(mesh_path)

        point_cloud, _colors, _mesh, recenter = load_mesh_data(
            str(mesh_path), 1.0, 3500
        )
        start = time.perf_counter()
        poses, confidences = GraspGenXSampler.run_inference(
            point_cloud,
            sampler,
            grasp_threshold=-1.0,
            num_grasps=args.num_grasps,
            topk_num_grasps=args.topk,
            remove_outliers=False,
        )
        elapsed = time.perf_counter() - start
        poses_np = poses.detach().cpu().numpy()
        confidence_np = confidences.detach().cpu().numpy()
        poses_np[:, 3, 3] = 1.0
        original_frame = np.array(
            [tra.inverse_matrix(recenter) @ pose for pose in poses_np]
        )
        output_path = args.out / f"{object_name}.yml"
        save_to_isaac_grasp_format(original_frame, confidence_np, str(output_path))
        row = {
            "object": object_name,
            "seed": object_seed,
            "candidates": int(len(original_frame)),
            "confidence_min": float(confidence_np.min()),
            "confidence_max": float(confidence_np.max()),
            "inference_seconds": elapsed,
            "yaml": str(output_path.resolve()),
            "sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        }
        rows.append(row)
        print(
            f"{object_name:25s}: {len(original_frame):3d} candidates, "
            f"score {confidence_np.min():.3f}-{confidence_np.max():.3f}, {elapsed:.2f}s"
        )

    manifest = {
        "graspgenx_revision": _git_revision(args.graspgenx_root),
        "checkpoint_root": checkpoint_root,
        "gripper": args.gripper,
        "base_seed": args.seed,
        "num_grasps": args.num_grasps,
        "topk": args.topk,
        "mesh_manifest": str((args.mesh_dir / "manifest.json").resolve()),
        "objects": rows,
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
