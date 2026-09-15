#!/usr/bin/env python3
"""Extend a partial-grid GraspGen-X sweep over the full quarter-ball grid.

The k = 100 can sweep was run on the runner's default extents (xy +-240 mm,
z 0-330 mm), which clip the 300 mm quarter ball the figure draws. Its
unreachable cells sit low and close to the object, inside the clipped grid, so
the figure's GraspGen-X layer is built by writing that result onto the full
grid and drawing the cells outside the clipped grid, which are far from the
object, as reachable. Every other layer is copied from the full-grid sweep.

The output carries an assumptions file that says exactly this, and the
renderer reads it through --tag.

Usage:
    uv run python merge_ggx_volume.py --object paper_can \\
        --full ../results/nogo_volume_paper_can.npz \\
        --partial ../results/ggx100/nogo_volume_paper_can.npz \\
        --out ../results/nogo_volume_paper_can_ggx100merged.npz
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--object", required=True)
    ap.add_argument("--full", type=Path, required=True, help="Full-grid sweep (.npz).")
    ap.add_argument("--partial", type=Path, required=True,
                    help="Sweep whose GraspGen-X layers replace the full sweep's.")
    ap.add_argument("--layers", nargs="+", default=["ggx_best", "ggx_worst"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--snap-mm", type=float, default=15.0,
                    help="Largest per-axis distance to a partial-grid cell that still "
                         "counts as the same cell; half the 30 mm step by default.")
    args = ap.parse_args()

    full = dict(np.load(args.full))
    part = np.load(args.partial)
    axes = [(full["xs_mm"], part["xs_mm"]), (full["ys_mm"], part["ys_mm"]),
            (full["zs_mm"], part["zs_mm"])]
    # For each full-grid coordinate, the index of the nearest partial-grid
    # coordinate, or -1 when none lies within --snap-mm.
    index = []
    for fa, pa in axes:
        near = np.abs(fa[:, None] - pa[None, :])
        j = near.argmin(axis=1)
        j[near.min(axis=1) > args.snap_mm] = -1
        index.append(j)
    ix, iy, iz = np.meshgrid(*index, indexing="ij")
    covered = (ix >= 0) & (iy >= 0) & (iz >= 0)
    # A partial-grid cell outside that sweep's own sampled region was never
    # evaluated and is stored as not admissible; it does not cover anything.
    part_sampled = part["sampled"].astype(bool)
    covered[covered] = part_sampled[ix[covered], iy[covered], iz[covered]]
    sampled = full["sampled"].astype(bool)

    report = {}
    for layer in args.layers:
        key = f"admissible_{layer}"
        grid = np.ones(sampled.shape, dtype=bool)
        src = part[key].astype(bool)
        grid[covered] = src[ix[covered], iy[covered], iz[covered]]
        if layer == "ggx_worst":
            # "Every proposal executes" held nowhere on the partial grid, so it is
            # not credited anywhere outside it either.
            grid[~covered] = False
        full[key] = grid
        n_cov = int((sampled & covered).sum())
        report[layer] = {
            "unreachable_in_covered": int((sampled & covered & ~grid).sum()),
            "covered_cells": n_cov,
            "assumed_reachable_outside": int((sampled & ~covered).sum()) if layer != "ggx_worst" else 0,
            "locked_fraction": float((sampled & ~grid).sum() / sampled.sum()),
        }
        floors = full.get(f"floor_mm_{layer}")
        if floors is not None:
            # Lowest admissible height per column, as the sweep would have written it.
            zs = full["zs_mm"]
            hits = grid & sampled
            has = hits.any(axis=2)
            full[f"floor_mm_{layer}"] = np.where(has, zs[np.argmax(hits, axis=2)],
                                                 zs[-1] + (zs[1] - zs[0]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **full)
    base = args.full.with_name(args.full.stem + "_assumptions.json")
    assumptions = json.loads(base.read_text()) if base.exists() else {}
    part_ap = args.partial.with_name(args.partial.stem + "_assumptions.json")
    part_assumptions = json.loads(part_ap.read_text()) if part_ap.exists() else {}
    assumptions.update({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "merged_from": {
            "full_grid": str(args.full), "partial_grid": str(args.partial),
            "layers_from_partial": args.layers,
            "partial_grid_extent": part_assumptions.get("grid"),
            "cells_outside_partial_drawn_reachable": True,
            "snap_mm": args.snap_mm,
        },
        "ggx_topk": part_assumptions.get("ggx_topk"),
        "n_ggx_candidates": part_assumptions.get("n_ggx_candidates"),
        "merge_report": report,
    })
    args.out.with_name(args.out.stem + "_assumptions.json").write_text(
        json.dumps(assumptions, indent=2))
    for layer, r in report.items():
        print(f"  {layer:10s} unreachable {r['unreachable_in_covered']} of {r['covered_cells']} "
              f"covered cells, {r['assumed_reachable_outside']} outside drawn reachable, "
              f"locked {r['locked_fraction']:.1%} of {int(sampled.sum())}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
