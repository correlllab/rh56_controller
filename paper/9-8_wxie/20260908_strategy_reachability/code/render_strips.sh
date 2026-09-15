#!/usr/bin/env bash
# Re-render the four no-go strips for the doc page and compose the paper
# figure from their bare panels. The camera angles and the framing margin are
# part of the figure, not defaults, so they live here rather than in anyone's
# shell history.
set -e
cd "$(dirname "$0")"

COMMON="--strip --volume-mode voxel --smooth --object-pin --strip-margin 1.04 \
  --scale-bar-mm 100 --elevation -34 --azimuth 235 --plain-floor --width 460 --height 400"
PANELS=../figures/panels

MUJOCO_GL=egl uv run python render_nogo_volume.py --object paper_strawberry --pin-height-mm 90 \
  --layers iterative_closure thumb_reflex naive \
  --out ../figures/20260908-nogo_strip_strawberry.png $COMMON
MUJOCO_GL=egl uv run python render_nogo_volume.py --object paper_strawberry --pin-height-mm 90 \
  --layers iterative_closure thumb_reflex naive ggx_best --panels-dir $PANELS \
  --out ../figures/20260908-nogo_strip_strawberry_ggx.png $COMMON
MUJOCO_GL=egl uv run python render_nogo_volume.py --object paper_can --pin-height-mm 190 \
  --layers iterative_closure thumb_reflex naive \
  --out ../figures/20260908-nogo_strip_can.png $COMMON
MUJOCO_GL=egl uv run python render_nogo_volume.py --object paper_can --pin-height-mm 190 \
  --layers iterative_closure thumb_reflex naive ggx_best --panels-dir $PANELS \
  --out ../figures/20260908-nogo_strip_can_ggx.png $COMMON

# The paper figure: both objects in one lettered grid, strategies named once.
uv run python compose_nogo_figure.py --panels-dir $PANELS \
  --row A Strawberry 20260908-nogo_strip_strawberry_ggx \
  --row B Can 20260908-nogo_strip_can_ggx \
  --out ../figures/20260908-nogo_grid_strawberry_can.png --pdf
