#!/usr/bin/env python3
"""Render the no-go start volume of each strategy around one object.

Reads the 3-D sweep written by run_nogo_volume.py and turns each strategy's
locked-out region into a solid: for every column of the start-pose grid the
lowest admissible height is found, and the volume beneath it is extruded into a
closed mesh.  Standing anywhere inside that solid, the strategy cannot execute
its commanded approach and closure without hitting the object or the table.

The solids are added to the MuJoCo grasp scene alongside the object proxy and
rendered together, so the volume is shown against the hand and object that
produced it rather than as an abstract plot.

Usage:
    MUJOCO_GL=egl uv run python render_nogo_volume.py --object paper_nut \\
        --out ../figures/20260908-nogo_volumes_nut.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mj_snap_rh56 import VIEWS, add_object_geom, label, render, set_scene_state, tile  # noqa: E402
from strategy_admissibility import (  # noqa: E402
    ClosureGeometry,
    InspireHandFK,
    build_strategy_plans,
    grasp_yaw_for_object,
)
from rh56_controller.paper_v2_objects import BUILTIN_OBJECTS, tabletop_grasp_target  # noqa: E402

# Categorical slots 1-4 of the reference palette, plus a muted tone for the
# pessimistic GraspGen-X layer so the pair reads as one planner.
LAYER_STYLE = {
    "naive": ("Naive", (0.165, 0.471, 0.839)),
    "iterative_closure": ("Iterative", (0.922, 0.408, 0.204)),
    "thumb_reflex": ("Reflex", (0.106, 0.686, 0.478)),
    "ggx_best": ("GraspGen-X, best of k", (0.929, 0.631, 0.000)),
    "ggx_worst": ("GraspGen-X, all of k", (0.910, 0.286, 0.282)),
}


# A strip panel is 460 px wide and carries its own name, so the legend label
# for the GraspGen-X layers is split: the planner names the panel and the
# candidate rule moves to the second line.
STRIP_LABEL = {
    "ggx_best": ("GraspGen-X", "best of k, "),
    "ggx_worst": ("GraspGen-X", "all of k, "),
}

# Every panel of a strip shows the same quantity, the start poses one strategy
# cannot reach from, so the panels share one colour and the reader compares
# extent instead of matching hues. The per-strategy palette above is kept for
# the nested overlay, where colour is the only thing separating the layers.
NOGO_RGB = (0.788, 0.114, 0.110)

# The complement of the no-go solid inside the sampled region. Drawing it says
# what the red is a fraction of: without it a small red solid could be a small
# unreachable set or a small sample.
REACH_RGB = (0.145, 0.635, 0.322)

# The object proxy is drawn in the colour of the real object so that the thing
# being grasped reads as an object rather than as one more region. The proxy is
# still the axis-aligned cylinder or box the admissibility model collides
# against; only its colour is taken from the real item.
OBJECT_RGB = {
    "paper_can": (0.60, 0.63, 0.68),
    "paper_strawberry": (0.62, 0.04, 0.10),
}

# Groups 3 to 5 are hidden by default and the hand scene keeps its collision
# geoms in group 3, so the no-go solid goes in group 1, which the scene leaves
# empty. Clearing that one group removes the solid and nothing else.
VOLUME_GROUP = 1


def floor_heights(admissible: np.ndarray, zs: np.ndarray) -> np.ndarray:
    """Lowest admissible height per column, or above the grid when never admissible."""

    n_x, n_y, n_z = admissible.shape
    ceiling = float(zs[-1] + (zs[1] - zs[0]))
    out = np.full((n_x, n_y), ceiling, dtype=float)
    for i in range(n_x):
        for j in range(n_y):
            hits = np.flatnonzero(admissible[i, j])
            if hits.size:
                out[i, j] = float(zs[hits[0]])
    return out


def build_extruded_mesh(
    xs: np.ndarray, ys: np.ndarray, heights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Closed mesh of the solid under a height field.

    Returns (vertices about the centroid, triangle indices, centroid).  The
    vertices are handed to MuJoCo directly rather than written as OBJ: its OBJ
    importer applies a coordinate convention of its own, which silently permutes
    the axes of a mesh authored in world coordinates.  MuJoCo also re-centres a
    mesh asset on its own centroid, so the caller places the body at the returned
    centroid.  None when the region is empty.
    """

    if not np.any(heights > 1e-9):
        return None

    n_x, n_y = len(xs), len(ys)
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    top = np.empty((n_x, n_y), dtype=int)
    bottom = np.empty((n_x, n_y), dtype=int)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            top[i, j] = len(verts)
            verts.append((float(x), float(y), float(heights[i, j])))
            bottom[i, j] = len(verts)
            verts.append((float(x), float(y), 0.0))

    for i in range(n_x - 1):
        for j in range(n_y - 1):
            a, b, c, d = top[i, j], top[i + 1, j], top[i + 1, j + 1], top[i, j + 1]
            faces += [(a, b, c), (a, c, d)]
            a, b, c, d = bottom[i, j], bottom[i + 1, j], bottom[i + 1, j + 1], bottom[i, j + 1]
            faces += [(a, c, b), (a, d, c)]
    for i in range(n_x - 1):
        for j in (0, n_y - 1):
            t0, t1, b0, b1 = top[i, j], top[i + 1, j], bottom[i, j], bottom[i + 1, j]
            faces += [(t0, t1, b1), (t0, b1, b0)]
    for j in range(n_y - 1):
        for i in (0, n_x - 1):
            t0, t1, b0, b1 = top[i, j], top[i, j + 1], bottom[i, j], bottom[i, j + 1]
            faces += [(t0, b1, t1), (t0, b0, b1)]

    points = np.asarray(verts, dtype=np.float64)
    centroid = points.mean(axis=0)
    return points - centroid, np.asarray(faces, dtype=np.int32), centroid


def build_voxel_surface_mesh(
    occupied: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Closed mesh of the exposed boundary of an occupied voxel set.

    The height-field mesher can only represent a set that is a solid under a
    surface, which forces every no-go volume to be a sheet regardless of what the
    sweep found. This walks the occupancy array and emits the faces of occupied
    cells whose neighbour is empty, so overhangs, cavities and detached lobes
    survive into the render. Returns (vertices about the centroid, triangles,
    centroid), or None when nothing is occupied.
    """

    if not occupied.any():
        return None

    def edges(centres: np.ndarray) -> np.ndarray:
        step = float(centres[1] - centres[0]) if len(centres) > 1 else 0.03
        return np.concatenate([centres - step / 2.0, [centres[-1] + step / 2.0]])

    ex, ey, ez = edges(xs), edges(ys), edges(zs)
    verts: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []

    def vid(i: int, j: int, k: int) -> int:
        key = (i, j, k)
        got = verts.get(key)
        if got is None:
            got = len(points)
            verts[key] = got
            points.append((float(ex[i]), float(ey[j]), float(ez[k])))
        return got

    def quad(a, b, c, d) -> None:
        faces.append((a, b, c))
        faces.append((a, c, d))

    pad = np.zeros(np.asarray(occupied.shape) + 2, dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = occupied
    for i, j, k in zip(*np.nonzero(occupied)):
        i, j, k = int(i), int(j), int(k)
        p, q, r = i + 1, j + 1, k + 1
        if not pad[p - 1, q, r]:
            quad(vid(i, j, k), vid(i, j, k + 1), vid(i, j + 1, k + 1), vid(i, j + 1, k))
        if not pad[p + 1, q, r]:
            quad(vid(i + 1, j, k), vid(i + 1, j + 1, k), vid(i + 1, j + 1, k + 1), vid(i + 1, j, k + 1))
        if not pad[p, q - 1, r]:
            quad(vid(i, j, k), vid(i + 1, j, k), vid(i + 1, j, k + 1), vid(i, j, k + 1))
        if not pad[p, q + 1, r]:
            quad(vid(i, j + 1, k), vid(i, j + 1, k + 1), vid(i + 1, j + 1, k + 1), vid(i + 1, j + 1, k))
        if not pad[p, q, r - 1]:
            quad(vid(i, j, k), vid(i, j + 1, k), vid(i + 1, j + 1, k), vid(i + 1, j, k))
        if not pad[p, q, r + 1]:
            quad(vid(i, j, k + 1), vid(i + 1, j, k + 1), vid(i + 1, j + 1, k + 1), vid(i, j + 1, k + 1))

    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    return pts - centroid, np.asarray(faces, dtype=np.int32), centroid


_FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation")


def _font(size: int, bold: bool = False):
    """A scalable face, falling back to PIL's bitmap font if none is installed.

    The default face is a fixed 11 px bitmap. A strip panel is reduced to about
    a third of a column on the page, so the caption has to be set at a size that
    survives the reduction.
    """

    from PIL import ImageFont

    names = ("DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf") if bold else \
            ("DejaVuSans.ttf", "LiberationSans-Regular.ttf")
    for directory in _FONT_DIRS:
        for name in names:
            path = Path(directory) / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _text_width(font, text: str) -> float:
    """Rendered width of ``text``, for the bitmap fallback face as well."""

    try:
        return float(font.getlength(text))
    except AttributeError:  # PIL's bitmap default on very old Pillow
        return 0.6 * getattr(font, "size", 11) * len(text)


def _fit_font(text: str, target_px: int, max_width: int, bold: bool = False):
    """Largest face at or below ``target_px`` whose ``text`` fits ``max_width``."""

    size = int(target_px)
    while size > 9:
        font = _font(size, bold=bold)
        if _text_width(font, text) <= max_width:
            return font
        size -= 1
    return _font(9, bold=bold)


def _wrap(text: str, font, max_width: int) -> list[str]:
    """Greedy word wrap at a pixel width."""

    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if current and _text_width(font, trial) > max_width:
            lines.append(current)
            current = word
        else:
            current = trial
    if current:
        lines.append(current)
    return lines


def swatch_legend(img: Image.Image, title: str,
                  entries: list[tuple[str, list[tuple], str]]) -> Image.Image:
    """Colour-keyed legend strip above the render.

    The layers are semi-transparent and nested, so a text-only caption leaves the
    reader matching percentages to colours by guesswork.
    """

    pad, row_h, swatch = 9, 17, 13
    height = pad * 2 + 16 + row_h * len(entries)
    out = Image.new("RGB", (img.width, img.height + height), (255, 255, 255))
    out.paste(img, (0, height))
    draw = ImageDraw.Draw(out)
    draw.text((pad, pad), title, fill=(20, 20, 20))
    for idx, (name, rgbs, note) in enumerate(entries):
        y = pad + 16 + idx * row_h
        # A row standing for two strategies whose volumes coincide keeps both
        # colours, so a strategy's colour means the same thing in every figure.
        band = swatch / max(len(rgbs), 1)
        for k, rgb in enumerate(rgbs):
            colour = tuple(int(round(c * 255)) for c in rgb)
            draw.rectangle([pad + k * band, y + 2, pad + (k + 1) * band, y + 2 + swatch],
                           fill=colour)
        draw.rectangle([pad, y + 2, pad + swatch, y + 2 + swatch],
                       outline=(90, 90, 90))
        draw.text((pad + swatch + 8, y + 3), name, fill=(20, 20, 20))
        draw.text((pad + swatch + 150, y + 3), note, fill=(80, 80, 80))
    draw.line([0, height - 1, out.width, height - 1], fill=(210, 210, 210))
    return out


def refine_occupancy(
    occupied: np.ndarray, xs: np.ndarray, ys: np.ndarray, zs: np.ndarray,
    upsample: int, sigma_cells: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Resample a boolean voxel set onto a finer grid and blur it.

    A 30 mm sweep grid meshed cell by cell is a staircase, and the staircase is
    an artefact of the sampling rather than of the geometry. Interpolating the
    indicator onto a finer grid and blurring it before taking the half level
    replaces the steps with a surface at the resolution the sweep actually
    resolves, without moving the boundary: the 0.5 level of a symmetric blur sits
    where the original faces were.
    """

    from scipy.ndimage import gaussian_filter, map_coordinates

    def fine_axis(centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        step = float(centres[1] - centres[0]) if len(centres) > 1 else 1.0
        lo, hi = centres[0] - step / 2.0, centres[-1] + step / 2.0
        n = len(centres) * upsample
        fstep = (hi - lo) / n
        fine = lo + fstep * (np.arange(n) + 0.5)
        return fine, (fine - centres[0]) / step

    fx, ix = fine_axis(xs)
    fy, iy = fine_axis(ys)
    fz, iz = fine_axis(zs)
    grid = np.meshgrid(ix, iy, iz, indexing="ij")
    field = map_coordinates(occupied.astype(float),
                            np.stack([g.ravel() for g in grid]),
                            order=1, mode="nearest").reshape(len(fx), len(fy), len(fz))
    if sigma_cells > 0:
        field = gaussian_filter(field, sigma=sigma_cells, mode="nearest")
    return field >= 0.5, fx, fy, fz


def taubin_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int,
                  lam: float = 0.55, mu: float = -0.58) -> np.ndarray:
    """Laplacian smoothing with Taubin's alternating step.

    Plain Laplacian smoothing shrinks a closed surface toward its centroid, which
    would quietly understate every volume in the figure. Alternating a positive
    and a slightly larger negative step is the standard low-pass that keeps the
    enclosed volume roughly fixed.
    """

    if iterations <= 0 or len(faces) == 0:
        return vertices
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.vstack([edges, edges[:, ::-1]])
    n = len(vertices)
    counts = np.bincount(edges[:, 0], minlength=n).astype(float)
    counts[counts == 0] = 1.0
    out = vertices.astype(np.float64).copy()
    for step in range(iterations):
        neighbour_mean = np.stack(
            [np.bincount(edges[:, 0], weights=out[edges[:, 1], k], minlength=n) / counts
             for k in range(3)], axis=1)
        out += (lam if step % 2 == 0 else mu) * (neighbour_mean - out)
    return out


def scale_bar(panel: Image.Image, length_mm: float, distance_m: float,
              fovy_deg: float = 45.0, pad: int = 14) -> Image.Image:
    """Draw a scale bar valid at the depth the camera is focused on.

    Nothing else in a panel gives the size of the region: the benchmark object
    is a few tens of millimetres inside a sweep several hundred across, so it
    reads as a speck rather than as a ruler.
    """

    if length_mm <= 0:
        return panel
    px_per_m = panel.height / (2.0 * distance_m * math.tan(math.radians(fovy_deg) / 2.0))
    length_px = int(round(length_mm / 1000.0 * px_per_m))
    draw = ImageDraw.Draw(panel)
    # A strip is reduced to a single column on the page, so the bar and its
    # label are sized as a fraction of the panel rather than in fixed pixels.
    size_px = max(14, round(panel.width * 0.055))
    rule = max(3, round(panel.width * 0.008))
    y = panel.height - pad - 6
    x0 = pad
    x1 = x0 + length_px
    draw.line([x0, y, x1, y], fill=(40, 40, 40), width=rule)
    for x in (x0, x1):
        draw.line([x, y - 2 * rule, x, y + 2 * rule], fill=(40, 40, 40), width=rule)
    font = _font(size_px)
    draw.text((x0, y - round(size_px * 1.45)), f"{length_mm:.0f} mm",
              fill=(40, 40, 40), font=font)
    return panel


def surface_from_set(args, draw_set: np.ndarray, gx: np.ndarray, gy: np.ndarray,
                     zs: np.ndarray, grasp_target: np.ndarray):
    """Triangulated boundary of one voxel set, refined when --smooth is set."""

    if args.smooth:
        fine, fx, fy, fz = refine_occupancy(
            draw_set, gx + grasp_target[0], gy + grasp_target[1], zs,
            args.smooth_upsample, args.smooth_sigma)
        built = build_voxel_surface_mesh(fine, fx, fy, fz)
        if built is None:
            return None
        verts, faces_, centre = built
        return taubin_smooth(verts, faces_, args.smooth_iters), faces_, centre
    return build_voxel_surface_mesh(draw_set, gx + grasp_target[0],
                                    gy + grasp_target[1], zs)


def _camera(view: str, lookat: np.ndarray, distance: float) -> mujoco.MjvCamera:
    azimuth, elevation = VIEWS[view]
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_pass(model, data, view: str, lookat: np.ndarray, distance: float,
                width: int, height: int, *, hide_volume: bool = False,
                segmentation: bool = False) -> np.ndarray:
    """Colour or segmentation buffer, with the no-go solid optionally removed."""

    option = mujoco.MjvOption()
    if hide_volume:
        option.geomgroup[VOLUME_GROUP] = 0
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        if segmentation:
            renderer.enable_segmentation_rendering()
        renderer.update_scene(data, _camera(view, lookat, distance), option)
        return renderer.render()


def geom_mask(segment: np.ndarray, geom_ids: list[int]) -> np.ndarray:
    """Pixels covered by the given geoms in a segmentation buffer."""

    if not geom_ids:
        return np.zeros(segment.shape[:2], dtype=bool)
    return (np.isin(segment[..., 0], geom_ids)
            & (segment[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)))


def foreground(base: np.ndarray, plain: np.ndarray, mask: np.ndarray,
               edge_px: int = 2, edge_rgb: tuple[int, int, int] = (25, 25, 25)) -> np.ndarray:
    """Paste the object back over the solid at full opacity, with a dark edge.

    The solid encloses the object, so in depth order the thing being grasped is
    seen through a translucent surface and on the naive strategy it disappears
    entirely. ``plain`` is the same frame rendered with the solid removed;
    copying the object's pixels from it is the only way to keep the object
    opaque without also making the solid opaque. The edge separates it from a
    surface of a similar colour.
    """

    out = base.copy()
    out[mask] = plain[mask]
    if edge_px > 0 and mask.any():
        grown = np.asarray(
            Image.fromarray((mask * 255).astype(np.uint8))
            .filter(ImageFilter.MaxFilter(2 * edge_px + 1))) > 0
        out[grown & ~mask] = edge_rgb
    return out


def hatch(image: np.ndarray, mask: np.ndarray, spacing: int, width: int,
          rgb: tuple[int, int, int] = (120, 12, 12), strength: float = 0.34) -> np.ndarray:
    """Diagonal rule over the shaded region, drawn in the plane of the page."""

    if spacing <= 0 or not mask.any():
        return image
    rows, cols = np.indices(mask.shape)
    lines = ((rows + cols) % spacing) < max(1, width)
    sel = mask & lines
    out = image.copy()
    out[sel] = np.clip((1.0 - strength) * out[sel].astype(float)
                       + strength * np.array(rgb, dtype=float), 0, 255).astype(np.uint8)
    return out


def frame_camera(points: np.ndarray, azimuth: float, elevation: float,
                 width: int, height: int, fovy_deg: float,
                 margin: float = 1.10) -> tuple[np.ndarray, float]:
    """Camera lookat and distance that just contain ``points`` in one panel.

    Panels in a strip are only useful if they share a camera, and a strip that
    is a third of a column wide cannot afford the slack of a hand-set distance.
    Projecting the points onto the camera axes gives the tightest framing that
    holds every panel, so the volumes are directly comparable in size.
    """

    az, el = math.radians(azimuth), math.radians(elevation)
    forward = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    right = np.array([-math.sin(az), math.cos(az), 0.0])
    up = np.cross(right, forward)
    centre = 0.5 * (points.min(axis=0) + points.max(axis=0))
    local = points - centre
    half_r = 0.5 * float(np.ptp(local @ right))
    half_u = 0.5 * float(np.ptp(local @ up))
    # Re-centre on the projected extent rather than the bounding-box centre, so
    # an asymmetric shell sits in the middle of the frame.
    centre = centre + right * float(np.mean([(local @ right).min(), (local @ right).max()])) \
                    + up * float(np.mean([(local @ up).min(), (local @ up).max()]))
    tan_half = math.tan(math.radians(fovy_deg) / 2.0)
    distance = max(half_u / tan_half, half_r / (tan_half * (width / height)))
    return centre, distance * margin


def caption_fonts(panel_width: int, names: list[str], notes: list[str]) -> tuple:
    """One name and note face for the whole strip.

    Fitting each panel's caption on its own gives the panel with the longest
    name smaller type than its neighbours, which reads as emphasis on the short
    ones. The sizes are fractions of the panel width because a strip is reduced
    to one column on the page: a caption set at 20 px prints at under four
    points.
    """

    avail = panel_width - caption_text_x(panel_width) - round(panel_width * 0.040)
    name_font = min((_fit_font(n, round(panel_width * 0.100), avail, bold=True)
                     for n in names), key=lambda f: f.size)
    note_font = min((_fit_font(n, round(panel_width * 0.078), avail)
                     for n in notes), key=lambda f: f.size)
    return name_font, note_font


def caption_text_x(panel_width: int) -> int:
    """Left edge of the caption text, clear of the colour rule."""

    pad = max(8, round(panel_width * 0.020))
    return pad + max(6, round(panel_width * 0.017)) + round(panel_width * 0.030)


def strip_caption(panel: Image.Image, name: str, note: str,
                  rgb: tuple[float, float, float],
                  name_font, note_font) -> Image.Image:
    """Ruled caption above one strip panel, in place of a legend.

    At strip size a shared legend costs more space than it saves and forces the
    reader to match a percentage to a colour; naming each panel binds the two.
    The rule takes the colour of the solid below it, which is the same in every
    panel, so it ties the caption to the drawing rather than identifying it.
    """

    pad = max(8, round(panel.width * 0.020))
    rule_w = max(6, round(panel.width * 0.017))
    text_x = caption_text_x(panel.width)
    name_h = round(name_font.size * 1.20)
    note_h = round(note_font.size * 1.20)
    gap = round(name_font.size * 0.16)
    bar = pad + name_h + gap + note_h + pad

    out = Image.new("RGB", (panel.width, panel.height + bar), (255, 255, 255))
    out.paste(panel, (0, bar))
    draw = ImageDraw.Draw(out)
    colour = tuple(int(round(c * 255)) for c in rgb)
    draw.rectangle([pad, pad, pad + rule_w, bar - pad], fill=colour)
    draw.text((text_x, pad), name, fill=(20, 20, 20), font=name_font)
    draw.text((text_x, pad + name_h + gap), note, fill=(95, 95, 95), font=note_font)
    return out


def legend_width(entries: list[tuple], font, swatch: int, gap: int) -> int:
    """Pixel width of a one-row swatch legend."""

    if not entries:
        return 0
    each = [swatch + round(swatch * 0.45) + _text_width(font, label) for _, label in entries]
    return round(sum(each) + gap * (len(entries) - 1))


def _fit_font_parts(parts: list[tuple], target_px: int, max_width: int, bold: bool = False):
    """Largest face at or below ``target_px`` whose segments fit ``max_width``."""

    size = int(target_px)
    while size > 9:
        font = _font(size, bold=bold)
        if sum(_text_width(font, text) for text, _ in parts) <= max_width:
            return font
        size -= 1
    return _font(9, bold=bold)


def strip_header(sheet_width: int, panel_width: int, title_parts: list[tuple],
                 subtitle: str | None,
                 legend: list[tuple] | None = None) -> Image.Image:
    """Title block above a strip, naming what the shaded solids are.

    Sized from the panel width rather than the sheet width, so the three-panel
    and four-panel versions of the same strip carry the same type. A title
    segment carrying a colour is set in that colour, which lets the heading name
    both solids and do the work of a legend in the space of a heading. The
    legend, when one is drawn, sits on the title row right-aligned to the sheet.
    """

    legend = legend or []
    title = "".join(text for text, _ in title_parts)
    if not title and not subtitle and not legend:
        return Image.new("RGB", (sheet_width, 0), (255, 255, 255))
    pad = max(8, round(panel_width * 0.020))
    avail = sheet_width - pad - round(panel_width * 0.045)
    key_font = _font(round(panel_width * 0.078))
    key_swatch = round(panel_width * 0.055)
    key_gap = round(panel_width * 0.048)
    key_w = legend_width(legend, key_font, key_swatch, key_gap)
    if key_w:
        avail -= key_w + key_gap
    title_font = _fit_font_parts(title_parts, round(panel_width * 0.118), avail, bold=True)
    sub_font = _font(round(panel_width * 0.075))
    sub_lines = _wrap(subtitle, sub_font, avail) if subtitle else []
    title_h = round(title_font.size * 1.25) if title else 0
    sub_h = round(sub_font.size * 1.30)
    height = pad + title_h + sub_h * len(sub_lines) + pad

    out = Image.new("RGB", (sheet_width, height), (255, 255, 255))
    draw = ImageDraw.Draw(out)
    y = pad
    x = pad
    for text, rgb in title_parts:
        fill = (20, 20, 20) if rgb is None else tuple(int(round(c * 255)) for c in rgb)
        draw.text((x, y), text, fill=fill, font=title_font)
        x += _text_width(title_font, text)
    if legend:
        x = sheet_width - round(panel_width * 0.045) - key_w
        mid = y + round(title_font.size * 0.60)
        for rgb, label in legend:
            colour = tuple(int(round(c * 255)) for c in rgb)
            draw.rectangle([x, mid - key_swatch // 2, x + key_swatch,
                            mid + key_swatch // 2], fill=colour)
            x += key_swatch + round(key_swatch * 0.45)
            draw.text((x, mid - round(key_font.size * 0.62)), label,
                      fill=(60, 60, 60), font=key_font)
            x += _text_width(key_font, label) + key_gap
    y += title_h
    for line in sub_lines:
        draw.text((pad, y), line, fill=(85, 85, 85), font=sub_font)
        y += sub_h
    return out


def _scene(args, obj, data, cap, grasp_target):
    """Grasp scene with the object, an optional neighbour, and a matte floor."""

    from strategy_admissibility import InspireHandFK as _FK

    spec = mujoco.MjSpec.from_file(str(_FK().xml_path))
    add_object_geom(spec, obj)
    # The object proxy ships in the orange used for a strategy series, which
    # would read as one more region. It is drawn in the colour of the real item
    # instead, opaque, and a specular material keeps a metal can from flattening
    # into a grey card.
    rgb = OBJECT_RGB.get(args.object, (0.99, 0.99, 0.99))
    material = spec.add_material()
    material.name = "object_proxy_mat"
    material.rgba = np.array([*rgb, 1.0])
    material.specular = 0.6
    material.shininess = 0.55
    material.reflectance = 0.0
    for body in spec.bodies:
        for geom in body.geoms:
            if geom.name == "object_proxy_geom":
                geom.material = "object_proxy_mat"
                geom.rgba = np.array([*rgb, 1.0])
    if getattr(args, "object_pin", False):
        # A benchmark object is a few tens of millimetres inside a region a few
        # hundred across, so even an opaque proxy is a smudge. A thin post
        # through it fixes where the object is without inflating it.
        pin_h = getattr(args, "pin_height_mm", 0.0) / 1000.0 or cap
        pin = spec.worldbody.add_body(name="object_pin",
                                      pos=np.array([0.0, 0.0, pin_h / 2.0]))
        rod = pin.add_geom(name="object_pin_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                           size=np.array([0.0025, pin_h / 2.0, 0.0]))
        rod.rgba = np.array([0.10, 0.10, 0.12, 1.0])
        rod.contype = 0
        rod.conaffinity = 0
    spec.visual.global_.offwidth = max(args.width, 640)
    spec.visual.global_.offheight = max(args.height, 480)
    if args.plain_floor:
        for geom in spec.worldbody.geoms:
            if geom.name == "floor":
                geom.material = ""
                geom.rgba = np.array([0.72, 0.72, 0.70, 1.0])
        # The default skybox is a blue gradient, which reads as a horizon behind a
        # scene that has none and competes with the translucent layers for
        # contrast. Flatten it to the same neutral as the page.
        for tex in spec.textures:
            if tex.type == mujoco.mjtTexture.mjTEXTURE_SKYBOX:
                tex.rgb1 = np.array([1.0, 1.0, 1.0])
                tex.rgb2 = np.array([1.0, 1.0, 1.0])
    if "clutter_center_m" in data:
        centre = data["clutter_center_m"]
        half = np.asarray(data["clutter_size_m"], dtype=float) / 2.0
        body = spec.worldbody.add_body(name="clutter", pos=centre)
        geom = body.add_geom(name="clutter_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=half)
        geom.rgba = np.array([0.45, 0.42, 0.40, 0.85])
        geom.contype = 0
        geom.conaffinity = 0
    return spec


def render_strip(args, data, xs, ys, zs, grasp_target, obj, plans, cap) -> int:
    """One panel per strategy, side by side, on a single shared camera.

    The nested overlay is legible on a monitor and not at column width, where
    three translucent surfaces of similar extent become one mass. Drawing each
    strategy's no-go set on its own removes the transparency problem entirely;
    what is lost is the containment, which the percentages state instead.
    """

    layers = [l for l in args.layers if f"admissible_{l}" in data]
    if not layers:
        raise SystemExit("none of the requested layers are present in the sweep")

    sampled_all = data["sampled"] if "sampled" in data else None
    surfaces: dict[str, tuple] = {}
    reachable: dict[str, tuple] = {}
    stats: dict[str, dict[str, float]] = {}
    for layer in layers:
        admissible = data[f"admissible_{layer}"]
        sampled = np.ones_like(admissible) if sampled_all is None else sampled_all
        nogo = sampled & ~admissible
        stats[layer] = {
            "locked_fraction": float(nogo.sum()) / max(int(sampled.sum()), 1),
            "never_admissible_fraction": float((floor_heights(admissible, zs) > zs[-1]).mean()),
        }
        built = surface_from_set(args, nogo, xs, ys, zs, grasp_target)
        if built is not None:
            surfaces[layer] = built
        if args.strip_reachable:
            built = surface_from_set(args, sampled & admissible, xs, ys, zs, grasp_target)
            if built is not None:
                reachable[layer] = built

    if not surfaces and not reachable:
        raise SystemExit("every requested layer is empty")

    # One camera for the whole strip, framed on the union of the panels so a
    # larger no-go set reads as larger rather than as the same size at a
    # different zoom.
    if args.azimuth is not None or args.elevation is not None:
        base_az, base_el = VIEWS[args.view]
        VIEWS["_custom"] = (base_az if args.azimuth is None else args.azimuth,
                            base_el if args.elevation is None else args.elevation)
        view_name = "_custom"
    else:
        view_name = args.view
    azimuth, elevation = VIEWS[view_name]
    union = np.vstack([verts + centroid for verts, _, centroid
                       in (*surfaces.values(), *reachable.values())])
    # The object sits inside every volume, but on a tall object the pin runs to
    # the top of the grid, so frame on the volumes and the object only.
    union = np.vstack([union, np.array([grasp_target, grasp_target + [0, 0, 0.001]])])
    lookat, distance = frame_camera(union, azimuth, elevation,
                                    args.width, args.height, 45.0, args.strip_margin)

    renders: list[Image.Image] = []
    bare: list[Image.Image] = []
    for layer in layers:
        spec = _scene(args, obj, data, cap, grasp_target)
        # The reachable shell wraps the no-go solid, so it is added first and
        # kept far more transparent; at equal opacity the red reads as a stain
        # on the green rather than as a solid inside it.
        for kind, store, rgb, alpha in (
                ("reach", reachable, REACH_RGB, args.strip_reach_alpha),
                ("nogo", surfaces, NOGO_RGB, args.strip_alpha)):
            if layer not in store:
                continue
            vertices, faces, centroid = store[layer]
            mesh = spec.add_mesh()
            mesh.name = f"{kind}_{layer}"
            mesh.uservert = vertices.flatten()
            mesh.userface = faces.flatten()
            body = spec.worldbody.add_body(name=f"{kind}_body_{layer}", pos=centroid)
            geom = body.add_geom(name=f"{kind}_geom_{layer}")
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = f"{kind}_{layer}"
            # A panel shows one surface per set, so each need only be
            # translucent enough to keep the far wall of the shell readable;
            # the object is drawn over both afterwards rather than through them.
            geom.rgba = np.array([*rgb, alpha])
            geom.group = VOLUME_GROUP
            geom.contype = 0
            geom.conaffinity = 0
        model = spec.compile()
        mj_data = mujoco.MjData(model)
        plan = plans["thumb_reflex"]
        if args.start_pose_mm is not None:
            start = grasp_target + np.asarray(args.start_pose_mm, dtype=float) / 1000.0
            yaw = math.radians(args.hand_yaw_deg)
            spin = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                             [math.sin(yaw), math.cos(yaw), 0.0],
                             [0.0, 0.0, 1.0]])
            set_scene_state(model, mj_data, plan.waypoints[0].ctrl, start,
                            spin @ plan.waypoints[0].rotation)
        else:
            set_scene_state(model, mj_data, plan.terminal_ctrl,
                            np.array([0.0, 0.0, -5.0]), np.eye(3))
        mujoco.mj_forward(model, mj_data)
        shot = dict(view=view_name, lookat=lookat, distance=distance,
                    width=args.width, height=args.height)
        frame = render_pass(model, mj_data, **shot)
        if layer in surfaces or layer in reachable:
            if args.strip_hatch and layer in surfaces:
                solid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,
                                          f"nogo_geom_{layer}")
                frame = hatch(frame,
                              geom_mask(render_pass(model, mj_data, segmentation=True, **shot),
                                        [solid]),
                              args.strip_hatch, max(1, args.strip_hatch // 5))
            # The object is rendered a second time with the solid removed, then
            # pasted back over the panel, so it stays opaque inside a volume
            # that covers it.
            marks = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                     for name in ("object_proxy_geom", "object_pin_geom")]
            frame = foreground(
                frame,
                render_pass(model, mj_data, hide_volume=True, **shot),
                geom_mask(render_pass(model, mj_data, hide_volume=True,
                                      segmentation=True, **shot),
                          [g for g in marks if g >= 0]),
                edge_px=args.object_edge_px)
        img = Image.fromarray(frame)
        bare.append(img.copy())
        if not renders:
            img = scale_bar(img, args.scale_bar_mm, distance)
        renders.append(img)

    names, notes = [], []
    for layer in layers:
        name, prefix = STRIP_LABEL.get(layer, (LAYER_STYLE[layer][0], ""))
        names.append(name)
        notes.append(f"{prefix}{stats[layer]['locked_fraction']:.0%} unreachable")

    if args.panels_dir is not None:
        # The bare frames, without caption or scale bar, for compose_nogo_figure.py
        # to lay out as a multi-object grid. The sidecar carries everything the
        # compositor needs that only the renderer knows: the panel labels and the
        # pixel scale at the focus depth, so it can draw its own scale bar.
        args.panels_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for layer, img, name, note in zip(layers, bare, names, notes):
            file = f"{args.out.stem}_{layer}.png"
            img.save(args.panels_dir / file)
            entries.append({"layer": layer, "name": name, "note": note, "file": file,
                            "locked_fraction": stats[layer]["locked_fraction"]})
        sidecar = {
            "object": args.object,
            "grasp_height_mm": float(grasp_target[2] * 1000.0),
            "px_per_m": bare[0].height / (2.0 * distance * math.tan(math.radians(45.0) / 2.0)),
            "panel_size": [bare[0].width, bare[0].height],
            "colours": {"reachable": REACH_RGB, "unreachable": NOGO_RGB},
            "panels": entries,
        }
        (args.panels_dir / f"{args.out.stem}_panels.json").write_text(json.dumps(sidecar, indent=2))
        print(f"wrote {len(entries)} bare panels to {args.panels_dir}")
    name_font, note_font = caption_fonts(renders[0].width, names, notes)
    panels = [strip_caption(img, name, note, NOGO_RGB, name_font, note_font)
              for img, name, note in zip(renders, names, notes)]

    gutter = 6
    width = sum(p.width for p in panels) + gutter * (len(panels) - 1)
    # The shaded solid is the set of start poses the strategy cannot execute
    # from, which is the opposite reading of a rendered volume by default, so
    # the heading says so rather than leaving it to the caption. It also carries
    # the word the panel notes leave out: the percentages are of start poses.
    name = args.object.replace("paper_", "").replace("_", " ").title()
    legend = []
    if args.strip_title is not None:
        title_parts = [(args.strip_title, None)]
        if reachable:
            legend = [(NOGO_RGB, "unreachable"), (REACH_RGB, "reachable")]
    elif reachable:
        # Setting the two words in the colours of the two solids says which is
        # which in the space a heading already occupies. A swatch legend beside
        # a heading this long would take a third of the row and force the
        # heading below the size of the panel names underneath it.
        title_parts = [("Reachable", REACH_RGB), (" and ", None),
                       ("unreachable", NOGO_RGB), (f" start poses: {name}", None)]
    else:
        title_parts = [(f"Unreachable start poses: {name}", None)]
    # The heading names what the solids are; what they are measured over and
    # where the grasp point sits belong to the caption, which has room for them.
    header = strip_header(width, panels[0].width, title_parts,
                          args.strip_subtitle or "", legend)
    sheet = Image.new("RGB", (width, panels[0].height + header.height), (255, 255, 255))
    sheet.paste(header, (0, 0))
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, header.height))
        x += panel.width + gutter
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out)
    stats_path = args.results / f"{args.out.stem}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    for layer in layers:
        print(f"  {LAYER_STYLE[layer][0]:24s} locked {stats[layer]['locked_fraction']:6.1%}")
    print(f"wrote {args.out}")
    print(f"wrote {stats_path}")
    return 0


def render_overlay(args, data, xs, ys, zs, grasp_target, obj, plans, cap) -> int:
    """All layers in one scene, nested, with the start pose drawn."""

    spec = _scene(args, obj, data, cap, grasp_target)
    stats: dict[str, dict[str, float]] = {}
    drawn = []
    centroids: list[np.ndarray] = []
    alphas = dict(zip(args.layers, args.alphas)) if args.alphas else {}
    # Score every layer the sweep recorded, not only the ones drawn: captions
    # quote strategies that are deduped into another mesh or left out of the
    # render entirely because their volume fills the region.
    for layer in LAYER_STYLE:
        key = f"admissible_{layer}"
        if key not in data:
            continue
        adm = data[key]
        sampled_all = data["sampled"] if "sampled" in data else np.ones_like(adm)
        if args.crop_mm is not None:
            keep_x = np.abs(data["xs_mm"]) <= args.crop_mm + 1e-9
            keep_y = np.abs(data["ys_mm"]) <= args.crop_mm + 1e-9
            sel = np.ix_(keep_x, keep_y, np.ones(len(zs), bool))
            adm, sampled_all = adm[sel], sampled_all[sel]
        heights_all = floor_heights(adm, zs)
        stats[layer] = {
            "locked_fraction": float((sampled_all & ~adm).sum())
            / max(int(sampled_all.sum()), 1),
            "median_floor_mm": float(np.median(np.minimum(heights_all, cap)) * 1000.0),
            "never_admissible_fraction": float((heights_all > zs[-1]).mean()),
        }
    # Strategies whose no-go sets coincide would be drawn as two coincident
    # meshes, which z-fight and read as one layer of unclear identity. Draw the
    # first and name both in the legend.
    same_as: dict[str, list[str]] = {}
    seen: list[tuple[str, np.ndarray]] = []
    keep: list[str] = []
    for layer in args.layers:
        key = f"admissible_{layer}"
        if key not in data:
            continue
        arr = data[key]
        match = next((n for n, a in seen if np.array_equal(a, arr)), None)
        if match is None:
            seen.append((layer, arr))
            keep.append(layer)
            same_as[layer] = []
        else:
            same_as[match].append(layer)
    # In disjoint mode a layer is drawn as what it locks out beyond the largest
    # layer strictly inside it. Ordering by size makes the peel well defined even
    # where the sets are not perfectly nested, as GraspGen-X is not.
    shells: dict[str, np.ndarray] = {}
    if args.disjoint and args.volume_mode == "voxel":
        sampled_full = data["sampled"] if "sampled" in data else None
        nogo_all = {}
        for layer in keep:
            adm = data[f"admissible_{layer}"]
            samp = np.ones_like(adm) if sampled_full is None else sampled_full
            nogo_all[layer] = samp & ~adm
        accumulated = np.zeros_like(next(iter(nogo_all.values())))
        for layer in sorted(keep, key=lambda l: nogo_all[l].sum()):
            shells[layer] = nogo_all[layer] & ~accumulated
            accumulated = accumulated | nogo_all[layer]

    for layer in keep:
        key = f"admissible_{layer}"
        if key not in data:
            continue
        admissible = data[key]
        if args.crop_mm is not None:
            keep_x = np.abs(data["xs_mm"]) <= args.crop_mm + 1e-9
            keep_y = np.abs(data["ys_mm"]) <= args.crop_mm + 1e-9
            admissible = admissible[np.ix_(keep_x, keep_y, np.ones(len(zs), bool))]
            gx, gy = xs[keep_x], ys[keep_y]
        else:
            gx, gy = xs, ys
        heights = floor_heights(admissible, zs)
        never = float((heights > zs[-1]).mean())
        if args.volume_mode == "voxel":
            sampled = data["sampled"] if "sampled" in data else np.ones_like(admissible)
            if args.crop_mm is not None:
                sampled = sampled[np.ix_(keep_x, keep_y, np.ones(len(zs), bool))]
            n_sampled = max(int(sampled.sum()), 1)
            nogo = sampled & ~admissible
            stats[layer] = {
                "locked_fraction": float(nogo.sum()) / n_sampled,
                "median_floor_mm": float(np.median(np.minimum(heights, cap)) * 1000.0),
                "never_admissible_fraction": never,
            }
            draw_set = shells.get(layer, nogo)
            if args.crop_mm is not None and layer in shells:
                draw_set = draw_set[np.ix_(keep_x, keep_y, np.ones(len(zs), bool))]
            if args.cutaway_x_mm is not None:
                draw_set = draw_set & (
                    (gx * 1000.0 <= args.cutaway_x_mm)[:, None, None])
            built = surface_from_set(args, draw_set, gx, gy, zs, grasp_target)
        else:
            stats[layer] = {
                "locked_fraction": 1.0 - float(admissible.mean()),
                "median_floor_mm": float(np.median(np.minimum(heights, cap)) * 1000.0),
                "never_admissible_fraction": never,
            }
            built = build_extruded_mesh(gx + grasp_target[0], gy + grasp_target[1],
                                        np.minimum(heights, cap))
        if built is None:
            continue
        vertices, faces, centroid = built
        centroids.append(centroid)
        mesh = spec.add_mesh()
        mesh.name = f"nogo_{layer}"
        mesh.uservert = vertices.flatten()
        mesh.userface = faces.flatten()
        body = spec.worldbody.add_body(name=f"nogo_body_{layer}", pos=centroid)
        geom = body.add_geom(name=f"nogo_geom_{layer}")
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = f"nogo_{layer}"
        geom.rgba = np.array([*LAYER_STYLE[layer][1],
                              alphas.get(layer, args.alpha)])
        geom.contype = 0
        geom.conaffinity = 0
        drawn.append(layer)

    model = spec.compile()
    mj_data = mujoco.MjData(model)

    plan = plans["thumb_reflex"]
    if args.start_pose_mm is not None:
        start = grasp_target + np.asarray(args.start_pose_mm, dtype=float) / 1000.0
        yaw = math.radians(args.hand_yaw_deg)
        spin = np.array([[math.cos(yaw), -math.sin(yaw), 0.0],
                         [math.sin(yaw), math.cos(yaw), 0.0],
                         [0.0, 0.0, 1.0]])
        set_scene_state(model, mj_data, plan.waypoints[0].ctrl, start,
                        spin @ plan.waypoints[0].rotation)
    elif args.show_hand:
        set_scene_state(model, mj_data, plan.terminal_ctrl,
                        plan.terminal_base, plan.terminal_rotation)
    else:
        set_scene_state(model, mj_data, plan.terminal_ctrl,
                        np.array([0.0, 0.0, -5.0]), np.eye(3))
    mujoco.mj_forward(model, mj_data)

    if args.azimuth is not None or args.elevation is not None:
        base_az, base_el = VIEWS[args.view]
        VIEWS["_custom"] = (base_az if args.azimuth is None else args.azimuth,
                            base_el if args.elevation is None else args.elevation)
        view_name = "_custom"
    else:
        view_name = args.view
    _, elevation = VIEWS[view_name]
    lookat = grasp_target + np.array([0.0, 0.0, cap / 2.0])
    if args.volume_mode == "voxel" and centroids:
        # Aim at the middle of what was actually drawn: a quarter-ball shell is
        # not centred on the grasp point, so the height-field lookat leaves it
        # hanging off one edge of the frame.
        lookat = np.mean(np.stack(centroids), axis=0)
    if args.volume_mode == "voxel":
        # The height-field framing pulls the camera back far enough to clear a
        # solid extruded to the grid ceiling. A voxel set is a shell around the
        # grasp point, so frame it on the grid extent instead.
        span = max(float(xs[-1] - xs[0]), float(ys[-1] - ys[0]), float(zs[-1] - zs[0]))
        distance = args.distance if args.distance > 0 else span * 1.15
    else:
        distance = max(args.distance,
                       cap / max(math.sin(math.radians(-elevation)), 1e-3) * 1.6)

    img = render(model, mj_data, view_name, lookat, distance, args.width, args.height)
    label_name = args.object.replace("paper_", "").replace("_", " ")
    img = swatch_legend(
        img,
        f"No-go start volumes  -  {label_name}, grasp point "
        f"{grasp_target[2] * 1000:.0f} mm above the table",
        [(" / ".join([LAYER_STYLE[l][0]] + [LAYER_STYLE[o][0] for o in same_as[l]]),
          [LAYER_STYLE[l][1]] + [LAYER_STYLE[o][1] for o in same_as[l]],
          f"locks out {stats[l]['locked_fraction']:.1%}")
         for l in drawn])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    stats_path = args.results / f"nogo_volume_{args.object}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    for layer in drawn:
        st = stats[layer]
        print(f"  {LAYER_STYLE[layer][0]:24s} locked {st['locked_fraction']:6.1%}  "
              f"never admissible {st['never_admissible_fraction']:5.1%}")
    print(f"wrote {args.out}")
    print(f"wrote {stats_path}")
    return 0


def main() -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", default="paper_nut", choices=sorted(BUILTIN_OBJECTS))
    ap.add_argument("--results", type=Path, default=here.parent / "results")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--layers", nargs="+", default=list(LAYER_STYLE))
    ap.add_argument(
        "--volume-mode",
        choices=["height-field", "voxel"],
        default="height-field",
        help=(
            "height-field extrudes the lowest admissible height per column, which "
            "is exact only where admissibility is monotone in height. voxel meshes "
            "the swept cells the strategy cannot start from directly, and is the "
            "one to use with a --full-grid sweep."
        ),
    )
    ap.add_argument(
        "--hand-yaw-deg",
        type=float,
        default=0.0,
        help=(
            "Rotate the drawn hand about the world z axis. The closure axis of a "
            "rotationally symmetric object admits either facing, so this only "
            "changes which side the arm is shown coming from."
        ),
    )
    ap.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-layer opacity, in --layers order. Nested translucent solids wash "
            "each other out at a single opacity; giving the outermost layer the "
            "lowest value keeps the inner boundaries readable."
        ),
    )
    ap.add_argument(
        "--disjoint",
        action="store_true",
        help=(
            "Draw each layer as the region it locks out and the next-smaller "
            "layer does not, instead of as a full nested solid. The strategy "
            "no-go sets nest, so stacked solids put three translucent surfaces "
            "over the same voxels and no colour belongs to one strategy; the "
            "shells are disjoint and each colour then names exactly one region."
        ),
    )
    ap.add_argument(
        "--cutaway-x-mm",
        type=float,
        default=None,
        help=(
            "Drop volume cells whose x offset exceeds this, leaving a cross "
            "section. The no-go sets nest downward rather than concentrically, "
            "so nested solids seen from outside read as slabs stacked on one "
            "another; a section through them shows the containment, and it "
            "uncovers the object, which otherwise sits inside the innermost "
            "solid."
        ),
    )
    ap.add_argument("--pin-height-mm", type=float, default=0.0,
                    help="Height of the object post; defaults to the top of the sweep grid.")
    ap.add_argument("--object-pin", action="store_true",
                    help="Draw a thin post through the object so its position is legible.")
    ap.add_argument("--smooth", action="store_true",
                    help="Refine and blur the voxel set, then low-pass the surface.")
    ap.add_argument("--smooth-upsample", type=int, default=4)
    ap.add_argument("--smooth-sigma", type=float, default=0.9,
                    help="Blur width in refined cells.")
    ap.add_argument("--smooth-iters", type=int, default=8)
    ap.add_argument("--view", default="iso")
    ap.add_argument("--azimuth", type=float, default=None, help="Override the view azimuth.")
    ap.add_argument("--elevation", type=float, default=None, help="Override the view elevation.")
    ap.add_argument(
        "--crop-mm",
        type=float,
        default=None,
        help=(
            "Restrict the rendered volume to this half-width around the grasp "
            "point. A tighter window makes the slab read as a solid rather than "
            "a sheet."
        ),
    )
    ap.add_argument("--width", type=int, default=620)
    ap.add_argument("--height", type=int, default=560)
    ap.add_argument("--distance", type=float, default=0.95)
    ap.add_argument("--plain-floor", action="store_true",
                    help="Replace the checkerboard ground with a matte plane.")
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--strip", action="store_true",
                    help="Draw one panel per strategy side by side on a shared camera, "
                         "instead of nesting every layer in one scene.")
    ap.add_argument("--strip-alpha", type=float, default=0.40,
                    help="Surface opacity in strip mode; one surface per panel needs far "
                         "less transparency than the nested overlay.")
    ap.add_argument("--scale-bar-mm", type=float, default=100.0,
                    help="Length of the scale bar drawn on the first strip panel; 0 to omit.")
    ap.add_argument("--strip-title", default=None,
                    help="Row heading above a strip; defaults to \"Reachable and "
                         "unreachable start poses: <Object>\", with the two words set "
                         "in the colours of the two solids. Passing one here sets it in "
                         "one colour and adds a swatch legend instead; an empty string "
                         "omits the heading.")
    ap.add_argument("--strip-reachable", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="Also draw the admissible part of the sampled region, in "
                         "green, so the red solid is read as a fraction of a "
                         "region rather than on its own.")
    ap.add_argument("--strip-reach-alpha", type=float, default=0.56,
                    help="Surface opacity of the reachable shell, which encloses "
                         "the no-go solid and so has to stay far lighter than it.")
    ap.add_argument("--strip-hatch", type=int, default=0, metavar="PX",
                    help="Rule the shaded solid with diagonal lines at this pixel "
                         "spacing; 0 draws it as a plain surface.")
    ap.add_argument("--object-edge-px", type=int, default=3,
                    help="Width of the dark outline drawn around the object where "
                         "it is pasted over the solid; 0 to omit. Three pixels is "
                         "half a point at the reduction a strip takes on the page.")
    ap.add_argument("--strip-subtitle", default=None,
                    help="Optional second heading line under the strip title; "
                         "omitted by default, since the caption carries the "
                         "sampling region and the grasp height.")
    ap.add_argument("--strip-margin", type=float, default=1.10,
                    help="Slack around the tightest framing that holds every panel.")
    ap.add_argument("--panels-dir", type=Path, default=None,
                    help="Also write each strip panel bare (no caption, no scale bar) "
                         "into this directory with a <stem>_panels.json sidecar, for "
                         "compose_nogo_figure.py.")
    ap.add_argument("--show-hand", action="store_true",
                    help="Draw the hand at the strategy's terminal pose inside the volume.")
    ap.add_argument(
        "--overlay",
        action="store_true",
        help=(
            "Draw every layer in one scene instead of one panel each. The "
            "volumes nest, so the outermost surface is the most restrictive "
            "strategy."
        ),
    )
    ap.add_argument(
        "--start-pose-mm",
        type=float,
        nargs=3,
        default=None,
        help=(
            "Draw the hand at this start hand-base position, relative to the "
            "grasp point, to show where the analysis is measured from."
        ),
    )
    ap.add_argument("--alpha", type=float, default=0.34)
    ap.add_argument("--tag", default=None,
                    help="Sweep file stem, when it differs from the object name "
                         "(a clutter run is saved as <object>_<clutter>).")
    ap.add_argument("--clutter", default=None, choices=sorted(BUILTIN_OBJECTS),
                    help="Draw a neighbouring object at the swept clutter position.")
    args = ap.parse_args()

    stem = args.object if args.tag is None else args.tag
    data = np.load(args.results / f"nogo_volume_{stem}.npz")
    xs = data["xs_mm"] / 1000.0
    ys = data["ys_mm"] / 1000.0
    zs = data["zs_mm"] / 1000.0
    grasp_target = data["grasp_target_m"]
    obj = BUILTIN_OBJECTS[args.object]

    fk = InspireHandFK()
    closure = ClosureGeometry(fk)
    yaw = grasp_yaw_for_object(obj)
    plans = build_strategy_plans(
        fk, closure, mode=obj.mode,
        target_width_m=obj.grasp_width_m + 0.020,
        grasp_target=grasp_target, yaw_rad=yaw,
    )

    frames: list[Image.Image] = []
    stats: dict[str, dict[str, float]] = {}

    # Columns that admit nothing anywhere in the sampled box are drawn capped at
    # the top of that box, so every panel shares a vertical scale and the label
    # carries the "never admissible" fraction instead.
    cap = float(zs[-1])
    print(f"object={args.object}  grid {len(xs)}x{len(ys)}x{len(zs)}  "
          f"grasp point {grasp_target[2] * 1000:.0f} mm above the table")

    if args.strip:
        return render_strip(args, data, xs, ys, zs, grasp_target, obj, plans, cap)
    if args.overlay:
        return render_overlay(args, data, xs, ys, zs, grasp_target, obj, plans, cap)

    for layer in args.layers:
        key = f"admissible_{layer}"
        if key not in data:
            print(f"  {layer}: absent from the sweep, skipped")
            continue
        admissible = data[key]
        if args.crop_mm is not None:
            keep_x = np.abs(data["xs_mm"]) <= args.crop_mm + 1e-9
            keep_y = np.abs(data["ys_mm"]) <= args.crop_mm + 1e-9
            admissible = admissible[np.ix_(keep_x, keep_y, np.ones(len(zs), bool))]
            grid_xs, grid_ys = xs[keep_x], ys[keep_y]
        else:
            grid_xs, grid_ys = xs, ys
        heights = floor_heights(admissible, zs)
        never = float((heights > zs[-1]).mean())
        volume_fraction = 1.0 - float(admissible.mean())
        heights = np.minimum(heights, cap)

        # Grid coordinates are offsets from the grasp point in x and y.
        world_xs = grid_xs + grasp_target[0]
        world_ys = grid_ys + grasp_target[1]
        built = build_extruded_mesh(world_xs, world_ys, heights)

        spec = mujoco.MjSpec.from_file(str(fk.xml_path))
        add_object_geom(spec, obj)
        if args.plain_floor:
            # The checkerboard reads as visual noise beside a translucent solid.
            for geom in spec.worldbody.geoms:
                if geom.name == "floor":
                    geom.material = ""
                    geom.rgba = np.array([0.90, 0.90, 0.88, 1.0])
        # The scene's default offscreen buffer is 640x480; renders larger than
        # that come back blank unless the framebuffer is grown first.
        spec.visual.global_.offwidth = max(args.width, 640)
        spec.visual.global_.offheight = max(args.height, 480)
        if built is not None:
            vertices, faces, centroid = built
            mesh = spec.add_mesh()
            mesh.name = f"nogo_{layer}"
            mesh.uservert = vertices.flatten()
            mesh.userface = faces.flatten()
            body = spec.worldbody.add_body(name=f"nogo_body_{layer}", pos=centroid)
            geom = body.add_geom(name=f"nogo_geom_{layer}")
            geom.type = mujoco.mjtGeom.mjGEOM_MESH
            geom.meshname = f"nogo_{layer}"
            geom.rgba = np.array([*LAYER_STYLE[layer][1], 0.42])
            geom.contype = 0
            geom.conaffinity = 0
        model = spec.compile()
        mj_data = mujoco.MjData(model)

        plan = plans.get(layer if layer in plans else "thumb_reflex")
        if args.show_hand and plan is not None:
            set_scene_state(model, mj_data, plan.terminal_ctrl,
                            plan.terminal_base, plan.terminal_rotation)
        else:
            # Park the hand far outside the view so only the volume is shown.
            set_scene_state(model, mj_data, plan.terminal_ctrl,
                            np.array([0.0, 0.0, -5.0]), np.eye(3))
        mujoco.mj_forward(model, mj_data)

        label_name, _ = LAYER_STYLE[layer]
        if args.azimuth is not None or args.elevation is not None:
            base_az, base_el = VIEWS[args.view]
            VIEWS["_custom"] = (
                base_az if args.azimuth is None else args.azimuth,
                base_el if args.elevation is None else args.elevation,
            )
            view_name = "_custom"
        else:
            view_name = args.view
        # One camera for every panel so the volumes are directly comparable,
        # framed on the middle of the sampled box rather than on the grasp point.
        _, elevation = VIEWS[view_name]
        lookat = grasp_target + np.array([0.0, 0.0, cap / 2.0])
        needed = cap / max(math.sin(math.radians(-elevation)), 1e-3)
        distance = max(args.distance, needed * 1.6)
        img = render(model, mj_data, view_name, lookat, distance,
                     args.width, args.height)
        caption = f"{volume_fraction:.0%} of sampled start poses locked out"
        if never > 0.999:
            caption += "  (no admissible height anywhere)"
        frames.append(label(img, f"{label_name}\n{caption}"))
        stats[layer] = {
            "locked_fraction": volume_fraction,
            "median_floor_mm": float(np.median(heights) * 1000.0),
            "never_admissible_fraction": never,
        }
        print(f"  {label_name:24s} locked out {volume_fraction:6.1%} of poses; "
              f"median floor {heights.mean() * 1000:5.0f} mm; "
              f"columns with no admissible height: {never:.0%}")

    if not frames:
        raise SystemExit("no layers rendered")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tile(frames, args.cols or len(frames)).save(args.out)
    stats_path = args.results / f"nogo_volume_{args.tag or args.object}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"wrote {args.out}  ({len(frames)} panels)")
    print(f"wrote {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
