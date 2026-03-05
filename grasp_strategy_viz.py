"""
grasp_strategy_viz.py — 2D XZ comparison of Naive, Thumb-Reflex, and Plan grasp strategies.

Object: 28 mm wide × 20 mm tall rectangle on the ground.
Grasp target: 28 mm, approach width (Plan): 38 mm.
Starting pose: fully open, 45° extra orientation offset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from rh56_controller.grasp_geometry import InspireHandFK, ClosureGeometry

XML_PATH = _ROOT / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml"

# ---------------------------------------------------------------------------
# Scene constants
# ---------------------------------------------------------------------------
OBJECT_W   = 0.028   # 28 mm
OBJECT_H   = 0.020   # 20 mm  (tall enough so plan works, naive still fails)
APPROACH_W = 0.038   # 38 mm approach (Plan)
INTERMED_W = 0.033   # 33 mm intermediate (Plan)

GZ         = OBJECT_H / 2   # 10 mm — grasp midpoint = object centre
GZ_START   = 0.055          # 5.25 cm — starting height (lower keeps start index in viewport)

EXTRA_TILT = np.radians(15)          # starting orientation offset
THUMB_YAW  = 1.16                    # fixed for 2-finger line (all states)

# ---------------------------------------------------------------------------
# Rotation helpers (ZY-order for world_R matches ClosureResult convention)
# ---------------------------------------------------------------------------
def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def world_R(tilt_y: float, extra_ry: float = 0.0) -> np.ndarray:
    """Hand-base → world rotation: plane_ry @ tilt_y @ Rx(π)."""
    return _Ry(extra_ry) @ _Ry(tilt_y) @ _Rx(np.pi)

def hand_to_world(
    tips_h: dict,
    ref_mid_h: np.ndarray,
    tilt_y: float,
    extra_ry: float,
    gz: float,
) -> dict:
    """
    Convert hand-frame positions to world frame.
    Places *ref_mid_h* at world (0, 0, gz).
    Returns dict with 'base' added.
    """
    R = world_R(tilt_y, extra_ry)
    offset = np.array([0.0, 0.0, gz]) - R @ ref_mid_h
    out: dict = {}
    for k, v in tips_h.items():
        out[k] = R @ v + offset
    out["base"] = R @ np.zeros(3) + offset
    return out

# ---------------------------------------------------------------------------
# Load FK + solve grasps
# ---------------------------------------------------------------------------
print("Loading FK (this may take ~2 s on first run) …")
fk = InspireHandFK(str(XML_PATH))
cg = ClosureGeometry(fk)

r_28 = cg.solve("2-finger line", OBJECT_W)
r_33 = cg.solve("2-finger line", INTERMED_W)
r_38 = cg.solve("2-finger line", APPROACH_W)

# Open-finger positions in hand frame (thumb yaw fixed at THUMB_YAW)
thumb_open_h  = fk.thumb_tip(fk.ctrl_min["thumb_proximal"], THUMB_YAW)
index_open_h  = fk.finger_tip("index", fk.ctrl_min["index"])
tips_open     = {"thumb": thumb_open_h, "index": index_open_h}

tips_reflex   = {"thumb": r_28.tip_positions["thumb"], "index": index_open_h}
tips_28       = r_28.tip_positions
tips_33       = r_33.tip_positions
tips_38       = r_38.tip_positions

# ---------------------------------------------------------------------------
# Naive crash height: gz where open thumb tip hits Z = 0
#   thumb_world_z = gz + R[2,:] @ (thumb_open_h − ref_mid_h) = 0
#   ⟹  gz_crash = − R[2,:] @ (thumb_open_h − r_28.midpoint)
# ---------------------------------------------------------------------------
R_28 = world_R(r_28.base_tilt_y, 0.0)
gz_thumb_crash = -float(R_28[2, :] @ (thumb_open_h - r_28.midpoint))
gz_index_crash = -float(R_28[2, :] @ (index_open_h - r_28.midpoint))
gz_crash = max(gz_thumb_crash, gz_index_crash)

print(f"  r_28 tilt  : {np.degrees(r_28.base_tilt_y):+.1f}°")
print(f"  r_33 tilt  : {np.degrees(r_33.base_tilt_y):+.1f}°")
print(f"  r_38 tilt  : {np.degrees(r_38.base_tilt_y):+.1f}°")
print(f"  gz_crash   : {gz_crash*1e3:.1f} mm  (GZ = {GZ*1e3:.1f} mm)")

if gz_crash <= GZ:
    print("  ⚠  No natural crash — bumping gz_crash for illustration.")
    gz_crash = GZ + 0.018   # artificial: 18 mm above grasp Z

gz_half = (GZ_START + GZ) / 2

# ---------------------------------------------------------------------------
# Build all waypoint positions
# ---------------------------------------------------------------------------
WP_start = hand_to_world(tips_open,   r_28.midpoint, r_28.base_tilt_y, EXTRA_TILT,   GZ_START)

naive = {
    "start":  WP_start,
    "half":   hand_to_world(tips_open,   r_28.midpoint, r_28.base_tilt_y, EXTRA_TILT/2, gz_half),
    "crash":  hand_to_world(tips_open,   r_28.midpoint, r_28.base_tilt_y, 0.0,          gz_crash),
    "whiff":  hand_to_world(tips_28,     r_28.midpoint, r_28.base_tilt_y, 0.0,          gz_crash),
}

reflex = {
    # "start":    WP_start,
    "start":    hand_to_world(tips_reflex, r_28.midpoint, r_28.base_tilt_y, EXTRA_TILT/2, GZ_START),
    "half":     hand_to_world(tips_reflex, r_28.midpoint, r_28.base_tilt_y, EXTRA_TILT/2, gz_half),
    "at_grasp": hand_to_world(tips_reflex, r_28.midpoint, r_28.base_tilt_y, 0.0,          GZ),
    "grasp":    hand_to_world(tips_28,     r_28.midpoint, r_28.base_tilt_y, 0.0,          GZ),
}

plan = {
    "start":    WP_start,
    "half":     hand_to_world(tips_38, r_38.midpoint, r_38.base_tilt_y, EXTRA_TILT/2, GZ_START),
    "approach": hand_to_world(tips_38, r_38.midpoint, r_38.base_tilt_y, 0.0,          GZ),
    "step1":    hand_to_world(tips_33, r_33.midpoint, r_33.base_tilt_y, 0.0,          GZ),
    "grasp":    hand_to_world(tips_28, r_28.midpoint, r_28.base_tilt_y, 0.0,          GZ),
}

# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams.update({
    "font.family":  "sans-serif",
    "font.size":    9,
    "axes.linewidth": 0,
})

# Colour palette
C_THUMB  = "#F01E0A"   # vivid red
C_INDEX  = "#0A7FFF"   # vivid blue
C_BASE   = "#2C2C2C"   # near-black
C_PINCH  = "#999999"   # grey connector
C_OBJ    = "#F2C14E"   # amber
C_OBJ_E  = "#C89020"   # amber edge
C_GND    = "#CCCCCC"   # ground line
C_GND_F  = "#F0F0F0"   # ground fill
C_COLL   = "#CC2222"   # collision marker
C_PATH   = "#BBBBBB"   # trajectory faint

def _xz(p: np.ndarray):
    """Return (x, z) in mm from a world-frame 3-vector (x negated so wrist is on left)."""
    return -p[0] * 1e3, p[2] * 1e3


_XLO, _XHI = -100.0, 80.0   # mm
_ZLO, _ZHI = -8.0,  133.0   # mm


def draw_scene(ax):
    """Ground fill + line + object rectangle."""
    ax.fill_between([_XLO, _XHI], _ZLO, 0, color=C_GND_F, zorder=0)
    ax.axhline(0, color=C_GND, lw=1.5, zorder=1)
    rect = Rectangle(
        (-OBJECT_W / 2 * 1e3, 0), OBJECT_W * 1e3, OBJECT_H * 1e3,
        fc=C_OBJ, ec=C_OBJ_E, lw=1.0, alpha=0.92, zorder=2
    )
    ax.add_patch(rect)


def _proxy_base(pos: dict, offset_mm: float = 45.0, proxy_mid_world=None) -> tuple[float, float]:
    """
    Visible proxy for the (off-screen) wrist: step offset_mm from the tip
    midpoint toward the actual base, clamped inside the viewport.

    proxy_mid_world: optional world-frame reference point (metres, 3-vector)
    to use instead of the current tip midpoint.  Pass this to keep the proxy
    fixed when only finger positions change (e.g. reflex at_grasp → grasp).
    """
    if proxy_mid_world is not None:
        mx, mz = float(proxy_mid_world[0]) * 1e3, float(proxy_mid_world[2]) * 1e3
    else:
        tx, tz = _xz(pos["thumb"])
        ix, iz = _xz(pos["index"])
        mx, mz = (tx + ix) / 2, (tz + iz) / 2
    bx, bz = _xz(pos["base"])
    dx, dz = bx - mx, bz - mz
    length = np.hypot(dx, dz)
    if length < 1e-9:
        return float(np.clip(bx, _XLO + 3, _XHI - 3)), float(np.clip(bz, _ZLO + 3, _ZHI - 3))
    ux, uz = dx / length, dz / length
    px = float(np.clip(mx + ux * offset_mm, _XLO + 3, _XHI - 3))
    pz = float(np.clip(mz + uz * offset_mm, _ZLO + 3, _ZHI - 3))
    return px, pz


def draw_hand(ax, pos, alpha=1.0, zorder=5, draw_lines=True, proxy_mid_world=None):
    """Three dots: thumb tip (red), index tip (blue), wrist proxy (black).
    draw_lines: add semi-transparent grey dashed connectors (skip when dots are very close).
    proxy_mid_world: forwarded to _proxy_base to pin the proxy reference point."""
    tx, tz = _xz(pos["thumb"])
    ix, iz = _xz(pos["index"])
    px, pz = _proxy_base(pos, proxy_mid_world=proxy_mid_world)

    # if draw_lines:
    la = alpha * 1.0
    lkw = dict(color="#AAAAAA", lw=0.7, ls=(0, (3, 4)), alpha=la,
                zorder=zorder - 1, clip_on=True)
    ax.plot([px, tx], [pz, tz], **lkw)
    ax.plot([px, ix], [pz, iz], **lkw)

    ms = 8.0
    ax.plot(px, pz, "o", color=C_BASE,  ms=ms, alpha=alpha, mec="white", mew=1.2, zorder=zorder,     clip_on=True)
    ax.plot(tx, tz, "o", color=C_THUMB, ms=ms, alpha=alpha, mec="white", mew=1.2, zorder=zorder + 1, clip_on=True)
    ax.plot(ix, iz, "o", color=C_INDEX, ms=ms, alpha=alpha, mec="white", mew=1.2, zorder=zorder + 1, clip_on=True)


def setup_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(_XLO, _XHI)
    ax.set_ylim(_ZLO, _ZHI)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=16, fontweight="bold", color="#222222", pad=10,
                 fontfamily="sans-serif")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(13, 5.8))
fig.patch.set_facecolor("white")
plt.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.17, wspace=0.06)

GZ_MM = GZ * 1e3

def _subtitle(ax, text):
    """Small italic subtitle centred just below axes bottom."""
    ax.text(0.5, -0.055, text, transform=ax.transAxes,
            fontsize=14, color="#888888", ha="center", va="top", style="italic")

def _pinch_width_label(ax, pos, label, z_abs, alpha=1.0, fontsize=7.0):
    """Label at a fixed absolute Z, centered between the two tips in X."""
    tx, _ = _xz(pos["thumb"])
    ix, _ = _xz(pos["index"])
    ax.text((tx + ix) / 2, z_abs, label, fontsize=fontsize, color="#555555",
            ha="center", va="bottom", alpha=alpha)

# ── Column 1 · Plan ─────────────────────────────────────────────────────────
ax = axes[0]
setup_ax(ax, "Plan")
_subtitle(ax, "offset pregrasp → iterative closure")
draw_scene(ax)

# Plan starts at 38 mm approach pose (no fully-open start/half shown)
draw_hand(ax, plan["half"], alpha=0.35, zorder=6)
draw_hand(ax, plan["approach"], alpha=0.55, zorder=6)
draw_hand(ax, plan["step1"],    alpha=0.75, zorder=7, draw_lines=False)
draw_hand(ax, plan["grasp"],    alpha=1.00, zorder=8, draw_lines=False)

# Width step labels between each pair of tips (at gz level)
# Labels above the object (top at 20 mm), stacked with 4 mm spacing
_pinch_width_label(ax, plan["half"], "38 mm", z_abs=46, alpha=0.35, fontsize=9)
_pinch_width_label(ax, plan["approach"], "38 mm", z_abs=32, alpha=0.55, fontsize=9)
_pinch_width_label(ax, plan["step1"],    "33 mm", z_abs=27, alpha=0.75, fontsize=9)
_pinch_width_label(ax, plan["grasp"],    "28 mm", z_abs=22, alpha=1.00, fontsize=9)

ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Column 2 · Thumb Reflex ─────────────────────────────────────────────────
ax = axes[1]
setup_ax(ax, "Thumb Reflex")
_subtitle(ax, "pre-set thumb → reactive index closure")
draw_scene(ax)

# at_grasp and grasp share the same gz, R, and base position.
# Pin proxy_mid_world to the world-frame grasp centre so the wrist proxy
# dot is identical across those two steps (only index moves).
_GZ_WORLD = np.array([0., 0., GZ])

draw_hand(ax, reflex["start"],    alpha=0.35, zorder=4,
          proxy_mid_world=np.array([0., 0., GZ_START]))
draw_hand(ax, reflex["half"],     alpha=0.55, zorder=5,
          proxy_mid_world=np.array([0., 0., gz_half]))
draw_hand(ax, reflex["at_grasp"], alpha=0.75, zorder=6, proxy_mid_world=_GZ_WORLD)
draw_hand(ax, reflex["grasp"],    alpha=1.00, zorder=7, proxy_mid_world=_GZ_WORLD)

# Arrow: index snaps closed
ig_x, ig_z = _xz(reflex["at_grasp"]["index"])
fg_x, fg_z = _xz(reflex["grasp"]["index"])
ax.annotate("", xy=(fg_x, fg_z), xytext=(ig_x, ig_z),
            arrowprops=dict(arrowstyle="-|>", color=C_INDEX, lw=1.3,
                            connectionstyle="arc3,rad=-0.4"),
            zorder=8)

ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Column 3 · Naive ────────────────────────────────────────────────────────
ax = axes[2]
setup_ax(ax, "Naive")
_subtitle(ax, "open fingers → close at target")
draw_scene(ax)

draw_hand(ax, naive["start"], alpha=0.35, zorder=4)
draw_hand(ax, naive["half"],  alpha=0.55, zorder=5)
draw_hand(ax, naive["crash"], alpha=0.75, zorder=6)
# naive never reaches full opacity — failure stops at ground collision

# Collision marker — thumb hits ground
th_crash_x, _ = _xz(naive["crash"]["thumb"])
ax.plot(th_crash_x, 0, "x", color=C_COLL, ms=10, mew=2.2, zorder=9, alpha=0.9)
ax.text(th_crash_x, 2, "ground\ncollision", fontsize=9.5, color=C_COLL,
        ha="center", va="bottom", zorder=10)
ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Shared legend ─────────────────────────────────────────────────────────────
# Row 1: dot type colours (single marker each)
# Row 2: step progression — each entry shows all three marker colours at that
#         step's transparency so the reader can identify any dot type per step.
def _step_markers(alpha):
    return (
        Line2D([0], [0], marker='o', ls='none', mfc=C_BASE,  mec="white", mew=1.0, ms=9, alpha=alpha),
        Line2D([0], [0], marker='o', ls='none', mfc=C_THUMB, mec="white", mew=1.0, ms=9, alpha=alpha),
        Line2D([0], [0], marker='o', ls='none', mfc=C_INDEX, mec="white", mew=1.0, ms=9, alpha=alpha),
    )

leg_items = [
    Line2D([0],[0], marker='o', ls='none', mfc=C_THUMB, mec="white", mew=1.0, ms=9, label="Thumb tip"),
    Line2D([0],[0], marker='o', ls='none', mfc=C_INDEX, mec="white", mew=1.0, ms=9, label="Index tip"),
    Line2D([0],[0], marker='o', ls='none', mfc=C_BASE,  mec="white", mew=1.0, ms=9, label="Wrist (proxy)"),
    mpatches.Patch(fc=C_OBJ, ec=C_OBJ_E, lw=0.8, label="Object (28 × 20 mm)"),
    _step_markers(0.35),
    _step_markers(0.55),
    _step_markers(0.75),
    _step_markers(1.00),
]
leg_labels = [
    "Thumb tip", "Index tip", "Wrist (proxy)", "Object (28 × 20 mm)",
    "Step 1", "Step 2", "Step 3", "Final",
]
fig.legend(
    handles=leg_items, labels=leg_labels,
    handler_map={tuple: HandlerTuple(ndivide=None, pad=0.5)},
    loc="lower center", ncol=4,
    fontsize=8.5, frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    handlelength=1.8, columnspacing=1.6,
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = _ROOT / "grasp_strategy_comparison.pdf"
out_png = _ROOT / "grasp_strategy_comparison.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")
# plt.show()  # disabled for headless; open the saved files instead

# =============================================================================
# SECOND FIGURE — Plan Fan vs Naive
# =============================================================================
_DISP_OFFSET  = 12                                          # mm; displayed = actual − 12
_PLAN_ACT_MM  = [12 + i * 10 for i in range(11)]           # [12, 22, …, 112]
_PLAN_DSP_MM  = [w - _DISP_OFFSET for w in _PLAN_ACT_MM]   # [0, 10, …, 100]
_NAIVE_ACT_MM = 122                                         # displayed as 110

# ── Solve closures ────────────────────────────────────────────────────────────
print("Solving fan closures…")
_fan_r  = [cg.solve("2-finger line", w / 1000) for w in _PLAN_ACT_MM]
_r_wide = cg.solve("2-finger line", _NAIVE_ACT_MM / 1000)

# Plan fan: all at GZ, each with its tilt-corrected orientation
_fan_pos = [
    hand_to_world(r.tip_positions, r.midpoint, r.base_tilt_y, 0.0, GZ)
    for r in _fan_r
]

# Naive whiff: close to 12 mm WITHOUT re-tilting (wrist stays fixed at 122 mm pose)
_R_wide = world_R(_r_wide.base_tilt_y, 0.0)
_r_min  = _fan_r[0]                                         # 12 mm closure

# Raise both naive poses so whiff tips clear z=0 by at least 3 mm
_NAIVE_MARGIN = 0.003
_whiff_zs = [
    GZ + float((_R_wide @ (_r_min.tip_positions[k] - _r_wide.midpoint))[2])
    for k in _r_min.tip_positions
]
_NAIVE_GZ = GZ + max(0.0, _NAIVE_MARGIN - min(_whiff_zs))

_naive_open = hand_to_world(
    _r_wide.tip_positions, _r_wide.midpoint, _r_wide.base_tilt_y, 0.0, _NAIVE_GZ
)

_wrist_naive = np.array([0.0, 0.0, _NAIVE_GZ]) - _R_wide @ _r_wide.midpoint
_whiff: dict = {"base": _wrist_naive.copy()}
for _k, _v in _r_min.tip_positions.items():
    _whiff[_k] = _R_wide @ _v + _wrist_naive

_whiff_mid   = (_whiff["thumb"] + _whiff["index"]) / 2
_whiff_xd    = float(-_whiff_mid[0] * 1e3)                 # display X
_whiff_zd    = float( _whiff_mid[2] * 1e3)                 # display Z
_delta_z_mm  = _whiff_zd - GZ * 1e3
_delta_x_mm  = _whiff_xd                                   # object centre is display-x = 0

print(f"  whiff midpoint: xd={_whiff_xd:.1f} mm  zd={_whiff_zd:.1f} mm")
print(f"  Δz = {_delta_z_mm:.1f} mm   Δx = {_delta_x_mm:.1f} mm")

# ── Viewport ─────────────────────────────────────────────────────────────────
_F2XLO, _F2XHI = -82.0, 82.0
_F2ZLO, _F2ZHI = -12.0, max(108.0, _whiff_zd + 22.0)

# ── Figure ───────────────────────────────────────────────────────────────────
fig2, (axP, axN) = plt.subplots(1, 2, figsize=(9.6, 6.2))
fig2.patch.set_facecolor("white")
plt.subplots_adjust(left=0.02, right=0.98, top=0.91, bottom=0.08, wspace=0.05)


def _setup_fan_ax(ax, title):
    ax.set_aspect("equal")
    ax.set_xlim(_F2XLO, _F2XHI)
    ax.set_ylim(_F2ZLO, _F2ZHI)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=16, fontweight="bold", color="#222222", pad=10,
                 fontfamily="sans-serif")


def _draw_fan_hand(ax, pos, alpha=1.0, zorder=5, proxy_mid=None):
    """draw_hand with clamping to the fan-figure viewport bounds.
    proxy_mid: optional (mx_mm, mz_mm) display-coord reference to use instead
    of the current tip midpoint when computing the proxy base position.
    Pass the same value for poses that should share a wrist dot location."""
    tx, tz = _xz(pos["thumb"])
    ix, iz = _xz(pos["index"])
    if proxy_mid is not None:
        mx, mz = proxy_mid
    else:
        mx, mz = (tx + ix) / 2, (tz + iz) / 2
    bx, bz = _xz(pos["base"])
    dx, dz = bx - mx, bz - mz
    dist = np.hypot(dx, dz)
    if dist > 1e-9:
        ux, uz = dx / dist, dz / dist
        px = float(np.clip(mx + ux * 80, _F2XLO + 3, _F2XHI - 3))
        pz = float(np.clip(mz + uz * 80, _F2ZLO + 3, _F2ZHI - 3))
    else:
        px = float(np.clip(bx, _F2XLO + 3, _F2XHI - 3))
        pz = float(np.clip(bz, _F2ZLO + 3, _F2ZHI - 3))
    lkw = dict(color="#AAAAAA", lw=0.7, ls=(0, (3, 4)), alpha=alpha,
               zorder=zorder - 1, clip_on=True)
    ax.plot([px, tx], [pz, tz], **lkw)
    ax.plot([px, ix], [pz, iz], **lkw)
    ms = 8.0
    ax.plot(px, pz, "o", color=C_BASE,  ms=ms, alpha=alpha, mec="white", mew=1.2,
            zorder=zorder,     clip_on=True)
    ax.plot(tx, tz, "o", color=C_THUMB, ms=ms, alpha=alpha, mec="white", mew=1.2,
            zorder=zorder + 1, clip_on=True)
    ax.plot(ix, iz, "o", color=C_INDEX, ms=ms, alpha=alpha, mec="white", mew=1.2,
            zorder=zorder + 1, clip_on=True)


def _draw_scene_fan(ax):
    ax.fill_between([_F2XLO, _F2XHI], _F2ZLO, 0, color=C_GND_F, zorder=0)
    ax.axhline(0, color=C_GND, lw=1.5, zorder=1)
    ax.axhline(GZ * 1e3, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)


# ── Plan column (fan) ─────────────────────────────────────────────────────────
_setup_fan_ax(axP, "Plan")
_subtitle(axP, "width-adaptive tilt at each closure step")
_draw_scene_fan(axP)

_N = len(_fan_pos)
# Draw widest → narrowest (back to front); narrowest gets highest alpha
for _i, (_pos, _dsp) in enumerate(zip(_fan_pos[::-1], _PLAN_DSP_MM[::-1])):
    _alpha = 0.14 + 0.82 * (_i / (_N - 1))
    _draw_fan_hand(axP, _pos, alpha=_alpha, zorder=5 + _i)


# ── Naive column ──────────────────────────────────────────────────────────────
_setup_fan_ax(axN, "Naive")
_subtitle(axN, "fixed orientation → close fingers in place")
_draw_scene_fan(axN)

# Shared proxy reference: both naive poses have the same wrist, so pin the
# proxy dot to the open-pose tip midpoint (display coords) for both draws.
_naive_proxy_mid = (0.0, _NAIVE_GZ * 1e3)

# 110 mm open grasp
_draw_fan_hand(axN, _naive_open, alpha=0.55, zorder=6, proxy_mid=_naive_proxy_mid)

# Label for the open pose width
_n_tx, _ = _xz(_naive_open["thumb"])
_n_ix, _ = _xz(_naive_open["index"])
axN.text((_n_tx + _n_ix) / 2, _NAIVE_GZ * 1e3 - 4, "110 mm",
         fontsize=9.5, color="#888888", ha="center", va="top", alpha=0.8)

# Whiff (close to 12 mm, no tilt change) — same proxy_mid keeps wrist dot fixed
_draw_fan_hand(axN, _whiff, alpha=1.00, zorder=8, proxy_mid=_naive_proxy_mid)

# ── Δz annotation (vertical double-headed arrow) ─────────────────────────────
_gz_z   = GZ * 1e3
# Place arrow to the right of whiff if space, otherwise left
_arr_x  = _whiff_xd + 16 if _whiff_xd < _F2XHI - 20 else _whiff_xd - 16

# Horizontal tick lines at the two Z levels
for _zl in [_gz_z, _whiff_zd]:
    axN.plot([_arr_x - 4, _arr_x + 4], [_zl, _zl], color="#444444", lw=0.9, zorder=10)

axN.annotate(
    "", xy=(_arr_x, _whiff_zd), xytext=(_arr_x, _gz_z),
    arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4, mutation_scale=9),
    zorder=11,
)

# ── Δx annotation (horizontal double-headed arrow, only if meaningful) ────────
if abs(_delta_x_mm) > 4:
    _arr_z_x = _whiff_zd + 14
    # tick marks at x=0 and x=_whiff_xd
    for _xl in [0.0, _whiff_xd]:
        axN.plot([_xl, _xl], [_arr_z_x - 3, _arr_z_x + 3], color="#444444", lw=0.9, zorder=10)
    axN.annotate(
        "", xy=(_whiff_xd, _arr_z_x), xytext=(0.0, _arr_z_x),
        arrowprops=dict(arrowstyle="<->", color="#444444", lw=1.4, mutation_scale=9),
        zorder=11,
    )

# ── Save ─────────────────────────────────────────────────────────────────────
out2_pdf = _ROOT / "plan_vs_naive_comparison.pdf"
out2_png = _ROOT / "plan_vs_naive_comparison.png"
fig2.savefig(out2_pdf, dpi=150, bbox_inches="tight", facecolor="white")
fig2.savefig(out2_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out2_pdf}")
print(f"Saved → {out2_png}")
