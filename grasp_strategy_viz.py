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
GZ_START   = 0.060          # 6 cm — starting height (lower keeps start index in viewport)

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
    "start":    WP_start,
    "half":     hand_to_world(tips_reflex, r_28.midpoint, r_28.base_tilt_y, EXTRA_TILT/2, gz_half),
    "at_grasp": hand_to_world(tips_reflex, r_28.midpoint, r_28.base_tilt_y, 0.0,          GZ),
    "grasp":    hand_to_world(tips_28,     r_28.midpoint, r_28.base_tilt_y, 0.0,          GZ),
}

plan = {
    "start":    WP_start,
    "half":     hand_to_world(tips_38, r_38.midpoint, r_38.base_tilt_y, EXTRA_TILT/2, gz_half),
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
C_THUMB  = "#D64B3A"   # warm red
C_INDEX  = "#2D78C8"   # steel blue
C_BASE   = "#2C2C2C"   # near-black
C_PINCH  = "#999999"   # grey connector
C_OBJ    = "#F2C14E"   # amber
C_OBJ_E  = "#C89020"   # amber edge
C_GND    = "#CCCCCC"   # ground line
C_GND_F  = "#F0F0F0"   # ground fill
C_COLL   = "#CC2222"   # collision marker
C_PATH   = "#BBBBBB"   # trajectory faint

def _xz(p: np.ndarray):
    """Return (x, z) in mm from a world-frame 3-vector."""
    return p[0] * 1e3, p[2] * 1e3


_XLO, _XHI = -80.0, 100.0   # mm
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


def _proxy_base(pos: dict, offset_mm: float = 45.0) -> tuple[float, float]:
    """
    Visible proxy for the (off-screen) wrist: step offset_mm from the tip
    midpoint toward the actual base, clamped inside the viewport.
    """
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


def draw_hand(ax, pos, alpha=1.0, zorder=5, draw_lines=True):
    """Three dots: thumb tip (red), index tip (blue), wrist proxy (black).
    draw_lines: add semi-transparent grey dashed connectors (skip when dots are very close)."""
    tx, tz = _xz(pos["thumb"])
    ix, iz = _xz(pos["index"])
    px, pz = _proxy_base(pos)

    # if draw_lines:
    la = alpha * 1.0
    lkw = dict(color="#AAAAAA", lw=0.7, ls=(0, (3, 4)), alpha=la,
                zorder=zorder - 1, clip_on=True)
    ax.plot([px, tx], [pz, tz], **lkw)
    ax.plot([px, ix], [pz, iz], **lkw)

    ms = 5.5
    ax.plot(px, pz, "o", color=C_BASE,  ms=ms, alpha=alpha, zorder=zorder,     clip_on=True)
    ax.plot(tx, tz, "o", color=C_THUMB, ms=ms, alpha=alpha, zorder=zorder + 1, clip_on=True)
    ax.plot(ix, iz, "o", color=C_INDEX, ms=ms, alpha=alpha, zorder=zorder + 1, clip_on=True)


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

# ── Column 1 · Naive ────────────────────────────────────────────────────────
ax = axes[0]
setup_ax(ax, "Naive")
_subtitle(ax, "open fingers → close at target")
draw_scene(ax)

draw_hand(ax, naive["start"], alpha=0.15, zorder=4)
draw_hand(ax, naive["half"],  alpha=0.30, zorder=5)
draw_hand(ax, naive["crash"], alpha=0.62, zorder=6)
draw_hand(ax, naive["whiff"], alpha=1.00, zorder=7)

# Collision marker — thumb hits ground
th_crash_x, _ = _xz(naive["crash"]["thumb"])
ax.plot(th_crash_x, 0, "x", color=C_COLL, ms=10, mew=2.2, zorder=9, alpha=0.9)
ax.text(th_crash_x, 2, "ground\ncollision", fontsize=9.5, color=C_COLL,
        ha="center", va="bottom", zorder=10)
ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Column 2 · Thumb Reflex ─────────────────────────────────────────────────
ax = axes[1]
setup_ax(ax, "Thumb Reflex")
_subtitle(ax, "pre-set thumb → reactive index closure")
draw_scene(ax)

draw_hand(ax, reflex["start"],    alpha=0.15, zorder=4)
draw_hand(ax, reflex["half"],     alpha=0.30, zorder=5)
draw_hand(ax, reflex["at_grasp"], alpha=0.62, zorder=6)
draw_hand(ax, reflex["grasp"],    alpha=1.00, zorder=7)

# Arrow: index snaps closed
ig_x, ig_z = _xz(reflex["at_grasp"]["index"])
fg_x, fg_z = _xz(reflex["grasp"]["index"])
ax.annotate("", xy=(fg_x, fg_z), xytext=(ig_x, ig_z),
            arrowprops=dict(arrowstyle="-|>", color=C_INDEX, lw=1.3,
                            connectionstyle="arc3,rad=0.4"),
            zorder=8)

ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Column 3 · Plan ─────────────────────────────────────────────────────────
ax = axes[2]
setup_ax(ax, "Plan")
_subtitle(ax, "offset pregrasp → iterative closure")
draw_scene(ax)

draw_hand(ax, plan["start"],    alpha=0.15, zorder=4)
draw_hand(ax, plan["half"],     alpha=0.28, zorder=5)
draw_hand(ax, plan["approach"], alpha=0.48, zorder=6, draw_lines=False)
draw_hand(ax, plan["step1"],    alpha=0.72, zorder=7, draw_lines=False)
draw_hand(ax, plan["grasp"],    alpha=1.00, zorder=8, draw_lines=False)

# Width step labels between each pair of tips (at gz level)
# Labels above the object (top at 20 mm), stacked with 4 mm spacing
_pinch_width_label(ax, plan["approach"], "38 mm", z_abs=32, alpha=0.55, fontsize=9)
_pinch_width_label(ax, plan["step1"],    "33 mm", z_abs=27, alpha=0.78, fontsize=9)
_pinch_width_label(ax, plan["grasp"],    "28 mm", z_abs=22, alpha=1.00, fontsize=9)

ax.axhline(GZ_MM, color="#BBBBBB", lw=0.7, ls=(0, (3, 4)), alpha=0.7, zorder=2)

# ── Shared legend ─────────────────────────────────────────────────────────────
# Row 1: dot type colours
# Row 2: step progression (four shades of grey, lightest → darkest)
leg_items = [
    Line2D([0],[0], marker='o', ls='none', mfc=C_THUMB, mec=C_THUMB, ms=7, label="Thumb tip"),
    Line2D([0],[0], marker='o', ls='none', mfc=C_INDEX, mec=C_INDEX, ms=7, label="Index tip"),
    Line2D([0],[0], marker='o', ls='none', mfc=C_BASE,  mec=C_BASE,  ms=7, label="Wrist (proxy)"),
    mpatches.Patch(fc=C_OBJ, ec=C_OBJ_E, lw=0.8, label="Object (28 × 20 mm)"),
    Line2D([0],[0], marker='o', ls='none', mfc='#555555', mec='#555555', ms=7, alpha=0.15, label="Step 1"),
    Line2D([0],[0], marker='o', ls='none', mfc='#555555', mec='#555555', ms=7, alpha=0.40, label="Step 2"),
    Line2D([0],[0], marker='o', ls='none', mfc='#555555', mec='#555555', ms=7, alpha=0.70, label="Step 3"),
    Line2D([0],[0], marker='o', ls='none', mfc='#555555', mec='#555555', ms=7, alpha=1.00, label="Final"),
]
fig.legend(
    handles=leg_items, loc="lower center", ncol=4,
    fontsize=8.5, frameon=False,
    bbox_to_anchor=(0.5, 0.01),
    handlelength=0.8, columnspacing=1.6,
)

# ── Save ─────────────────────────────────────────────────────────────────────
out_pdf = _ROOT / "grasp_strategy_comparison.pdf"
out_png = _ROOT / "grasp_strategy_comparison.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_pdf}")
print(f"Saved → {out_png}")
# plt.show()  # disabled for headless; open the saved files instead
