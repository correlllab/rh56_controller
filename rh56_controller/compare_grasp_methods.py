"""
compare_grasp_methods.py — Analytical vs mink IK grasp planner comparison.

Sweeps the full width/diameter range for all grasp modes (2, 3, 4, 5 finger,
cylinder), runs both planners with identical targets, and produces:

  mink_vs_analytical/
    grasp_comparison.png        — mega overview (5 modes × 5 metrics)
    2/comparison.png            — 2-finger line: per-finger ctrl + metrics
    3/comparison.png            — 3-finger plane
    4/comparison.png            — 4-finger plane
    5/comparison.png            — 5-finger plane
    cyl/comparison.png          — cylinder (power grasp)

Usage:
    uv run python -m rh56_controller.compare_grasp_methods

Metrics per grasp mode:
  • Tip position error  (mm)  — how close each planner's tips are to the target
  • World-frame Z span  (mm)  — max–min Z of all active fingertips in world frame
  • Mink iterations           — IK iterations to convergence per width
  • Per-finger ctrl    (deg)  — proximal joint ctrl vs width, both planners
  • Per-finger smoothness     — |Δctrl/Δwidth| derivative (smoothness check)

Analytical planner (ClosureGeometry):
  Analytical FK tables + brentq root-finding.  Zero tip error by construction.
  Sub-millisecond per grasp.

Mink planner (MinkGraspPlanner):
  Differential IK starting from open hand.  EqualityConstraintTask enforces
  joint coupling automatically.  Iterates until convergence.
  Targets: identical to analytical planner's computed tip positions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from .grasp_geometry import (
    ClosureGeometry,
    ClosureResult,
    InspireHandFK,
    NON_THUMB_FINGERS,
    GRASP_FINGER_SETS,
)
from .mink_grasp_planner import MinkGraspPlanner, MinkGraspResult

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE      = Path(__file__).parent.parent
_GRASP_XML = _HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml"
_RIGHT_XML = _HERE / "h1_mujoco" / "inspire" / "inspire_right.xml"
_OUT_ROOT  = _HERE / "plots" / "mink_vs_analytical"

# ---------------------------------------------------------------------------
# Ctrl index mapping
# mink ctrl array: [pinky=0, ring=1, middle=2, index=3, thumb_bend=4, thumb_yaw=5]
# ---------------------------------------------------------------------------
_FINGER_TO_MINK_IDX: Dict[str, int] = {
    "pinky": 0, "ring": 1, "middle": 2, "index": 3, "thumb": 4,
}


def _finger_ctrl_key(f: str) -> str:
    """Map finger display-name to ClosureResult.ctrl_values key."""
    return "thumb_proximal" if f == "thumb" else f


# ---------------------------------------------------------------------------
# Colors / style
# ---------------------------------------------------------------------------
_C_ANALYTICAL = "#2196F3"   # blue — ClosureGeometry
_C_MINK      = "#FF5722"   # orange — mink IK
_ALPHA_SHADE = 0.15


# ---------------------------------------------------------------------------
# World-frame helpers
# ---------------------------------------------------------------------------

def _world_tips_mink(mink_r: MinkGraspResult, c_res: ClosureResult) -> Dict[str, np.ndarray]:
    """Apply analytical planner's tilt rotation to mink's base-frame tips → world frame."""
    R      = ClosureResult._rot_matrix(c_res.base_tilt_y)
    base_w = -(R @ c_res.midpoint)   # gz = 0
    return {f: R @ pos + base_w for f, pos in mink_r.tip_positions.items()}


def _z_span_mm(world_tips: Dict[str, np.ndarray], fingers: List[str]) -> float:
    """World-frame Z span (max − min) over given fingers [mm]."""
    zvals = [world_tips[f][2] for f in fingers if f in world_tips]
    return (max(zvals) - min(zvals)) * 1000.0 if len(zvals) >= 2 else 0.0


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _analytical_pos_error(result: ClosureResult, target_tips: Dict) -> float:
    errs = [
        float(np.linalg.norm(result.tip_positions[f] - target_tips[f]))
        for f in target_tips
        if f in result.tip_positions
    ]
    return float(np.mean(errs)) * 1000.0


def _analytical_coplan_err(result: ClosureResult, non_thumb_fingers: List[str]) -> float:
    if len(non_thumb_fingers) < 2:
        return 0.0
    wtips = result.world_tips(world_grasp_z=0.0)
    zvals = np.array([wtips[f][2] for f in non_thumb_fingers if f in wtips])
    return float(np.mean(np.abs(zvals - zvals.mean()))) * 1000.0 if len(zvals) >= 2 else 0.0


def _mink_coplan_err(mink_r: MinkGraspResult, c_res: ClosureResult,
                     non_thumb_fingers: List[str]) -> float:
    present = [f for f in non_thumb_fingers if f in mink_r.tip_positions]
    if len(present) < 2:
        return 0.0
    wtips = _world_tips_mink(mink_r, c_res)
    zvals = np.array([wtips[f][2] for f in present])
    return float(np.mean(np.abs(zvals - zvals.mean()))) * 1000.0


def _mink_mean_pos_error(mink_r: MinkGraspResult) -> float:
    return float(np.mean(list(mink_r.position_errors_m.values()))) * 1000.0


def _mink_max_pos_error(mink_r: MinkGraspResult) -> float:
    return float(np.max(list(mink_r.position_errors_m.values()))) * 1000.0


def _ctrl_smoothness(ctrl_vals: np.ndarray, widths_m: np.ndarray) -> np.ndarray:
    """Discrete |Δctrl/Δwidth|."""
    if len(ctrl_vals) < 2:
        return np.zeros(len(ctrl_vals))
    dctrl = np.abs(np.diff(ctrl_vals))
    dw = np.diff(widths_m)
    dw = np.where(dw == 0, 1e-9, dw)
    return np.concatenate([[0.0], dctrl / dw])


def _collect_per_finger(
    c_res: ClosureResult,
    m_res: MinkGraspResult,
    active_fingers: List[str],
    pfc_analytical: Dict[str, list],
    pfc_mink: Dict[str, list],
) -> None:
    """Append per-finger proximal ctrl for one width sample."""
    for f in active_fingers:
        pfc_analytical[f].append(c_res.ctrl_values.get(_finger_ctrl_key(f), 0.0))
        pfc_mink[f].append(float(m_res.ctrl[_FINGER_TO_MINK_IDX[f]]))


# ---------------------------------------------------------------------------
# Per-mode sweep functions
# ---------------------------------------------------------------------------

def sweep_line(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
) -> dict:
    """Sweep 2-finger line grasp."""
    widths = np.linspace(0.012, 0.110, n_widths)
    active = ["index", "thumb"]

    analytical_pos_errs, analytical_coplan, analytical_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times, mink_conv = [], [], [], []
    valid_widths = []
    analytical_thumb_ctrl, mink_thumb_ctrl = [], []
    z_span_analytical, z_span_mink = [], []
    mink_ctrl_arrs = []
    pfc_analytical = {f: [] for f in active}
    pfc_mink   = {f: [] for f in active}

    for w in widths:
        try:
            t0 = time.perf_counter()
            c_res = closure.line(w)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [line] analytical FAIL @ w={w*1000:.1f}mm: {e}")
            continue

        targets = {f: c_res.tip_positions[f] for f in c_res.tip_positions}
        m_res = mink_planner.solve_line(
            w,
            thumb_target=targets["thumb"],
            index_target=targets["index"],
        )

        valid_widths.append(w)
        analytical_pos_errs.append(_analytical_pos_error(c_res, targets))
        analytical_coplan.append(0.0)   # only 1 non-thumb finger — coplan not applicable
        analytical_times.append(c_time * 1000)
        analytical_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(0.0)
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))
        mink_ctrl_arrs.append(m_res.ctrl.copy())

        wtips_c = c_res.world_tips(world_grasp_z=0.0)
        z_span_analytical.append(_z_span_mm(wtips_c, active))
        z_span_mink.append(_z_span_mm(_world_tips_mink(m_res, c_res), active))
        _collect_per_finger(c_res, m_res, active, pfc_analytical, pfc_mink)

    w_arr = np.array(valid_widths) * 1000
    return dict(
        widths_mm=w_arr,
        active_fingers=active,
        analytical_pos_errs=np.array(analytical_pos_errs),
        analytical_coplan=np.array(analytical_coplan),
        analytical_times=np.array(analytical_times),
        analytical_thumb_ctrl=np.array(analytical_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_coplan=np.array(mink_coplan),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
        mink_ctrl_arr=np.array(mink_ctrl_arrs),
        z_span_analytical=np.array(z_span_analytical),
        z_span_mink=np.array(z_span_mink),
        per_finger_ctrl_analytical={f: np.array(v) for f, v in pfc_analytical.items()},
        per_finger_ctrl_mink={f: np.array(v) for f, v in pfc_mink.items()},
    )


def sweep_plane(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
    n_fingers: int = 4,
) -> dict:
    """Sweep n-finger plane grasp (n ∈ {3, 4, 5})."""
    widths  = np.linspace(0.020, 0.125, n_widths)
    fingers = GRASP_FINGER_SETS[n_fingers]
    active  = fingers + ["thumb"]

    analytical_pos_errs, analytical_coplan, analytical_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times, mink_conv = [], [], [], []
    valid_widths = []
    analytical_thumb_ctrl, mink_thumb_ctrl = [], []
    z_span_analytical, z_span_mink = [], []
    mink_ctrl_arrs = []
    pfc_analytical = {f: [] for f in active}
    pfc_mink   = {f: [] for f in active}

    for w in widths:
        try:
            t0 = time.perf_counter()
            c_res = closure.plane(w, n_fingers=n_fingers)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [plane-{n_fingers}f] analytical FAIL @ w={w*1000:.1f}mm: {e}")
            continue

        targets = {f: c_res.tip_positions[f] for f in active if f in c_res.tip_positions}
        m_res = mink_planner.solve_plane(
            w,
            target_positions=targets,
            active_fingers=list(targets.keys()),
        )

        valid_widths.append(w)
        analytical_pos_errs.append(_analytical_pos_error(c_res, targets))
        analytical_coplan.append(_analytical_coplan_err(c_res, fingers))
        analytical_times.append(c_time * 1000)
        analytical_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(_mink_coplan_err(m_res, c_res, fingers))
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))
        mink_ctrl_arrs.append(m_res.ctrl.copy())

        wtips_c = c_res.world_tips(world_grasp_z=0.0)
        z_span_analytical.append(_z_span_mm(wtips_c, active))
        z_span_mink.append(_z_span_mm(_world_tips_mink(m_res, c_res), active))
        _collect_per_finger(c_res, m_res, active, pfc_analytical, pfc_mink)

    w_arr = np.array(valid_widths) * 1000
    return dict(
        widths_mm=w_arr,
        active_fingers=active,
        analytical_pos_errs=np.array(analytical_pos_errs),
        analytical_coplan=np.array(analytical_coplan),
        analytical_times=np.array(analytical_times),
        analytical_thumb_ctrl=np.array(analytical_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_coplan=np.array(mink_coplan),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
        mink_ctrl_arr=np.array(mink_ctrl_arrs),
        z_span_analytical=np.array(z_span_analytical),
        z_span_mink=np.array(z_span_mink),
        per_finger_ctrl_analytical={f: np.array(v) for f, v in pfc_analytical.items()},
        per_finger_ctrl_mink={f: np.array(v) for f, v in pfc_mink.items()},
    )


def sweep_cylinder(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
) -> dict:
    """Sweep cylinder / power grasp."""
    diameters = np.linspace(0.030, 0.100, n_widths)
    active    = NON_THUMB_FINGERS + ["thumb"]   # index, middle, ring, pinky, thumb

    analytical_pos_errs, analytical_coplan, analytical_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times, mink_conv = [], [], [], []
    valid_diams = []
    analytical_thumb_ctrl, mink_thumb_ctrl = [], []
    z_span_analytical, z_span_mink = [], []
    mink_ctrl_arrs = []
    pfc_analytical = {f: [] for f in active}
    pfc_mink   = {f: [] for f in active}

    for d in diameters:
        try:
            t0 = time.perf_counter()
            c_res = closure.cylinder(d)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [cylinder] analytical FAIL @ d={d*1000:.1f}mm: {e}")
            continue

        targets = {f: c_res.tip_positions[f] for f in active if f in c_res.tip_positions}
        m_res = mink_planner.solve_cylinder(
            d,
            target_positions=targets,
            active_fingers=list(targets.keys()),
        )

        valid_diams.append(d)
        analytical_pos_errs.append(_analytical_pos_error(c_res, targets))
        analytical_coplan.append(_analytical_coplan_err(c_res, NON_THUMB_FINGERS))
        analytical_times.append(c_time * 1000)
        analytical_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(_mink_coplan_err(m_res, c_res, NON_THUMB_FINGERS))
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))
        mink_ctrl_arrs.append(m_res.ctrl.copy())

        wtips_c = c_res.world_tips(world_grasp_z=0.0)
        z_span_analytical.append(_z_span_mm(wtips_c, active))
        z_span_mink.append(_z_span_mm(_world_tips_mink(m_res, c_res), active))
        _collect_per_finger(c_res, m_res, active, pfc_analytical, pfc_mink)

    d_arr = np.array(valid_diams) * 1000
    return dict(
        widths_mm=d_arr,
        active_fingers=active,
        analytical_pos_errs=np.array(analytical_pos_errs),
        analytical_coplan=np.array(analytical_coplan),
        analytical_times=np.array(analytical_times),
        analytical_thumb_ctrl=np.array(analytical_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_coplan=np.array(mink_coplan),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
        mink_ctrl_arr=np.array(mink_ctrl_arrs),
        z_span_analytical=np.array(z_span_analytical),
        z_span_mink=np.array(z_span_mink),
        per_finger_ctrl_analytical={f: np.array(v) for f, v in pfc_analytical.items()},
        per_finger_ctrl_mink={f: np.array(v) for f, v in pfc_mink.items()},
    )


# ---------------------------------------------------------------------------
# Single-width convergence curve
# ---------------------------------------------------------------------------

def convergence_curve(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    width_m: float,
    mode: str = "line",
    n_fingers: int = 4,
) -> dict:
    """Return mink error_history for a single representative width."""
    if mode == "line":
        c_res = closure.line(width_m)
        targets = {f: c_res.tip_positions[f] for f in ["thumb", "index"]}
        m_res = mink_planner.solve_line(
            width_m,
            thumb_target=targets["thumb"],
            index_target=targets["index"],
        )
    elif mode == "plane":
        c_res = closure.plane(width_m, n_fingers=n_fingers)
        fingers = GRASP_FINGER_SETS[n_fingers]
        active = fingers + ["thumb"]
        targets = {f: c_res.tip_positions[f] for f in active if f in c_res.tip_positions}
        m_res = mink_planner.solve_plane(width_m, target_positions=targets)
    else:
        c_res = closure.cylinder(width_m)
        active = NON_THUMB_FINGERS + ["thumb"]
        targets = {f: c_res.tip_positions[f] for f in active if f in c_res.tip_positions}
        m_res = mink_planner.solve_cylinder(width_m, target_positions=targets)

    return dict(
        error_history_mm=np.array(m_res.error_history) * 1000,
        n_iters=m_res.n_iters,
        converged=m_res.converged,
        width_mm=width_m * 1000,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Per-mode figure (saved to subfolder)
# ---------------------------------------------------------------------------

def save_mode_figure(
    data: dict,
    mode_label: str,
    out_dir: Path,
    x_label: str = "Width [mm]",
) -> None:
    """
    Save a 3-row figure for one grasp mode:
      Row 0 (4 panels): tip pos error | world Z span | mink iters | timing
      Row 1 (N panels): per-finger ctrl (analytical vs mink)
      Row 2 (N panels): per-finger ctrl smoothness
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    active = data["active_fingers"]
    n_af   = len(active)
    w      = data["widths_mm"]
    dw_m   = w / 1000.0

    fig_w = max(16, 4 * n_af)
    fig = plt.figure(figsize=(fig_w, 13))
    fig.suptitle(
        f"{mode_label}  —  ClosureGeometry (analytical) vs Mink IK\n"
        f"Active fingers: {', '.join(active)}",
        fontsize=12,
    )

    outer    = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50, height_ratios=[1, 1, 1])
    gs_metr  = gridspec.GridSpecFromSubplotSpec(1, 4,    subplot_spec=outer[0], wspace=0.35)
    gs_ctrl  = gridspec.GridSpecFromSubplotSpec(1, n_af, subplot_spec=outer[1], wspace=0.35)
    gs_smth  = gridspec.GridSpecFromSubplotSpec(1, n_af, subplot_spec=outer[2], wspace=0.35)

    # ---- Row 0: overview metrics ----------------------------------------
    ax_pos  = fig.add_subplot(gs_metr[0, 0])
    ax_zsp  = fig.add_subplot(gs_metr[0, 1])
    ax_iter = fig.add_subplot(gs_metr[0, 2])
    ax_time = fig.add_subplot(gs_metr[0, 3])

    ax_pos.axhline(0, color=_C_ANALYTICAL, lw=1.5, label="Analytical (ClosureGeometry)")
    ax_pos.plot(w, data["mink_pos_errs_mean"], color=_C_MINK, lw=1.5, label="Mink (mean)")
    ax_pos.fill_between(w, data["mink_pos_errs_mean"], data["mink_pos_errs_max"],
                        alpha=_ALPHA_SHADE, color=_C_MINK, label="Mink (max)")
    nc = ~data["mink_conv"]
    if nc.any():
        ax_pos.scatter(w[nc], data["mink_pos_errs_mean"][nc],
                       color="red", s=30, zorder=5, label="Not converged")
    ax_pos.set_title("Tip position error")
    ax_pos.set_ylabel("Error [mm]")
    ax_pos.set_xlabel(x_label)
    ax_pos.legend(fontsize=7)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim(bottom=0)

    ax_zsp.plot(w, data["z_span_analytical"], color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_zsp.plot(w, data["z_span_mink"],   color=_C_MINK,   lw=1.5, ls="--", label="Mink")
    ax_zsp.set_title("World-frame Z span (all active tips)")
    ax_zsp.set_ylabel("Z_max − Z_min [mm]")
    ax_zsp.set_xlabel(x_label)
    ax_zsp.legend(fontsize=7)
    ax_zsp.grid(True, alpha=0.3)
    ax_zsp.set_ylim(bottom=0)

    ax_iter.plot(w, data["mink_iters"], color=_C_MINK, lw=1.5)
    ax_iter.set_title("Mink iterations to convergence")
    ax_iter.set_ylabel("Iterations")
    ax_iter.set_xlabel(x_label)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_ylim(bottom=0)

    ax_time.semilogy(w, data["analytical_times"], color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_time.semilogy(w, data["mink_times"],   color=_C_MINK,   lw=1.5, label="Mink")
    ax_time.set_title("Computation time (log scale)")
    ax_time.set_ylabel("Time [ms]")
    ax_time.set_xlabel(x_label)
    ax_time.legend(fontsize=7)
    ax_time.grid(True, alpha=0.3)

    # ---- Rows 1 & 2: per-finger ctrl and smoothness ---------------------
    for i, fname in enumerate(active):
        ax_c = fig.add_subplot(gs_ctrl[0, i])
        ax_s = fig.add_subplot(gs_smth[0, i])
        lbl  = "thumb bend" if fname == "thumb" else fname

        c_ctrl = data["per_finger_ctrl_analytical"][fname]
        m_ctrl = data["per_finger_ctrl_mink"][fname]

        ax_c.plot(w, np.degrees(c_ctrl), color=_C_ANALYTICAL, lw=1.5, label="Analytical")
        ax_c.plot(w, np.degrees(m_ctrl), color=_C_MINK,   lw=1.5, ls="--", label="Mink")
        ax_c.set_title(f"{lbl} ctrl")
        ax_c.set_ylabel("Ctrl [deg]")
        ax_c.set_xlabel(x_label)
        ax_c.legend(fontsize=7)
        ax_c.grid(True, alpha=0.3)

        c_smth = _ctrl_smoothness(c_ctrl, dw_m)
        m_smth = _ctrl_smoothness(m_ctrl, dw_m)
        ax_s.plot(w, np.degrees(c_smth), color=_C_ANALYTICAL, lw=1.5, label="Analytical")
        ax_s.plot(w, np.degrees(m_smth), color=_C_MINK,   lw=1.5, ls="--", label="Mink")
        ax_s.set_title(f"|Δ{lbl} / Δwidth|")
        ax_s.set_ylabel("[deg / m]")
        ax_s.set_xlabel(x_label)
        ax_s.legend(fontsize=7)
        ax_s.grid(True, alpha=0.3)
        ax_s.set_ylim(bottom=0)

    fig.savefig(str(out_dir / "comparison.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'comparison.png'}")


# ---------------------------------------------------------------------------
# Mega overview figure
# ---------------------------------------------------------------------------

def _plot_mode_row(
    axes: np.ndarray,
    data: dict,
    mode_label: str,
    x_label: str = "Width [mm]",
) -> None:
    """Fill one row of the mega figure (5 axes): pos_err | z_span | iters | thumb_ctrl | thumb_smooth."""
    w = data["widths_mm"]
    ax_pos, ax_zsp, ax_iter, ax_ctrl, ax_smth = axes

    ax_pos.axhline(0, color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_pos.plot(w, data["mink_pos_errs_mean"], color=_C_MINK, lw=1.5, label="Mink (mean)")
    ax_pos.fill_between(w, data["mink_pos_errs_mean"], data["mink_pos_errs_max"],
                        alpha=_ALPHA_SHADE, color=_C_MINK)
    nc = ~data["mink_conv"]
    if nc.any():
        ax_pos.scatter(w[nc], data["mink_pos_errs_mean"][nc], color="red", s=20, zorder=5)
    ax_pos.set_title(f"{mode_label} — tip error")
    ax_pos.set_ylabel("Error [mm]")
    ax_pos.set_xlabel(x_label)
    ax_pos.legend(fontsize=6)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim(bottom=0)

    ax_zsp.plot(w, data["z_span_analytical"], color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_zsp.plot(w, data["z_span_mink"],   color=_C_MINK,   lw=1.5, ls="--", label="Mink")
    ax_zsp.set_title("World-frame Z span")
    ax_zsp.set_ylabel("Z_max−Z_min [mm]")
    ax_zsp.set_xlabel(x_label)
    ax_zsp.legend(fontsize=6)
    ax_zsp.grid(True, alpha=0.3)
    ax_zsp.set_ylim(bottom=0)

    ax_iter.plot(w, data["mink_iters"], color=_C_MINK, lw=1.5)
    ax_iter.set_title("Mink iterations")
    ax_iter.set_ylabel("Iters")
    ax_iter.set_xlabel(x_label)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_ylim(bottom=0)

    ax_ctrl.plot(w, np.degrees(data["analytical_thumb_ctrl"]), color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_ctrl.plot(w, np.degrees(data["mink_thumb_ctrl"]),   color=_C_MINK,   lw=1.5, ls="--", label="Mink")
    ax_ctrl.set_title("Thumb bend ctrl")
    ax_ctrl.set_ylabel("Ctrl [deg]")
    ax_ctrl.set_xlabel(x_label)
    ax_ctrl.legend(fontsize=6)
    ax_ctrl.grid(True, alpha=0.3)

    dw_m = w / 1000.0
    c_smth = _ctrl_smoothness(data["analytical_thumb_ctrl"], dw_m)
    m_smth = _ctrl_smoothness(data["mink_thumb_ctrl"],   dw_m)
    ax_smth.plot(w, np.degrees(c_smth), color=_C_ANALYTICAL, lw=1.5, label="Analytical")
    ax_smth.plot(w, np.degrees(m_smth), color=_C_MINK,   lw=1.5, ls="--", label="Mink")
    ax_smth.set_title("|Δthumb / Δwidth|")
    ax_smth.set_ylabel("[deg / m]")
    ax_smth.set_xlabel(x_label)
    ax_smth.legend(fontsize=6)
    ax_smth.grid(True, alpha=0.3)
    ax_smth.set_ylim(bottom=0)


def plot_convergence_inset(ax, conv_data: dict) -> None:
    history = conv_data["error_history_mm"]
    iters   = np.arange(len(history))
    ax.semilogy(iters, history, color=_C_MINK, lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total tip error [mm] (log)")
    ax.set_title(
        f"Mink convergence @ {conv_data['width_mm']:.0f} mm "
        f"({'OK' if conv_data['converged'] else 'FAIL'} in {conv_data['n_iters']} iters)"
    )
    ax.grid(True, alpha=0.3)


def _pct(arr):
    return f"{100 * arr.mean():.0f}%"

def _mean(arr, scale=1):
    return f"{arr.mean() * scale:.1f}"


# ---------------------------------------------------------------------------
# Sweep cache — save / load
# ---------------------------------------------------------------------------

def save_sweep_cache(data: dict, path: Path) -> None:
    """
    Save sweep results to a .npz file for offline ctrl lookup.

    Stored arrays:
        widths_mm    (N,)    — sweep x-axis
        mink_ctrl_arr (N,6)  — full mink ctrl vector per sample
        mink_conv    (N,)    — convergence flags (bool)
        active_fingers       — list of finger names (stored as object array)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(path),
        widths_mm=data["widths_mm"],
        mink_ctrl_arr=data["mink_ctrl_arr"],
        mink_conv=data["mink_conv"],
        active_fingers=np.array(data["active_fingers"]),
    )
    print(f"  Cache saved: {path}")


# ---------------------------------------------------------------------------
# Pinch-family cross-mode figure (2f / 3f / 4f / 5f overlaid)
# ---------------------------------------------------------------------------

_MODE_COLORS = {
    "2f": "#1f77b4",   # blue
    "3f": "#2ca02c",   # green
    "4f": "#ff7f0e",   # orange
    "5f": "#d62728",   # red
}
_ALL_FINGERS = ["thumb", "index", "middle", "ring", "pinky"]


def _interp(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """Linear interpolation, clamped at edges."""
    return np.interp(x_tgt, x_src, y_src)


def save_pinch_family_figure(
    modes_dict: Dict[str, dict],
    out_dir: Path,
) -> None:
    """
    Save a cross-mode comparison figure for 2f/3f/4f/5f pinch grasps.

    Layout (3 rows using nested GridSpec):
      Row 0 (4 panels): tip pos error | world Z span | mink iters | timing
      Row 1 (5 panels): per-finger ctrl  (thumb, index, middle, ring, pinky)
      Row 2 (5 panels): per-finger ctrl smoothness

    For each panel:
      - Thin lines (alpha=0.5): individual modes (solid=analytical, dashed=mink)
      - Thick line (alpha=1.0): average across applicable modes
      - Fingers absent from a mode are simply not plotted in that mode's line.

    The common x-axis for averaging is the overlap of all mode width ranges.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Common width grid = overlap of all mode ranges
    w_min = max(d["widths_mm"].min() for d in modes_dict.values())
    w_max = min(d["widths_mm"].max() for d in modes_dict.values())
    w_common = np.linspace(w_min, w_max, 100)

    fig = plt.figure(figsize=(25, 13))
    fig.suptitle(
        "Pinch Grasp Family: 2f / 3f / 4f / 5f — ClosureGeometry vs Mink IK\n"
        "Thin lines = individual modes  |  Thick = average across applicable modes  "
        "|  Solid = Analytical  |  Dashed = Mink",
        fontsize=11,
    )

    outer   = gridspec.GridSpec(3, 1, figure=fig, hspace=0.50, height_ratios=[1, 1, 1])
    gs_metr = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.35)
    gs_ctrl = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[1], wspace=0.30)
    gs_smth = gridspec.GridSpecFromSubplotSpec(1, 5, subplot_spec=outer[2], wspace=0.30)

    # ---- Helper: plot all modes + average for a scalar metric ---------------

    def plot_metric(ax, key_analytical, key_mink, ylabel, title, log=False):
        c_interps, m_interps = [], []
        for mode_key, d in modes_dict.items():
            w = d["widths_mm"]
            color = _MODE_COLORS[mode_key]
            y_c = d[key_analytical]
            y_m = d[key_mink]
            plot_fn = ax.semilogy if log else ax.plot
            plot_fn(w, y_c, color=color, lw=1.0, alpha=0.45, label=f"{mode_key} analytical")
            plot_fn(w, y_m, color=color, lw=1.0, alpha=0.45, ls="--", label=f"{mode_key} mink")
            # interpolate to common grid for averaging (only within mode's range)
            mask = (w_common >= w.min()) & (w_common <= w.max())
            if mask.sum() >= 2:
                c_interps.append(_interp(w, y_c, w_common[mask]))
                m_interps.append(_interp(w, y_m, w_common[mask]))

        # Average
        if c_interps:
            avg_c = np.mean(c_interps, axis=0)
            avg_m = np.mean(m_interps, axis=0)
            if log:
                ax.semilogy(w_common, avg_c, color="black", lw=2.2, alpha=0.9, label="avg analytical")
                ax.semilogy(w_common, avg_m, color="#555555", lw=2.2, alpha=0.9, ls="--",
                            label="avg mink")
            else:
                ax.plot(w_common, avg_c, color="black", lw=2.2, alpha=0.9, label="avg analytical")
                ax.plot(w_common, avg_m, color="#555555", lw=2.2, alpha=0.9, ls="--",
                        label="avg mink")

        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Width [mm]")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        if not log:
            ax.set_ylim(bottom=0)

    # ---- Helper: plot per-finger ctrl / smoothness --------------------------

    def plot_finger(ax, fname, ctrl_or_smooth, ylabel, title_suffix):
        """Plot ctrl (or smoothness) for a given finger across all modes where active."""
        c_interps, m_interps = [], []
        has_data = False
        for mode_key, d in modes_dict.items():
            if fname not in d["active_fingers"]:
                continue
            has_data = True
            w = d["widths_mm"]
            color = _MODE_COLORS[mode_key]
            if ctrl_or_smooth == "ctrl":
                y_c = np.degrees(d["per_finger_ctrl_analytical"][fname])
                y_m = np.degrees(d["per_finger_ctrl_mink"][fname])
            else:
                dw_m = w / 1000.0
                y_c = np.degrees(_ctrl_smoothness(d["per_finger_ctrl_analytical"][fname], dw_m))
                y_m = np.degrees(_ctrl_smoothness(d["per_finger_ctrl_mink"][fname], dw_m))
            ax.plot(w, y_c, color=color, lw=1.0, alpha=0.45, label=f"{mode_key} analytical")
            ax.plot(w, y_m, color=color, lw=1.0, alpha=0.45, ls="--", label=f"{mode_key} mink")
            mask = (w_common >= w.min()) & (w_common <= w.max())
            if mask.sum() >= 2:
                c_interps.append(_interp(w, y_c, w_common[mask]))
                m_interps.append(_interp(w, y_m, w_common[mask]))

        n_modes_for_finger = len(c_interps)
        if n_modes_for_finger > 1:
            avg_c = np.mean(c_interps, axis=0)
            avg_m = np.mean(m_interps, axis=0)
            ax.plot(w_common, avg_c, color="black",   lw=2.2, alpha=0.9, label="avg analytical")
            ax.plot(w_common, avg_m, color="#555555", lw=2.2, alpha=0.9, ls="--", label="avg mink")
        elif n_modes_for_finger == 1:
            # only pinky (5f only) — label as "no avg"
            pass

        lbl = "thumb bend" if fname == "thumb" else fname
        ax.set_title(f"{lbl} {title_suffix}" + ("" if has_data else " (absent)"))
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Width [mm]")
        if has_data:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)
        if ctrl_or_smooth == "smooth":
            ax.set_ylim(bottom=0)

    # ---- Row 0: overview metrics -------------------------------------------
    plot_metric(fig.add_subplot(gs_metr[0, 0]),
                "analytical_pos_errs", "mink_pos_errs_mean", "Error [mm]", "Tip position error")
    plot_metric(fig.add_subplot(gs_metr[0, 1]),
                "z_span_analytical", "z_span_mink", "Z_max−Z_min [mm]", "World-frame Z span")

    ax_iter = fig.add_subplot(gs_metr[0, 2])
    for mode_key, d in modes_dict.items():
        ax_iter.plot(d["widths_mm"], d["mink_iters"],
                     color=_MODE_COLORS[mode_key], lw=1.2, alpha=0.7, label=mode_key)
    iters_avg = np.mean([_interp(d["widths_mm"], d["mink_iters"], w_common)
                         for d in modes_dict.values()], axis=0)
    ax_iter.plot(w_common, iters_avg, color="black", lw=2.2, alpha=0.9, label="avg")
    ax_iter.set_title("Mink iterations")
    ax_iter.set_ylabel("Iterations")
    ax_iter.set_xlabel("Width [mm]")
    ax_iter.legend(fontsize=6)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_ylim(bottom=0)

    plot_metric(fig.add_subplot(gs_metr[0, 3]),
                "analytical_times", "mink_times", "Time [ms]", "Computation time (log)", log=True)

    # ---- Rows 1 & 2: per-finger ctrl / smoothness --------------------------
    for i, fname in enumerate(_ALL_FINGERS):
        plot_finger(fig.add_subplot(gs_ctrl[0, i]), fname, "ctrl",   "Ctrl [deg]",    "ctrl")
        plot_finger(fig.add_subplot(gs_smth[0, i]), fname, "smooth", "[deg / m]", "|Δctrl/Δwidth|")

    fig.savefig(str(out_dir / "pinch_family.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_dir / 'pinch_family.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(n_widths: int = 25, show: bool = True) -> None:
    """
    Run full comparison across all 5 grasp modes and save per-mode + mega figures.

    Output structure:
        mink_vs_analytical/
            grasp_comparison.png       — 5-mode × 5-metric mega overview
            2/comparison.png           — 2-finger line (per-finger ctrl + metrics)
            3/comparison.png           — 3-finger plane
            4/comparison.png           — 4-finger plane
            5/comparison.png           — 5-finger plane
            cyl/comparison.png         — cylinder power grasp
            pinch_family/
              pinch_family.png         — 2f/3f/4f/5f overlaid + averages
            cache/
              line.npz                 — mink ctrl lookup table (line)
              plane_3f.npz             — mink ctrl lookup table (plane 3f)
              plane_4f.npz             — etc.
              plane_5f.npz
              cylinder.npz
    """
    print("Loading FK tables (may take a moment on first run)...")
    fk      = InspireHandFK(str(_GRASP_XML))
    closure = ClosureGeometry(fk)

    print("Loading mink planner...")
    mink_planner = MinkGraspPlanner(
        str(_RIGHT_XML),
        dt=0.005,
        max_iters=1000,
        conv_thr=5e-4,
        solver="daqp",
    )

    # ---- Sweeps -----------------------------------------------------------
    modes_data = []

    print(f"\nSweeping 2-finger LINE grasp ({n_widths} widths)...")
    line_data = sweep_line(closure, mink_planner, n_widths)
    print(f"  conv: {_pct(line_data['mink_conv'])}  "
          f"iters: {_mean(line_data['mink_iters'])}  "
          f"tip_err: {_mean(line_data['mink_pos_errs_mean'])} mm")
    modes_data.append((line_data, "2-finger Line",  _OUT_ROOT / "2",   "Width [mm]"))

    print(f"\nSweeping 3-finger PLANE grasp ({n_widths} widths)...")
    plane3_data = sweep_plane(closure, mink_planner, n_widths, n_fingers=3)
    print(f"  conv: {_pct(plane3_data['mink_conv'])}  "
          f"iters: {_mean(plane3_data['mink_iters'])}  "
          f"tip_err: {_mean(plane3_data['mink_pos_errs_mean'])} mm")
    modes_data.append((plane3_data, "3-finger Plane", _OUT_ROOT / "3",   "Width [mm]"))

    print(f"\nSweeping 4-finger PLANE grasp ({n_widths} widths)...")
    plane4_data = sweep_plane(closure, mink_planner, n_widths, n_fingers=4)
    print(f"  conv: {_pct(plane4_data['mink_conv'])}  "
          f"iters: {_mean(plane4_data['mink_iters'])}  "
          f"tip_err: {_mean(plane4_data['mink_pos_errs_mean'])} mm")
    modes_data.append((plane4_data, "4-finger Plane", _OUT_ROOT / "4",   "Width [mm]"))

    print(f"\nSweeping 5-finger PLANE grasp ({n_widths} widths)...")
    plane5_data = sweep_plane(closure, mink_planner, n_widths, n_fingers=5)
    print(f"  conv: {_pct(plane5_data['mink_conv'])}  "
          f"iters: {_mean(plane5_data['mink_iters'])}  "
          f"tip_err: {_mean(plane5_data['mink_pos_errs_mean'])} mm")
    modes_data.append((plane5_data, "5-finger Plane", _OUT_ROOT / "5",   "Width [mm]"))

    print(f"\nSweeping CYLINDER (power) grasp ({n_widths} diameters)...")
    cyl_data = sweep_cylinder(closure, mink_planner, n_widths)
    print(f"  conv: {_pct(cyl_data['mink_conv'])}  "
          f"iters: {_mean(cyl_data['mink_iters'])}  "
          f"tip_err: {_mean(cyl_data['mink_pos_errs_mean'])} mm")
    modes_data.append((cyl_data, "Cylinder (power)", _OUT_ROOT / "cyl", "Diameter [mm]"))

    # ---- Save sweep caches (offline ctrl lookup) ---------------------------
    cache_dir = _OUT_ROOT / "cache"
    print("\nSaving sweep caches...")
    cache_names = ["line", "plane_3f", "plane_4f", "plane_5f", "cylinder"]
    for (d, _, _, _), name in zip(modes_data, cache_names):
        save_sweep_cache(d, cache_dir / f"{name}.npz")

    # ---- Per-mode figures -------------------------------------------------
    print("\nSaving per-mode figures...")
    for d, label, out_dir, xlabel in modes_data:
        save_mode_figure(d, label, out_dir, xlabel)

    # ---- Pinch family cross-mode figure (2f/3f/4f/5f) ---------------------
    print("\nSaving pinch family figure...")
    pinch_modes = {
        "2f": line_data,
        "3f": plane3_data,
        "4f": plane4_data,
        "5f": plane5_data,
    }
    save_pinch_family_figure(pinch_modes, _OUT_ROOT / "pinch_family")

    # ---- Convergence curves (line, 4f-plane, cylinder) --------------------
    print("\nRecording convergence curves...")
    conv_line  = convergence_curve(closure, mink_planner, 0.060, mode="line")
    conv_plane = convergence_curve(closure, mink_planner, 0.080, mode="plane", n_fingers=4)
    conv_cyl   = convergence_curve(closure, mink_planner, 0.070, mode="cylinder")

    # ---- Mega overview figure: 5 modes × 5 metrics + convergence ----------
    print("\nBuilding mega overview figure...")
    n_modes = len(modes_data)
    ncols   = 5
    fig = plt.figure(figsize=(22, 4 * n_modes + 5))
    fig.suptitle(
        "Grasp Planner Comparison: ClosureGeometry (analytical) vs Mink IK\n"
        f"Inspire RH56 Hand  |  n_widths={n_widths}  |  "
        f"mink: dt={mink_planner.dt}s, max_iters={mink_planner.max_iters}, "
        f"conv_thr={mink_planner.conv_thr * 1000:.1f}mm",
        fontsize=11,
    )

    height_ratios = [1] * n_modes + [0.8]
    axs = fig.subplots(n_modes + 1, ncols,
                       gridspec_kw={"height_ratios": height_ratios})

    for row_i, (d, label, _, xlabel) in enumerate(modes_data):
        _plot_mode_row(axs[row_i], d, label, x_label=xlabel)

    # Convergence row
    plot_convergence_inset(axs[n_modes][0], conv_line)
    plot_convergence_inset(axs[n_modes][1], conv_plane)
    plot_convergence_inset(axs[n_modes][2], conv_cyl)

    # Summary text
    axs[n_modes][3].axis("off")
    axs[n_modes][4].axis("off")

    summary_lines = [
        "Summary (mean over sweep range)",
        "",
        f"{'Mode':<18} {'Conv%':>6} {'Iters':>6} {'TipErr[mm]':>10} {'ZSpan[mm]':>10}",
        "-" * 56,
    ]
    for d, label, _, _ in modes_data:
        summary_lines.append(
            f"{label:<18} {_pct(d['mink_conv']):>6} "
            f"{_mean(d['mink_iters']):>6} "
            f"{_mean(d['mink_pos_errs_mean']):>10} "
            f"{_mean(d['z_span_mink']):>10}"
        )
    summary_lines += [
        "",
        "ZSpan = max(Z) − min(Z) of active tips in world frame.",
        "Analytical tip error = 0 by construction (brentq solver).",
        f"Analytical mean time: {line_data['analytical_times'].mean():.2f} ms",
        f"Mink line mean: {line_data['mink_times'].mean():.1f} ms",
        f"Mink plane-4f mean: {plane4_data['mink_times'].mean():.1f} ms",
        f"Mink cyl mean: {cyl_data['mink_times'].mean():.1f} ms",
    ]
    axs[n_modes][3].text(
        0.0, 1.0, "\n".join(summary_lines),
        transform=axs[n_modes][3].transAxes,
        fontsize=8, family="monospace",
        va="top", ha="left",
    )
    axs[n_modes][4].text(
        0.0, 1.0,
        "Design notes\n\n"
        "Analytical (ClosureGeometry):\n"
        " + Analytical: zero tip error by design\n"
        " + Guaranteed smooth ctrl trajectory\n"
        " + Sub-millisecond per grasp\n"
        " + Proportional parameterization\n"
        " ~ ZSpan non-zero due to finger-length\n"
        "   differences after tilt (hardware limit)\n\n"
        "Mink IK (MinkGraspPlanner):\n"
        " + No FK table precomputation needed\n"
        " + Joint coupling via EqualityConstraintTask\n"
        " + Simultaneous multi-finger optimisation\n"
        " - Iterative (slower at runtime)\n"
        " - May not converge at all widths\n"
        " - Smoothness depends on initialization\n\n"
        "Per-mode detail plots in:\n"
        "  mink_vs_analytical/{2,3,4,5,cyl}/",
        transform=axs[n_modes][4].transAxes,
        fontsize=8,
        va="top", ha="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _OUT_ROOT.mkdir(parents=True, exist_ok=True)
    mega_path = _OUT_ROOT / "grasp_comparison.png"
    fig.savefig(str(mega_path), dpi=120, bbox_inches="tight")
    print(f"\nMega figure saved to: {mega_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ClosureGeometry vs mink IK grasp planners"
    )
    parser.add_argument("--n-widths", type=int, default=120,
                        help="Number of width samples per mode (default 25)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures but do not call plt.show()")
    args = parser.parse_args()

    run_comparison(n_widths=args.n_widths, show=not args.no_show)
