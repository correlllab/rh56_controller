"""
compare_grasp_methods.py — Analytical vs mink IK grasp planner comparison.

Sweeps the full width/diameter range for each grasp mode, runs both planners
with identical targets, and produces a multi-panel comparison figure.

Usage:
    uv run python -m rh56_controller.compare_grasp_methods

Metrics reported per grasp mode:
  • Tip position error  (mm)  — how close each planner's tips are to the target
  • Coplanarity error   (mm)  — Z-spread of active fingertips (plane mode)
  • Mink iterations           — IK iterations to convergence per width
  • Ctrl comparison           — thumb_proximal ctrl value vs width (both planners)
  • Ctrl smoothness           — discrete dctrl/dwidth derivative (smoothness check)

Custom planner (ClosureGeometry):
  Analytical FK tables + brentq root-finding.  Width error = 0 by construction.
  Tip targets computed analytically.  Sub-millisecond per grasp.

Mink planner (MinkGraspPlanner):
  Differential IK (mink) starting from open hand.  EqualityConstraintTask
  enforces joint coupling automatically.  Iterates until convergence.
  Targets: identical to custom planner's computed tip positions.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

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
_HERE = Path(__file__).parent.parent
_GRASP_XML = _HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml"
_RIGHT_XML = _HERE / "h1_mujoco" / "inspire" / "inspire_right.xml"


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def _custom_pos_error(result: ClosureResult, target_tips: Dict) -> float:
    """Mean Euclidean tip error of custom planner vs its own targets.

    Always 0 by definition — included for completeness and axis consistency.
    """
    errs = [
        float(np.linalg.norm(result.tip_positions[f] - target_tips[f]))
        for f in target_tips
        if f in result.tip_positions
    ]
    return float(np.mean(errs)) * 1000.0  # → mm


def _custom_coplan_err(result: ClosureResult, active_fingers: List[str]) -> float:
    """Mean absolute world-frame Z deviation of non-thumb tips from their mean Z [mm].

    The custom planner enforces equal *base-frame* Z for non-thumb fingers via
    sequential brentq root-finding, making base-frame Z std exactly zero.  However,
    after the hand-tilt rotation is applied, fingers with different base-frame X
    positions land at different world-frame Z values (different finger lengths).
    This metric captures that residual world-frame coplanarity error honestly.
    """
    non_thumb = [f for f in active_fingers if f != "thumb"]
    if len(non_thumb) < 2:
        return 0.0
    wtips = result.world_tips(world_grasp_z=0.0)
    zvals = np.array([wtips[f][2] for f in non_thumb if f in wtips])
    if len(zvals) < 2:
        return 0.0
    return float(np.mean(np.abs(zvals - zvals.mean()))) * 1000.0


def _mink_coplan_err(mink_r: MinkGraspResult, result: ClosureResult,
                     active_fingers: List[str]) -> float:
    """Mean absolute world-frame Z deviation of mink non-thumb tips from mean Z [mm].

    Mink tip positions are in base frame.  We apply the same hand-tilt rotation as
    the custom planner (result.base_tilt_y) to convert to world frame and compute
    the same metric as _custom_coplan_err for a fair comparison.
    """
    non_thumb = [f for f in active_fingers
                 if f != "thumb" and f in mink_r.tip_positions]
    if len(non_thumb) < 2:
        return 0.0
    R     = ClosureResult._rot_matrix(result.base_tilt_y)
    mid_w = R @ result.midpoint
    base_w = np.array([-mid_w[0], -mid_w[1], -mid_w[2]])  # gz=0
    zvals = np.array([(R @ mink_r.tip_positions[f] + base_w)[2]
                      for f in non_thumb])
    return float(np.mean(np.abs(zvals - zvals.mean()))) * 1000.0


def _mink_mean_pos_error(mink_r: MinkGraspResult) -> float:
    """Mean per-finger tip error [mm]."""
    return float(np.mean(list(mink_r.position_errors_m.values()))) * 1000.0


def _mink_max_pos_error(mink_r: MinkGraspResult) -> float:
    """Max per-finger tip error [mm]."""
    return float(np.max(list(mink_r.position_errors_m.values()))) * 1000.0


def _ctrl_smoothness(ctrl_vals: np.ndarray, widths: np.ndarray) -> np.ndarray:
    """Discrete derivative: |Δctrl/Δwidth| for the thumb_proximal channel."""
    if len(ctrl_vals) < 2:
        return np.zeros(len(ctrl_vals))
    dctrl = np.abs(np.diff(ctrl_vals))
    dw = np.diff(widths)
    dw = np.where(dw == 0, 1e-9, dw)
    return np.concatenate([[0.0], dctrl / dw])


# ---------------------------------------------------------------------------
# Per-mode sweep functions
# ---------------------------------------------------------------------------

def sweep_line(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
) -> dict:
    """Sweep 2-finger line grasp from ~12 mm to ~110 mm."""
    widths = np.linspace(0.012, 0.110, n_widths)

    custom_pos_errs, custom_coplan, custom_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times = [], [], []
    mink_conv, valid_widths = [], []
    custom_thumb_ctrl, mink_thumb_ctrl = [], []

    for w in widths:
        try:
            t0 = time.perf_counter()
            c_res = closure.line(w)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [line] custom FAIL @ w={w*1000:.1f}mm: {e}")
            continue

        # Targets for mink = custom planner's computed tip positions
        targets = {f: c_res.tip_positions[f] for f in c_res.tip_positions}
        m_res = mink_planner.solve_line(
            w,
            thumb_target=targets["thumb"],
            index_target=targets["index"],
        )

        valid_widths.append(w)
        custom_pos_errs.append(_custom_pos_error(c_res, targets))
        custom_coplan.append(_custom_coplan_err(c_res, ["index"]))
        custom_times.append(c_time * 1000)  # → ms
        custom_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(_mink_coplan_err(m_res, c_res, ["index"]))
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))  # thumb_bend index

    widths_arr = np.array(valid_widths) * 1000  # → mm
    return dict(
        widths_mm=widths_arr,
        custom_pos_errs=np.array(custom_pos_errs),
        custom_times=np.array(custom_times),
        custom_thumb_ctrl=np.array(custom_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
    )


def sweep_plane(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
    n_fingers: int = 4,
) -> dict:
    """Sweep n-finger plane grasp."""
    widths = np.linspace(0.020, 0.125, n_widths)
    fingers = GRASP_FINGER_SETS[n_fingers]
    active = fingers + ["thumb"]

    custom_pos_errs, custom_coplan, custom_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times = [], [], []
    mink_conv, valid_widths = [], []
    custom_thumb_ctrl, mink_thumb_ctrl = [], []

    for w in widths:
        try:
            t0 = time.perf_counter()
            c_res = closure.plane(w, n_fingers=n_fingers)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [plane-{n_fingers}f] custom FAIL @ w={w*1000:.1f}mm: {e}")
            continue

        targets = {f: c_res.tip_positions[f] for f in active if f in c_res.tip_positions}
        m_res = mink_planner.solve_plane(
            w,
            target_positions=targets,
            active_fingers=list(targets.keys()),
        )

        valid_widths.append(w)
        custom_pos_errs.append(_custom_pos_error(c_res, targets))
        custom_coplan.append(_custom_coplan_err(c_res, fingers))
        custom_times.append(c_time * 1000)
        custom_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(_mink_coplan_err(m_res, c_res, fingers))
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))

    widths_arr = np.array(valid_widths) * 1000
    return dict(
        widths_mm=widths_arr,
        custom_pos_errs=np.array(custom_pos_errs),
        custom_coplan=np.array(custom_coplan),
        custom_times=np.array(custom_times),
        custom_thumb_ctrl=np.array(custom_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_coplan=np.array(mink_coplan),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
    )


def sweep_cylinder(
    closure: ClosureGeometry,
    mink_planner: MinkGraspPlanner,
    n_widths: int,
) -> dict:
    """Sweep cylinder / power grasp."""
    diameters = np.linspace(0.030, 0.100, n_widths)
    active_all = NON_THUMB_FINGERS + ["thumb"]

    custom_pos_errs, custom_coplan, custom_times = [], [], []
    mink_pos_errs_mean, mink_pos_errs_max = [], []
    mink_coplan, mink_iters, mink_times = [], [], []
    mink_conv, valid_diams = [], []
    custom_thumb_ctrl, mink_thumb_ctrl = [], []

    for d in diameters:
        try:
            t0 = time.perf_counter()
            c_res = closure.cylinder(d)
            c_time = time.perf_counter() - t0
        except Exception as e:
            print(f"  [cylinder] custom FAIL @ d={d*1000:.1f}mm: {e}")
            continue

        targets = {f: c_res.tip_positions[f] for f in active_all if f in c_res.tip_positions}
        active = list(targets.keys())
        m_res = mink_planner.solve_cylinder(
            d,
            target_positions=targets,
            active_fingers=active,
        )

        valid_diams.append(d)
        custom_pos_errs.append(_custom_pos_error(c_res, targets))
        custom_coplan.append(_custom_coplan_err(c_res, NON_THUMB_FINGERS))
        custom_times.append(c_time * 1000)
        custom_thumb_ctrl.append(c_res.ctrl_values["thumb_proximal"])

        mink_pos_errs_mean.append(_mink_mean_pos_error(m_res))
        mink_pos_errs_max.append(_mink_max_pos_error(m_res))
        mink_coplan.append(_mink_coplan_err(m_res, c_res, NON_THUMB_FINGERS))
        mink_iters.append(m_res.n_iters)
        mink_times.append(m_res.wall_time_s * 1000)
        mink_conv.append(m_res.converged)
        mink_thumb_ctrl.append(float(m_res.ctrl[4]))

    diam_mm = np.array(valid_diams) * 1000
    return dict(
        widths_mm=diam_mm,
        custom_pos_errs=np.array(custom_pos_errs),
        custom_coplan=np.array(custom_coplan),
        custom_times=np.array(custom_times),
        custom_thumb_ctrl=np.array(custom_thumb_ctrl),
        mink_pos_errs_mean=np.array(mink_pos_errs_mean),
        mink_pos_errs_max=np.array(mink_pos_errs_max),
        mink_coplan=np.array(mink_coplan),
        mink_iters=np.array(mink_iters),
        mink_times=np.array(mink_times),
        mink_conv=np.array(mink_conv),
        mink_thumb_ctrl=np.array(mink_thumb_ctrl),
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
# Plotting
# ---------------------------------------------------------------------------

_C_CUSTOM = "#2196F3"   # blue — custom (ClosureGeometry)
_C_MINK   = "#FF5722"   # orange — mink IK
_ALPHA_SHADE = 0.15


def _plot_mode(
    axes: np.ndarray,
    data: dict,
    mode_label: str,
    has_coplan: bool = True,
    x_label: str = "Width [mm]",
) -> None:
    """Fill one row of the comparison figure (5 axes)."""
    w = data["widths_mm"]
    ax_pos, ax_cop, ax_iter, ax_ctrl, ax_smth = axes

    # --- Tip position error ---
    ax_pos.axhline(0, color=_C_CUSTOM, lw=1.5, label="Custom (ClosureGeometry)")
    ax_pos.plot(w, data["mink_pos_errs_mean"], color=_C_MINK, lw=1.5,
                label="Mink IK (mean)")
    ax_pos.fill_between(w, data["mink_pos_errs_mean"], data["mink_pos_errs_max"],
                        alpha=_ALPHA_SHADE, color=_C_MINK, label="Mink IK (max)")
    # Mark non-converged points
    nc = ~data["mink_conv"]
    if nc.any():
        ax_pos.scatter(w[nc], data["mink_pos_errs_mean"][nc],
                       color="red", s=30, zorder=5, label="Not converged")
    ax_pos.set_title(f"{mode_label} — Tip position error")
    ax_pos.set_ylabel("Error [mm]")
    ax_pos.set_xlabel(x_label)
    ax_pos.legend(fontsize=7)
    ax_pos.grid(True, alpha=0.3)
    ax_pos.set_ylim(bottom=0)

    # --- Coplanarity error ---
    if has_coplan and "custom_coplan" in data:
        ax_cop.plot(w, data["custom_coplan"], color=_C_CUSTOM, lw=1.5, label="Custom")
        ax_cop.plot(w, data["mink_coplan"],   color=_C_MINK,   lw=1.5, label="Mink IK")
        ax_cop.set_title("World-frame Z spread (non-thumb tips)")
        ax_cop.set_ylabel("Mean |ΔZ| [mm]")
    else:
        # Line grasp: only 2 tips, coplanarity not meaningful — show timing instead
        ax_cop.semilogy(w, data["custom_times"], color=_C_CUSTOM, lw=1.5,
                        label="Custom (brentq)")
        ax_cop.semilogy(w, data["mink_times"], color=_C_MINK, lw=1.5,
                        label="Mink IK")
        ax_cop.set_title("Computation time (log scale)")
        ax_cop.set_ylabel("Time [ms]")
    ax_cop.set_xlabel(x_label)
    ax_cop.legend(fontsize=7)
    ax_cop.grid(True, alpha=0.3)
    ax_cop.set_ylim(bottom=0)

    # --- Mink iterations ---
    ax_iter.plot(w, data["mink_iters"], color=_C_MINK, lw=1.5)
    ax_iter.set_title("Mink IK iterations to convergence")
    ax_iter.set_ylabel("Iterations")
    ax_iter.set_xlabel(x_label)
    ax_iter.grid(True, alpha=0.3)
    ax_iter.set_ylim(bottom=0)

    # --- Thumb ctrl comparison ---
    ax_ctrl.plot(w, np.degrees(data["custom_thumb_ctrl"]),
                 color=_C_CUSTOM, lw=1.5, label="Custom")
    ax_ctrl.plot(w, np.degrees(data["mink_thumb_ctrl"]),
                 color=_C_MINK, lw=1.5, ls="--", label="Mink IK")
    ax_ctrl.set_title("Thumb proximal ctrl")
    ax_ctrl.set_ylabel("Ctrl [deg]")
    ax_ctrl.set_xlabel(x_label)
    ax_ctrl.legend(fontsize=7)
    ax_ctrl.grid(True, alpha=0.3)

    # --- Ctrl smoothness (discrete derivative) ---
    dw = data["widths_mm"]
    c_smth = _ctrl_smoothness(data["custom_thumb_ctrl"], dw / 1000)
    m_smth = _ctrl_smoothness(data["mink_thumb_ctrl"], dw / 1000)
    ax_smth.plot(w, np.degrees(c_smth), color=_C_CUSTOM, lw=1.5, label="Custom")
    ax_smth.plot(w, np.degrees(m_smth), color=_C_MINK, lw=1.5, ls="--",
                 label="Mink IK")
    ax_smth.set_title("|Δctrl / Δwidth| (thumb proximal)")
    ax_smth.set_ylabel("[deg / m]")
    ax_smth.set_xlabel(x_label)
    ax_smth.legend(fontsize=7)
    ax_smth.grid(True, alpha=0.3)
    ax_smth.set_ylim(bottom=0)


def plot_convergence_inset(ax, conv_data: dict) -> None:
    """Plot mink error convergence curve as an inset."""
    history = conv_data["error_history_mm"]
    iters = np.arange(len(history))
    ax.semilogy(iters, history, color=_C_MINK, lw=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total tip error [mm] (log)")
    ax.set_title(
        f"Mink convergence @ {conv_data['width_mm']:.0f} mm "
        f"({'OK' if conv_data['converged'] else 'FAIL'} in {conv_data['n_iters']} iters)"
    )
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_comparison(n_widths: int = 25, show: bool = True) -> None:
    """
    Run full comparison and generate a 3-mode × 5-metric figure.

    Args:
        n_widths: Number of width samples per grasp mode.
        show:     If True, call plt.show() at the end.
    """
    print("Loading FK tables (may take a moment on first run)...")
    fk = InspireHandFK(str(_GRASP_XML))
    closure = ClosureGeometry(fk)

    print("Loading mink planner...")
    mink_planner = MinkGraspPlanner(
        str(_RIGHT_XML),
        dt=0.005,
        max_iters=500,
        conv_thr=3e-3,
        solver="daqp",
    )

    # --- Sweeps ---
    print(f"\nSweeping 2-finger LINE grasp ({n_widths} widths)...")
    line_data = sweep_line(closure, mink_planner, n_widths)
    pct_conv = 100 * line_data["mink_conv"].mean()
    print(f"  Mink convergence: {pct_conv:.0f}%  |  "
          f"mean iters: {line_data['mink_iters'].mean():.1f}  |  "
          f"mean tip err: {line_data['mink_pos_errs_mean'].mean():.2f} mm")

    print(f"\nSweeping 4-finger PLANE grasp ({n_widths} widths)...")
    plane_data = sweep_plane(closure, mink_planner, n_widths, n_fingers=4)
    pct_conv = 100 * plane_data["mink_conv"].mean()
    print(f"  Mink convergence: {pct_conv:.0f}%  |  "
          f"mean iters: {plane_data['mink_iters'].mean():.1f}  |  "
          f"mean tip err: {plane_data['mink_pos_errs_mean'].mean():.2f} mm")

    print(f"\nSweeping CYLINDER (power) grasp ({n_widths} diameters)...")
    cyl_data = sweep_cylinder(closure, mink_planner, n_widths)
    pct_conv = 100 * cyl_data["mink_conv"].mean()
    print(f"  Mink convergence: {pct_conv:.0f}%  |  "
          f"mean iters: {cyl_data['mink_iters'].mean():.1f}  |  "
          f"mean tip err: {cyl_data['mink_pos_errs_mean'].mean():.2f} mm")

    # --- Convergence curves ---
    print("\nRecording convergence curves...")
    conv_line = convergence_curve(closure, mink_planner, 0.060, mode="line")
    conv_plane = convergence_curve(closure, mink_planner, 0.080, mode="plane")
    conv_cyl = convergence_curve(closure, mink_planner, 0.070, mode="cylinder")

    # --- Figure layout: 4 rows × 5 cols ---
    # Row 0: Line grasp (5 metrics)
    # Row 1: Plane grasp (5 metrics)
    # Row 2: Cylinder grasp (5 metrics)
    # Row 3: Convergence curves (3 × wider plots)
    fig = plt.figure(figsize=(22, 18))
    fig.suptitle(
        "Grasp Planner Comparison: ClosureGeometry (analytical) vs Mink IK\n"
        f"Inspire RH56 Hand  |  n_widths={n_widths}  |  "
        f"mink: dt={mink_planner.dt}s, max_iters={mink_planner.max_iters}, "
        f"conv_thr={mink_planner.conv_thr*1000:.0f}mm",
        fontsize=11,
    )

    ncols = 5
    nrows = 4
    axs = fig.subplots(nrows, ncols, gridspec_kw={"height_ratios": [1, 1, 1, 0.8]})

    _plot_mode(axs[0], line_data, "2-finger Line",
               has_coplan=False, x_label="Width [mm]")
    _plot_mode(axs[1], plane_data, "4-finger Plane",
               has_coplan=True, x_label="Width [mm]")
    _plot_mode(axs[2], cyl_data, "Cylinder (power)",
               has_coplan=True, x_label="Diameter [mm]")

    # Convergence curves: row 3 — three pairs merged into 5 axes
    plot_convergence_inset(axs[3][0], conv_line)
    plot_convergence_inset(axs[3][1], conv_plane)
    plot_convergence_inset(axs[3][2], conv_cyl)

    # Summary table in the last two slots of row 3
    axs[3][3].axis("off")
    axs[3][4].axis("off")

    def _pct(arr):
        return f"{100*arr.mean():.0f}%"
    def _mean(arr, scale=1):
        return f"{arr.mean()*scale:.1f}"

    summary_lines = [
        "Summary (mean over width range)",
        "",
        f"{'Mode':<16} {'Conv%':>6} {'Iters':>6} {'TipErr[mm]':>10} {'ZSpread[mm]':>12}",
        "-" * 56,
        f"{'Line (2f)':<16} {_pct(line_data['mink_conv']):>6} "
        f"{_mean(line_data['mink_iters']):>6} "
        f"{_mean(line_data['mink_pos_errs_mean']):>10} {'n/a':>12}",
        f"{'Plane (4f, mink)':<16} {_pct(plane_data['mink_conv']):>6} "
        f"{_mean(plane_data['mink_iters']):>6} "
        f"{_mean(plane_data['mink_pos_errs_mean']):>10} "
        f"{_mean(plane_data['mink_coplan']):>12}",
        f"{'Plane (4f, ours)':<16} {'—':>6} {'—':>6} {'0.0':>10} "
        f"{_mean(plane_data['custom_coplan']):>12}",
        f"{'Cylinder, mink':<16} {_pct(cyl_data['mink_conv']):>6} "
        f"{_mean(cyl_data['mink_iters']):>6} "
        f"{_mean(cyl_data['mink_pos_errs_mean']):>10} "
        f"{_mean(cyl_data['mink_coplan']):>12}",
        f"{'Cylinder, ours':<16} {'—':>6} {'—':>6} {'0.0':>10} "
        f"{_mean(cyl_data['custom_coplan']):>12}",
        "",
        "ZSpread = mean|ΔZ| in world frame (non-thumb tips).",
        "Custom tip error = 0 by construction (brentq solver).",
        "Custom ZSpread > 0 due to per-finger length differences",
        "after tilt rotation (irreducible hardware constraint).",
        f"Custom mean time: {line_data['custom_times'].mean():.2f} ms",
        f"Mink line mean time: {line_data['mink_times'].mean():.1f} ms",
        f"Mink plane mean time: {plane_data['mink_times'].mean():.1f} ms",
        f"Mink cyl mean time: {cyl_data['mink_times'].mean():.1f} ms",
    ]

    axs[3][3].text(
        0.0, 1.0, "\n".join(summary_lines),
        transform=axs[3][3].transAxes,
        fontsize=8, family="monospace",
        va="top", ha="left",
    )
    axs[3][4].text(
        0.0, 1.0,
        "Design notes\n\n"
        "Custom (ClosureGeometry):\n"
        " + Analytical: zero tip error by design\n"
        " + Guaranteed smooth ctrl trajectory\n"
        " + Sub-millisecond per grasp\n"
        " + Proportional parameterization\n"
        " ~ ZSpread non-zero due to finger-length\n"
        "   differences after tilt (hardware limit)\n\n"
        "Mink IK (MinkGraspPlanner):\n"
        " + No FK table pre-computation needed\n"
        " + Joint coupling via EqualityConstraintTask\n"
        " + Simultaneous multi-finger optimisation\n"
        " + Flexible cost structure\n"
        " - Iterative (slower at runtime)\n"
        " - May not converge at all widths\n"
        " - Smoothness depends on initialization\n\n"
        "Coplanarity metric: mean|Z_i - Z_mean|\n"
        "of non-thumb tips in world frame (gz=0).\n"
        "Both planners show residual spread due to\n"
        "the tilt × finger-length interaction.",
        transform=axs[3][4].transAxes,
        fontsize=8,
        va="top", ha="left",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = _HERE / "grasp_comparison.png"
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path}")

    if show:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare ClosureGeometry vs mink IK grasp planners"
    )
    parser.add_argument("--n-widths", type=int, default=25,
                        help="Number of width samples per mode (default 25)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figure but do not call plt.show()")
    args = parser.parse_args()

    run_comparison(n_widths=args.n_widths, show=not args.no_show)
