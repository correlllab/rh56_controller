#!/usr/bin/env python3
"""
Real2Sim Grasp Force Visualizer — Inspire RH56 Hand.

Runs the MuJoCo inspire_scene.xml simulation in parallel with the real RH56
hand (read via serial) and compares grasp quality metrics in real-time:

  Simulation (full cone):
    • Each finger contact → k linearized friction-cone primitive wrenches
    • Ferrari-Canny Q_sim = radius of largest ball inside 6D wrench hull

  Real hand (normal only — the real2sim gap):
    • Each finger contact → 1 primitive wrench (normal direction only)
    • Contact positions / normals borrowed from sim (shared geometry assumption)
    • Ferrari-Canny Q_real (necessarily ≤ Q_sim; gap reveals information loss)

Visualization (Matplotlib + MuJoCo viewer):
  Panel 1 — Sim wrench force-subspace hull (3D)
  Panel 2 — Real wrench force-subspace (normal vectors, 3D)
  Panel 3 — Per-finger normal force bar chart (real N vs sim sensor N)
  Panel 4 — Ferrari-Canny time series (Q_sim and Q_real)

Usage:
    python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0
    python -m rh56_controller.real2sim_viz --port /dev/ttyUSB0 --mirror-angles
    python -m rh56_controller.real2sim_viz --sim-only

Controls (MuJoCo viewer):
    P         — Pause / resume simulation
    R         — Reset to home keyframe
    S         — Print detailed status
    Q / Esc   — Quit
"""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ── Path setup so the script can be run directly ─────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from rh56_controller.mujoco_bridge import SimAnalyzer, ContactInfo  # noqa: E402

try:
    import mujoco
    import mujoco.viewer
except ImportError as e:
    sys.exit(f"mujoco not found: {e}")

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.widgets import CheckButtons
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import warnings
    warnings.filterwarnings("ignore", message=".*Matplotlib.*main thread.*")
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available — external visualization disabled")

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── Finger ordering bridge (real hand ↔ sim) ──────────────────────────────────

# force_act() index → finger name (thumb_rot excluded: no fingertip contact)
REAL_IDX_TO_FINGER: Dict[int, str] = {
    0: "pinky",
    1: "ring",
    2: "middle",
    3: "index",
    4: "thumb",
}

# finger name → force_act() index
FINGER_TO_REAL_IDX: Dict[str, int] = {v: k for k, v in REAL_IDX_TO_FINGER.items()}

FINGER_COLORS_RGB = {
    "thumb":  (1.0, 0.2, 0.2),
    "index":  (0.2, 0.4, 1.0),
    "middle": (0.2, 0.8, 0.2),
    "ring":   (0.8, 0.2, 0.8),
    "pinky":  (1.0, 0.6, 0.1),
}

# Calibrated linear map: raw force_act() units → Newtons   F = a*raw + b
# Source: README.md "Force Mapping — Initial Coefficients (2025-11-04)"
# Measured with a force meter; valid inside the listed raw ranges.
# Fingers not yet calibrated fall back to the legacy Inspire formula (raw/1000*9.81).
_FORCE_CALIB: Dict[str, tuple] = {
    "index":  (0.007478, -0.414),
    "middle": (0.006452,  0.018),
    "thumb":  (0.012547,  0.384),  # thumb_bend channel
    # "ring" and "pinky": TBD — using fallback below
}
_FORCE_CALIB_FALLBACK = (9.81 / 1000.0, 0.0)  # Inspire default: raw/1000 * g

# Thumb yaw tangential force calibration (interim — borrows thumb_bend scale)
# tau_yaw [N·m] = a_yaw * raw_yaw + b_yaw
_THUMB_YAW_CALIB = (0.012547, -0.384)

# r_eff(theta_flex_rad) polynomial from tools/thumb_lever_arm.py
# r_eff = C2*theta² + C1*theta + C0  (metres)
_THUMB_YAW_REFF_POLY = (-0.029209, -0.063028, 0.091856)


def _raw_to_newtons(raw: float, finger: str) -> float:
    """Convert a raw force_act() reading to Newtons using calibrated coefficients."""
    a, b = _FORCE_CALIB.get(finger, _FORCE_CALIB_FALLBACK)
    return max(0.0, a * raw + b)


def _raw_yaw_to_tangential_N(raw_yaw: float, theta_flex_raw: int) -> float:
    """Convert raw yaw force reading to signed tangential force (N).

    Sign convention follows the raw signal: positive = adduction torque.
    Uses interim calibration (_THUMB_YAW_CALIB) and the MuJoCo FK polynomial
    (_THUMB_YAW_REFF_POLY) for the lever arm.
    """
    a, b = _THUMB_YAW_CALIB
    tau_yaw = a * float(raw_yaw) + b
    theta_flex_rad = (float(theta_flex_raw) / 1000.0) * 0.60
    c2, c1, c0 = _THUMB_YAW_REFF_POLY
    r_eff = c2 * theta_flex_rad ** 2 + c1 * theta_flex_rad + c0
    r_eff = max(r_eff, 0.010)
    return tau_yaw / r_eff


# ── Shared state ──────────────────────────────────────────────────────────────

@dataclass
class SharedState:
    """Thread-safe snapshot exchanged between sim loop and viz thread.

    Two sim wrench representations are carried:
      pw_sim_cone  — normalized unit-vector cone (geometric/scale-free force closure)
      pw_sim_actual — actual contact force wrenches in Newtons (comparable to pw_real)

    For the force-cone panels and viewer sphere we use the geometric cone.
    For the Ferrari-Canny Q comparison we use pw_sim_actual vs pw_real (both in N).
    """
    sim_time: float = 0.0
    contacts: List[ContactInfo] = field(default_factory=list)
    real_forces_N: Dict[str, float] = field(default_factory=dict)    # finger → N
    sim_sensor_N: Dict[str, float] = field(default_factory=dict)     # finger → N
    # Geometric (normalized) — for cone panel + viewer sphere
    pw_sim_cone: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    # Actual Newton wrenches — for Q comparison
    pw_sim_actual: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    pw_real: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    # Geometric force closure (from normalized cone)
    fc_sim: bool = False
    q_sim_geo: float = 0.0
    # Newton-comparable force closure (actual sim vs friction-disk real)
    fc_sim_actual: bool = False
    q_sim_actual: float = 0.0
    fc_real: bool = False
    q_real: float = 0.0
    # Thumb tangential force + constrained wrench cone (toggled)
    thumb_tangential_signed_N: float = 0.0
    pw_real_tangential: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    fc_real_tangential: bool = False
    q_real_tangential: float = 0.0
    object_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    num_contacts: int = 0


@dataclass
class TimeSeriesBuffer:
    """Rolling buffer for time-series plots."""
    max_len: int = 2000
    times: List[float] = field(default_factory=list)
    q_sim_actual: List[float] = field(default_factory=list)   # Newton units
    q_real: List[float] = field(default_factory=list)         # Newton units
    q_real_tangential: List[float] = field(default_factory=list)  # constrained
    q_sim_geo: List[float] = field(default_factory=list)      # geometric (normalized)
    fc_sim: List[bool] = field(default_factory=list)
    fc_real: List[bool] = field(default_factory=list)
    fc_real_tangential: List[bool] = field(default_factory=list)
    num_contacts: List[int] = field(default_factory=list)
    real_forces: Dict[str, List[float]] = field(default_factory=dict)
    sim_sensor: Dict[str, List[float]] = field(default_factory=dict)
    thumb_tangential_N: List[float] = field(default_factory=list)

    def append(self, s: SharedState):
        self.times.append(s.sim_time)
        self.q_sim_actual.append(s.q_sim_actual)
        self.q_real.append(s.q_real)
        self.q_real_tangential.append(s.q_real_tangential)
        self.q_sim_geo.append(s.q_sim_geo)
        self.fc_sim.append(s.fc_sim)
        self.fc_real.append(s.fc_real)
        self.fc_real_tangential.append(s.fc_real_tangential)
        self.num_contacts.append(s.num_contacts)
        for f in SimAnalyzer.FINGER_NAMES:
            self.real_forces.setdefault(f, []).append(s.real_forces_N.get(f, 0.0))
            self.sim_sensor.setdefault(f, []).append(s.sim_sensor_N.get(f, 0.0))
        self.thumb_tangential_N.append(abs(s.thumb_tangential_signed_N))
        if len(self.times) > self.max_len:
            self._trim()

    def _trim(self):
        n = len(self.times) - self.max_len
        self.times = self.times[n:]
        self.q_sim_actual = self.q_sim_actual[n:]
        self.q_real = self.q_real[n:]
        self.q_real_tangential = self.q_real_tangential[n:]
        self.q_sim_geo = self.q_sim_geo[n:]
        self.fc_sim = self.fc_sim[n:]
        self.fc_real = self.fc_real[n:]
        self.fc_real_tangential = self.fc_real_tangential[n:]
        self.num_contacts = self.num_contacts[n:]
        for f in list(self.real_forces):
            self.real_forces[f] = self.real_forces[f][n:]
            self.sim_sensor[f] = self.sim_sensor[f][n:]
        self.thumb_tangential_N = self.thumb_tangential_N[n:]


# ── Main class ────────────────────────────────────────────────────────────────

class Real2SimViz:
    """Real-time real2sim grasp force visualizer."""

    def __init__(
        self,
        port: Optional[str],
        hand_id: int,
        xml_path: Optional[str],
        friction_cone_edges: int,
        mirror_angles: bool,
        sim_only: bool,
        record_path: Optional[str],
        thumb_tangential: bool = False,
    ):
        self.sim_only = sim_only
        self.mirror_angles = mirror_angles and not sim_only
        self.record_path = record_path

        # Simulation
        self.sim = SimAnalyzer(xml_path=xml_path, friction_cone_edges=friction_cone_edges)

        # Real hardware
        self.hand = None
        if not sim_only:
            try:
                from rh56_controller.rh56_hand import RH56Hand
                self.hand = RH56Hand(port=port, hand_id=hand_id)
                print(f"[Real2Sim] Connected to real hand on {port} (ID={hand_id})")
                self.hand.force_set([1000] * 6)
                print("[Real2Sim] Force limits set to max (1000)")
            except Exception as e:
                print(f"[Real2Sim] Warning: could not open hand on {port}: {e}")
                print("[Real2Sim] Falling back to sim-only mode")
                self.hand = None

        # Shared state
        self._lock = threading.Lock()
        self._state = SharedState()
        self._history = TimeSeriesBuffer()
        self._recording: List[dict] = []

        # Control flags
        self.running = True
        self.paused = False
        self.show_tangential = thumb_tangential  # toggled by matplotlib checkbox
        self.viewer = None
        self._viz_thread = None
        self._reader_thread = None

    # ── Real hand reader thread ───────────────────────────────────────────────

    def _real_reader_thread(self):
        """Poll force_act() (and angle_read() when tangential toggle is on) at ~50 Hz."""
        while self.running:
            if self.hand is None:
                time.sleep(0.1)
                continue

            forces_raw = None
            try:
                forces_raw = self.hand.force_act()
            except Exception as e:
                print(f"[Real2Sim] force_act error: {e}")

            angles_raw = None
            if self.show_tangential:
                try:
                    angles_raw = self.hand.angle_read()
                except Exception as e:
                    print(f"[Real2Sim] angle_read error: {e}")

            if forces_raw is not None and len(forces_raw) >= 5:
                real_N: Dict[str, float] = {}
                for real_idx, fname in REAL_IDX_TO_FINGER.items():
                    raw = float(forces_raw[real_idx])
                    real_N[fname] = _raw_to_newtons(raw, fname)

                # Compute signed tangential force if toggle is on, data available,
                # and thumb has actual contact (avoids bias artifact at raw_yaw=0
                # caused by the calibration offset term; proper fix: dedicated calib).
                thumb_tang_N = 0.0
                if (self.show_tangential
                        and len(forces_raw) >= 6
                        and angles_raw is not None
                        and len(angles_raw) >= 5
                        and real_N.get("thumb", 0.0) > 0.1):
                    raw_yaw_force = float(forces_raw[5])
                    raw_flex_angle = int(angles_raw[4])
                    thumb_tang_N = _raw_yaw_to_tangential_N(raw_yaw_force, raw_flex_angle)

                with self._lock:
                    self._state.real_forces_N = real_N
                    if self.show_tangential:
                        self._state.thumb_tangential_signed_N = thumb_tang_N

            time.sleep(0.02)  # ~50 Hz

    # ── Sim analysis step ─────────────────────────────────────────────────────

    def _analysis_step(self):
        """Run one sim step and update shared state."""
        self.sim.step()

        contacts = self.sim.get_contacts()
        sensor_N = self.sim.get_sensor_forces_N()
        sensor_vecs = self.sim.get_sensor_force_vectors_N()
        tip_pos = self.sim.get_tip_site_positions()
        object_pos = self.sim.get_object_pos()

        with self._lock:
            real_forces_N = dict(self._state.real_forces_N)
            thumb_tangential_signed_N = self._state.thumb_tangential_signed_N

        # In sim-only mode, use sim sensor magnitudes as "real" proxy
        if self.sim_only or self.hand is None:
            real_forces_N = {f: v for f, v in sensor_N.items()}

        # Geometric cone: normalized unit vectors (scale-free force closure)
        pw_sim_cone = self.sim.compute_sim_wrench_cone(contacts, object_pos)
        # Actual Newton wrenches: exact sim forces (for Q comparison with real)
        pw_sim_actual = self.sim.compute_sim_actual_wrenches(contacts, object_pos)
        # Real friction-disk boundary: Fn known, Ft unknown (both in Newtons)
        pw_real = self.sim.compute_real_wrench_cone(contacts, real_forces_N, object_pos)

        # Constrained wrench cone with thumb tangential — only when toggle is on
        pw_real_tangential = np.zeros((0, 6))
        fc_real_tangential = False
        q_real_tangential = 0.0
        if self.show_tangential:
            pw_real_tangential = self.sim.compute_real_wrench_cone_with_tangential(
                contacts, real_forces_N, thumb_tangential_signed_N, object_pos)
            fc_real_tangential, q_real_tangential = self.sim.evaluate_force_closure(
                pw_real_tangential)

        fc_sim, q_sim_geo     = self.sim.evaluate_force_closure(pw_sim_cone)
        fc_sim_actual, q_sim_actual = self.sim.evaluate_force_closure(pw_sim_actual)
        fc_real, q_real       = self.sim.evaluate_force_closure(pw_real)

        new_state = SharedState(
            sim_time=self.sim.get_time(),
            contacts=contacts,
            real_forces_N=real_forces_N,
            sim_sensor_N=sensor_N,
            pw_sim_cone=pw_sim_cone,
            pw_sim_actual=pw_sim_actual,
            pw_real=pw_real,
            fc_sim=fc_sim,
            q_sim_geo=q_sim_geo,
            fc_sim_actual=fc_sim_actual,
            q_sim_actual=q_sim_actual,
            fc_real=fc_real,
            q_real=q_real,
            thumb_tangential_signed_N=thumb_tangential_signed_N,
            pw_real_tangential=pw_real_tangential,
            fc_real_tangential=fc_real_tangential,
            q_real_tangential=q_real_tangential,
            object_pos=object_pos,
            num_contacts=len(contacts),
        )

        with self._lock:
            self._state = new_state
            self._history.append(new_state)

        # Mirror angles to real hand
        if self.mirror_angles and self.hand is not None:
            try:
                angles = self.sim.get_ctrl_as_real_angles()
                self.hand.angle_set(angles)
            except Exception as e:
                print(f"[Real2Sim] angle_set error: {e}")

        # Record
        if self.record_path:
            self._recording.append({
                "time": new_state.sim_time,
                "fc_sim": fc_sim,
                "fc_real": fc_real,
                "fc_real_tangential": fc_real_tangential,
                "q_sim_geo": q_sim_geo,
                "q_sim_actual": q_sim_actual,
                "q_real": q_real,
                "q_real_tangential": q_real_tangential,
                "thumb_tangential_signed_N": thumb_tangential_signed_N,
                "num_contacts": len(contacts),
                "real_forces_N": {f: v for f, v in real_forces_N.items()},
                "sim_sensor_N": {f: v for f, v in sensor_N.items()},
                "object_pos": object_pos.copy(),
            })

        return new_state, sensor_vecs, tip_pos

    # ── Keyboard handler ──────────────────────────────────────────────────────

    def _key_callback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_P:
            self.paused = not self.paused
            print(f"[Real2Sim] {'Paused' if self.paused else 'Resumed'}")
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            self.running = False
        elif key == glfw.KEY_R:
            self.sim.reset()
            print("[Real2Sim] Reset to home keyframe")
        elif key == glfw.KEY_S:
            self._print_status()

    def _print_status(self):
        with self._lock:
            s = self._state
        print(f"\n{'='*72}")
        print(f"t={s.sim_time:.3f}s | Contacts={s.num_contacts} | "
              f"FC_sim={s.fc_sim} Q_geo={s.q_sim_geo:.5f} | "
              f"Q_sim(N)={s.q_sim_actual:.5f} Q_real(N)={s.q_real:.5f} | "
              f"FC_real={s.fc_real}")
        if self.show_tangential:
            print(f"  [TANGENTIAL ON] thumb_tan={s.thumb_tangential_signed_N:+.3f} N | "
                  f"Q_real_tan={s.q_real_tangential:.5f} FC_real_tan={s.fc_real_tangential}")
        if s.contacts:
            print(f"\n  {'Finger':>8}  {'Normal(sim)':>12}  {'Normal(real)':>13}")
            for fname in SimAnalyzer.FINGER_NAMES:
                sim_n = s.sim_sensor_N.get(fname, 0.0)
                real_n = s.real_forces_N.get(fname, 0.0)
                if sim_n > 0.001 or real_n > 0.001:
                    print(f"  {fname:>8}  {sim_n:12.4f} N  {real_n:12.4f} N")
        print(f"{'='*72}")

    # ── Matplotlib visualization thread ───────────────────────────────────────

    def _viz_thread_func(self):
        if not HAS_MATPLOTLIB:
            return

        fig = None
        check_widget = None
        try:
            plt.ion()
            fig = plt.figure(figsize=(16, 9))
            fig.suptitle("Real2Sim Grasp Force Analysis — Inspire Hand", fontsize=13)
            # Main 2×2 grid; leave right margin for the toggle widget
            gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.32,
                                  left=0.06, right=0.84, top=0.92, bottom=0.07)

            ax_sim_cone  = fig.add_subplot(gs[0, 0], projection="3d")
            ax_real_cone = fig.add_subplot(gs[0, 1], projection="3d")
            ax_forces    = fig.add_subplot(gs[1, 0])
            ax_quality   = fig.add_subplot(gs[1, 1])

            # Toggle widget (persists across redraws — not cleared with cla)
            ax_check = fig.add_axes([0.86, 0.82, 0.12, 0.08])
            check_widget = CheckButtons(ax_check, ["Thumb\nTangential"],
                                        [self.show_tangential])

            def _on_check(label):
                self.show_tangential = not self.show_tangential

            check_widget.on_clicked(_on_check)

            while self.running:
                with self._lock:
                    s = SharedState(**self._state.__dict__)
                    hist = TimeSeriesBuffer(
                        max_len=self._history.max_len,
                        times=list(self._history.times),
                        q_sim_actual=list(self._history.q_sim_actual),
                        q_real=list(self._history.q_real),
                        q_real_tangential=list(self._history.q_real_tangential),
                        q_sim_geo=list(self._history.q_sim_geo),
                        fc_sim=list(self._history.fc_sim),
                        fc_real=list(self._history.fc_real),
                        fc_real_tangential=list(self._history.fc_real_tangential),
                        num_contacts=list(self._history.num_contacts),
                        real_forces={f: list(v) for f, v in self._history.real_forces.items()},
                        sim_sensor={f: list(v) for f, v in self._history.sim_sensor.items()},
                        thumb_tangential_N=list(self._history.thumb_tangential_N),
                    )

                try:
                    # ── Panel 1: Sim geometric wrench cone (unit-normalized) ────
                    ax_sim_cone.cla()
                    ax_sim_cone.set_title(
                        f"Sim Cone (geometric)  FC={s.fc_sim}  Q={s.q_sim_geo:.4f}",
                        fontsize=9)
                    ax_sim_cone.set_xlabel("Fx̂", fontsize=7)
                    ax_sim_cone.set_ylabel("Fŷ", fontsize=7)
                    ax_sim_cone.set_zlabel("Fẑ", fontsize=7)
                    ax_sim_cone.tick_params(labelsize=6)

                    lim = 2.0
                    ax_sim_cone.set_xlim([-lim, lim])
                    ax_sim_cone.set_ylim([-lim, lim])
                    ax_sim_cone.set_zlim([-lim, lim])
                    ax_sim_cone.scatter(0, 0, 0, c="black", s=60, marker="x",
                                        linewidths=2, zorder=10)

                    if len(s.pw_sim_cone) > 0:
                        forces = s.pw_sim_cone[:, :3]
                        col = "green" if s.fc_sim else "red"
                        ax_sim_cone.scatter(
                            forces[:, 0], forces[:, 1], forces[:, 2],
                            c=col, alpha=0.25, s=8)
                        if HAS_SCIPY and len(forces) >= 4:
                            try:
                                hull = ConvexHull(forces)
                                for simplex in hull.simplices:
                                    tri = forces[simplex]
                                    poly = Poly3DCollection(
                                        [tri], alpha=0.07, linewidths=0.2)
                                    poly.set_facecolor(col)
                                    poly.set_edgecolor(col)
                                    ax_sim_cone.add_collection3d(poly)
                            except Exception:
                                pass

                    # ── Panel 2: Real friction-disk vs sim actual (Newton units) ─
                    ax_real_cone.cla()
                    ax_real_cone.set_title(
                        f"Actual forces (N):  Sim Q={s.q_sim_actual:.4f}  "
                        f"Real Q={s.q_real:.4f}",
                        fontsize=9)
                    ax_real_cone.set_xlabel("Fx (N)", fontsize=7)
                    ax_real_cone.set_ylabel("Fy (N)", fontsize=7)
                    ax_real_cone.set_zlabel("Fz (N)", fontsize=7)
                    ax_real_cone.tick_params(labelsize=6)
                    ax_real_cone.scatter(0, 0, 0, c="black", s=60, marker="x",
                                         linewidths=2, zorder=10)

                    # Real: friction-disk boundary (ring of possibilities)
                    if len(s.pw_real) > 0:
                        forces_r = s.pw_real[:, :3]
                        col_r = "blue" if s.fc_real else "orange"
                        ax_real_cone.scatter(
                            forces_r[:, 0], forces_r[:, 1], forces_r[:, 2],
                            c=col_r, s=20, marker="o", alpha=0.6,
                            label=f"Real disk (FC={'✓' if s.fc_real else '✗'})")
                        if HAS_SCIPY and len(forces_r) >= 4:
                            try:
                                hull_r = ConvexHull(forces_r)
                                for simplex in hull_r.simplices:
                                    tri = forces_r[simplex]
                                    poly = Poly3DCollection(
                                        [tri], alpha=0.06, linewidths=0.2)
                                    poly.set_facecolor(col_r)
                                    poly.set_edgecolor(col_r)
                                    ax_real_cone.add_collection3d(poly)
                            except Exception:
                                pass

                    # Sim: actual contact force point (single point — what Ft really is)
                    if len(s.pw_sim_actual) > 0:
                        forces_sa = s.pw_sim_actual[:, :3]
                        col_sa = "green" if s.fc_sim_actual else "red"
                        ax_real_cone.scatter(
                            forces_sa[:, 0], forces_sa[:, 1], forces_sa[:, 2],
                            c=col_sa, s=80, marker="*", alpha=1.0, zorder=10,
                            label=f"Sim actual (FC={'✓' if s.fc_sim_actual else '✗'})")

                    # Auto-scale axes to data
                    all_pts = []
                    if len(s.pw_real) > 0:
                        all_pts.append(s.pw_real[:, :3])
                    if len(s.pw_sim_actual) > 0:
                        all_pts.append(s.pw_sim_actual[:, :3])
                    if all_pts:
                        pts = np.vstack(all_pts)
                        rng = max(np.abs(pts).max() * 1.3, 0.05)
                        ax_real_cone.set_xlim([-rng, rng])
                        ax_real_cone.set_ylim([-rng, rng])
                        ax_real_cone.set_zlim([-rng, rng])
                    ax_real_cone.legend(fontsize=6, loc="upper left")

                    # ── Panel 3: Per-finger force bar chart ───────────────────
                    ax_forces.cla()
                    show_tan = self.show_tangential
                    ax_forces.set_title(
                        "Per-Finger Normal Force (N)"
                        + ("  |  Thumb Tangential ON" if show_tan else ""),
                        fontsize=10)
                    ax_forces.set_ylabel("|F| (N)", fontsize=9)

                    fingers = SimAnalyzer.FINGER_NAMES
                    x = np.arange(len(fingers))
                    w = 0.28 if show_tan else 0.38
                    real_vals = [s.real_forces_N.get(f, 0.0) for f in fingers]
                    sim_vals  = [s.sim_sensor_N.get(f, 0.0) for f in fingers]
                    colors    = [FINGER_COLORS_RGB.get(f, (0.5, 0.5, 0.5)) for f in fingers]

                    if show_tan:
                        ax_forces.bar(x - w, real_vals, w,
                                      color=colors, label="Real (Fn)", alpha=0.85)
                        ax_forces.bar(x, sim_vals, w,
                                      color=colors, label="Sim sensor |F|", alpha=0.40,
                                      hatch="//", edgecolor="gray")
                        # Thumb tangential bar (orange, only thumb slot)
                        thumb_idx = fingers.index("thumb")
                        ax_forces.bar(
                            thumb_idx + w, abs(s.thumb_tangential_signed_N), w,
                            color=(1.0, 0.55, 0.0), alpha=0.9,
                            label=f"Thumb Ft_yaw ({s.thumb_tangential_signed_N:+.2f} N)")
                    else:
                        ax_forces.bar(x - w / 2, real_vals, w,
                                      color=colors, label="Real (Fn)", alpha=0.85)
                        ax_forces.bar(x + w / 2, sim_vals, w,
                                      color=colors, label="Sim sensor |F|", alpha=0.40,
                                      hatch="//", edgecolor="gray")
                    ax_forces.set_xticks(x)
                    ax_forces.set_xticklabels(fingers, rotation=25, fontsize=8)
                    ax_forces.legend(fontsize=7)
                    ax_forces.set_ylim(bottom=0)

                    # ── Panel 4: Ferrari-Canny time series ────────────────────
                    ax_quality.cla()
                    show_tan = self.show_tangential
                    ax_quality.set_title(
                        "Ferrari-Canny Q  [sim actual (N) vs real disk (N)"
                        + (" vs tangential (N)]" if show_tan else "]"),
                        fontsize=10)
                    ax_quality.set_xlabel("Time (s)", fontsize=8)
                    ax_quality.set_ylabel("Q (N)", fontsize=8)

                    if hist.times:
                        t = np.array(hist.times)
                        qs = np.array(hist.q_sim_actual)
                        qr = np.array(hist.q_real)

                        ax_quality.plot(t, qs, color="green", linewidth=1.5,
                                        label="Sim actual (Fn+Ft known)")
                        ax_quality.plot(t, qr, color="blue", linewidth=1.5,
                                        linestyle="--", label="Real disk (Ft unknown)")
                        ax_quality.axhline(y=0, color="red", linestyle=":",
                                           alpha=0.5, linewidth=0.8)
                        ax_quality.fill_between(t, 0, qs, where=(qs > 0),
                                                color="green", alpha=0.12,
                                                interpolate=True)
                        ax_quality.fill_between(t, 0, qr, where=(qr > 0),
                                                color="blue", alpha=0.10,
                                                interpolate=True)

                        if show_tan and hist.q_real_tangential:
                            qt = np.array(hist.q_real_tangential)
                            ax_quality.plot(t, qt, color="darkorange", linewidth=1.5,
                                            linestyle="-.",
                                            label="Real+thumb Ft_yaw (constrained)")
                            ax_quality.fill_between(t, 0, qt, where=(qt > 0),
                                                    color="orange", alpha=0.10,
                                                    interpolate=True)

                        ax_quality.legend(fontsize=8)

                    # Force closure status annotation
                    fc_text = (f"FC_sim={'✓' if s.fc_sim else '✗'}  "
                               f"FC_real={'✓' if s.fc_real else '✗'}")
                    if show_tan:
                        fc_text += f"  FC_tan={'✓' if s.fc_real_tangential else '✗'}"
                    ax_quality.text(
                        0.02, 0.97, fc_text,
                        transform=ax_quality.transAxes,
                        fontsize=9, verticalalignment="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7))

                    fig.tight_layout(rect=[0, 0, 1, 0.95])
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    plt.pause(0.1)

                except Exception as e:
                    if self.running:
                        print(f"[Real2Sim] Viz error: {e}")
                    time.sleep(0.3)

        except Exception as e:
            print(f"[Real2Sim] Matplotlib thread error: {e}")
        finally:
            if fig is not None:
                try:
                    plt.close(fig)
                except Exception:
                    pass

    # ── Recording ─────────────────────────────────────────────────────────────

    def _save_recording(self):
        if not self.record_path or not self._recording:
            return
        d = self._recording
        fingers = SimAnalyzer.FINGER_NAMES

        times                  = np.array([r["time"] for r in d])
        fc_sim                 = np.array([r["fc_sim"] for r in d])
        fc_real                = np.array([r["fc_real"] for r in d])
        fc_real_tangential     = np.array([r["fc_real_tangential"] for r in d])
        q_sim_geo              = np.array([r["q_sim_geo"] for r in d])
        q_sim_actual           = np.array([r["q_sim_actual"] for r in d])
        q_real                 = np.array([r["q_real"] for r in d])
        q_real_tangential      = np.array([r["q_real_tangential"] for r in d])
        thumb_tangential_N     = np.array([r["thumb_tangential_signed_N"] for r in d])
        num_contacts           = np.array([r["num_contacts"] for r in d])
        object_pos             = np.array([r["object_pos"] for r in d])

        real_forces = np.array([[r["real_forces_N"].get(f, 0.0) for f in fingers]
                                 for r in d])
        sim_sensor  = np.array([[r["sim_sensor_N"].get(f, 0.0) for f in fingers]
                                 for r in d])

        np.savez(
            self.record_path,
            times=times, fc_sim=fc_sim, fc_real=fc_real,
            fc_real_tangential=fc_real_tangential,
            q_sim_geo=q_sim_geo, q_sim_actual=q_sim_actual, q_real=q_real,
            q_real_tangential=q_real_tangential,
            thumb_tangential_signed_N=thumb_tangential_N,
            num_contacts=num_contacts, object_pos=object_pos,
            real_forces_N=real_forces, sim_sensor_N=sim_sensor,
        )
        print(f"[Real2Sim] Saved {len(d)} timesteps to {self.record_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, show_viz: bool = True):
        """Entry point: set up viewer, threads, and run the main loop."""
        self.sim.reset()

        print("\nControls:")
        print("  P     — Pause / resume simulation")
        print("  R     — Reset to home keyframe")
        print("  S     — Print detailed status")
        print("  Q/Esc — Quit")
        print("  [Matplotlib] Check 'Thumb Tangential' to enable yaw tangential force")
        if self.sim_only or self.hand is None:
            print("  [Running in SIM-ONLY mode — real forces from sim sensors]")
        if self.mirror_angles:
            print("  [Mirror-angles ON — sim ctrl sent to real hand each step]")
        if self.show_tangential:
            print("  [Thumb Tangential ON at startup]")
        print()

        # Start real-hand reader thread
        if not self.sim_only and self.hand is not None:
            self._reader_thread = threading.Thread(
                target=self._real_reader_thread, daemon=True)
            self._reader_thread.start()

        # Launch MuJoCo viewer (passive, so we control the loop)
        self.viewer = mujoco.viewer.launch_passive(
            self.sim.model, self.sim.data,
            key_callback=self._key_callback,
        )
        self.viewer.cam.azimuth = 90
        self.viewer.cam.distance = 0.5
        self.viewer.cam.elevation = -20
        self.viewer.cam.lookat[:] = [0.07, 0.029, 0.12]
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

        # Start matplotlib thread
        if show_viz and HAS_MATPLOTLIB:
            self._viz_thread = threading.Thread(
                target=self._viz_thread_func, daemon=True)
            self._viz_thread.start()

        # Batch sim steps per display frame to run at real time.
        # Analysis (contacts, wrench cones, ConvexHull) runs once per frame.
        _sim_dt = self.sim.model.opt.timestep
        _steps_per_frame = max(1, round((1 / 60) / _sim_dt))
        print(f"[Real2Sim] timestep={_sim_dt*1000:.1f} ms  "
              f"steps_per_frame={_steps_per_frame}")

        last_print = 0.0
        try:
            while self.viewer.is_running() and self.running:
                if not self.paused:
                    # Sub-steps: advance sim without expensive analysis
                    for _ in range(_steps_per_frame - 1):
                        self.sim.step()
                    # Final step: step + full contact/force/FC analysis
                    state, sensor_vecs, tip_pos = self._analysis_step()

                    # Update viewer overlays
                    self.sim.fill_viewer_geoms(
                        scn=self.viewer.user_scn,
                        contacts=state.contacts,
                        sensor_forces_world=sensor_vecs,
                        tip_positions=tip_pos,
                        object_pos=state.object_pos,
                        force_closure_sim=state.fc_sim,
                        ferrari_canny_sim=state.q_sim_geo,
                    )

                    # Periodic console status
                    if (state.sim_time - last_print > 3.0
                            and state.num_contacts > 0):
                        self._print_status()
                        last_print = state.sim_time

                self.viewer.sync()
                time.sleep(1 / 60)

        except KeyboardInterrupt:
            print("\n[Real2Sim] Interrupted")
        finally:
            self.running = False
            if self._viz_thread:
                self._viz_thread.join(timeout=2.0)
            self._save_recording()
            print("[Real2Sim] Done")


# ── Replay ────────────────────────────────────────────────────────────────────

def replay_recording(npz_path: str):
    """Static replay of a saved .npz file."""
    if not HAS_MATPLOTLIB:
        sys.exit("matplotlib required for replay")

    print(f"Loading {npz_path}...")
    d = np.load(npz_path, allow_pickle=True)
    times        = d["times"]
    q_sim_actual = d["q_sim_actual"]
    q_sim_geo    = d.get("q_sim_geo", np.zeros_like(q_sim_actual))
    q_real       = d["q_real"]
    fc_sim       = d["fc_sim"]
    fc_real      = d["fc_real"]
    num_contacts = d["num_contacts"]
    real_forces  = d.get("real_forces_N", np.zeros((len(times), 5)))
    sim_sensor   = d.get("sim_sensor_N",  np.zeros((len(times), 5)))
    fingers      = SimAnalyzer.FINGER_NAMES

    print(f"  {len(times)} timesteps, t=[{times[0]:.2f}, {times[-1]:.2f}]s")
    print(f"  FC_sim: {np.sum(fc_sim)}/{len(fc_sim)}")
    print(f"  FC_real: {np.sum(fc_real)}/{len(fc_real)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Replay: {Path(npz_path).name}", fontsize=13)

    # Panel 1 — Ferrari-Canny time series
    ax = axes[0, 0]
    ax.set_title("Ferrari-Canny Q over time (Newton units)", fontsize=10)
    ax.plot(times, q_sim_actual, color="green", linewidth=1.5,
            label="Sim actual (Fn+Ft known, N)")
    ax.plot(times, q_real,       color="blue",  linewidth=1.5, linestyle="--",
            label="Real disk (Ft unknown, N)")
    ax.axhline(0, color="red", linestyle=":", alpha=0.5)
    ax.fill_between(times, 0, q_sim_actual, where=(q_sim_actual > 0),
                    color="green", alpha=0.12)
    ax.fill_between(times, 0, q_real, where=(q_real > 0), color="blue",  alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Q (N)")
    ax.legend(fontsize=8)

    # Panel 2 — Force closure status
    ax2 = axes[0, 1]
    ax2.set_title("Force Closure Status", fontsize=10)
    ax2.fill_between(times, 0, fc_sim.astype(float),  color="green", alpha=0.35,
                     label="FC_sim", step="post")
    ax2.fill_between(times, 0, fc_real.astype(float), color="blue",  alpha=0.25,
                     label="FC_real", step="post")
    ax2.set_ylim([-0.05, 1.2])
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=8)

    # Panel 3 — Per-finger force magnitudes over time
    ax3 = axes[1, 0]
    ax3.set_title("Per-Finger Normal Force (N)", fontsize=10)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("|F| (N)")
    for i, f in enumerate(fingers):
        col = FINGER_COLORS_RGB.get(f, (0.5, 0.5, 0.5))
        ax3.plot(times, real_forces[:, i], color=col, linewidth=1.5, label=f"{f} (real)")
        ax3.plot(times, sim_sensor[:, i],  color=col, linewidth=1.0, linestyle="--",
                 alpha=0.55, label=f"{f} (sim sensor)")
    ax3.legend(fontsize=6, ncol=2, loc="upper left")

    # Panel 4 — Contact count + Q_sim gap
    ax4 = axes[1, 1]
    ax4.set_title("Contact Count & Q_sim − Q_real gap", fontsize=10)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("# Contacts", color="gray")
    ax4.plot(times, num_contacts, "k-", alpha=0.35, linewidth=0.8)
    ax4.fill_between(times, 0, num_contacts, alpha=0.1, color="gray")
    ax4b = ax4.twinx()
    gap = np.array(q_sim_actual) - np.array(q_real)
    ax4b.plot(times, gap, color="orange", linewidth=1.5, label="Q_sim_actual − Q_real")
    ax4b.axhline(0, color="orange", linestyle=":", alpha=0.4)
    ax4b.fill_between(times, 0, gap, where=(gap > 0), color="orange", alpha=0.15)
    ax4b.set_ylabel("Q gap", color="orange")
    ax4b.tick_params(axis="y", labelcolor="orange")
    ax4b.legend(fontsize=8, loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real2Sim Grasp Force Visualizer — Inspire RH56 Hand")
    parser.add_argument("--port", type=str, default=None,
                        help="Serial port for real hand (e.g. /dev/ttyUSB0)")
    parser.add_argument("--hand-id", type=int, default=1,
                        help="Hand ID on the serial bus (default: 1)")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to inspire_scene.xml (default: auto-detect)")
    parser.add_argument("--cone-edges", type=int, default=8,
                        help="Friction cone linearization edges (default: 8)")
    parser.add_argument("--mirror-angles", action="store_true",
                        help="Send sim ctrl angles to real hand each step")
    parser.add_argument("--sim-only", action="store_true",
                        help="Disable real hardware; use sim sensor forces as 'real'")
    parser.add_argument("--no-viz", action="store_true",
                        help="Disable Matplotlib window")
    parser.add_argument("--record", type=str, default=None,
                        help="Save metrics to .npz file")
    parser.add_argument("--replay", type=str, default=None,
                        help="Replay a previously saved .npz recording")
    parser.add_argument("--thumb-tangential", action="store_true",
                        help="Enable thumb yaw tangential force at startup "
                             "(can also toggle in the Matplotlib window)")
    args = parser.parse_args()

    if args.replay:
        replay_recording(args.replay)
        return

    if not args.sim_only and args.port is None:
        print("Warning: --port not specified and --sim-only not set.")
        print("Defaulting to --sim-only mode.\n")
        args.sim_only = True

    viz = Real2SimViz(
        port=args.port,
        hand_id=args.hand_id,
        xml_path=args.model,
        friction_cone_edges=args.cone_edges,
        mirror_angles=args.mirror_angles,
        sim_only=args.sim_only,
        record_path=args.record,
        thumb_tangential=args.thumb_tangential,
    )
    viz.run(show_viz=not args.no_viz)


if __name__ == "__main__":
    main()
