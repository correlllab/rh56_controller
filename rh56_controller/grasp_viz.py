"""
grasp_viz.py — Interactive antipodal grasp geometry visualizer for Inspire RH56.

Usage:
    python -m rh56_controller.grasp_viz [--xml path/to/inspire_right.xml] [--rebuild]

Controls:
    Radio buttons : select grasp mode (2-finger line / 3/4/5-finger plane / cylinder)
    Width slider  : target object width or diameter (mm)
    Z slider      : world-frame height of the grasp midplane (mm)
    MuJoCo button : open floating-hand viewer synced to current settings

World frame convention (matplotlib):
    +Z up  (world gravity down)
    Hand hangs fingers-down above the grasp plane.
    The hand base origin appears above the fingertips.
"""

import argparse
import pathlib
import threading
import time
from typing import Optional, Dict

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons, Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mujoco
import mujoco.viewer

from .grasp_geometry import (
    InspireHandFK, ClosureGeometry, ClosureResult,
    CTRL_MAX, NON_THUMB_FINGERS, _DEFAULT_XML,
)

_HERE = pathlib.Path(__file__).parent.parent
_GRASP_SCENE = str(_HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml")

# Colour palette per finger
FINGER_COLORS = {
    "thumb":  "#e74c3c",   # red
    "index":  "#e67e22",   # orange
    "middle": "#2ecc71",   # green
    "ring":   "#3498db",   # blue
    "pinky":  "#9b59b6",   # purple
}

MODES = [
    "2-finger line",
    "3-finger plane",
    "4-finger plane",
    "5-finger plane",
    "cylinder",
]


# ---------------------------------------------------------------------------
# GraspViz
# ---------------------------------------------------------------------------
class GraspViz:
    def __init__(self, xml_path: str = _DEFAULT_XML, rebuild: bool = False):
        print("[GraspViz] Initialising FK model...")
        self.fk       = InspireHandFK(xml_path=xml_path, rebuild=rebuild)
        self.closure  = ClosureGeometry(self.fk)

        # State
        self._mode     = "4-finger plane"
        self._width_m  = 0.040          # metres
        self._grasp_z  = 0.0            # world Z of grasp midplane (metres)
        self._result: Optional[ClosureResult] = None

        # MuJoCo viewer state
        self._viewer_thread: Optional[threading.Thread] = None
        self._viewer_stop   = threading.Event()
        self._viewer_lock   = threading.Lock()
        self._viewer_ctrl: Optional[np.ndarray] = None  # ctrl to push to viewer

        # Width slider range tracking (updated on mode change)
        self._width_range = (0.015, 0.090)  # metres

        # Compute initial result
        self._recompute()

    # ------------------------------------------------------------------
    # Compute closure
    # ------------------------------------------------------------------
    def _recompute(self):
        """Recompute ClosureResult for current mode + width."""
        mode = self._mode
        w    = self._width_m
        try:
            self._result = self.closure.solve(mode, w)
        except Exception as e:
            print(f"[GraspViz] solve failed: {e}")
            self._result = None

        # Push ctrl update to viewer if open
        if self._result is not None:
            self._push_viewer_ctrl()

    # ------------------------------------------------------------------
    # MuJoCo viewer integration
    # ------------------------------------------------------------------
    def _push_viewer_ctrl(self):
        """Package current ctrl + base pose for the viewer thread."""
        if self._result is None:
            return
        r  = self._result
        gz = self._grasp_z

        # World-frame position of the hand base, accounting for tilt.
        # world_base() applies Ry(base_tilt_y) @ Rx(π) and places the
        # grasp midpoint at (0, 0, gz).
        wbase = r.world_base(gz)

        # Build 12-element ctrl vector:
        # [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
        #  pinky, ring, middle, index, thumb_prox, thumb_yaw]
        #
        # MuJoCo joint order: Rx(rot_x) @ Ry(rot_y)  (axes in world/parent frame).
        # Our FK model uses R = Ry(tilt) @ Rx(π).
        # Identity: Rx(π) @ Ry(−tilt) = Ry(tilt) @ Rx(π), so negate rot_y.
        ctrl = np.array([
            wbase[0],           # pos_x
            wbase[1],           # pos_y
            wbase[2],           # pos_z
            np.pi,              # rot_x  (180° → fingers hang down)
            -r.base_tilt_y,     # rot_y  (negated: Rx(π)@Ry(−tilt)=Ry(tilt)@Rx(π))
            0.0,                # rot_z
            r.ctrl_values.get("pinky",  0.0),
            r.ctrl_values.get("ring",   0.0),
            r.ctrl_values.get("middle", 0.0),
            r.ctrl_values.get("index",  0.0),
            r.ctrl_values.get("thumb_proximal", 0.0),
            r.ctrl_values.get("thumb_yaw",      0.0),
        ], dtype=float)

        with self._viewer_lock:
            self._viewer_ctrl = ctrl

    def _launch_viewer(self):
        """Launch MuJoCo passive viewer in a background thread."""
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            print("[GraspViz] Viewer already open.")
            return

        self._viewer_stop.clear()
        self._viewer_thread = threading.Thread(
            target=self._viewer_loop, daemon=True, name="mujoco-viewer")
        self._viewer_thread.start()

    @staticmethod
    def _setup_jnt_map(model: mujoco.MjModel) -> dict:
        """Cache qpos addresses for the 12 joints we control in the grasp scene."""
        def jadr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise ValueError(f"Joint '{name}' not found in grasp scene model")
            return int(model.jnt_qposadr[jid])
        return {
            "pos_x":        jadr("right_pos_x"),
            "pos_y":        jadr("right_pos_y"),
            "pos_z":        jadr("right_pos_z"),
            "rot_x":        jadr("right_rot_x"),
            "rot_y":        jadr("right_rot_y"),
            "rot_z":        jadr("right_rot_z"),
            "pinky":        jadr("pinky_proximal_joint"),
            "pinky_inter":  jadr("pinky_intermediate_joint"),
            "ring":         jadr("ring_proximal_joint"),
            "ring_inter":   jadr("ring_intermediate_joint"),
            "middle":       jadr("middle_proximal_joint"),
            "middle_inter": jadr("middle_intermediate_joint"),
            "index":        jadr("index_proximal_joint"),
            "index_inter":  jadr("index_intermediate_joint"),
            "thumb_yaw":    jadr("thumb_proximal_yaw_joint"),
            "thumb_pitch":  jadr("thumb_proximal_pitch_joint"),
            "thumb_inter":  jadr("thumb_intermediate_joint"),
            "thumb_distal": jadr("thumb_distal_joint"),
        }

    @staticmethod
    def _apply_qpos(jm: dict, data: mujoco.MjData, ctrl: np.ndarray,
                    model: mujoco.MjModel):
        """
        Directly write qpos from the 12-element ctrl vector (no servo dynamics).
        ctrl layout: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
                      pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        After writing, call mj_kinematics to update FK (no equality-constraint solver
        needed — coupled joints are handled explicitly with the same formulas as the
        FK sweep code).
        """
        pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = ctrl[0:6]
        pinky, ring, middle, index, pitch, yaw     = ctrl[6:12]

        # Base 6-DOF
        data.qpos[jm["pos_x"]] = pos_x
        data.qpos[jm["pos_y"]] = pos_y
        data.qpos[jm["pos_z"]] = pos_z
        data.qpos[jm["rot_x"]] = rot_x
        data.qpos[jm["rot_y"]] = rot_y
        data.qpos[jm["rot_z"]] = rot_z

        # Non-thumb fingers (1:1 coupling except index)
        data.qpos[jm["pinky"]]       = pinky
        data.qpos[jm["pinky_inter"]] = pinky
        data.qpos[jm["ring"]]        = ring
        data.qpos[jm["ring_inter"]]  = ring
        data.qpos[jm["middle"]]      = middle
        data.qpos[jm["middle_inter"]] = middle
        data.qpos[jm["index"]]       = index
        # polycoef="0.15 1 0 0 0": index_intermediate = 0.15 + 1.0 * index_proximal
        data.qpos[jm["index_inter"]] = index + 0.15

        # Thumb (coupling from inspire_right.xml)
        data.qpos[jm["thumb_yaw"]]   = yaw
        data.qpos[jm["thumb_pitch"]] = pitch
        # polycoef="0.15 1.25 0 0": thumb_intermediate = 0.15 + 1.25 * pitch
        data.qpos[jm["thumb_inter"]] = 0.15 + 1.25 * pitch
        # polycoef="0.15 0.75 0 0 0": thumb_distal = 0.15 + 0.75 * pitch
        data.qpos[jm["thumb_distal"]] = 0.15 + 0.75 * pitch

        mujoco.mj_kinematics(model, data)

    def _viewer_loop(self):
        """Background thread: open passive viewer and sync hand configuration."""
        try:
            model = mujoco.MjModel.from_xml_path(_GRASP_SCENE)
            data  = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            jm = self._setup_jnt_map(model)

            # Apply initial pose
            self._push_viewer_ctrl()
            with self._viewer_lock:
                if self._viewer_ctrl is not None:
                    self._apply_qpos(jm, data, self._viewer_ctrl, model)

            with mujoco.viewer.launch_passive(model, data) as v:
                while v.is_running() and not self._viewer_stop.is_set():
                    with self._viewer_lock:
                        ctrl = self._viewer_ctrl.copy() \
                               if self._viewer_ctrl is not None else None
                    if ctrl is not None:
                        self._apply_qpos(jm, data, ctrl, model)
                    v.sync()
                    time.sleep(0.033)   # ~30 Hz
        except Exception as e:
            print(f"[GraspViz] Viewer error: {e}")

    # ------------------------------------------------------------------
    # matplotlib setup
    # ------------------------------------------------------------------
    def run(self):
        """Build and show the interactive matplotlib figure."""
        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Inspire RH56 — Antipodal Grasp Geometry Planner", fontsize=12)

        # 3D axes: left 65%
        ax3d: Axes3D = fig.add_axes([0.02, 0.05, 0.60, 0.88], projection="3d")
        ax3d.set_xlabel("X  (closure direction)")
        ax3d.set_ylabel("Y  (finger spread)")
        ax3d.set_zlabel("Z  (world up)")

        self._ax3d = ax3d

        # ---- Radio buttons ----
        ax_radio = fig.add_axes([0.65, 0.60, 0.16, 0.30])
        self._radio = RadioButtons(ax_radio, MODES, active=MODES.index(self._mode))
        self._radio.on_clicked(self._on_mode)

        # ---- Width slider ----
        ax_wmin = 0.65
        ax_wmin_txt = fig.add_axes([0.65, 0.52, 0.16, 0.04])
        ax_wmin_txt.axis("off")
        ax_wmin_txt.text(0.0, 0.5, "Width / Diameter (mm):", fontsize=9)
        ax_wslide = fig.add_axes([0.65, 0.47, 0.28, 0.03])
        wmin_mm, wmax_mm = (x * 1000 for x in self._width_range)
        self._slider_w = Slider(
            ax_wslide, "", wmin_mm, wmax_mm,
            valinit=self._width_m * 1000, valstep=1.0,
        )
        self._slider_w.on_changed(self._on_width)

        # ---- Z slider ----
        ax_ztxt = fig.add_axes([0.65, 0.37, 0.16, 0.04])
        ax_ztxt.axis("off")
        ax_ztxt.text(0.0, 0.5, "Grasp plane Z (mm):", fontsize=9)
        ax_zslide = fig.add_axes([0.65, 0.32, 0.28, 0.03])
        self._slider_z = Slider(
            ax_zslide, "", -200.0, 200.0,
            valinit=self._grasp_z * 1000, valstep=5.0,
        )
        self._slider_z.on_changed(self._on_z)

        # ---- Info text ----
        self._ax_info = fig.add_axes([0.65, 0.12, 0.30, 0.18])
        self._ax_info.axis("off")
        self._info_text = self._ax_info.text(
            0.0, 1.0, "", fontsize=8,
            verticalalignment="top", family="monospace",
        )

        # ---- MuJoCo viewer button ----
        ax_btn = fig.add_axes([0.72, 0.04, 0.20, 0.05])
        self._btn_mujoco = Button(ax_btn, "Open MuJoCo Viewer")
        self._btn_mujoco.on_clicked(self._on_open_viewer)

        # Initial draw
        self._update_plot()
        plt.show()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_mode(self, label: str):
        self._mode = label
        # Update width slider range for the new mode
        n = int(label[0]) if label[0].isdigit() else 4
        wrange = self.closure.width_range(label, n_fingers=n)
        self._width_range = wrange
        wmin_mm, wmax_mm = wrange[0] * 1000, wrange[1] * 1000
        # Clamp current width to new range
        self._width_m = float(np.clip(self._width_m, wrange[0], wrange[1]))
        # Rebuild slider with new range
        self._slider_w.valmin = wmin_mm
        self._slider_w.valmax = wmax_mm
        self._slider_w.set_val(self._width_m * 1000)
        self._slider_w.ax.set_xlim(wmin_mm, wmax_mm)
        self._recompute()
        self._update_plot()

    def _on_width(self, val: float):
        self._width_m = float(val) / 1000.0
        self._recompute()
        self._update_plot()

    def _on_z(self, val: float):
        self._grasp_z = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_open_viewer(self, event):
        self._launch_viewer()

    # ------------------------------------------------------------------
    # 3D plot update
    # ------------------------------------------------------------------
    def _update_plot(self):
        ax = self._ax3d
        ax.cla()
        ax.set_xlabel("X  (closure direction)")
        ax.set_ylabel("Y  (finger spread)")
        ax.set_zlabel("Z  (world up)")

        if self._result is None:
            ax.set_title("No solution found")
            plt.draw()
            return

        r   = self._result
        gz  = self._grasp_z
        wtips = r.world_tips(gz)    # {fname: (3,) in world frame}
        wbase = r.world_base(gz)    # hand base origin in world frame

        # ---------- Fingertip dots ----------
        for fname, pos in wtips.items():
            col = FINGER_COLORS.get(fname, "gray")
            ax.scatter(*pos, color=col, s=60, zorder=5)
            ax.text(pos[0]+0.003, pos[1], pos[2]+0.003, fname[:3],
                    fontsize=7, color=col)

        # ---------- Hand base origin ----------
        ax.scatter(*wbase, color="black", s=80, marker="x", zorder=6)
        ax.text(wbase[0]+0.003, wbase[1], wbase[2]+0.003, "base", fontsize=7)

        # ---------- Grasp plane Z reference line ----------
        ys = np.array([
            min(p[1] for p in wtips.values()) - 0.01,
            max(p[1] for p in wtips.values()) + 0.01,
        ])
        ax.plot([-0.02, 0.12], ys[[0, 0]], [gz, gz], "g--", lw=0.8, alpha=0.6)
        ax.plot([-0.02, 0.12], ys[[1, 1]], [gz, gz], "g--", lw=0.8, alpha=0.6)

        mode = r.mode
        if mode == "2-finger line":
            self._draw_line_closure(ax, wtips, gz)
        elif "plane" in mode:
            self._draw_plane_closure(ax, wtips, gz, r)
        elif mode == "cylinder":
            self._draw_cylinder_closure(ax, wtips, gz, r)

        # ---------- Info text ----------
        lines = [
            f"Mode:   {r.mode}",
            f"Width:  {r.width*1000:.1f} mm",
            f"Span:   {r.finger_span*1000:.1f} mm",
        ]
        if r.cylinder_radius > 0:
            lines.append(f"Radius: {r.cylinder_radius*1000:.1f} mm")
        lines.append(f"Tilt Y: {r.tilt_deg:.1f}°")
        lines.append(f"Base Z: {wbase[2]*1000:.1f} mm")
        lines.append("")
        lines.append("Ctrl values (rad):")
        for k in ["index", "middle", "ring", "pinky",
                  "thumb_proximal", "thumb_yaw"]:
            v = r.ctrl_values.get(k, 0.0)
            if v > 0.001:
                lines.append(f"  {k[:12]:12s}: {v:.3f}")
        self._info_text.set_text("\n".join(lines))

        # Axis limits
        all_pts = np.array(list(wtips.values()) + [wbase])
        margin = 0.025
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

        plt.draw()

    # ------------------------------------------------------------------
    # Geometry overlays
    # ------------------------------------------------------------------
    def _draw_line_closure(self, ax, wtips, gz):
        """Draw the 2-finger pinch line segment."""
        t = wtips["thumb"]
        i = wtips["index"]
        ax.plot([t[0], i[0]], [t[1], i[1]], [t[2], i[2]],
                "r-", lw=2.5, label="pinch line")
        mid = (t + i) / 2
        ax.scatter(*mid, color="gold", s=90, marker="*", zorder=7)

    def _draw_plane_closure(self, ax, wtips, gz, r: ClosureResult):
        """Draw the box plane as a filled translucent rectangle."""
        n = int(r.mode[0])
        from .grasp_geometry import GRASP_FINGER_SETS
        fnames = GRASP_FINGER_SETS[n]

        # Non-thumb side: a horizontal bar connecting the active fingertips
        fpts = np.array([wtips[f] for f in fnames])
        ax.plot(fpts[:, 0], fpts[:, 1], fpts[:, 2],
                "b-o", lw=2, ms=5, label="non-thumb row")

        # Thumb tip
        th = wtips["thumb"]

        # Project thumb to the plane of non-thumb tips (same Z — coplanar by construction)
        # Draw thumb connector to each finger
        for f in fnames:
            fp = wtips[f]
            ax.plot([th[0], fp[0]], [th[1], fp[1]], [th[2], fp[2]],
                    "--", color="gray", lw=0.8, alpha=0.6)

        # Draw the bounding rectangle of the closure box
        y_min = fpts[:, 1].min()
        y_max = fpts[:, 1].max()
        z_ref = gz

        # Corners: (x_nonthumb, y_min/max, gz) and (x_thumb, y_min/max, gz)
        x_nf  = fpts[:, 0].mean()
        x_th  = th[0]
        corners = np.array([
            [x_nf, y_min, z_ref],
            [x_nf, y_max, z_ref],
            [x_th, y_max, z_ref],
            [x_th, y_min, z_ref],
        ])
        poly = Poly3DCollection([corners], alpha=0.10, facecolor="cyan",
                                edgecolor="steelblue", linewidth=1.2)
        ax.add_collection3d(poly)

        # Midplane line
        mid_x = (x_nf + x_th) / 2
        ax.plot([mid_x, mid_x], [y_min - 0.01, y_max + 0.01], [z_ref, z_ref],
                "g:", lw=1.5, label="midplane")

    def _draw_cylinder_closure(self, ax, wtips, gz, r: ClosureResult):
        """Draw the cylinder closure as a wireframe cylinder."""
        from .grasp_geometry import NON_THUMB_FINGERS
        fpts = np.array([wtips[f] for f in NON_THUMB_FINGERS])

        # Cylinder axis is along Y; draw circular arcs at min and max Y
        centroid_x = fpts[:, 0].mean()
        centroid_z = fpts[:, 2].mean()
        radius = r.cylinder_radius

        y_min = fpts[:, 1].min() - 0.005
        y_max = fpts[:, 1].max() + 0.005
        theta = np.linspace(0, np.pi, 60)  # semi-circle on the finger side

        # Draw two arcs (top = finger arc, bottom = palm)
        for y in [y_min, y_max]:
            xc = centroid_x + radius * np.cos(theta)
            zc = gz + (centroid_z - gz) + radius * np.sin(theta)
            ax.plot(xc, np.full_like(xc, y), zc,
                    "b-", lw=1.5, alpha=0.7)

        # Connect the arcs with vertical lines at endpoints
        for ang in [0, np.pi]:
            xp = centroid_x + radius * np.cos(ang)
            zp = gz + (centroid_z - gz) + radius * np.sin(ang)
            ax.plot([xp, xp], [y_min, y_max], [zp, zp], "b-", lw=1.0, alpha=0.5)

        # Centroid axis
        ax.plot([centroid_x, centroid_x], [y_min, y_max],
                [centroid_z, centroid_z], "g:", lw=1.5, label="cyl axis")

        # Thumb connector
        centroid_y = fpts[:, 1].mean()
        th = wtips["thumb"]
        ax.plot([th[0], centroid_x], [th[1], centroid_y],
                [th[2], centroid_z], "--r", lw=1.5, alpha=0.7)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inspire RH56 Grasp Geometry Visualizer")
    parser.add_argument("--xml", default=_DEFAULT_XML,
                        help="Path to inspire_right.xml")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild of FK cache")
    args = parser.parse_args()

    viz = GraspViz(xml_path=args.xml, rebuild=args.rebuild)
    viz.run()


if __name__ == "__main__":
    main()
