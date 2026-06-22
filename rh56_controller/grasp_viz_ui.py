"""
grasp_viz_ui.py — Tkinter UI layer for GraspViz.

GraspVizUI inherits GraspVizCore and adds:
  - A tk.Tk() root window with three columns:
      Col 0: embedded matplotlib 3D axes (FigureCanvasTkAgg)
      Col 1: parameter sliders (ttk.Scale + tk.Entry pairs)
      Col 2: mode/strategy radios, viewer buttons, real-robot panel
  - Debounced slider callbacks (recompute + plot at most every 40 ms)
  - Native tk.Entry for all text input (responsive; no coordinate math)
  - ScrolledText status area replacing matplotlib TextBox
  - Dedicated "Width Target" entry decoupled from the slider (item 1)
  - Per-grasp JSONL logging via GraspLogger (item 2)
"""

import os
import queue
import sys
import threading
import time
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import ttk
from typing import Optional

import matplotlib
matplotlib.use("TkAgg")  # must be set before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D           # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .grasp_geometry import ClosureResult, GRASP_FINGER_SETS, NON_THUMB_FINGERS
from .grasp_viz_core import GraspVizCore
from .grasp_viz_workers import FINGER_COLORS, MODES
from .grasp_logger import GraspLogger
from .grasp_viz_force_panel import ForceVizPanel
from .force_control_ui import ForceControlUI

# Active finger indices (actuator order: pinky=0 ring=1 middle=2 index=3 thumb_bend=4)
# per closure mode, used when setting force targets
_MODE_ACTIVE_FINGERS = {
    "2-finger line":  [3, 4],
    "3-finger plane": [2, 3, 4],
    "4-finger plane": [1, 2, 3, 4],
    "5-finger plane": [0, 1, 2, 3, 4],
    "cylinder":       [0, 1, 2, 3, 4],
}

_POLL_MS    = 150   # status queue poll interval
_RECOMP_MS  = 40    # debounce for slider → recompute + plot
_BUTTON_TEXT_ON_COLOR = "black" if sys.platform == "darwin" else "white"


class GraspVizUI(GraspVizCore):
    """Interactive tkinter UI wrapping GraspVizCore."""

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def run(self):
        self._root = tk.Tk()
        self._root.title("Inspire RH56 — Antipodal Grasp Geometry Planner")
        self._root.resizable(True, True)

        # Debounce handles
        self._debounce_id: Optional[str] = None

        # ---- Main layout: 3 columns ----
        self._root.columnconfigure(0, weight=3)
        self._root.columnconfigure(1, weight=1)
        self._root.columnconfigure(2, weight=1)
        self._root.rowconfigure(0, weight=1)

        self._build_plot_column()
        self._build_slider_column()
        self._build_control_column()

        # Initial plot
        self._update_plot()

        # Intercept the close button
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start status poll loop
        self._poll_id: Optional[str] = self._root.after(_POLL_MS, self._poll_status_queue)

        self._root.mainloop()

    def _on_close(self):
        """Graceful shutdown sequence."""
        # 1. Cancel all pending after-callbacks before anything blocks
        if hasattr(self, "_debounce_id") and self._debounce_id:
            self._root.after_cancel(self._debounce_id)
            self._debounce_id = None
        if hasattr(self, "_poll_id") and self._poll_id:
            self._root.after_cancel(self._poll_id)
            self._poll_id = None

        # 2. Stop the poll loop from rescheduling itself
        self._running = False

        # 3. Call the core cleanup (closes robot/hand/viewers)
        self.cleanup()

        # 4. Destroy the UI
        self._root.destroy()

        # 5. Force-exit: magpie_control UR5_Interface has non-daemon RTDE
        # threads that prevent a clean Python exit.  os._exit bypasses them
        # and also ensures the RTDE socket is closed by the OS immediately,
        # so the robot controller can accept a new connection right away.
        os._exit(0)

    # ------------------------------------------------------------------
    # Column 0: 3D matplotlib plot
    # ------------------------------------------------------------------
    def _build_plot_column(self):
        frame = ttk.Frame(self._root)
        frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self._fig = plt.figure(figsize=(6, 6))
        self._ax3d = self._fig.add_subplot(111, projection="3d")
        self._ax3d.set_xlabel("X  (closure direction)")
        self._ax3d.set_ylabel("Y  (finger spread)")
        self._ax3d.set_zlabel("Z  (world up)")

        self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------
    # Column 1: sliders
    # ------------------------------------------------------------------
    def _build_slider_column(self):
        outer = ttk.LabelFrame(self._root, text="Parameters", padding=6)
        outer.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        outer.columnconfigure(1, weight=1)

        r = 0

        # Width slider
        wmin_mm, wmax_mm = (x * 1000 for x in self._width_range)
        ttk.Label(outer, text="Width / Diam (mm):").grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self._var_w = tk.DoubleVar(value=self._width_m * 1000)
        self._sl_w  = ttk.Scale(outer, from_=wmin_mm, to=wmax_mm,
                                variable=self._var_w, orient="horizontal", length=160,
                                command=self._on_width)
        self._sl_w.grid(row=r, column=0, columnspan=2, sticky="ew")
        self._ent_w = tk.Entry(outer, width=7)
        self._ent_w.insert(0, f"{self._width_m * 1000:.1f}")
        self._ent_w.grid(row=r, column=2, padx=2)
        self._ent_w.bind("<Return>", lambda e: self._sl_w.set(
            float(self._ent_w.get() or self._var_w.get())))
        self._ent_w.bind("<FocusOut>", lambda e: self._sl_w.set(
            float(self._ent_w.get() or self._var_w.get())))
        r += 1

        # Width Target entry (decoupled from slider — item 1)
        ttk.Label(outer, text="Target width (mm):", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w")
        self._ent_width_target = tk.Entry(outer, width=7,
                                          bg="#fffbe6", relief="solid")
        self._ent_width_target.insert(0, f"{self._width_target_m * 1000:.1f}")
        self._ent_width_target.grid(row=r, column=2, padx=2, pady=2)
        self._ent_width_target.bind("<Return>",   self._on_width_target_submit)
        self._ent_width_target.bind("<FocusOut>", self._on_width_target_submit)
        r += 1

        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4)
        r += 1

        # Grasp Z
        _has_arm = self._robot_mode or self._h12_mode
        z_max = 400.0 if _has_arm else 200.0
        if self._h12_mode:
            z_min = -400.0
        else:
            z_min = 0.0 if _has_arm else -200.0
        r = self._add_slider_row(outer, r, "Grasp Z (mm):",
                                 z_min, z_max, self._grasp_z * 1000, 5.0,
                                 "_var_z", "_sl_z", "_ent_z", self._on_z)
        if _has_arm:
            btn_up = tk.Button(outer, text="+10cm", command=self._on_move_up)
            btn_up.grid(row=r - 1, column=3, padx=2)

        # X, Y (robot mode or h12 mode)
        if _has_arm:
            r = self._add_slider_row(outer, r, "Grasp X (mm):",
                                     -850.0, 850.0, self._grasp_x * 1000, 5.0,
                                     "_var_x", "_sl_x", "_ent_x", self._on_x)
            r = self._add_slider_row(outer, r, "Grasp Y (mm):",
                                     -850.0, 850.0, self._grasp_y * 1000, 5.0,
                                     "_var_y", "_sl_y", "_ent_y", self._on_y)

        ttk.Label(outer, text="── Plane orientation ──",
                  foreground="#777").grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1

        r = self._add_slider_row(outer, r, "Plane Rx (°):",
                                 -180.0, 180.0, 0.0, 1.0,
                                 "_var_rx", "_sl_rx", "_ent_rx", self._on_plane_rx)
        r = self._add_slider_row(outer, r, "Plane Ry (°):",
                                 -180.0, 180.0, 0.0, 1.0,
                                 "_var_ry", "_sl_ry", "_ent_ry", self._on_plane_ry)
        r = self._add_slider_row(outer, r, "Plane Rz (°):",
                                 -180.0, 180.0, 0.0, 1.0,
                                 "_var_rz", "_sl_rz", "_ent_rz", self._on_plane_rz)

        if self._h12_mode:
            ttk.Label(outer, text="── Wrist orientation (real, from /ee_pose) ──",
                      foreground="#777").grid(row=r, column=0, columnspan=3, sticky="w")
            r += 1
            r = self._add_slider_row(outer, r, "Wrist Rx (°):",
                                     -180.0, 180.0, 0.0, 1.0,
                                     "_var_wrist_rx", "_sl_wrist_rx", "_ent_wrist_rx", lambda _v: None)
            r = self._add_slider_row(outer, r, "Wrist Ry (°):",
                                     -180.0, 180.0, 0.0, 1.0,
                                     "_var_wrist_ry", "_sl_wrist_ry", "_ent_wrist_ry", lambda _v: None)
            r = self._add_slider_row(outer, r, "Wrist Rz (°):",
                                     -180.0, 180.0, 0.0, 1.0,
                                     "_var_wrist_rz", "_sl_wrist_rz", "_ent_wrist_rz", lambda _v: None)
            for wid in [
                getattr(self, "_sl_wrist_rx", None),
                getattr(self, "_sl_wrist_ry", None),
                getattr(self, "_sl_wrist_rz", None),
                getattr(self, "_ent_wrist_rx", None),
                getattr(self, "_ent_wrist_ry", None),
                getattr(self, "_ent_wrist_rz", None),
            ]:
                if wid is not None:
                    try:
                        wid.config(state="disabled")
                    except Exception:
                        pass

    def _add_slider_row(self, parent, row, label, vmin, vmax, vinit, vstep,
                        var_attr, sl_attr, ent_attr, callback):
        """Add a label + Scale + Entry row.  Returns next available row."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w")
        var = tk.DoubleVar(value=vinit)
        sl  = ttk.Scale(parent, from_=vmin, to=vmax, variable=var,
                        orient="horizontal", length=160, command=callback)
        sl.grid(row=row, column=1, sticky="ew")
        ent = tk.Entry(parent, width=7)
        ent.insert(0, f"{vinit:.1f}")
        ent.grid(row=row, column=2, padx=2)

        def _on_entry_commit(event=None, s=sl, v=var, e=ent):
            text = e.get().strip()
            if not text:
                # Empty input: revert to current variable value
                value = v.get()
            else:
                try:
                    value = float(text)
                except ValueError:
                    # Invalid input: revert entry to last valid value
                    value = v.get()
                    e.delete(0, tk.END)
                    e.insert(0, f"{value:.1f}")
                    return
            s.set(value)
            # Normalize entry text to the parsed numeric value
            e.delete(0, tk.END)
            e.insert(0, f"{value:.1f}")

        ent.bind("<Return>", _on_entry_commit)
        ent.bind("<FocusOut>", _on_entry_commit)
        setattr(self, var_attr, var)
        setattr(self, sl_attr,  sl)
        setattr(self, ent_attr, ent)
        return row + 1

    # ------------------------------------------------------------------
    # Column 2: controls
    # ------------------------------------------------------------------
    def _build_control_column(self):
        outer = ttk.Frame(self._root, padding=4)
        outer.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        r = 0

        # ── Mode ──
        ttk.Label(outer, text="── Mode ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._mode_var = tk.StringVar(value=self._mode)
        for m in MODES:
            tk.Radiobutton(outer, text=m, variable=self._mode_var, value=m,
                           command=self._on_mode_radio).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(outer, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1

        _mink_ready = self._mink_enabled and self._mink_planner is not None

        # ── Viewer buttons ──
        ttk.Label(outer, text="── Viewers ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        tk.Button(outer, text="Hand: Ours",
                  command=self._launch_hand_viewer_ours).grid(
            row=r, column=0, sticky="ew", padx=2, pady=1)
        tk.Button(outer, text="Hand: Mink" if _mink_ready else "Hand: Mink N/A",
                  state="normal" if _mink_ready else "disabled",
                  command=self._launch_hand_viewer_mink).grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1
        tk.Button(outer, text="Robot: Ours",
                  command=self._launch_robot_viewer_ours).grid(
            row=r, column=0, sticky="ew", padx=2, pady=1)
        tk.Button(outer, text="Robot: Mink" if _mink_ready else "Robot: Mink N/A",
                  state="normal" if _mink_ready else "disabled",
                  command=self._launch_robot_viewer_mink).grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1
        tk.Button(outer, text="H1-2: Ours",
                  bg="#1a6b3c", fg=_BUTTON_TEXT_ON_COLOR,
                  command=self._launch_h12_viewer).grid(
            row=r, column=0, sticky="ew", padx=2, pady=1)
        tk.Button(outer, text="H1-2: Mink" if _mink_ready else "H1-2: Mink N/A",
                  state="normal" if _mink_ready else "disabled",
                  command=self._launch_h12_viewer_mink).grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        # Bimanual toggle + viewer (only in h12 mode)
        if self._h12_mode:
            self._bimanual_var = tk.BooleanVar(value=self._bimanual_mode)
            ttk.Checkbutton(outer, text="Bimanual mode",
                            variable=self._bimanual_var,
                            command=self._on_bimanual_toggle).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1
            tk.Button(outer, text="H1-2: Bimanual",
                      bg="#1a3c6b", fg=_BUTTON_TEXT_ON_COLOR,
                      command=self._launch_h12_bimanual_viewer).grid(
                row=r, column=0, columnspan=2, sticky="ew", padx=2, pady=1); r += 1

        tk.Button(outer, text="Force Viz",
                  command=self._open_force_viz_panel).grid(
            row=r, column=0, sticky="ew", padx=2, pady=1)
        tk.Button(outer, text="Force Control",
                  bg="#2980b9", fg=_BUTTON_TEXT_ON_COLOR,
                  command=self._open_force_control_ui).grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        # Send to Real checkbox (serial hand or H12 ROS hand)
        if self._hand is not None or getattr(self, "_h12_arm", None) is not None:
            self._send_real_var = tk.BooleanVar(value=self._send_real)
            ttk.Checkbutton(outer, text="Send Hand (Real)",
                            variable=self._send_real_var,
                            command=self._on_send_real).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        # ── H1-2 real robot panel ──
        if self._h12_mode and (self._real_h12_mode or self._h12_ros_mode):
            self._build_h12_real_panel(outer, r)
            return

        # ── H1-2 sim-only panel ──
        if self._h12_mode:
            self._build_h12_sim_panel(outer, r)
            return

        # ── UR5 real robot panel ──
        if self._real_robot_mode:
            self._build_real_robot_panel(outer, r)
            return   # real robot panel manages its own rows

    def _build_h12_sim_panel(self, parent, start_row):
        """H1-2 simulation controls for running the paper grasp strategies."""
        r = start_row

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── H1-2 Sim ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        self._btn_sim_h12 = tk.Button(
            parent, text="Sim H1-2",
            bg="#1a3c6b", fg=_BUTTON_TEXT_ON_COLOR,
            command=self._on_sim_h12)
        self._btn_sim_h12.grid(row=r, column=0, columnspan=2,
                               sticky="ew", padx=2, pady=1); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Strategy ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        self._strategy_var = tk.StringVar(value=self._grasp_strategy)
        for s in ["Naive", "Plan", "Thumb Reflex"]:
            tk.Radiobutton(parent, text=s, variable=self._strategy_var, value=s,
                           command=self._on_strategy_radio).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Parameters ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        for label, attr, default in [
            ("Step (mm):",     "_ent_step",     "10"),
            ("Approach (mm):", "_ent_approach", ""),
        ]:
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            ent = tk.Entry(parent, width=9)
            ent.insert(0, default)
            ent.grid(row=r, column=1, sticky="ew", padx=2)
            setattr(self, attr, ent)
            r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Status ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._status_text = scrolledtext.ScrolledText(
            parent, height=10, width=26, state="disabled",
            font=("Courier", 7), wrap="word")
        self._status_text.grid(row=r, column=0, columnspan=2,
                               sticky="nsew", padx=2, pady=2); r += 1
        parent.rowconfigure(r - 1, weight=1)

    def _build_real_robot_panel(self, parent, start_row):
        r = start_row

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Real Robot ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        _arm_ok = self._arm is not None and self._arm.connected

        self._btn_teach = tk.Button(parent, text="Teach Mode",
                                    command=self._on_teach_mode,
                                    state="normal" if _arm_ok else "disabled")
        self._btn_teach.grid(row=r, column=0, sticky="ew", padx=2, pady=1)
        self._btn_setpose = tk.Button(parent, text="Set Pose",
                                      command=self._on_set_pose_from_robot,
                                      state="normal" if _arm_ok else "disabled")
        self._btn_setpose.grid(row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        self._btn_sendarm = tk.Button(parent, text="Send Arm",
                                      command=self._on_send_arm,
                                      state="normal" if _arm_ok else "disabled")
        self._btn_sendarm.grid(row=r, column=0, sticky="ew", padx=2, pady=1)
        self._btn_simtraj = tk.Button(parent, text="Sim Traj",
                                      command=self._on_simulate_trajectory,
                                      state="normal" if _arm_ok else "disabled")
        self._btn_simtraj.grid(row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        self._btn_reconnect = tk.Button(parent, text="Reconnect Arm",
                                        command=self._reconnect_arm,
                                        state="disabled" if _arm_ok else "normal")
        self._btn_reconnect.grid(row=r, column=0, columnspan=2,
                                 sticky="ew", padx=2, pady=1); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Strategy ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._strategy_var = tk.StringVar(value=self._grasp_strategy)
        for s in ["Naive", "Plan", "Thumb Reflex"]:
            tk.Radiobutton(parent, text=s, variable=self._strategy_var, value=s,
                           command=self._on_strategy_radio).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Parameters ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        for label, attr, default in [
            ("Force (N):", "_ent_force", "0"),
            ("Step (mm):", "_ent_step",  "10"),
            ("Approach (mm):", "_ent_approach", ""),
        ]:
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            ent = tk.Entry(parent, width=9)
            ent.insert(0, default)
            ent.grid(row=r, column=1, sticky="ew", padx=2)
            setattr(self, attr, ent)
            r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Logging ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(parent, text="Log name:").grid(row=r, column=0, sticky="w")
        self._ent_log_name = tk.Entry(parent, width=12)
        self._ent_log_name.insert(0, "test")
        self._ent_log_name.grid(row=r, column=1, sticky="ew", padx=2); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1

        self._btn_grasp = tk.Button(parent, text="GRASP!", font=("TkDefaultFont", 10, "bold"),
                                    bg="#2ecc71", fg=_BUTTON_TEXT_ON_COLOR,
                                    command=self._on_grasp,
                                    state="normal" if _arm_ok else "disabled")
        self._btn_grasp.grid(row=r, column=0, columnspan=2, sticky="ew",
                             padx=2, pady=4); r += 1
        self._set_robot_only_ui_state(self._robot_only_mode)

        if not _arm_ok:
            self._update_status("No UR5 connection. Use --ur5-ip to connect.")

        # Status text area
        ttk.Label(parent, text="── Status ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._status_text = scrolledtext.ScrolledText(
            parent, height=10, width=26, state="disabled",
            font=("Courier", 7), wrap="word")
        self._status_text.grid(row=r, column=0, columnspan=2,
                               sticky="nsew", padx=2, pady=2); r += 1
        parent.rowconfigure(r - 1, weight=1)

    def _build_h12_real_panel(self, parent, start_row):
        """H1-2 real robot control panel (replaces UR5 panel when --real-h12/--h12-ros)."""
        r = start_row
        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Real H1-2 ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1

        self._robot_only_var = tk.BooleanVar(value=self._robot_only_mode)
        ttk.Checkbutton(
            parent,
            text="Robot-only pose (bypass planner pose)",
            variable=self._robot_only_var,
            command=self._on_robot_only_toggle,
        ).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        _h12_ok = getattr(self, "_h12_arm", None) is not None

        self._btn_send_h12 = tk.Button(
            parent, text="Send H1-2",
            bg="#1a6b3c" if _h12_ok else "#aaa", fg=_BUTTON_TEXT_ON_COLOR,
            command=self._on_send_h12,
            state="normal" if _h12_ok else "disabled")
        self._btn_send_h12.grid(row=r, column=0, sticky="ew", padx=2, pady=1)

        self._btn_sim_h12 = tk.Button(
            parent, text="Sim H1-2",
            bg="#1a3c6b", fg=_BUTTON_TEXT_ON_COLOR,
            command=self._on_sim_h12)
        self._btn_sim_h12.grid(row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        self._btn_setpose_h12 = tk.Button(
            parent, text="Set Pose",
            command=self._on_set_pose_from_h12,
            state="normal" if _h12_ok else "disabled")
        self._btn_setpose_h12.grid(row=r, column=0, columnspan=2, sticky="ew",
                                   padx=2, pady=1); r += 1

        ttk.Label(parent, text="── Wrist Pose Sanity ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._h12_pose_frame_var = tk.StringVar(value="frame: n/a")
        self._h12_pose_pelvis_var = tk.StringVar(value="pelvis xyz(mm): n/a")
        self._h12_pose_planner_var = tk.StringVar(value="planner xyz(mm): n/a")
        self._h12_pose_quat_var = tk.StringVar(value="planner quat(xyzw): n/a")
        self._h12_pose_rpy_var = tk.StringVar(value="planner rpy(deg): n/a")
        ttk.Label(parent, textvariable=self._h12_pose_frame_var).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(parent, textvariable=self._h12_pose_pelvis_var).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(parent, textvariable=self._h12_pose_planner_var).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(parent, textvariable=self._h12_pose_quat_var).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        ttk.Label(parent, textvariable=self._h12_pose_rpy_var).grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._btn_refresh_h12_pose = tk.Button(
            parent, text="Refresh Wrist Pose",
            command=self._on_refresh_h12_wrist_pose,
            state="normal" if _h12_ok else "disabled")
        self._btn_refresh_h12_pose.grid(row=r, column=0, columnspan=2, sticky="ew",
                                        padx=2, pady=1); r += 1
        if _h12_ok:
            self._refresh_h12_wrist_pose_panel(log_on_fail=False)

        ttk.Label(parent, text="Active arm:").grid(row=r, column=0, sticky="w")
        self._active_arm_mode_var = tk.StringVar(value="auto")
        arm_mode = ttk.Combobox(
            parent,
            textvariable=self._active_arm_mode_var,
            values=["auto", "right", "left"],
            state="readonly",
            width=8,
        )
        arm_mode.grid(row=r, column=1, sticky="ew", padx=2)
        arm_mode.bind("<<ComboboxSelected>>", self._on_active_arm_mode)
        r += 1

        self._btn_gravity_h12 = tk.Button(
            parent,
            text="Gravity Comp",
            command=self._on_toggle_h12_gravity_comp,
            state="normal" if _h12_ok else "disabled",
        )
        self._btn_gravity_h12.grid(row=r, column=0, columnspan=2, sticky="ew",
                                   padx=2, pady=1); r += 1
        self._refresh_h12_gravity_button()

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1

        self._btn_grasp = tk.Button(
            parent, text="GRASP!", font=("TkDefaultFont", 10, "bold"),
            bg="#2ecc71", fg=_BUTTON_TEXT_ON_COLOR,
            command=self._on_grasp_h12,
            state="normal" if _h12_ok else "disabled")
        self._btn_grasp.grid(row=r, column=0, columnspan=2, sticky="ew",
                             padx=2, pady=4); r += 1

        if not _h12_ok:
            self._update_status("No H1-2 connection. Start frame_task_server + use --real-h12.")

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Strategy ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._strategy_var = tk.StringVar(value=self._grasp_strategy)
        for s in ["Naive", "Plan", "Thumb Reflex"]:
            tk.Radiobutton(parent, text=s, variable=self._strategy_var, value=s,
                           command=self._on_strategy_radio).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        ttk.Separator(parent, orient="horizontal").grid(
            row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1
        ttk.Label(parent, text="── Parameters ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        for label, attr, default in [
            ("Force (N):",     "_ent_force",    "0"),
            ("Step (mm):",     "_ent_step",     "10"),
            ("Approach (mm):", "_ent_approach", ""),
        ]:
            ttk.Label(parent, text=label).grid(row=r, column=0, sticky="w")
            ent = tk.Entry(parent, width=9)
            ent.insert(0, default)
            ent.grid(row=r, column=1, sticky="ew", padx=2)
            setattr(self, attr, ent)
            r += 1

        ttk.Label(parent, text="── Status ──", foreground="#555").grid(
            row=r, column=0, columnspan=2, sticky="w"); r += 1
        self._status_text = scrolledtext.ScrolledText(
            parent, height=10, width=26, state="disabled",
            font=("Courier", 7), wrap="word")
        self._status_text.grid(row=r, column=0, columnspan=2,
                               sticky="nsew", padx=2, pady=2); r += 1
        parent.rowconfigure(r - 1, weight=1)

    # ------------------------------------------------------------------
    # H1-2 control callbacks
    # ------------------------------------------------------------------
    def _on_bimanual_toggle(self):
        self._bimanual_mode = self._bimanual_var.get()
        arm_label = "right" if self._active_arm() == 0 else "left"
        self._update_status(
            f"Bimanual mode {'ON' if self._bimanual_mode else 'OFF'}."
            + (f" Active arm: {arm_label}" if self._bimanual_mode else ""))

    def _on_send_h12(self):
        """Send arm to current slider position on real H1-2 via ROS2."""
        self._update_status("Sending H1-2 arm…")
        threading.Thread(target=self._send_h12_arm, daemon=True,
                         name="send-h12").start()

    def _on_robot_only_toggle(self):
        self._robot_only_mode = bool(self._robot_only_var.get())
        if hasattr(self, "_h12_robot_only_mode"):
            self._h12_robot_only_mode.value = 1 if self._robot_only_mode else 0
        self._set_robot_only_ui_state(self._robot_only_mode)
        self._push_viewer_ctrl()
        self._update_status(
            f"H1-2 robot-only pose {'ON' if self._robot_only_mode else 'OFF'}."
        )

    def _set_robot_only_ui_state(self, enabled: bool):
        state = "disabled" if enabled else "normal"
        try:
            self._sl_w.config(state=state)
            self._ent_w.config(state=state)
            self._ent_width_target.config(state=state)
            self._sl_rx.config(state=state)
            self._sl_ry.config(state=state)
            self._sl_rz.config(state=state)
            self._ent_rx.config(state=state)
            self._ent_ry.config(state=state)
            self._ent_rz.config(state=state)
        except Exception:
            pass
        if hasattr(self, "_btn_grasp"):
            self._btn_grasp.config(state="disabled" if enabled else "normal")

    def _on_active_arm_mode(self, _event=None):
        mode = self._active_arm_mode_var.get()
        self.set_active_arm_override(mode)
        arm = self._active_arm()
        arm_label = "right" if arm == 0 else "left"
        self._update_active_arm()
        self._refresh_h12_wrist_pose_panel(log_on_fail=False)
        self._update_status(f"H1-2 active arm mode: {mode} (current={arm_label}).")

    def _refresh_h12_gravity_button(self):
        if not hasattr(self, "_btn_gravity_h12"):
            return
        active = self.is_h12_gravity_comp_active()
        if active:
            self._btn_gravity_h12.config(text="GC ACTIVE", bg="#ff4444", fg=_BUTTON_TEXT_ON_COLOR)
        else:
            self._btn_gravity_h12.config(text="Gravity Comp", bg="#f0f0f0", fg="black")

    def _on_toggle_h12_gravity_comp(self):
        if getattr(self, "_h12_arm", None) is None:
            self._update_status("No H1-2 connection.")
            return
        self._update_status("Toggling H1-2 gravity compensation…")

        def _do_toggle():
            self.toggle_h12_gravity_compensation(sport_mode=True)
            self._status_queue.put("__refresh_h12_gravity_btn__")

        threading.Thread(target=_do_toggle, daemon=True,
                         name="toggle-h12-gravity").start()

    def _on_grasp_h12(self):
        """Execute grasp on real H1-2: send arm then close fingers."""
        if self._robot_only_mode:
            self._update_status("Robot-only mode: GRASP is disabled.")
            return
        if self._result is None:
            return
        self._update_status("H1-2 GRASP! sequence started…")
        threading.Thread(target=self._execute_h12_grasp, daemon=True,
                         name="grasp-h12").start()

    def _on_refresh_h12_wrist_pose(self):
        if getattr(self, "_h12_arm", None) is None:
            self._update_status("No H1-2 connection.")
            return
        if self._refresh_h12_wrist_pose_panel(log_on_fail=True):
            self._update_status("H1-2 wrist pose panel refreshed from /right_ee_pose or /left_ee_pose.")

    def _refresh_h12_wrist_pose_panel(self, log_on_fail: bool = False) -> bool:
        dbg = self.get_h12_wrist_pose_debug(timeout=0.5)
        if not dbg:
            self._h12_pose_frame_var.set("frame: n/a")
            self._h12_pose_pelvis_var.set("pelvis xyz(mm): n/a")
            self._h12_pose_planner_var.set("planner xyz(mm): n/a")
            self._h12_pose_quat_var.set("pelvis quat(xyzw): n/a")
            self._h12_pose_rpy_var.set("pelvis rpy(deg): n/a")
            if hasattr(self, "_sl_wrist_rx"):
                self._sl_wrist_rx.set(0.0)
            if hasattr(self, "_sl_wrist_ry"):
                self._sl_wrist_ry.set(0.0)
            if hasattr(self, "_sl_wrist_rz"):
                self._sl_wrist_rz.set(0.0)
            if hasattr(self, "_ent_wrist_rx"):
                self._ent_wrist_rx.config(state="normal")
                self._ent_wrist_rx.delete(0, tk.END)
                self._ent_wrist_rx.insert(0, "0.0")
                self._ent_wrist_rx.config(state="disabled")
            if hasattr(self, "_ent_wrist_ry"):
                self._ent_wrist_ry.config(state="normal")
                self._ent_wrist_ry.delete(0, tk.END)
                self._ent_wrist_ry.insert(0, "0.0")
                self._ent_wrist_ry.config(state="disabled")
            if hasattr(self, "_ent_wrist_rz"):
                self._ent_wrist_rz.config(state="normal")
                self._ent_wrist_rz.delete(0, tk.END)
                self._ent_wrist_rz.insert(0, "0.0")
                self._ent_wrist_rz.config(state="disabled")
            if log_on_fail:
                self._update_status("Failed to refresh H1-2 wrist pose panel.")
            return False

        p_p = dbg["pelvis_xyz_m"] * 1000.0
        p_w = dbg["planner_xyz_m"] * 1000.0
        q_p = dbg["pelvis_quat_xyzw"]
        rpy_p = np.degrees(dbg["pelvis_rpy_rad"])
        self._h12_pose_frame_var.set(f"frame: {dbg['frame']}")
        self._h12_pose_pelvis_var.set(
            f"pelvis xyz(mm): [{p_p[0]:.1f}, {p_p[1]:.1f}, {p_p[2]:.1f}]"
        )
        self._h12_pose_planner_var.set(
            f"planner xyz(mm): [{p_w[0]:.1f}, {p_w[1]:.1f}, {p_w[2]:.1f}]"
        )
        self._h12_pose_quat_var.set(
            f"pelvis quat(xyzw): [{q_p[0]:.4f}, {q_p[1]:.4f}, {q_p[2]:.4f}, {q_p[3]:.4f}]"
        )
        self._h12_pose_rpy_var.set(
            f"pelvis rpy(deg): [{rpy_p[0]:.1f}, {rpy_p[1]:.1f}, {rpy_p[2]:.1f}]"
        )
        if hasattr(self, "_sl_wrist_rx"):
            self._sl_wrist_rx.set(float(rpy_p[0]))
        if hasattr(self, "_sl_wrist_ry"):
            self._sl_wrist_ry.set(float(rpy_p[1]))
        if hasattr(self, "_sl_wrist_rz"):
            self._sl_wrist_rz.set(float(rpy_p[2]))
        if hasattr(self, "_ent_wrist_rx"):
            self._ent_wrist_rx.config(state="normal")
            self._ent_wrist_rx.delete(0, tk.END)
            self._ent_wrist_rx.insert(0, f"{float(rpy_p[0]):.1f}")
            self._ent_wrist_rx.config(state="disabled")
        if hasattr(self, "_ent_wrist_ry"):
            self._ent_wrist_ry.config(state="normal")
            self._ent_wrist_ry.delete(0, tk.END)
            self._ent_wrist_ry.insert(0, f"{float(rpy_p[1]):.1f}")
            self._ent_wrist_ry.config(state="disabled")
        if hasattr(self, "_ent_wrist_rz"):
            self._ent_wrist_rz.config(state="normal")
            self._ent_wrist_rz.delete(0, tk.END)
            self._ent_wrist_rz.insert(0, f"{float(rpy_p[2]):.1f}")
            self._ent_wrist_rz.config(state="disabled")
        return True

    def _h12_read_params(self):
        """Read force_N, step_mm, approach_m from UI widgets (safe from bg thread)."""
        try:
            force_N = float(getattr(self, "_ent_force", None) and
                            self._ent_force.get().strip() or "0")
        except (ValueError, AttributeError):
            force_N = 0.0
        try:
            step_mm = float(getattr(self, "_ent_step", None) and
                            self._ent_step.get().strip() or "10")
        except (ValueError, AttributeError):
            step_mm = 10.0
        try:
            s = getattr(self, "_ent_approach", None)
            approach_m = float(s.get().strip()) / 1000.0 if s and s.get().strip() else None
        except (ValueError, AttributeError):
            approach_m = None
        return force_N, step_mm, approach_m

    def _h12_close_fingers(self, cmd, force_N: float, active_fingers=None):
        """
        Send finger close command with optional force-adaptive stop.

        If force_N > 0 and a hand is connected: sets the per-finger force limit,
        sends the position command, then polls force_act() until any active
        calibrated finger reaches the threshold (mirrors GraspExecutor behaviour).
        """
        if self._hand is None:
            return
        from .grasp_executor import _force_N_to_raw, _FORCE_CALIB
        if force_N > 0.0:
            self._hand.force_set([_force_N_to_raw(i, force_N) for i in range(6)])
        self._hand.angle_set(cmd)
        if force_N <= 0.0:
            return
        # Poll force_act until threshold reached on any active calibrated finger
        self._update_status(f"H1-2: monitoring force (threshold {force_N:.1f} N)…")
        while True:
            raw = self._hand.force_act()
            if raw is None:
                time.sleep(0.05)
                continue
            for idx, (a, b) in _FORCE_CALIB.items():
                if active_fingers is not None and idx not in active_fingers:
                    continue
                f_N = max(0.0, a * raw[idx] + b)
                if f_N >= force_N:
                    self._update_status(
                        f"H1-2: force threshold reached (finger[{idx}]={f_N:.2f} N)")
                    return
            time.sleep(0.05)

    def _execute_h12_grasp(self):
        """Background thread: dispatch to the selected strategy."""
        strategy = self._grasp_strategy
        if strategy == "Plan":
            self._execute_h12_plan()
        elif strategy == "Thumb Reflex":
            self._execute_h12_thumb_reflex()
        else:
            self._execute_h12_naive()

    def _execute_h12_naive(self):
        """H1-2 Naive: arm → pose, then close fingers (with optional force stop)."""
        force_N, _, _ = self._h12_read_params()
        active_fingers = _MODE_ACTIVE_FINGERS.get(self._mode, [2, 3, 4])
        self._send_h12_arm()
        with self._state_lock:
            r = self._result
        if r is not None:
            self._h12_close_fingers(self._h12_finger_cmd(r), force_N, active_fingers)
        self._update_status("H1-2 Naive complete.")

    def _execute_h12_plan(self):
        """H1-2 Plan: fingers → approach, arm → approach, then step arm+fingers."""
        force_N, step_mm, approach_m = self._h12_read_params()
        active_fingers = _MODE_ACTIVE_FINGERS.get(self._mode, [2, 3, 4])

        r_target = self.closure.solve(self._mode, self._width_target_m)
        closures = self._compute_plan_closures(step_mm, r_target, approach_m)
        if not closures:
            self._execute_h12_naive()
            return

        r_approach = closures[0]
        final_fc = dict(r_target.ctrl_values)
        final_thumb_yaw = final_fc.get("thumb_yaw", self.fk.ctrl_min["thumb_yaw"])
        approach_fc = dict(r_approach.ctrl_values)
        approach_fc["thumb_yaw"] = final_thumb_yaw
        # Phase 1: fingers to approach config (no force limit on approach)
        if self._hand is not None:
            self._hand.angle_set(self._h12_finger_cmd(r_approach, approach_fc))
        # Phase 2: arm to approach pose
        self._update_status(f"H1-2 Plan: approach {r_approach.width*1000:.1f} mm…")
        self._h12_send_arm_for_result(r_approach)
        # Phase 3: step through waypoints; force control only on final step
        for i, r_i in enumerate(closures[1:], 1):
            is_last = (i == len(closures) - 1)
            self._update_status(
                f"H1-2 Plan: step {i}/{len(closures)-1} → {r_i.width*1000:.1f} mm")
            self._h12_send_arm_for_result(r_i)
            fn = force_N if is_last else 0.0
            step_fc = dict(r_i.ctrl_values)
            step_fc["thumb_yaw"] = final_thumb_yaw
            self._h12_close_fingers(self._h12_finger_cmd(r_i, step_fc), fn, active_fingers)
            if not is_last:
                time.sleep(0.2)
        self._update_status("H1-2 Plan complete.")

    def _execute_h12_thumb_reflex(self):
        """H1-2 Thumb Reflex: thumb to final, arm, then all fingers (with force stop)."""
        force_N, _, _ = self._h12_read_params()
        active_fingers = _MODE_ACTIVE_FINGERS.get(self._mode, [2, 3, 4])
        with self._state_lock:
            r = self._result
        if r is None:
            return
        final_cmd = self._h12_finger_cmd(r)
        # Phase 1: thumb to final config, all other fingers open
        if self._hand is not None:
            thumb_cmd = [1000, 1000, 1000, 1000, final_cmd[4], final_cmd[5]]
            self._hand.angle_set(thumb_cmd)
            time.sleep(0.4)
        # Phase 2: arm to final pose
        self._send_h12_arm()
        # Phase 3: all fingers close (with optional force stop)
        self._h12_close_fingers(final_cmd, force_N, active_fingers)
        self._update_status("H1-2 Thumb Reflex complete.")

    def _on_set_pose_from_h12(self):
        """Read current H1-2 wrist pose via TF and set sliders to match."""
        if getattr(self, "_h12_arm", None) is None:
            self._update_status("No H1-2 connection.")
            return
        with self._state_lock:
            r = self._result
        params = self.decode_h12_pose_to_grasp_params(current_result=r, include_offsets=False)
        if not params:
            self._update_status("Failed to read H1-2 arm pose — is dual_arm server running?")
            return
        q_ok = self.seed_h12_joints_from_bridge(timeout=1.0)
        self._grasp_x  = params["grasp_x"]
        self._grasp_y  = params["grasp_y"]
        self._grasp_z  = params["grasp_z"]
        self._plane_rx = params["plane_rx"]
        self._plane_ry = params["plane_ry"]
        self._plane_rz = params["plane_rz"]
        # Sync sliders
        if hasattr(self, "_sl_x"):
            self._sl_x.set(self._grasp_x * 1000)
        if hasattr(self, "_sl_y"):
            self._sl_y.set(self._grasp_y * 1000)
        if hasattr(self, "_sl_z"):
            self._sl_z.set(np.clip(self._grasp_z * 1000, -400.0, 400.0))
        if hasattr(self, "_sl_rx"):
            self._sl_rx.set(np.degrees(self._plane_rx))
        if hasattr(self, "_sl_ry"):
            self._sl_ry.set(np.degrees(self._plane_ry))
        if hasattr(self, "_sl_rz"):
            self._sl_rz.set(np.degrees(self._plane_rz))
        self._push_viewer_ctrl()
        self._schedule_plot_only()
        self._refresh_h12_wrist_pose_panel(log_on_fail=False)
        if q_ok:
            self._update_status(
                f"Pose set: wrist({params['grasp_x']*1000:.0f},"
                f"{params['grasp_y']*1000:.0f},{params['grasp_z']*1000:.0f})mm + joint seed")
        else:
            self._update_status(
                f"Pose set: wrist({params['grasp_x']*1000:.0f},"
                f"{params['grasp_y']*1000:.0f},{params['grasp_z']*1000:.0f})mm (no joint_states)")

    # ------------------------------------------------------------------
    # Status queue poll (called every _POLL_MS ms via root.after)
    # ------------------------------------------------------------------
    def _poll_status_queue(self):
        # Wrap in a check to ensure we don't schedule if closing
        if not getattr(self, "_running", True):
            return
        try:
            while True:
                msg = self._status_queue.get_nowait()
                if msg == "__reset_grasp_btn__":
                    if hasattr(self, "_btn_grasp"):
                        self._btn_grasp.config(text="GRASP!", bg="#2ecc71")
                elif msg == "__reset_teach_btn__":
                    self._teach_mode = False
                    self._manual_teach_override = False
                    if hasattr(self, "_btn_teach"):
                        self._btn_teach.config(text="Teach Mode", bg="#f0f0f0")
                elif msg == "__refresh_h12_gravity_btn__":
                    self._refresh_h12_gravity_button()
                else:
                    self._append_status(msg)
        except queue.Empty:
            pass
        finally:
            # Only schedule the next poll if we haven't been told to stop
            if getattr(self, "_running", True) and self._root.winfo_exists():
                self._poll_id = self._root.after(_POLL_MS, self._poll_status_queue)

    def _append_status(self, msg: str):
        if not hasattr(self, "_status_text"):
            print(f"[status] {msg}")
            return
        self._status_text.config(state="normal")
        self._status_text.insert("end", msg + "\n")
        self._status_text.see("end")
        self._status_text.config(state="disabled")

    # ------------------------------------------------------------------
    # Debounced recompute + plot
    # ------------------------------------------------------------------
    def _schedule_recompute(self):
        if self._debounce_id is not None:
            self._root.after_cancel(self._debounce_id)
        self._debounce_id = self._root.after(_RECOMP_MS, self._do_recompute)

    def _do_recompute(self):
        self._debounce_id = None
        self._recompute()
        self._update_plot()

    def _schedule_plot_only(self):
        """Debounced plot refresh without recomputing closure or sending to hand.
        Use for Z/X/Y/rotation changes — arm-pose only, finger config unchanged."""
        if self._debounce_id is not None:
            self._root.after_cancel(self._debounce_id)
        self._debounce_id = self._root.after(_RECOMP_MS, self._do_plot_only)

    def _do_plot_only(self):
        self._debounce_id = None
        self._push_viewer_ctrl()
        if self._mink_enabled:
            self._push_mink_viewer_ctrl()
        self._update_plot()

    # ------------------------------------------------------------------
    # Slider / control callbacks
    # ------------------------------------------------------------------
    def _on_mode_radio(self):
        if self._h12_mode and self._robot_only_mode:
            self._mode_var.set(self._mode)
            self._update_status("Robot-only mode: grasp mode changes are disabled.")
            return
        label = self._mode_var.get()
        self._mode = label
        n = int(label[0]) if label[0].isdigit() else 4
        wrange = self.closure.width_range(label, n_fingers=n)
        self._width_range = wrange
        wmin_mm, wmax_mm = wrange[0] * 1000, wrange[1] * 1000
        self._width_m = float(np.clip(self._width_m, wrange[0], wrange[1]))
        # Update width slider range + value
        self._sl_w.config(from_=wmin_mm, to=wmax_mm)
        self._sl_w.set(self._width_m * 1000)
        self._ent_w.delete(0, "end")
        self._ent_w.insert(0, f"{self._width_m * 1000:.1f}")
        # Reset width target on mode change
        self._width_target_m      = self._width_m
        self._width_target_edited = False
        self._ent_width_target.delete(0, "end")
        self._ent_width_target.insert(0, f"{self._width_m * 1000:.1f}")
        self._schedule_recompute()
        self._update_cylinder_guard()

    def _on_width(self, val_str):
        if self._h12_mode and self._robot_only_mode:
            self._sl_w.set(self._width_m * 1000.0)
            self._update_status("Robot-only mode: grasp width is disabled.")
            return
        val = float(val_str)
        self._width_m = val / 1000.0
        self._ent_w.delete(0, "end")
        self._ent_w.insert(0, f"{val:.1f}")
        # Sync target entry if user hasn't manually set it
        if not self._width_target_edited:
            self._width_target_m = self._width_m
            self._ent_width_target.delete(0, "end")
            self._ent_width_target.insert(0, f"{val:.1f}")
        self._schedule_recompute()
        self._update_cylinder_guard()

    def _on_width_target_submit(self, _event=None):
        if self._h12_mode and self._robot_only_mode:
            self._ent_width_target.delete(0, "end")
            self._ent_width_target.insert(0, f"{self._width_m * 1000:.1f}")
            self._update_status("Robot-only mode: target width is disabled.")
            return
        raw = self._ent_width_target.get().strip()
        try:
            val_mm = float(raw)
            self._width_target_m      = val_mm / 1000.0
            self._width_target_edited = True
        except ValueError:
            # Revert to current width
            self._ent_width_target.delete(0, "end")
            self._ent_width_target.insert(0, f"{self._width_m * 1000:.1f}")

    def _update_cylinder_guard(self):
        if not self._real_robot_mode or not hasattr(self, "_btn_grasp"):
            return
        bad   = self._is_cylinder_bad()
        color = "#cccccc" if bad else "#2ecc71"
        if self._btn_grasp.cget("text") not in ("RUNNING…",):
            self._btn_grasp.config(bg=color)
        arm_state = "disabled" if bad else "normal"
        if hasattr(self, "_btn_sendarm"):
            self._btn_sendarm.config(state=arm_state)
        if bad:
            self._update_status(
                f"WARN: cylinder {self._width_m * 2000:.0f}mm < 71mm — "
                "power grasp disabled")

    def _on_z(self, val_str):
        self._grasp_z = float(val_str) / 1000.0
        ent = getattr(self, "_ent_z", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_x(self, val_str):
        self._grasp_x = float(val_str) / 1000.0
        ent = getattr(self, "_ent_x", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_y(self, val_str):
        self._grasp_y = float(val_str) / 1000.0
        ent = getattr(self, "_ent_y", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_plane_rx(self, val_str):
        self._plane_rx = float(val_str) * np.pi / 180.0
        ent = getattr(self, "_ent_rx", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_plane_ry(self, val_str):
        self._plane_ry = float(val_str) * np.pi / 180.0
        ent = getattr(self, "_ent_ry", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_plane_rz(self, val_str):
        self._plane_rz = float(val_str) * np.pi / 180.0
        ent = getattr(self, "_ent_rz", None)
        if ent:
            ent.delete(0, "end"); ent.insert(0, f"{float(val_str):.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_move_up(self):
        z_max_mm = 400.0 if (self._robot_mode or self._h12_mode) else 200.0
        new_z_mm = min(z_max_mm, self._grasp_z * 1000.0 + 100.0)
        self._grasp_z = new_z_mm / 1000.0
        self._sl_z.set(new_z_mm)
        self._ent_z.delete(0, "end")
        self._ent_z.insert(0, f"{new_z_mm:.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _set_arm_buttons(self, connected: bool):
        """Enable/disable all arm-dependent buttons based on connection state."""
        arm_state    = "normal"   if connected else "disabled"
        reconn_state = "disabled" if connected else "normal"
        for btn in ("_btn_teach", "_btn_setpose", "_btn_sendarm", "_btn_simtraj"):
            if hasattr(self, btn):
                getattr(self, btn).config(state=arm_state)
        if hasattr(self, "_btn_reconnect"):
            self._btn_reconnect.config(state=reconn_state)

    def _reconnect_arm(self):
        """Reconnect the UR5 arm (after it was disconnected for force_control_ui)."""
        if self._arm is None:
            return
        ok = self._arm.connect()
        self._set_arm_buttons(ok)

    def _on_fc_ui_close(self):
        """Called when force_control_ui window is closed — reconnect the arm."""
        self._fc_ui_win = None
        self._reconnect_arm()

    def _open_force_control_ui(self):
        """Open (or raise) the Force Control debug UI popup."""
        if hasattr(self, "_fc_ui_win") and self._fc_ui_win is not None:
            try:
                self._fc_ui_win._root.lift()
                return
            except Exception:
                self._fc_ui_win = None
        # Disconnect the arm so the C++ binary can claim the sole RTDE slot.
        if self._arm is not None and self._arm.connected:
            self._arm.disconnect()
            self._set_arm_buttons(False)
        # Inherit robot IP from arm config if available
        robot_ip = "192.168.0.4"
        if self._arm is not None and hasattr(self._arm, "_ip"):
            robot_ip = self._arm._ip
        self._fc_ui_win = ForceControlUI(
            parent=self._root,
            robot_ip=robot_ip,
            on_close=self._on_fc_ui_close,
        )
        self._fc_ui_win.run()

    def _open_force_viz_panel(self):
        """Open (or raise) the Force Viz Toplevel window."""
        if hasattr(self, "_force_panel") and self._force_panel is not None:
            try:
                self._force_panel._win.lift()
                return
            except Exception:
                self._force_panel = None
        self._force_panel = ForceVizPanel(
            root=self._root,
            ctrl_arr=self._custom_ctrl_arr,
            state_arr=self._viewer_state_arr,
            mp_ctx=self._mp_ctx,
            hand=self._hand,
            fk=self.fk,
        )

    def _on_send_real(self):
        self._send_real = self._send_real_var.get()
        self._grasp_hand_locked = False  # explicit toggle always clears the post-grasp lock
        if self._send_real:
            self._send_real_hand()

    def _on_strategy_radio(self):
        if self._h12_mode and self._robot_only_mode:
            self._strategy_var.set(self._grasp_strategy)
            self._update_status("Robot-only mode: grasp strategy is disabled.")
            return
        self._grasp_strategy = self._strategy_var.get()

    def _on_teach_mode(self):
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._arm.disable_teach_mode()
            self._teach_mode = False
            self._manual_teach_override = False
            self._btn_teach.config(text="Teach Mode", bg="#f0f0f0")
        else:
            self._arm.enable_teach_mode()
            self._teach_mode = True
            self._manual_teach_override = True
            self._btn_teach.config(text="TEACH MODE ACTIVE", bg="#ff4444", fg=_BUTTON_TEXT_ON_COLOR)

    def _on_set_pose_from_robot(self):
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        with self._state_lock:
            r = self._result
        params = self._arm.decode_tcp_to_grasp_params(current_result=r)
        if not params:
            self._update_status("Failed to read arm pose.")
            return
        self._grasp_x  = params["grasp_x"]
        self._grasp_y  = params["grasp_y"]
        self._grasp_z  = params["grasp_z"]
        self._plane_rx = params["plane_rx"]
        self._plane_ry = params["plane_ry"]
        self._plane_rz = params["plane_rz"]
        # Sync sliders
        if hasattr(self, "_sl_x"):
            self._sl_x.set(self._grasp_x * 1000)
        if hasattr(self, "_sl_y"):
            self._sl_y.set(self._grasp_y * 1000)
        if hasattr(self, "_sl_z"):
            self._sl_z.set(np.clip(self._grasp_z * 1000, 0.0, 400.0))
        if hasattr(self, "_sl_rx"):
            self._sl_rx.set(np.degrees(self._plane_rx))
        if hasattr(self, "_sl_ry"):
            self._sl_ry.set(np.degrees(self._plane_ry))
        if hasattr(self, "_sl_rz"):
            self._sl_rz.set(np.degrees(self._plane_rz))
        q = self._arm.snapshot_joints(self._real_q_arr)
        if q is not None:
            self._real_tracking.value = 1
        self._push_viewer_ctrl()
        # Overwrite finger ctrl with real hand angles so the viewer shows
        # the actual hand pose, not the theoretical closure geometry.
        self._sync_real_hand_to_ctrl(self._custom_ctrl_arr)
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers
        self._update_status(
            f"Pose set: hand({params['grasp_x']*1000:.0f},"
            f"{params['grasp_y']*1000:.0f},{params['grasp_z']*1000:.0f})mm")

    def _on_send_arm(self):
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._update_status("BLOCKED: teach mode is active.")
            return
        if self._is_cylinder_bad():
            self._update_status("BLOCKED: cylinder < 71 mm — power grasp disabled.")
            return
        with self._state_lock:
            r = self._result
        if r is None:
            self._update_status("No grasp result — adjust sliders first.")
            return
        world_T_hand = self._build_world_T_hand(r)
        warns = self._arm.check_pose_workspace(world_T_hand)
        for w in warns:
            self._update_status(w)
        self._approach_width_m = self._width_m
        self._update_status("Moving arm to planned pose...")
        self._real_tracking.value = 0

        def _do_move():
            warns2 = self._arm.move_to_hand_pose(world_T_hand, blocking=True)
            for w in warns2:
                self._status_queue.put(w)
            self._arm.snapshot_joints(self._real_q_arr)
            self._real_tracking.value = 1
            self._status_queue.put("Arm move complete.")

        threading.Thread(target=_do_move, daemon=True, name="arm-move").start()

    def _on_simulate_trajectory(self):
        # If real joint data is available, seed the viewer arm at the actual robot
        # configuration so mink IK starts from the correct solution (not home pose).
        q_real = np.array(self._real_q_arr[:])
        has_real = not np.all(q_real == 0)
        if has_real:
            self._real_tracking.value = 1   # arm follows real joints initially
        self._launch_robot_viewer_ours()
        self._start_sim_animation(has_real=has_real)

    def _on_sim_h12(self):
        """Open (or refresh) the H1-2 viewer at current pose and animate the grasp."""
        if hasattr(self, "_sim_arm_t"):
            self._sim_arm_t.value = 0.0
        self._sim_grasp_t.value = 0.0
        if self._bimanual_mode:
            self._update_active_arm()
        if self._bimanual_mode:
            self._launch_h12_bimanual_viewer()
        else:
            self._launch_h12_viewer()
        self._start_sim_animation(has_real=False)

    def _start_sim_animation(self, has_real: bool = False):
        """Animate the current grasp strategy in the active sim viewer."""
        self._sim_grasp_gen += 1
        gen      = self._sim_grasp_gen
        strategy = self._grasp_strategy

        def _set_sim_ctrl(ctrl: np.ndarray):
            self._custom_ctrl_arr[:] = ctrl
            if self._bimanual_mode and self._active_arm() == 1:
                self._update_active_arm()

        def _set_arm_progress(value: float):
            if hasattr(self, "_sim_arm_t"):
                self._sim_arm_t.value = float(np.clip(value, 0.0, 1.0))

        def _animate_arm_to_target(duration_s: float = 2.0) -> bool:
            n_steps = max(1, int(duration_s / 0.033))
            dt = duration_s / n_steps
            for i in range(n_steps + 1):
                if self._sim_grasp_gen != gen:
                    return False
                _set_arm_progress(i / n_steps)
                time.sleep(dt)
            return True

        def _animate_ctrl(start_ctrl: np.ndarray, end_ctrl: np.ndarray,
                          duration_s: float = 1.0) -> bool:
            n_steps = max(1, int(duration_s / 0.033))
            dt = duration_s / n_steps
            start = np.array(start_ctrl, dtype=float)
            end = np.array(end_ctrl, dtype=float)
            for i in range(n_steps + 1):
                if self._sim_grasp_gen != gen:
                    return False
                alpha = i / n_steps
                _set_sim_ctrl(start + alpha * (end - start))
                time.sleep(dt)
            return True

        with self._state_lock:
            r = self._result
        if r is None:
            return

        try:
            step_mm = float(self._ent_step.get().strip() or "10") if hasattr(self, "_ent_step") else 10.0
        except (ValueError, AttributeError):
            step_mm = 10.0
        try:
            approach_m = (float(self._ent_approach.get().strip()) / 1000.0
                          if hasattr(self, "_ent_approach") and self._ent_approach.get().strip()
                          else None)
        except (ValueError, AttributeError):
            approach_m = None

        def _sync_then_ik():
            """Let viewer sync arm from real joints (~1 frame), then hand off to mink IK."""
            if has_real:
                time.sleep(0.3)
            if hasattr(self, "_real_tracking"):
                self._real_tracking.value = 0

        def _animate_naive():
            _sync_then_ik()
            _set_arm_progress(0.0)
            self._sim_grasp_t.value = 0.0
            self._push_viewer_ctrl()
            self._update_status("Sim Naive: moving arm home → grasp pose...")
            if not _animate_arm_to_target(duration_s=2.0):
                return
            self._update_status("Sim Naive: fingers closing...")
            n_steps = 40
            for i in range(n_steps + 1):
                if self._sim_grasp_gen != gen:
                    return
                self._sim_grasp_t.value = i / n_steps
                time.sleep(0.05)
            _set_arm_progress(1.0)
            self._status_queue.put("Sim Naive: complete.")

        def _animate_plan():
            try:
                r_target = self.closure.solve(self._mode, self._width_target_m)
            except Exception:
                r_target = r
            closures = self._compute_plan_closures(step_mm, r_target, approach_m)
            if not closures:
                return

            _sync_then_ik()
            _set_arm_progress(0.0)
            self._sim_grasp_t.value = 1.0

            final_cv = r_target.ctrl_values
            final_thumb_yaw = final_cv.get("thumb_yaw", 0.0)

            def _plan_ctrl(r_i):
                fc = dict(r_i.ctrl_values)
                fc["thumb_yaw"] = final_thumb_yaw
                return self._build_ctrl_array(r_i, fc)

            def _animate_plan_segment(r_from, r_to, duration_s: float = 0.5) -> bool:
                n_steps = max(1, int(duration_s / 0.033))
                dt = duration_s / n_steps
                for j in range(n_steps + 1):
                    if self._sim_grasp_gen != gen:
                        return False
                    alpha = j / n_steps
                    width = r_from.width + alpha * (r_to.width - r_from.width)
                    try:
                        r_mid = self.closure.solve(self._mode, width)
                    except Exception:
                        r_mid = r_to if alpha > 0.5 else r_from
                    _set_sim_ctrl(_plan_ctrl(r_mid))
                    time.sleep(dt)
                return True

            r_approach = closures[0]
            ctrl = _plan_ctrl(r_approach)
            _set_sim_ctrl(ctrl)
            self._update_status(
                f"Sim Plan: approach {r_approach.width*1000:.1f}mm → "
                f"{r_target.width*1000:.1f}mm ({len(closures)-1} steps)")
            if not _animate_arm_to_target(duration_s=2.5):
                return

            prev_r = r_approach
            for i, r_i in enumerate(closures[1:]):
                if self._sim_grasp_gen != gen:
                    return
                self._update_status(
                    f"Sim Plan: step {i+1}/{len(closures)-1} "
                    f"({r_i.width*1000:.1f}mm)")
                if not _animate_plan_segment(prev_r, r_i, duration_s=0.5):
                    return
                prev_r = r_i

            _set_arm_progress(1.0)
            self._status_queue.put("Sim Plan: complete.")

        def _animate_thumb_reflex():
            try:
                r_target = self.closure.solve(self._mode, self._width_target_m)
            except Exception:
                r_target = r

            _sync_then_ik()
            _set_arm_progress(0.0)
            self._sim_grasp_t.value = 1.0

            final_cv = r_target.ctrl_values
            all_open_fc = {k: self.fk.ctrl_min[k] for k in
                           ["pinky", "ring", "middle", "index",
                            "thumb_proximal", "thumb_yaw"]}
            thumb_fc = {
                "pinky":          self.fk.ctrl_min["pinky"],
                "ring":           self.fk.ctrl_min["ring"],
                "middle":         self.fk.ctrl_min["middle"],
                "index":          self.fk.ctrl_min["index"],
                "thumb_proximal": final_cv.get("thumb_proximal", 0.0),
                "thumb_yaw":      final_cv.get("thumb_yaw",      0.0),
            }
            start_ctrl = self._build_ctrl_array(r_target, all_open_fc)
            ctrl = self._build_ctrl_array(r_target, thumb_fc)
            _set_sim_ctrl(start_ctrl)
            self._update_status("Sim Thumb Reflex: thumb closing at home pose...")
            if not _animate_ctrl(start_ctrl, ctrl, duration_s=1.0):
                return
            self._update_status("Sim Thumb Reflex: moving arm home → grasp pose...")
            if not _animate_arm_to_target(duration_s=2.0):
                return

            self._update_status("Sim Thumb Reflex: all fingers closing...")
            final_ctrl = self._build_ctrl_array(r_target)
            if not _animate_ctrl(ctrl, final_ctrl, duration_s=1.0):
                return
            _set_arm_progress(1.0)
            self._status_queue.put("Sim Thumb Reflex: complete.")

        self._update_status(f"Sim [{strategy}]: starting...")
        if strategy == "Plan":
            threading.Thread(target=_animate_plan, daemon=True, name="sim-grasp").start()
        elif strategy == "Thumb Reflex":
            threading.Thread(target=_animate_thumb_reflex, daemon=True, name="sim-grasp").start()
        else:
            threading.Thread(target=_animate_naive, daemon=True, name="sim-grasp").start()

    # ------------------------------------------------------------------
    # GRASP! execution
    # ------------------------------------------------------------------
    def _on_grasp(self):
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._update_status("BLOCKED: teach mode is active.")
            return
        if self._is_cylinder_bad():
            self._update_status("BLOCKED: cylinder < 71 mm — power grasp disabled.")
            return
        if self._executor is None:
            self._update_status("GraspExecutor not available (hand connected?)")
            return
        if self._executor.is_running():
            self._update_status("Executor busy — abort first.")
            return
        with self._state_lock:
            r_vis = self._result
        if r_vis is None:
            self._update_status("No grasp result.")
            return

        # Parse parameters from UI entries
        try:
            force_N = float(self._ent_force.get().strip() or "0")
            step_mm = float(self._ent_step.get().strip()  or "10")
        except ValueError:
            force_N, step_mm = 0.0, 10.0
        approach_text = self._ent_approach.get().strip()
        try:
            approach_m = float(approach_text) / 1000.0 if approach_text else None
        except ValueError:
            approach_m = None

        # Recompute closure at the TARGET width (may differ from slider)
        try:
            r_target = self.closure.solve(self._mode, self._width_target_m)
        except Exception:
            r_target = r_vis   # fall back to current slider result

        world_T_hand = self._build_world_T_hand(r_target)
        warns = self._arm.check_pose_workspace(world_T_hand)
        for w in warns:
            self._update_status(w)

        # Active fingers for force control
        active_fingers = _MODE_ACTIVE_FINGERS.get(self._mode, [2, 3, 4])

        # Create logger
        log_name = self._ent_log_name.get().strip() or "test"
        strategy = self._grasp_strategy  # "Naive", "Plan", or "Thumb Reflex"
        strategy_tag = strategy.lower().replace(" ", "_")
        full_log_id = f"{log_name}_{strategy_tag}"

        logger = GraspLogger(full_log_id)
        logger.log_meta(
            mode=self._mode,
            width_target_m=self._width_target_m,
            grasp_z=self._grasp_z,
            grasp_x=self._grasp_x,
            grasp_y=self._grasp_y,
            plane_rx=self._plane_rx,
            plane_ry=self._plane_ry,
            plane_rz=self._plane_rz,
            force_N=force_N,
            step_mm=step_mm,
            approach_m=approach_m,
            strategy=self._grasp_strategy,
            name=log_name,
        )
        self._update_status(f"Log: {logger.path}")

        self._send_real = True
        self._grasp_hand_locked = True   # prevent slider recompute from opening the hand
        self._real_tracking.value = 0
        strategy = self._grasp_strategy

        self._btn_grasp.config(text="RUNNING…", bg="#e67e22")

        def _on_done(msg):
            self._status_queue.put(msg)
            if "complete" in msg.lower() or "aborted" in msg.lower():
                self._arm.snapshot_joints(self._real_q_arr)
                self._real_tracking.value = 1
                self._status_queue.put("__reset_grasp_btn__")

        orig_status_cb = self._executor._status

        def _wrapped_status(msg):
            orig_status_cb(msg)
            _on_done(msg)

        self._executor._status = _wrapped_status

        # Restore speed, force to max before each grasp
        self._hand.clear_errors()
        time.sleep(0.125)
        self._hand.speed_set([1000] * 6)
        time.sleep(0.125)
        self._hand.force_set([1000] * 6)
        time.sleep(0.125)

        if strategy == "Naive":
            self._executor.execute_naive(
                world_T_hand, force_N, move_arm=True,
                active_fingers=active_fingers, logger=logger)
        elif strategy == "Plan":
            closures  = self._compute_plan_closures(step_mm, r_target, approach_m)
            waypoints = [(self._build_world_T_hand(r_i), r_i) for r_i in closures]
            self._executor.execute_plan_waypoints(
                waypoints, force_N, move_arm=True,
                active_fingers=active_fingers, logger=logger)
        else:  # Thumb Reflex
            self._executor.execute_thumb_reflex(
                r_target, world_T_hand, force_N, move_arm=True,
                active_fingers=active_fingers, logger=logger)

        self._update_status(f"[{strategy}] grasp started…")

    # ------------------------------------------------------------------
    # 3D plot update
    # ------------------------------------------------------------------
    def _update_plot(self):
        ax = self._ax3d
        ax.cla()
        ax.set_xlabel("X  (closure direction)")
        ax.set_ylabel("Y  (finger spread)")
        ax.set_zlabel("Z  (world up)")

        with self._state_lock:
            r  = self._result
            gz = self._grasp_z

        if r is None:
            ax.set_title("No solution found")
            self._canvas.draw_idle()
            return

        wtips = r.world_tips(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)

        # Fingertip dots
        for fname, pos in wtips.items():
            col = FINGER_COLORS.get(fname, "gray")
            ax.scatter(*pos, color=col, s=60, zorder=5)
            ax.text(pos[0] + 0.003, pos[1], pos[2] + 0.003, fname[:3],
                    fontsize=7, color=col)

        # Mink comparison dots
        if self._mink_enabled:
            with self._mink_lock:
                m_res = self._mink_result
            if m_res is not None:
                R_base = ClosureResult._rot_matrix(r.base_tilt_y)
                R      = self._plane_R_matrix() @ R_base
                mid_w  = R @ r.midpoint
                base_w = np.array([-mid_w[0], -mid_w[1], gz - mid_w[2]])
                for fname, pos_base in m_res.tip_positions.items():
                    wpos = R @ pos_base + base_w
                    ax.scatter(*wpos, color="cyan", s=30, marker="D",
                               zorder=6, alpha=0.85)
                    if fname in wtips:
                        err = float(np.linalg.norm(wpos - wtips[fname]))
                        if err > 0.002:
                            ax.plot(
                                [wpos[0], wtips[fname][0]],
                                [wpos[1], wtips[fname][1]],
                                [wpos[2], wtips[fname][2]],
                                "--", color="gold", lw=0.8, alpha=0.7)

        # Hand base
        ax.scatter(*wbase, color="black", s=80, marker="x", zorder=6)
        ax.text(wbase[0] + 0.003, wbase[1], wbase[2] + 0.003, "base", fontsize=7)

        # Z reference lines
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

        # Info text
        prx_deg = np.degrees(self._plane_rx)
        pry_deg = np.degrees(self._plane_ry)
        prz_deg = np.degrees(self._plane_rz)
        lines = [
            f"Mode:   {r.mode}",
            f"Width:  {r.width * 1000:.1f} mm",
            f"Target: {self._width_target_m * 1000:.1f} mm",
            f"Span:   {r.finger_span * 1000:.1f} mm",
        ]
        if r.cylinder_radius > 0:
            lines.append(f"Radius: {r.cylinder_radius * 1000:.1f} mm")
        lines.append(f"Tilt Y: {r.tilt_deg:.1f}°")
        lines.append(f"Base Z: {wbase[2] * 1000:.1f} mm")
        if any(abs(v) > 0.1 for v in (prx_deg, pry_deg, prz_deg)):
            lines.append(f"Plane: Rx={prx_deg:.0f}° Ry={pry_deg:.0f}° Rz={prz_deg:.0f}°")
        if self._robot_mode:
            lines.append(f"[ROBOT] X={self._grasp_x*1000:.0f} Y={self._grasp_y*1000:.0f}")
        elif self._h12_mode:
            lines.append(f"[H1-2] X={self._grasp_x*1000:.0f} Y={self._grasp_y*1000:.0f}")
        if self._hand is not None:
            lines.append(f"Real: {'ON' if self._send_real else 'off'}")
        lines.append("Ctrl (rad):")
        for k in ["index", "middle", "ring", "pinky", "thumb_proximal", "thumb_yaw"]:
            v = r.ctrl_values.get(k, 0.0)
            if v > 0.001:
                lines.append(f"  {k[:12]:12s}: {v:.3f}")
        if self._mink_enabled:
            with self._mink_lock:
                m_res = self._mink_result
            if m_res is None:
                lines.append("Mink: solving…")
            else:
                status   = "✓" if m_res.converged else "✗"
                mean_err = float(np.mean(list(m_res.position_errors_m.values()))) * 1000
                lines.append(f"Mink: {status} {m_res.n_iters} iters | err {mean_err:.1f}mm")

        ax.text2D(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                  fontsize=7.0, family="monospace", verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

        all_pts = np.array(list(wtips.values()) + [wbase])
        margin  = 0.025
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    # Geometry overlay helpers (matplotlib 3D)
    # ------------------------------------------------------------------
    def _draw_line_closure(self, ax, wtips, gz):
        t = wtips["thumb"]; i = wtips["index"]
        ax.plot([t[0], i[0]], [t[1], i[1]], [t[2], i[2]], "r-", lw=2.5)
        ax.scatter(*(t + i) / 2, color="gold", s=90, marker="*", zorder=7)

    def _draw_plane_closure(self, ax, wtips, gz, r: ClosureResult):
        n      = int(r.mode[0])
        fnames = GRASP_FINGER_SETS[n]
        fpts   = np.array([wtips[f] for f in fnames])
        ax.plot(fpts[:, 0], fpts[:, 1], fpts[:, 2], "b-o", lw=2, ms=5)
        th = wtips["thumb"]
        for f in fnames:
            fp = wtips[f]
            ax.plot([th[0], fp[0]], [th[1], fp[1]], [th[2], fp[2]],
                    "--", color="gray", lw=0.8, alpha=0.6)
        y_min = fpts[:, 1].min(); y_max = fpts[:, 1].max()
        x_nf  = fpts[:, 0].mean(); x_th = th[0]
        corners = np.array([
            [x_nf, y_min, gz], [x_nf, y_max, gz],
            [x_th, y_max, gz], [x_th, y_min, gz],
        ])
        poly = Poly3DCollection([corners], alpha=0.10, facecolor="cyan",
                                edgecolor="steelblue", linewidth=1.2)
        ax.add_collection3d(poly)
        mid_x = (x_nf + x_th) / 2
        ax.plot([mid_x, mid_x], [y_min - 0.01, y_max + 0.01], [gz, gz], "g:", lw=1.5)

    def _draw_cylinder_closure(self, ax, wtips, gz, r: ClosureResult):
        fpts      = np.array([wtips[f] for f in NON_THUMB_FINGERS])
        cx        = fpts[:, 0].mean(); cz = fpts[:, 2].mean()
        radius    = r.cylinder_radius
        y_min     = fpts[:, 1].min() - 0.005
        y_max     = fpts[:, 1].max() + 0.005
        theta     = np.linspace(0, np.pi, 60)
        for y in [y_min, y_max]:
            ax.plot(cx + radius * np.cos(theta),
                    np.full_like(theta, y),
                    gz + (cz - gz) + radius * np.sin(theta),
                    "b-", lw=1.5, alpha=0.7)
        for ang in [0, np.pi]:
            xp = cx + radius * np.cos(ang)
            zp = gz + (cz - gz) + radius * np.sin(ang)
            ax.plot([xp, xp], [y_min, y_max], [zp, zp], "b-", lw=1.0, alpha=0.5)
        ax.plot([cx, cx], [y_min, y_max], [cz, cz], "g:", lw=1.5)
        cy_mean = fpts[:, 1].mean()
        th      = wtips["thumb"]
        ax.plot([th[0], cx], [th[1], cy_mean], [th[2], cz], "--r", lw=1.5, alpha=0.7)
