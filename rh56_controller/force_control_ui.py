"""
force_control_ui.py — Force control debug/command UI.

Frame conventions
-----------------
WRIST (EEF) frame:
  Wrench inputs interpreted as EEF-frame (no conversion → binary).
  Position delta inputs are in EEF frame → Python rotates by R_eef before
  sending RELD in world frame.

WORLD frame:
  Wrench inputs interpreted as world frame → Python rotates by R_eef.T to
  get EEF-frame wrench before sending WRENCH to binary.
  Position delta inputs sent as-is (already world frame).

Rotation reference (always in world frame for RELD; EEF frame for wrist mode
is converted via R_eef rotation):
  Expressed as Euler XYZ degrees in the UI, converted to axis-angle (rad) for
  the C++ RELD/REFA commands. Clipped to ±360°.

C++ stdin protocol
------------------
  WRENCH Fx Fy Fz Tx Ty Tz         (EEF frame always)
  AXIS   a0 a1 a2 a3 a4 a5         (compliance mask 0/1)
  RELD   dx dy dz drx dry drz       (world frame; all optional rot)
  REFA   x  y  z  ax  ay  az        (world frame; all optional rot)
  STIFF  k0..k5 / DAMP d0..d5
  STOP

Stdout telemetry
----------------
  Lines starting with "TELEM:" parsed into float32[37]:
  [t, tcp×6, ref×6, ws×6, wd×6, q×6, T×6]
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.filedialog as tkfiledialog
import tkinter.scrolledtext as scrolledtext
from tkinter import ttk
from typing import Optional, List, Deque
from collections import deque

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE     = pathlib.Path(__file__).parent
_REPO     = _HERE.parent
_LOGS_DIR = _REPO / "logs"
_DEFAULT_BINARY = str(_REPO / "magpie_force_control" / "build" / "force_control_demo")

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety & display constants
# ---------------------------------------------------------------------------
_MAX_F_N   = 20.0   # N  per axis (hard input clamp)
_MAX_T_NM  = 3.0    # N·m per axis
_MAX_ROT_DEG = 360.0  # degrees per axis (rotation reference)

_TELEM_BUF = 300    # rolling samples
_PLOT_MS   = 150    # plot refresh
_STATUS_MS = 100    # status/wrench text refresh

_FORCE_SCALE  = 0.05   # 1 N → 5 cm
_TORQUE_SCALE = 0.12   # 1 N·m → arc radius

_F_LABELS = ["Fx", "Fy", "Fz"]
_T_LABELS = ["Tx", "Ty", "Tz"]
_W_LABELS = _F_LABELS + _T_LABELS
_F_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
_T_COLORS = ["#e67e22", "#1abc9c", "#9b59b6"]
_W_COLORS = _F_COLORS + _T_COLORS

_DEF_STIFFNESS = [100.0, 100.0, 100.0,  1.0,  1.0,  1.0]
_DEF_DAMPING   = [  2.0,   2.0,   2.0,  0.2,  0.2,  0.2]
_DEF_INERTIA   = [  5.0,   5.0,   5.0, 0.005, 0.005, 0.005]
_DEF_STICTION  = [  0.0] * 6

# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def _aa_to_R(axis_angle: np.ndarray) -> np.ndarray:
    """Axis-angle vector → 3×3 rotation matrix (Rodrigues)."""
    aa = np.asarray(axis_angle, dtype=float)
    angle = np.linalg.norm(aa)
    if angle < 1e-9:
        return np.eye(3)
    ax = aa / angle
    c, s = np.cos(angle), np.sin(angle)
    K = np.array([[0, -ax[2], ax[1]],
                  [ax[2], 0, -ax[0]],
                  [-ax[1], ax[0], 0]])
    return c * np.eye(3) + s * K + (1 - c) * np.outer(ax, ax)


def _euler_xyz_to_aa(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """XYZ Euler angles (degrees) → axis-angle vector (radians, world frame)."""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0],
                   [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    # R → axis-angle
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    angle = math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))
    if angle < 1e-9:
        return np.zeros(3)
    axis = np.array([R[2, 1] - R[1, 2],
                     R[0, 2] - R[2, 0],
                     R[1, 0] - R[0, 1]]) / (2.0 * math.sin(angle))
    return axis * angle


def _get_R_eef(tcp: np.ndarray) -> np.ndarray:
    """Extract rotation matrix from TCP [x,y,z,rx,ry,rz] (axis-angle)."""
    return _aa_to_R(tcp[3:6])


def _wrench_world_to_eef(wrench_world: np.ndarray, R_eef: np.ndarray) -> np.ndarray:
    """Rotate a 6D wrench from world frame to EEF frame: w_eef = R.T @ w_world."""
    w = np.asarray(wrench_world, dtype=float)
    out = np.empty(6)
    out[:3] = R_eef.T @ w[:3]
    out[3:] = R_eef.T @ w[3:]
    return out


def _pos_eef_to_world(delta_eef: np.ndarray, R_eef: np.ndarray) -> np.ndarray:
    """Rotate a position delta from EEF frame to world frame."""
    return R_eef @ np.asarray(delta_eef, dtype=float)

# ---------------------------------------------------------------------------
# Telemetry parsing
# ---------------------------------------------------------------------------
# Array layout: [t, tcp×6, ref×6, ws×6, wd×6, q×6, T×6]  → 37 floats
#                0   1:7    7:13  13:19 19:25 25:31 31:37


def _parse_telem(line: str) -> Optional[np.ndarray]:
    try:
        parts = line[6:].strip().split()   # strip "TELEM:"
        d = {}
        for p in parts:
            k, v = p.split("=", 1)
            d[k] = [float(x) for x in v.split(",")]
        arr = np.zeros(37, dtype=np.float32)
        arr[0]    = d["t"][0]
        arr[1:7]  = d["tcp"][:6]
        arr[7:13] = d["ref"][:6]
        arr[13:19]= d["ws"][:6]
        arr[19:25]= d["wd"][:6]
        arr[25:31]= d.get("q", [0]*6)[:6]
        arr[31:37]= d.get("T", [0]*6)[:6]
        return arr
    except Exception:
        return None


def _write_config_yaml(path: str, rate_hz: int,
                       stiffness, damping, inertia, stiction,
                       max_spring_f: float, max_spring_t: float) -> None:
    dt = round(1.0 / rate_hz, 8)
    cfg = {
        "admittance_controller": {
            "dt": dt,
            "log_to_file": False,
            "log_file_path": "/tmp/admittance_controller.log",
            "alert_overrun": False,
            "compliance6d": {
                "stiffness": list(stiffness),
                "damping":   list(damping),
                "inertia":   list(inertia),
                "stiction":  list(stiction),
            },
            "max_spring_force_magnitude":  max_spring_f,
            "max_spring_torque_magnitude": max_spring_t,
            "direct_force_control_gains": {
                "P_trans": 0, "I_trans": 0, "D_trans": 0,
                "P_rot":   0, "I_rot":   0, "D_rot":   0,
            },
            "direct_force_control_I_limit": [0]*6,
        }
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=None)


# ---------------------------------------------------------------------------
# ForceControlUI
# ---------------------------------------------------------------------------

class ForceControlUI:
    """
    Force-control debug/command UI.

    Popup from grasp_viz:
        ui = ForceControlUI(parent=root_widget); ui.run()

    Standalone:
        ui = ForceControlUI(); ui.run()   # blocking mainloop
    """

    def __init__(self,
                 parent: Optional[tk.Widget] = None,
                 robot_ip: str = "192.168.0.4",
                 ft_ip:    str = "192.168.0.3",
                 binary:   str = _DEFAULT_BINARY,
                 on_close=None):
        self._parent    = parent
        self._robot_ip  = robot_ip
        self._ft_ip     = ft_ip
        self._binary    = binary
        self._on_close_cb = on_close

        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._telem_queue: queue.Queue = queue.Queue()
        self._running_proc = False

        self._buf: Deque[np.ndarray] = deque(maxlen=_TELEM_BUF)
        self._buf_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None

        self._log_enabled = False
        self._log_file: Optional[str] = None
        self._log_fh   = None
        self._log_writer = None

        # Frame vars are initialised in _section_frame; set defaults here so
        # other code that reads them before the UI is built won't crash.
        self._wf_var: Optional[tk.StringVar] = None  # wrench frame: "eef"/"world"
        self._pf_var: Optional[tk.StringVar] = None  # position frame: "world"/"eef"

        # Plot state
        self._show_6plot = False
        self._show_temp  = False   # True → show temp in top area instead of 3D

        self._tmp_config = tempfile.NamedTemporaryFile(
            suffix=".yaml", prefix="magpie_fc_", delete=False)
        self._tmp_config.close()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        if self._parent is None:
            self._root = tk.Tk()
            self._root.title("Force Control")
            standalone = True
        else:
            self._root = tk.Toplevel(self._parent)
            self._root.title("Force Control Panel")
            standalone = False

        self._root.resizable(True, True)
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self._schedule_plot_update()
        self._schedule_status_update()

        if standalone:
            self._root.mainloop()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._root.columnconfigure(0, weight=3, minsize=520)
        self._root.columnconfigure(1, weight=1, minsize=280)
        self._root.rowconfigure(0, weight=1)

        self._build_plot_panel()
        self._build_control_panel()

    # ── Left: plots ────────────────────────────────────────────────────

    def _build_plot_panel(self):
        self._plot_frame = ttk.Frame(self._root)
        self._plot_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self._plot_frame.rowconfigure(0, weight=2)
        self._plot_frame.rowconfigure(1, weight=1)
        self._plot_frame.columnconfigure(0, weight=1)

        # Top area: 3D EEF or temperature
        self._top_plot_frame = ttk.LabelFrame(
            self._plot_frame, text="End-Effector + Forces (EEF frame)")
        self._top_plot_frame.grid(row=0, column=0, sticky="nsew")
        self._top_plot_frame.rowconfigure(0, weight=1)
        self._top_plot_frame.columnconfigure(0, weight=1)
        self._build_3d_plot()

        # Bottom: time series
        self._ts_frame = ttk.LabelFrame(self._plot_frame, text="Time series")
        self._ts_frame.grid(row=1, column=0, sticky="nsew", pady=2)
        self._ts_frame.rowconfigure(0, weight=1)
        self._ts_frame.columnconfigure(0, weight=1)
        self._build_ts_2plot()

    def _build_3d_plot(self):
        self._fig3d = plt.figure(figsize=(5.5, 3.5))
        self._ax3d  = self._fig3d.add_subplot(111, projection="3d")
        self._ax3d.set_xlabel("X"); self._ax3d.set_ylabel("Y")
        self._ax3d.set_zlabel("Z")
        self._ax3d.set_title("EEF (solid=sensed, dashed=desired)", fontsize=8)
        self._canvas3d = FigureCanvasTkAgg(self._fig3d, master=self._top_plot_frame)
        self._canvas3d.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas_top = self._canvas3d

    def _build_temp_plot(self):
        """Replace top area with joint temperature history."""
        self._canvas3d.get_tk_widget().destroy()
        plt.close(self._fig3d)
        self._fig_temp = plt.figure(figsize=(5.5, 3.5))
        self._ax_temp  = self._fig_temp.add_subplot(111)
        self._ax_temp.set_title("Joint temperatures (°C)", fontsize=9)
        self._ax_temp.set_xlabel("t (s)", fontsize=7)
        self._ax_temp.tick_params(labelsize=7)
        self._top_plot_frame.config(text="Joint Temperatures")
        self._canvas_temp = FigureCanvasTkAgg(
            self._fig_temp, master=self._top_plot_frame)
        self._canvas_temp.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._canvas_top = self._canvas_temp

    def _build_ts_2plot(self):
        self._fig_ts = plt.figure(figsize=(5.5, 2.2))
        gs = gridspec.GridSpec(1, 2, figure=self._fig_ts,
                               left=0.08, right=0.98, wspace=0.35,
                               bottom=0.20, top=0.88)
        self._ax_f = self._fig_ts.add_subplot(gs[0])
        self._ax_t = self._fig_ts.add_subplot(gs[1])
        self._ax_f.set_title("Forces (N)", fontsize=8)
        self._ax_t.set_title("Torques (N·m)", fontsize=8)
        for ax in (self._ax_f, self._ax_t):
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.axhline(0, color="#bbb", linewidth=0.5)
        self._canvas_ts = FigureCanvasTkAgg(self._fig_ts, master=self._ts_frame)
        self._canvas_ts.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_ts_6plot(self):
        self._fig_ts = plt.figure(figsize=(5.5, 2.5))
        gs = gridspec.GridSpec(2, 3, figure=self._fig_ts,
                               left=0.08, right=0.98,
                               wspace=0.5, hspace=0.65,
                               bottom=0.12, top=0.92)
        self._axes6 = []
        for idx in range(6):
            ax = self._fig_ts.add_subplot(gs[idx // 3, idx % 3])
            ax.set_title(_W_LABELS[idx], fontsize=8, color=_W_COLORS[idx])
            ax.tick_params(labelsize=6)
            ax.axhline(0, color="#ccc", linewidth=0.5)
            self._axes6.append(ax)
        self._canvas_ts = FigureCanvasTkAgg(self._fig_ts, master=self._ts_frame)
        self._canvas_ts.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ── Right: scrollable control panel ───────────────────────────────

    def _build_control_panel(self):
        outer = ttk.Frame(self._root)
        outer.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        canvas  = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vsb     = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._ctrl = ttk.Frame(canvas, padding=(6, 4))
        canvas.create_window((0, 0), window=self._ctrl, anchor="nw")
        self._ctrl.bind("<Configure>",
                        lambda e: canvas.configure(
                            scrollregion=canvas.bbox("all")))
        # Mousewheel scroll
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(
                            int(-1 * (e.delta / 120)), "units"))

        r = 0
        r = self._section_connection(r)
        r = self._section_frame(r)
        r = self._section_wrench(r)
        r = self._section_compliance(r)
        r = self._section_pose(r)
        r = self._section_execution(r)
        r = self._section_advanced(r)
        r = self._section_logging(r)
        r = self._section_view(r)
        r = self._section_wrench_display(r)

    def _sep(self, r, pady=5):
        ttk.Separator(self._ctrl, orient="horizontal").grid(
            row=r, column=0, columnspan=4, sticky="ew", pady=pady)
        return r + 1

    # ── Sections ───────────────────────────────────────────────────────

    def _section_connection(self, r):
        ttk.Label(self._ctrl, text="── Connection ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(4, 2)); r += 1

        for lbl, attr in [("Robot IP:", "_var_robot_ip"),
                           ("FT IP:",    "_var_ft_ip")]:
            ttk.Label(self._ctrl, text=lbl).grid(
                row=r, column=0, sticky="w")
            v = tk.StringVar(value=getattr(self, lbl.split(":")[0].lower()
                                           .replace(" ", "_").replace("robot_ip", "_robot_ip")
                                           .replace("ft_ip", "_ft_ip"), ""))
            # simpler:
            default = self._robot_ip if "Robot" in lbl else self._ft_ip
            v = tk.StringVar(value=default)
            ttk.Entry(self._ctrl, textvariable=v, width=17).grid(
                row=r, column=1, columnspan=3, sticky="ew")
            setattr(self, attr, v); r += 1

        ttk.Label(self._ctrl, text="Binary:").grid(
            row=r, column=0, sticky="w")
        self._var_binary = tk.StringVar(value=self._binary)
        ttk.Entry(self._ctrl, textvariable=self._var_binary, width=14).grid(
            row=r, column=1, columnspan=2, sticky="ew")
        tk.Button(self._ctrl, text="…", width=2,
                  command=self._browse_binary).grid(
            row=r, column=3, padx=2); r += 1

        return self._sep(r)

    def _section_frame(self, r):
        """
        Separate frame selectors for wrench and position.

        Hou & Mason default (what the binary does naturally):
          Wrench → EEF frame   |   Position → WORLD frame
        This combination is green (recommended).
        The reverse (WORLD wrench + EEF position) shows a red warning.
        """
        # Wrench frame
        self._wf_var = tk.StringVar(value="eef")   # "eef" or "world"
        # Position frame
        self._pf_var = tk.StringVar(value="world")  # "world" or "eef"

        ttk.Label(self._ctrl, text="── Command Frames ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(4, 1)); r += 1

        # Wrench row
        frm_w = tk.Frame(self._ctrl, bd=1, relief="groove")
        frm_w.grid(row=r, column=0, columnspan=4, sticky="ew",
                   pady=1); r += 1
        self._wf_label = tk.Label(frm_w, text="Wrench:  EEF frame (H&M default)",
                                   bg="#16a085", fg="white",
                                   font=("TkDefaultFont", 8), padx=4, pady=3)
        self._wf_label.pack(side="left", fill="x", expand=True)
        tk.Radiobutton(frm_w, text="EEF", variable=self._wf_var,
                       value="eef", command=self._on_frame_toggle).pack(side="left")
        tk.Radiobutton(frm_w, text="WORLD", variable=self._wf_var,
                       value="world", command=self._on_frame_toggle).pack(side="left")

        # Position row
        frm_p = tk.Frame(self._ctrl, bd=1, relief="groove")
        frm_p.grid(row=r, column=0, columnspan=4, sticky="ew",
                   pady=1); r += 1
        self._pf_label = tk.Label(frm_p, text="Position:  WORLD frame (H&M default)",
                                   bg="#16a085", fg="white",
                                   font=("TkDefaultFont", 8), padx=4, pady=3)
        self._pf_label.pack(side="left", fill="x", expand=True)
        tk.Radiobutton(frm_p, text="WORLD", variable=self._pf_var,
                       value="world", command=self._on_frame_toggle).pack(side="left")
        tk.Radiobutton(frm_p, text="EEF", variable=self._pf_var,
                       value="eef", command=self._on_frame_toggle).pack(side="left")

        # Warning banner (hidden by default)
        self._frame_warn = tk.Label(
            self._ctrl,
            text="⚠  WORLD wrench + EEF position is non-standard.\n"
                 "   Hou & Mason use EEF wrench + WORLD position.",
            bg="#c0392b", fg="white",
            font=("TkDefaultFont", 8), justify="left",
            padx=4, pady=3, wraplength=230)
        self._frame_warn.grid(row=r, column=0, columnspan=4, sticky="ew")
        self._frame_warn.grid_remove()  # hidden until triggered
        r += 1

        ttk.Label(self._ctrl,
                  text="H&M default: EEF wrench + WORLD pos (green = recommended)",
                  foreground="#888", font=("TkDefaultFont", 7)).grid(
            row=r, column=0, columnspan=4, sticky="w"); r += 1

        return self._sep(r)

    def _section_wrench(self, r):
        ttk.Label(self._ctrl, text="── Desired Wrench ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 2)); r += 1

        # Safety max row
        frm_s = ttk.Frame(self._ctrl)
        frm_s.grid(row=r, column=0, columnspan=4, sticky="ew"); r += 1
        ttk.Label(frm_s, text="Max |F| (N):", font=("TkDefaultFont", 7)).pack(side="left")
        self._var_max_f = tk.DoubleVar(value=_MAX_F_N)
        ttk.Entry(frm_s, textvariable=self._var_max_f, width=5).pack(side="left", padx=2)
        ttk.Label(frm_s, text=" |T| (N·m):", font=("TkDefaultFont", 7)).pack(side="left")
        self._var_max_t = tk.DoubleVar(value=_MAX_T_NM)
        ttk.Entry(frm_s, textvariable=self._var_max_t, width=5).pack(side="left", padx=2)

        self._wrench_vars: List[tk.DoubleVar] = []
        # Row of 3 forces, then row of 3 torques
        for group_start, labels, colors, unit in [
            (0, _F_LABELS, _F_COLORS, "N"),
            (3, _T_LABELS, _T_COLORS, "N·m"),
        ]:
            frm = ttk.Frame(self._ctrl)
            frm.grid(row=r, column=0, columnspan=4, sticky="ew", pady=1); r += 1
            for i, (lbl, col) in enumerate(zip(labels, colors)):
                tk.Label(frm, text=f"{lbl}:", foreground=col,
                         font=("TkDefaultFont", 8, "bold"),
                         width=3, anchor="e").pack(side="left")
                v = tk.DoubleVar(value=0.0)
                ttk.Entry(frm, textvariable=v, width=7).pack(side="left", padx=1)
                self._wrench_vars.append(v)
            tk.Label(frm, text=unit, foreground="#666",
                     font=("TkDefaultFont", 7)).pack(side="left", padx=2)

        tk.Button(self._ctrl, text="Send Wrench",
                  bg="#27ae60", fg="white",
                  command=self._send_wrench).grid(
            row=r, column=0, columnspan=4, sticky="ew", pady=3); r += 1

        return self._sep(r)

    def _section_compliance(self, r):
        ttk.Label(self._ctrl, text="── Compliance Axes ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 1)); r += 1
        ttk.Label(self._ctrl, text="checked = force-controlled",
                  foreground="#888", font=("TkDefaultFont", 7)).grid(
            row=r, column=0, columnspan=4, sticky="w"); r += 1

        self._axis_vars: List[tk.IntVar] = []
        frm = ttk.Frame(self._ctrl)
        frm.grid(row=r, column=0, columnspan=4, sticky="w"); r += 1
        for lbl, col in zip(_W_LABELS, _W_COLORS):
            v = tk.IntVar(value=1)
            tk.Checkbutton(frm, text=lbl, variable=v,
                           foreground=col, command=self._send_axis).pack(
                side="left", padx=2)
            self._axis_vars.append(v)

        return self._sep(r)

    def _section_pose(self, r):
        ttk.Label(self._ctrl, text="── Pose Reference ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 1)); r += 1

        # Delta / Absolute toggle with visual indicator
        self._pose_mode_var = tk.StringVar(value="delta")
        mode_frm = tk.Frame(self._ctrl, bd=2, relief="groove")
        mode_frm.grid(row=r, column=0, columnspan=4, sticky="ew",
                      pady=2); r += 1
        self._pose_mode_label = tk.Label(
            mode_frm, text="DELTA  (from initial)",
            bg="#2980b9", fg="white",
            font=("TkDefaultFont", 8),
            padx=4, pady=3)
        self._pose_mode_label.pack(side="left", fill="x", expand=True)
        tk.Radiobutton(mode_frm, text="Δ", variable=self._pose_mode_var,
                       value="delta",
                       command=self._on_pose_mode).pack(side="left")
        tk.Radiobutton(mode_frm, text="ABS", variable=self._pose_mode_var,
                       value="absolute",
                       command=self._on_pose_mode).pack(side="left")

        # Position XYZ
        ttk.Label(self._ctrl, text="Position (m):",
                  font=("TkDefaultFont", 8)).grid(
            row=r, column=0, columnspan=4, sticky="w"); r += 1
        self._pos_vars: List[tk.DoubleVar] = []
        frm_pos = ttk.Frame(self._ctrl)
        frm_pos.grid(row=r, column=0, columnspan=4, sticky="ew",
                     pady=1); r += 1
        for lbl, col in zip(["X:", "Y:", "Z:"], _F_COLORS):
            tk.Label(frm_pos, text=lbl, foreground=col,
                     font=("TkDefaultFont", 8), width=2).pack(side="left")
            v = tk.DoubleVar(value=0.0)
            ttk.Entry(frm_pos, textvariable=v, width=7).pack(
                side="left", padx=1)
            self._pos_vars.append(v)
        tk.Label(frm_pos, text="m", foreground="#666",
                 font=("TkDefaultFont", 7)).pack(side="left", padx=2)

        # Rotation XYZ Euler (degrees)
        ttk.Label(self._ctrl, text="Rotation (°, Euler XYZ, ±360):  ⚠ be careful",
                  font=("TkDefaultFont", 8), foreground="#c0392b").grid(
            row=r, column=0, columnspan=4, sticky="w"); r += 1
        self._rot_vars: List[tk.DoubleVar] = []
        frm_rot = ttk.Frame(self._ctrl)
        frm_rot.grid(row=r, column=0, columnspan=4, sticky="ew",
                     pady=1); r += 1
        for lbl, col in zip(["Rx:", "Ry:", "Rz:"], _F_COLORS):
            tk.Label(frm_rot, text=lbl, foreground=col,
                     font=("TkDefaultFont", 8), width=3).pack(side="left")
            v = tk.DoubleVar(value=0.0)
            ttk.Entry(frm_rot, textvariable=v, width=6).pack(
                side="left", padx=1)
            self._rot_vars.append(v)
        tk.Label(frm_rot, text="°", foreground="#666",
                 font=("TkDefaultFont", 7)).pack(side="left", padx=2)

        tk.Button(self._ctrl, text="Send Pose Ref",
                  command=self._send_pose_ref).grid(
            row=r, column=0, columnspan=4, sticky="ew", pady=3); r += 1

        return self._sep(r)

    def _section_execution(self, r):
        ttk.Label(self._ctrl, text="── Execution ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 2)); r += 1

        for lbl, attr, val in [("Duration (s):", "_var_duration", 30.0),
                                ("Rate (Hz):",   "_var_rate",     500)]:
            ttk.Label(self._ctrl, text=lbl).grid(row=r, column=0,
                                                   columnspan=2, sticky="w")
            v = (tk.DoubleVar if "s" in lbl else tk.IntVar)(value=val)
            ttk.Entry(self._ctrl, textvariable=v, width=8).grid(
                row=r, column=2, columnspan=2, sticky="ew")
            setattr(self, attr, v); r += 1

        frm_btns = ttk.Frame(self._ctrl)
        frm_btns.grid(row=r, column=0, columnspan=4, sticky="ew",
                      pady=4); r += 1
        self._btn_start = tk.Button(
            frm_btns, text="▶ Start",
            bg="#27ae60", fg="white",
            width=10,
            command=self._start_controller)
        self._btn_start.pack(side="left", padx=2)
        self._btn_stop = tk.Button(
            frm_btns, text="■ Stop",
            bg="#e74c3c", fg="white",
            width=10,
            state="disabled",
            command=self._stop_controller)
        self._btn_stop.pack(side="left", padx=2)

        self._status_var = tk.StringVar(value="Idle")
        tk.Label(self._ctrl, textvariable=self._status_var,
                 font=("TkDefaultFont", 8), foreground="#555",
                 anchor="w", wraplength=220, justify="left").grid(
            row=r, column=0, columnspan=4, sticky="ew", pady=1); r += 1

        return self._sep(r)

    def _section_advanced(self, r):
        hdr = ttk.Frame(self._ctrl)
        hdr.grid(row=r, column=0, columnspan=4, sticky="ew"); r += 1
        self._adv_var = tk.IntVar(value=0)
        tk.Checkbutton(hdr, text="── Advanced Config ──",
                       variable=self._adv_var,
                       foreground="#555",
                       command=self._toggle_advanced).pack(side="left")

        self._adv_frame = ttk.Frame(self._ctrl)
        self._adv_frame.grid(row=r, column=0, columnspan=4,
                              sticky="ew"); r += 1

        for name, attr, defs in [
            ("Stiffness [Fx…Tz]:", "_stiff_vars", _DEF_STIFFNESS),
            ("Damping   [Fx…Tz]:", "_damp_vars",  _DEF_DAMPING),
            ("Inertia   [Fx…Tz]:", "_iner_vars",  _DEF_INERTIA),
        ]:
            ttk.Label(self._adv_frame, text=name,
                      font=("TkDefaultFont", 7)).grid(sticky="w")
            frm = ttk.Frame(self._adv_frame)
            frm.grid(sticky="ew")
            vs = []
            for d in defs:
                v = tk.DoubleVar(value=d)
                ttk.Entry(frm, textvariable=v, width=6).pack(
                    side="left", padx=1)
                vs.append(v)
            setattr(self, attr, vs)

        frm_sf = ttk.Frame(self._adv_frame)
        frm_sf.grid(sticky="ew", pady=2)
        ttk.Label(frm_sf, text="Spring max F (N):").pack(side="left")
        self._var_spring_f = tk.DoubleVar(value=50.0)
        ttk.Entry(frm_sf, textvariable=self._var_spring_f,
                  width=6).pack(side="left", padx=2)
        ttk.Label(frm_sf, text="T (N·m):").pack(side="left")
        self._var_spring_t = tk.DoubleVar(value=4.0)
        ttk.Entry(frm_sf, textvariable=self._var_spring_t,
                  width=6).pack(side="left", padx=2)

        tk.Button(self._adv_frame,
                  text="Apply Stiffness & Damping (live)",
                  command=self._send_stiff_damp).grid(
            sticky="ew", pady=3)

        self._adv_frame.grid_remove()
        return self._sep(r)

    def _section_logging(self, r):
        ttk.Label(self._ctrl, text="── Logging ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 1)); r += 1

        frm = ttk.Frame(self._ctrl)
        frm.grid(row=r, column=0, columnspan=4, sticky="ew"); r += 1
        self._log_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Enable CSV log",
                       variable=self._log_var,
                       command=self._toggle_log).pack(side="left")
        tk.Button(frm, text="…", width=2,
                  command=self._choose_log_path).pack(side="left", padx=4)

        self._log_path_var = tk.StringVar(value="(not set)")
        tk.Label(self._ctrl, textvariable=self._log_path_var,
                 foreground="#555", font=("TkDefaultFont", 7),
                 wraplength=220, anchor="w", justify="left").grid(
            row=r, column=0, columnspan=4, sticky="ew"); r += 1

        return self._sep(r)

    def _section_view(self, r):
        ttk.Label(self._ctrl, text="── Visualisation ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 1)); r += 1

        frm = ttk.Frame(self._ctrl)
        frm.grid(row=r, column=0, columnspan=4, sticky="ew"); r += 1

        self._des_var = tk.IntVar(value=1)
        tk.Checkbutton(frm, text="Show desired",
                       variable=self._des_var).pack(side="left")
        self._six_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="6-plot",
                       variable=self._six_var,
                       command=self._toggle_6plot).pack(side="left", padx=4)
        self._tmp_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Temps (replaces 3D)",
                       variable=self._tmp_var,
                       command=self._toggle_temp).pack(side="left")

        return self._sep(r)

    def _section_wrench_display(self, r):
        ttk.Label(self._ctrl, text="── Live Wrench ──",
                  foreground="#555").grid(row=r, column=0, columnspan=4,
                                          sticky="w", pady=(2, 1)); r += 1

        # Desired static header — compact, 2 decimal places, 3 per row
        self._des_display = tk.Text(self._ctrl, height=3, width=30,
                                    font=("Courier", 8), state="disabled",
                                    bg="#f5f5f5", relief="flat", padx=2)
        self._des_display.grid(row=r, column=0, columnspan=4,
                               sticky="ew"); r += 1

        tk.Label(self._ctrl, text="Sensed  (grn=in-range, red=over):",
                 font=("TkDefaultFont", 7), foreground="#555").grid(
            row=r, column=0, columnspan=4, sticky="w"); r += 1

        self._sensed_display = scrolledtext.ScrolledText(
            self._ctrl, height=7, width=30,
            font=("Courier", 8), state="disabled", wrap="none")
        self._sensed_display.grid(row=r, column=0, columnspan=4,
                                  sticky="ew"); r += 1

        for tag, col in [("grn", "#1a7a1a"),
                          ("red", "#c0392b"),
                          ("blk", "#111")]:
            self._sensed_display.tag_configure(
                tag, foreground=col,
                font=("Courier", 8, "bold" if tag != "blk" else "normal"))
        self._des_display.tag_configure(
            "hdr", foreground="#333",
            font=("Courier", 8, "bold"))

        return r

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _browse_binary(self):
        p = tkfiledialog.askopenfilename(
            title="Select force_control_demo binary")
        if p:
            self._var_binary.set(p)

    def _on_frame_toggle(self):
        wf = self._wf_var.get()   # "eef" or "world"
        pf = self._pf_var.get()   # "world" or "eef"

        if wf == "eef":
            self._wf_label.config(text="Wrench:  EEF frame", bg="#16a085")
        else:
            self._wf_label.config(text="Wrench:  WORLD frame", bg="#d35400")

        if pf == "world":
            self._pf_label.config(text="Position:  WORLD frame", bg="#2980b9")
        else:
            self._pf_label.config(text="Position:  EEF frame", bg="#8e44ad")

        # Show warning for non-standard combination
        if wf == "world" and pf == "eef":
            self._frame_warn.grid()
        else:
            self._frame_warn.grid_remove()

    def _on_pose_mode(self):
        m = self._pose_mode_var.get()
        if m == "delta":
            self._pose_mode_label.config(
                text="DELTA  (from initial)", bg="#2980b9")
        else:
            self._pose_mode_label.config(
                text="ABSOLUTE  (world)", bg="#8e44ad")

    def _toggle_advanced(self):
        if self._adv_var.get():
            self._adv_frame.grid()
        else:
            self._adv_frame.grid_remove()

    def _toggle_log(self):
        self._log_enabled = bool(self._log_var.get())
        if self._log_enabled and self._log_file is None:
            self._auto_set_log_path()
        if self._log_enabled and self._log_fh is None and self._log_file:
            self._open_log()
        elif not self._log_enabled and self._log_fh:
            self._close_log()

    def _auto_set_log_path(self):
        _LOGS_DIR.mkdir(exist_ok=True)
        self._log_file = str(
            _LOGS_DIR / f"fc_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        self._log_path_var.set(self._log_file)

    def _choose_log_path(self):
        _LOGS_DIR.mkdir(exist_ok=True)
        p = tkfiledialog.asksaveasfilename(
            title="Save log CSV",
            initialdir=str(_LOGS_DIR),
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self._log_file = p
            self._log_path_var.set(p)
            if self._log_enabled:
                self._close_log()
                self._open_log()

    def _toggle_6plot(self):
        self._canvas_ts.get_tk_widget().destroy()
        plt.close(self._fig_ts)
        if self._six_var.get():
            self._build_ts_6plot()
            self._show_6plot = True
        else:
            self._build_ts_2plot()
            self._show_6plot = False

    def _toggle_temp(self):
        self._show_temp = bool(self._tmp_var.get())
        if self._show_temp:
            self._build_temp_plot()
            self._top_plot_frame.config(text="Joint Temperatures")
        else:
            self._build_3d_plot()
            self._top_plot_frame.config(
                text="End-Effector + Forces (EEF frame)")

    # ------------------------------------------------------------------
    # Safety clamp
    # ------------------------------------------------------------------

    def _clamp_wrench(self, vals: List[float]) -> List[float]:
        mf = self._var_max_f.get()
        mt = self._var_max_t.get()
        out = list(vals)
        for i in range(3):
            out[i] = max(-mf, min(mf, out[i]))
        for i in range(3, 6):
            out[i] = max(-mt, min(mt, out[i]))
        return out

    def _clamp_rotation_deg(self, vals: List[float]) -> List[float]:
        return [max(-_MAX_ROT_DEG, min(_MAX_ROT_DEG, v)) for v in vals]

    # ------------------------------------------------------------------
    # Send commands → stdin
    # ------------------------------------------------------------------

    def _send_cmd(self, line: str):
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def _send_wrench(self):
        raw = [v.get() for v in self._wrench_vars]

        # Frame conversion: if world, rotate to EEF
        lat = self._latest
        wf = self._wf_var.get() if self._wf_var else "eef"
        if wf == "world" and lat is not None:
            R_eef = _get_R_eef(lat[1:7])  # from current TCP axis-angle
            raw = list(_wrench_world_to_eef(np.array(raw), R_eef))

        clamped = self._clamp_wrench(raw)
        if any(abs(c - r) > 1e-9 for c, r in zip(clamped, raw)):
            self._status_var.set("⚠ Wrench clamped to safety limits")
            # Update UI entries with clamped values (in original frame)
            clamped_original = self._clamp_wrench(
                [v.get() for v in self._wrench_vars])
            for i, (v, c) in enumerate(zip(self._wrench_vars,
                                            clamped_original)):
                v.set(round(c, 4))

        self._send_cmd("WRENCH " +
                       " ".join(f"{x:.4f}" for x in clamped))

    def _send_axis(self):
        self._send_cmd("AXIS " +
                       " ".join(str(v.get()) for v in self._axis_vars))

    def _send_pose_ref(self):
        pos = [v.get() for v in self._pos_vars]
        rot_deg = self._clamp_rotation_deg(
            [v.get() for v in self._rot_vars])
        # Warn on large rotations
        if any(abs(d) > 45 for d in rot_deg):
            self._status_var.set(
                f"⚠ Large rotation {[f'{d:.1f}°' for d in rot_deg]} — proceed carefully")

        # Rotation → axis-angle (radians)
        rot_aa = _euler_xyz_to_aa(*rot_deg)

        lat = self._latest
        mode = self._pose_mode_var.get()

        pf = self._pf_var.get() if self._pf_var else "world"
        if pf == "eef" and lat is not None:
            # Position: EEF frame → world
            R_eef = _get_R_eef(lat[1:7])
            pos_world = list(_pos_eef_to_world(np.array(pos), R_eef))
            # Rotation: EEF delta → world (pre-multiply by R_eef)
            rot_world = list(R_eef @ rot_aa)
        else:
            pos_world = pos
            rot_world = list(rot_aa)

        cmd = "RELD" if mode == "delta" else "REFA"
        self._send_cmd(
            f"{cmd} {pos_world[0]:.4f} {pos_world[1]:.4f} {pos_world[2]:.4f}"
            f" {rot_world[0]:.5f} {rot_world[1]:.5f} {rot_world[2]:.5f}")

    def _send_stiff_damp(self):
        if not hasattr(self, "_stiff_vars"):
            return
        self._send_cmd("STIFF " +
                       " ".join(f"{v.get():.4f}" for v in self._stiff_vars))
        self._send_cmd("DAMP " +
                       " ".join(f"{v.get():.4f}" for v in self._damp_vars))

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    def _start_controller(self):
        if self._proc is not None:
            self._status_var.set("Already running — stop first.")
            return

        binary = self._var_binary.get()
        if not os.path.isfile(binary):
            self._status_var.set(f"Binary not found:\n{binary}")
            return

        rate = self._var_rate.get()
        stiff = ([v.get() for v in self._stiff_vars]
                 if hasattr(self, "_stiff_vars") else _DEF_STIFFNESS)
        damp  = ([v.get() for v in self._damp_vars]
                 if hasattr(self, "_damp_vars")  else _DEF_DAMPING)
        iner  = ([v.get() for v in self._iner_vars]
                 if hasattr(self, "_iner_vars")  else _DEF_INERTIA)
        sf    = (self._var_spring_f.get()
                 if hasattr(self, "_var_spring_f") else 50.0)
        st    = (self._var_spring_t.get()
                 if hasattr(self, "_var_spring_t") else 4.0)
        _write_config_yaml(self._tmp_config.name, rate,
                           stiff, damp, iner, _DEF_STICTION, sf, st)

        # Initial wrench (EEF frame, clamped)
        raw_w = [v.get() for v in self._wrench_vars]
        wf = self._wf_var.get() if self._wf_var else "eef"
        if wf == "world" and self._latest is not None:
            R = _get_R_eef(self._latest[1:7])
            raw_w = list(_wrench_world_to_eef(np.array(raw_w), R))
        initial_w = self._clamp_wrench(raw_w)

        duration = self._var_duration.get()
        cmd = [
            binary,
            "--robot-ip", self._var_robot_ip.get(),
            "--ft-ip",    self._var_ft_ip.get(),
            "--rate",     str(rate),
            "--time",     str(int(duration)),
            "--config",   self._tmp_config.name,
            "--wrench",
        ] + [f"{x:.4f}" for x in initial_w]

        # Position delta at start
        pos = [v.get() for v in self._pos_vars]
        if any(abs(v) > 1e-9 for v in pos) and \
                self._pose_mode_var.get() == "delta":
            # Pass as --pose delta; identity quaternion (no rotation at start)
            cmd += ["--pose",
                    f"{pos[0]:.4f}", f"{pos[1]:.4f}", f"{pos[2]:.4f}",
                    "1", "0", "0", "0"]

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1)
        except Exception as e:
            self._status_var.set(f"Failed: {e}")
            self._proc = None
            return

        # Wipe telemetry buffer for fresh plots
        with self._buf_lock:
            self._buf.clear()
            self._latest = None

        self._running_proc = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        wf_str = (self._wf_var.get() if self._wf_var else "eef").upper()
        pf_str = (self._pf_var.get() if self._pf_var else "world").upper()
        self._status_var.set(
            f"Running  {rate} Hz  {duration} s  [W:{wf_str} P:{pf_str}]")

        if self._log_enabled and self._log_fh is None:
            if self._log_file is None:
                self._auto_set_log_path()
            self._open_log()

    def _stop_controller(self):
        self._send_cmd("STOP")
        time.sleep(0.1)
        self._cleanup_proc()

    def _cleanup_proc(self):
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None
        self._running_proc = False
        self._close_log()
        if self._root.winfo_exists():
            self._btn_start.config(state="normal")
            self._btn_stop.config(state="disabled")
            self._status_var.set("Idle")

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self):
        while self._running_proc and self._proc:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            if line.startswith("TELEM:"):
                s = _parse_telem(line)
                if s is not None:
                    with self._buf_lock:
                        self._buf.append(s)
                        self._latest = s
                    if self._log_enabled and self._log_writer:
                        self._write_log(s)
            else:
                self._telem_queue.put(("log", line))
        self._running_proc = False
        self._telem_queue.put(("done", ""))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _open_log(self):
        if not self._log_file:
            return
        _LOGS_DIR.mkdir(exist_ok=True)
        try:
            self._log_fh = open(self._log_file, "w", newline="")
            self._log_writer = csv.writer(self._log_fh)
            hdr = (["epoch_s", "t_s"] +
                   [f"tcp_{x}" for x in ("x","y","z","rx","ry","rz")] +
                   [f"ref_{x}" for x in ("x","y","z","rx","ry","rz")] +
                   [f"ws_{l}" for l in _W_LABELS] +
                   [f"wd_{l}" for l in _W_LABELS] +
                   [f"q{i}" for i in range(6)] +
                   [f"T{i}" for i in range(6)])
            self._log_writer.writerow(hdr)
            self._log_path_var.set(self._log_file)
        except Exception as e:
            self._status_var.set(f"Log open error: {e}")

    def _write_log(self, s: np.ndarray):
        try:
            row = ([f"{time.time():.4f}", f"{s[0]:.4f}"] +
                   [f"{x:.5f}" for x in s[1:]])
            self._log_writer.writerow(row)
            self._log_fh.flush()
        except Exception:
            pass

    def _close_log(self):
        if self._log_fh:
            self._log_fh.close()
            self._log_fh = None
            self._log_writer = None

    # ------------------------------------------------------------------
    # Plot updates
    # ------------------------------------------------------------------

    def _schedule_plot_update(self):
        if self._root.winfo_exists():
            self._update_plots()
            self._root.after(_PLOT_MS, self._schedule_plot_update)

    def _schedule_status_update(self):
        if self._root.winfo_exists():
            self._update_status_text()
            self._drain_log_queue()
            self._root.after(_STATUS_MS, self._schedule_status_update)

    def _update_plots(self):
        with self._buf_lock:
            buf  = list(self._buf)
            lat  = self._latest

        if not buf:
            return

        arr  = np.array(buf)
        ts   = arr[:, 0]
        ws   = arr[:, 13:19]
        wd   = arr[:, 19:25]
        T_j  = arr[:, 31:37]
        show_des = bool(self._des_var.get())

        # Top plot: 3D or temperature
        if self._show_temp:
            if hasattr(self, "_ax_temp"):
                self._draw_temp(ts, T_j)
                try:
                    self._canvas_top.draw_idle()
                except Exception:
                    pass
        else:
            self._draw_3d(lat, show_des)
            try:
                self._canvas3d.draw_idle()
            except Exception:
                pass

        # Time-series
        if self._show_6plot:
            if hasattr(self, "_axes6"):
                self._draw_6plot(ts, ws, wd, show_des)
        else:
            self._draw_2plot(ts, ws, wd, show_des)

        try:
            self._canvas_ts.draw_idle()
        except Exception:
            pass

    def _draw_3d(self, s: Optional[np.ndarray], show_des: bool):
        ax = self._ax3d
        ax.cla()
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("EEF  solid=sensed  dashed=desired", fontsize=8)

        if s is None:
            return

        tcp = s[1:7]
        ws  = s[13:19]
        wd  = s[19:25]
        pos = tcp[:3]
        R   = _aa_to_R(tcp[3:6])

        # Coordinate frame axes
        L = 0.04
        for i, col in enumerate(["r", "g", "b"]):
            ax.quiver(pos[0], pos[1], pos[2],
                      R[0, i]*L, R[1, i]*L, R[2, i]*L,
                      color=col, arrow_length_ratio=0.3,
                      linewidth=1.5)

        def _farrows(w, alpha, lw, ls="solid"):
            for i in range(3):
                v = R[:, i] * w[i] * _FORCE_SCALE
                ax.quiver(pos[0], pos[1], pos[2],
                          v[0], v[1], v[2],
                          color=_F_COLORS[i], alpha=alpha,
                          linewidth=lw, linestyle=ls,
                          arrow_length_ratio=0.25)

        def _tarcs(w, alpha, lw):
            for i in range(3):
                tau = w[3 + i]
                if abs(tau) < 0.05:
                    continue
                ax_w  = R[:, i] * np.sign(tau)
                r_arc = max(0.01, min(0.06, abs(tau) * _TORQUE_SCALE))
                perp  = (np.array([0,0,1]) if abs(ax_w[2]) < 0.9
                         else np.array([1,0,0]))
                u = np.cross(ax_w, perp); u /= np.linalg.norm(u)
                v = np.cross(ax_w, u)
                th = np.linspace(0, 1.5 * np.pi, 24)
                arc = pos + r_arc * (
                    np.outer(np.cos(th), u) + np.outer(np.sin(th), v))
                ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                        color=_T_COLORS[i], alpha=alpha, linewidth=lw)

        _farrows(ws, 1.0, 1.5)
        _tarcs(ws, 1.0, 1.5)
        if show_des:
            _farrows(wd, 0.40, 2.5, "dashed")
            _tarcs(wd, 0.40, 2.0)

        # Trail
        with self._buf_lock:
            trail = np.array([b[1:4] for b in list(self._buf)])
        if len(trail) > 2:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                    "k:", alpha=0.3, linewidth=0.8)

        pad = 0.10
        ax.set_xlim(pos[0]-pad, pos[0]+pad)
        ax.set_ylim(pos[1]-pad, pos[1]+pad)
        ax.set_zlim(pos[2]-pad, pos[2]+pad)

    def _draw_2plot(self, ts, ws, wd, show_des):
        for ax in (self._ax_f, self._ax_t):
            ax.cla(); ax.axhline(0, color="#ccc", linewidth=0.5)
        self._ax_f.set_title("Forces (N)", fontsize=8)
        self._ax_t.set_title("Torques (N·m)", fontsize=8)

        for i in range(3):
            self._ax_f.plot(ts, ws[:, i],   color=_F_COLORS[i],
                            linewidth=1.0, alpha=0.9, label=_F_LABELS[i])
            self._ax_t.plot(ts, ws[:, 3+i], color=_T_COLORS[i],
                            linewidth=1.0, alpha=0.9, label=_T_LABELS[i])
            if show_des:
                self._ax_f.plot(ts, wd[:, i],   color=_F_COLORS[i],
                                linewidth=2.0, alpha=0.35, linestyle="--")
                self._ax_t.plot(ts, wd[:, 3+i], color=_T_COLORS[i],
                                linewidth=2.0, alpha=0.35, linestyle="--")

        for ax in (self._ax_f, self._ax_t):
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper left", ncol=3)

    def _draw_6plot(self, ts, ws, wd, show_des):
        for i, ax in enumerate(self._axes6):
            ax.cla()
            ax.set_title(_W_LABELS[i], fontsize=8, color=_W_COLORS[i])
            ax.tick_params(labelsize=6)
            ax.axhline(0, color="#ccc", linewidth=0.5)
            ax.plot(ts, ws[:, i], color=_W_COLORS[i], linewidth=1.0)
            if show_des:
                ax.plot(ts, wd[:, i], color=_W_COLORS[i],
                        linewidth=2.0, alpha=0.35, linestyle="--")
            ax.set_ylabel("N" if i < 3 else "N·m", fontsize=6)

    def _draw_temp(self, ts, T_j):
        ax = self._ax_temp
        ax.cla()
        ax.set_title("Joint temperatures (°C)", fontsize=9)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=7)
        for j in range(6):
            ax.plot(ts, T_j[:, j], linewidth=1.2, label=f"J{j+1}")
        ax.legend(fontsize=7, loc="upper left", ncol=3)

    # ------------------------------------------------------------------
    # Status/wrench text
    # ------------------------------------------------------------------

    def _drain_log_queue(self):
        try:
            while True:
                kind, msg = self._telem_queue.get_nowait()
                if kind == "done":
                    self._cleanup_proc()
                elif kind == "log":
                    self._status_var.set(msg[:160])
        except queue.Empty:
            pass

    def _update_status_text(self):
        s = self._latest
        if s is None:
            return

        ws = s[13:19]  # sensed
        wd = s[19:25]  # desired

        # ── Desired (compact header, 2 dec, 3 per row) ──────────────
        self._des_display.config(state="normal")
        self._des_display.delete("1.0", tk.END)
        self._des_display.insert(tk.END, "Des: ", "hdr")
        f_parts = [f"{_W_LABELS[i]}={wd[i]:+.2f}N" for i in range(3)]
        self._des_display.insert(tk.END, "  ".join(f_parts) + "\n")
        t_parts = [f"{_W_LABELS[i]}={wd[i]:+.2f}Nm" for i in range(3, 6)]
        self._des_display.insert(tk.END, "     " + "  ".join(t_parts) + "\n")
        self._des_display.config(state="disabled")

        # ── Sensed stream (colored, 2 dec, 3 per row) ────────────────
        self._sensed_display.config(state="normal")
        t_str = f"{s[0]:.1f}s"
        self._sensed_display.insert(tk.END, f"{t_str:>6}: ")

        # Fx Fy Fz on same line
        for i in range(3):
            sv, dv = float(ws[i]), float(wd[i])
            tag = self._wrench_color_tag(sv, dv)
            self._sensed_display.insert(
                tk.END, f"{_W_LABELS[i]}={sv:+.2f}N  ", tag)
        self._sensed_display.insert(tk.END, "\n        ")

        # Tx Ty Tz on next line
        for i in range(3, 6):
            sv, dv = float(ws[i]), float(wd[i])
            tag = self._wrench_color_tag(sv, dv)
            self._sensed_display.insert(
                tk.END, f"{_W_LABELS[i]}={sv:+.2f}Nm ", tag)
        self._sensed_display.insert(tk.END, "\n")

        # Keep last ~30 reading pairs (60 lines)
        lines = int(self._sensed_display.index("end-1c").split(".")[0])
        if lines > 65:
            self._sensed_display.delete("1.0", "6.0")
        self._sensed_display.see(tk.END)
        self._sensed_display.config(state="disabled")

    @staticmethod
    def _wrench_color_tag(sv: float, dv: float) -> str:
        """
        green  → |sensed| within ±10% of |desired|
        red    → |sensed| > 110% of |desired|
        black  → |sensed| < 90% of |desired|  (also if desired ≈ 0)
        """
        adv = abs(dv)
        if adv < 0.05:
            return "blk"
        asv = abs(sv)
        if asv > 1.10 * adv:
            return "red"
        if asv >= 0.90 * adv:
            return "grn"
        return "blk"

    # ------------------------------------------------------------------
    # Close
    # ------------------------------------------------------------------

    def _on_close(self):
        self._running_proc = False
        if self._proc:
            try:
                self._send_cmd("STOP")
                time.sleep(0.05)
                self._proc.terminate()
                self._proc.wait(timeout=2)
            except Exception:
                pass
        self._close_log()
        try:
            os.unlink(self._tmp_config.name)
        except Exception:
            pass
        plt.close("all")
        self._root.destroy()
        if self._on_close_cb is not None:
            try:
                self._on_close_cb()
            except Exception as e:
                _log.warning("on_close callback raised: %s", e)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Force control debug UI")
    p.add_argument("--robot-ip", default="192.168.0.4")
    p.add_argument("--ft-ip",    default="192.168.0.3")
    p.add_argument("--binary",   default=_DEFAULT_BINARY)
    p.add_argument("--log-file", default=None,
                   help="Write Python logs to file (default: stderr only)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging (default: INFO)")
    return p.parse_args()


def main():
    args = _parse_args()
    _handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log_file:
        _handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        handlers=_handlers,
    )
    ui = ForceControlUI(robot_ip=args.robot_ip, ft_ip=args.ft_ip,
                        binary=args.binary)
    ui.run()


if __name__ == "__main__":
    main()
