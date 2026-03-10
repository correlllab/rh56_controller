"""
force_control_ui.py — Tkinter UI for magpie force control debugging & commanding.

Standalone:
    uv run python -m rh56_controller.force_control_ui \\
        [--robot-ip 192.168.0.4] [--ft-ip 192.168.0.3] [--binary /path/to/force_control_demo]

From grasp_viz — button calls ForceControlUI.open_popup(parent_tk_root).

Architecture
------------
- Spawns force_control_demo C++ binary as a subprocess.
- Commands sent via subprocess stdin (WRENCH, AXIS, RELD, REFA, STIFF, DAMP, STOP).
- Telemetry received via subprocess stdout (lines starting with "TELEM:").
- Matplotlib embedded via TkAgg backend for live plots.
- All wrench inputs are safety-clamped before sending.

Safety limits (defaults, editable in UI):
    Max |F| per axis:  20 N
    Max |T| per axis:   3 N·m
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import queue
import subprocess
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
_HERE    = pathlib.Path(__file__).parent
_REPO    = _HERE.parent
_DEFAULT_BINARY  = str(_REPO / "magpie_force_control" / "build" / "force_control_demo")
_DEFAULT_CONFIG  = str(_REPO / "magpie_force_control" / "src" / "config.yaml")

# ---------------------------------------------------------------------------
# Safety & UI constants
# ---------------------------------------------------------------------------
_MAX_F_N    = 20.0   # N per axis
_MAX_T_NM   = 3.0    # N·m per axis
_TELEM_BUF  = 300    # rolling telemetry samples kept for plots
_PLOT_MS    = 120    # matplotlib refresh interval
_STATUS_MS  = 80     # status text refresh interval

_FORCE_SCALE = 0.05  # 1 N → 5 cm in 3D plot
_TORQUE_SCALE = 0.12 # 1 N·m → arc radius scale

# Axis labels for display
_F_LABELS = ["Fx", "Fy", "Fz"]
_T_LABELS = ["Tx", "Ty", "Tz"]
_W_LABELS = _F_LABELS + _T_LABELS

# Colors per axis (F: red/green/blue, T: orange/cyan/magenta)
_F_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]
_T_COLORS = ["#e67e22", "#1abc9c", "#9b59b6"]
_W_COLORS = _F_COLORS + _T_COLORS

# Default admittance params
_DEF_STIFFNESS = [100.0, 100.0, 100.0,  1.0,  1.0,  1.0]
_DEF_DAMPING   = [  2.0,   2.0,   2.0,  0.2,  0.2,  0.2]
_DEF_INERTIA   = [  5.0,   5.0,   5.0, 0.005,0.005,0.005]
_DEF_STICTION  = [  0.0,   0.0,   0.0,  0.0,  0.0,  0.0]

# ---------------------------------------------------------------------------
# Telemetry sample dataclass (plain tuple for speed)
# ---------------------------------------------------------------------------
# Sample = (t, tcp[6], ref[6], ws[6], wd[6], q[6], T[6])
#           0   1:7    7:13   13:19  19:25  25:31 31:37


def _parse_telem(line: str) -> Optional[np.ndarray]:
    """Parse a TELEM: line into a flat float32 array of length 37."""
    # Format: TELEM: t=0.123 tcp=... ref=... ws=... wd=... q=... T=...
    try:
        parts = line[6:].strip().split()  # strip "TELEM:"
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
        arr[25:31]= (d["q"][:6]  if len(d.get("q", [])) >= 6  else [0]*6)
        arr[31:37]= (d["T"][:6]  if len(d.get("T", [])) >= 6  else [0]*6)
        return arr
    except Exception:
        return None


def _write_config_yaml(path: str, rate_hz: int,
                       stiffness: List[float], damping: List[float],
                       inertia: List[float], stiction: List[float],
                       max_spring_f: float, max_spring_t: float) -> None:
    """Write admittance config.yaml with correct dt matching rate."""
    dt = round(1.0 / rate_hz, 8)
    cfg = {
        "admittance_controller": {
            "dt": dt,
            "log_to_file": False,
            "log_file_path": "/tmp/admittance_controller.log",
            "alert_overrun": False,
            "compliance6d": {
                "stiffness": stiffness,
                "damping":   damping,
                "inertia":   inertia,
                "stiction":  stiction,
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

    Use as popup from grasp_viz:
        ui = ForceControlUI(parent=some_tk_widget)
        ui.run()          # opens Toplevel, non-blocking; parent keeps mainloop

    Use standalone:
        ui = ForceControlUI()
        ui.run()          # creates Tk root, blocking mainloop
    """

    def __init__(self,
                 parent: Optional[tk.Widget] = None,
                 robot_ip: str = "192.168.0.4",
                 ft_ip:    str = "192.168.0.3",
                 binary:   str = _DEFAULT_BINARY):
        self._parent    = parent
        self._robot_ip  = robot_ip
        self._ft_ip     = ft_ip
        self._binary    = binary

        # Subprocess
        self._proc: Optional[subprocess.Popen] = None
        self._reader_thread: Optional[threading.Thread] = None
        self._telem_queue: queue.Queue = queue.Queue()
        self._running_proc = False

        # Telemetry ring buffer
        self._buf: Deque[np.ndarray] = deque(maxlen=_TELEM_BUF)
        self._buf_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None  # most recent sample

        # Logging
        self._log_enabled  = False
        self._log_file: Optional[str] = None
        self._log_fh  = None
        self._log_writer = None

        # UI state
        self._pose_mode = "delta"   # "delta" or "absolute"
        self._show_adv  = False
        self._show_6plot= False
        self._show_temp = False
        self._show_des_force = True

        # Temp config path
        self._tmp_config = tempfile.NamedTemporaryFile(
            suffix=".yaml", prefix="magpie_fc_", delete=False)
        self._tmp_config.close()

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        """Open window and start mainloop (blocking for standalone)."""
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
    # UI layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._root.columnconfigure(0, weight=3)
        self._root.columnconfigure(1, weight=1)
        self._root.rowconfigure(0, weight=1)

        self._build_plot_panel()
        self._build_control_panel()

    # ── Left: plots ────────────────────────────────────────────────────

    def _build_plot_panel(self):
        frm = ttk.Frame(self._root)
        frm.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        frm.rowconfigure(0, weight=2)
        frm.rowconfigure(1, weight=1)
        frm.columnconfigure(0, weight=1)

        # Row 0: 3D EEF plot
        frm3d = ttk.LabelFrame(frm, text="End-Effector + Forces/Torques (EEF frame)")
        frm3d.grid(row=0, column=0, sticky="nsew")
        frm3d.rowconfigure(0, weight=1); frm3d.columnconfigure(0, weight=1)

        self._fig3d = plt.figure(figsize=(5.5, 4.0))
        self._ax3d  = self._fig3d.add_subplot(111, projection="3d")
        self._ax3d.set_xlabel("X"); self._ax3d.set_ylabel("Y"); self._ax3d.set_zlabel("Z")
        self._ax3d.set_title("EEF pose  (axes = TCP frame)", fontsize=9)
        self._canvas3d = FigureCanvasTkAgg(self._fig3d, master=frm3d)
        self._canvas3d.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Row 1: time-series plots (2-plot default, 6-plot optional)
        self._frmts = ttk.LabelFrame(frm, text="Time series")
        self._frmts.grid(row=1, column=0, sticky="nsew", pady=2)
        self._frmts.rowconfigure(0, weight=1); self._frmts.columnconfigure(0, weight=1)

        self._build_timeseries_2plot()

    def _build_timeseries_2plot(self):
        """2-panel view: forces + torques (combined desired vs sensed)."""
        self._fig_ts = plt.figure(figsize=(5.5, 2.5))
        gs = gridspec.GridSpec(1, 2, figure=self._fig_ts, left=0.08, right=0.98,
                               wspace=0.35, bottom=0.18, top=0.88)
        self._ax_f = self._fig_ts.add_subplot(gs[0])
        self._ax_t = self._fig_ts.add_subplot(gs[1])
        self._ax_f.set_title("Forces (N)", fontsize=8)
        self._ax_t.set_title("Torques (N·m)", fontsize=8)
        for ax in (self._ax_f, self._ax_t):
            ax.set_xlabel("t (s)", fontsize=7); ax.tick_params(labelsize=7)
            ax.axhline(0, color="#aaa", linewidth=0.5)
        self._canvas_ts = FigureCanvasTkAgg(self._fig_ts, master=self._frmts)
        self._canvas_ts.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_timeseries_6plot(self):
        """6-panel view: one subplot per axis."""
        self._fig_ts = plt.figure(figsize=(5.5, 2.8))
        gs = gridspec.GridSpec(2, 3, figure=self._fig_ts, left=0.08, right=0.98,
                               wspace=0.45, hspace=0.55, bottom=0.12, top=0.90)
        self._axes6 = []
        for idx in range(6):
            ax = self._fig_ts.add_subplot(gs[idx // 3, idx % 3])
            ax.set_title(_W_LABELS[idx], fontsize=8, color=_W_COLORS[idx])
            ax.tick_params(labelsize=6)
            ax.axhline(0, color="#bbb", linewidth=0.5)
            self._axes6.append(ax)
        self._canvas_ts = FigureCanvasTkAgg(self._fig_ts, master=self._frmts)
        self._canvas_ts.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    # ── Right: controls ────────────────────────────────────────────────

    def _build_control_panel(self):
        outer = ttk.Frame(self._root)
        outer.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._ctrl_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=self._ctrl_frame, anchor="nw")
        self._ctrl_frame.bind("<Configure>",
                              lambda e: canvas.configure(
                                  scrollregion=canvas.bbox("all")))

        r = 0
        r = self._build_connection_section(r)
        r = self._build_wrench_section(r)
        r = self._build_compliance_section(r)
        r = self._build_pose_section(r)
        r = self._build_execution_section(r)
        r = self._build_advanced_section(r)
        r = self._build_logging_section(r)
        r = self._build_view_toggles(r)
        r = self._build_wrench_display(r)

    def _sep(self, r):
        ttk.Separator(self._ctrl_frame, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3)
        return r + 1

    def _build_connection_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Connection ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(self._ctrl_frame, text="Robot IP:").grid(row=r, column=0, sticky="w")
        self._var_robot_ip = tk.StringVar(value=self._robot_ip)
        ttk.Entry(self._ctrl_frame, textvariable=self._var_robot_ip, width=16).grid(
            row=r, column=1, columnspan=2, sticky="ew"); r += 1

        ttk.Label(self._ctrl_frame, text="FT IP:").grid(row=r, column=0, sticky="w")
        self._var_ft_ip = tk.StringVar(value=self._ft_ip)
        ttk.Entry(self._ctrl_frame, textvariable=self._var_ft_ip, width=16).grid(
            row=r, column=1, columnspan=2, sticky="ew"); r += 1

        ttk.Label(self._ctrl_frame, text="Binary:").grid(row=r, column=0, sticky="w")
        self._var_binary = tk.StringVar(value=self._binary)
        ttk.Entry(self._ctrl_frame, textvariable=self._var_binary, width=16).grid(
            row=r, column=1, sticky="ew")
        tk.Button(self._ctrl_frame, text="…", width=2,
                  command=self._browse_binary).grid(row=r, column=2); r += 1

        return self._sep(r)

    def _build_wrench_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Desired Wrench ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(self._ctrl_frame, text=f"Max |F|: {_MAX_F_N} N   Max |T|: {_MAX_T_NM} N·m",
                  foreground="#888", font=("TkDefaultFont", 7)).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        # Safety overrides (per-axis max editable)
        self._var_max_f = tk.DoubleVar(value=_MAX_F_N)
        self._var_max_t = tk.DoubleVar(value=_MAX_T_NM)

        self._wrench_vars: List[tk.DoubleVar] = []
        labels = _F_LABELS + _T_LABELS
        units  = ["N"]*3 + ["N·m"]*3
        colors = _F_COLORS + _T_COLORS
        for i, (lbl, unit, col) in enumerate(zip(labels, units, colors)):
            ttk.Label(self._ctrl_frame, text=f"{lbl}:", foreground=col,
                      font=("TkDefaultFont", 9, "bold")).grid(
                row=r, column=0, sticky="e")
            v = tk.DoubleVar(value=0.0)
            ent = ttk.Entry(self._ctrl_frame, textvariable=v, width=8)
            ent.grid(row=r, column=1, sticky="ew", padx=2)
            ttk.Label(self._ctrl_frame, text=unit, foreground="#666").grid(
                row=r, column=2, sticky="w")
            self._wrench_vars.append(v)
            r += 1

        tk.Button(self._ctrl_frame, text="Send Wrench",
                  bg="#27ae60", fg="white",
                  command=self._send_wrench).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1

        # Safety clamp controls
        frm = ttk.Frame(self._ctrl_frame)
        frm.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        ttk.Label(frm, text="Safety max F (N):").pack(side="left")
        ttk.Entry(frm, textvariable=self._var_max_f, width=5).pack(side="left", padx=2)
        ttk.Label(frm, text="T (N·m):").pack(side="left")
        ttk.Entry(frm, textvariable=self._var_max_t, width=5).pack(side="left", padx=2)

        return self._sep(r)

    def _build_compliance_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Compliance Axes ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(self._ctrl_frame,
                  text="Checked = force-controlled, unchecked = position-held",
                  foreground="#888", font=("TkDefaultFont", 7)).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._axis_vars: List[tk.IntVar] = []
        frm = ttk.Frame(self._ctrl_frame)
        frm.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1
        for i, lbl in enumerate(_W_LABELS):
            v = tk.IntVar(value=1)
            cb = tk.Checkbutton(frm, text=lbl, variable=v,
                                foreground=_W_COLORS[i],
                                command=self._send_axis)
            cb.pack(side="left", padx=2)
            self._axis_vars.append(v)

        return self._sep(r)

    def _build_pose_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Pose Reference ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        # Mode toggle with large visual indicator
        self._pose_mode_var = tk.StringVar(value="delta")
        mode_frm = tk.Frame(self._ctrl_frame, bd=2, relief="groove")
        mode_frm.grid(row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1
        self._pose_mode_label = tk.Label(
            mode_frm, text="MODE: DELTA (relative)", width=28,
            bg="#2980b9", fg="white", font=("TkDefaultFont", 9, "bold"), padx=4)
        self._pose_mode_label.pack(side="left", fill="x", expand=True)
        tk.Radiobutton(mode_frm, text="Δ", variable=self._pose_mode_var,
                       value="delta", command=self._on_pose_mode).pack(side="left")
        tk.Radiobutton(mode_frm, text="ABS", variable=self._pose_mode_var,
                       value="absolute", command=self._on_pose_mode).pack(side="left")

        # XYZ inputs
        self._pose_vars: List[tk.DoubleVar] = []
        for lbl in ("dx (m)", "dy (m)", "dz (m)"):
            ttk.Label(self._ctrl_frame, text=lbl).grid(row=r, column=0, sticky="w")
            v = tk.DoubleVar(value=0.0)
            ttk.Entry(self._ctrl_frame, textvariable=v, width=10).grid(
                row=r, column=1, columnspan=2, sticky="ew", padx=2)
            self._pose_vars.append(v)
            r += 1

        tk.Button(self._ctrl_frame, text="Send Pose Ref",
                  command=self._send_pose_ref).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1

        return self._sep(r)

    def _build_execution_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Execution ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(self._ctrl_frame, text="Duration (s):").grid(row=r, column=0, sticky="w")
        self._var_duration = tk.DoubleVar(value=30.0)
        ttk.Entry(self._ctrl_frame, textvariable=self._var_duration, width=8).grid(
            row=r, column=1, columnspan=2, sticky="ew"); r += 1

        ttk.Label(self._ctrl_frame, text="Rate (Hz):").grid(row=r, column=0, sticky="w")
        self._var_rate = tk.IntVar(value=500)
        ttk.Entry(self._ctrl_frame, textvariable=self._var_rate, width=8).grid(
            row=r, column=1, columnspan=2, sticky="ew"); r += 1

        frm = ttk.Frame(self._ctrl_frame)
        frm.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self._btn_start = tk.Button(frm, text="▶  Start Force Control",
                                    bg="#27ae60", fg="white",
                                    font=("TkDefaultFont", 9, "bold"),
                                    command=self._start_controller)
        self._btn_start.pack(side="left", fill="x", expand=True, padx=1)
        self._btn_stop = tk.Button(frm, text="■  Stop",
                                   bg="#e74c3c", fg="white",
                                   state="disabled",
                                   command=self._stop_controller)
        self._btn_stop.pack(side="left", fill="x", expand=True, padx=1)

        self._status_var = tk.StringVar(value="Idle")
        tk.Label(self._ctrl_frame, textvariable=self._status_var,
                 font=("TkDefaultFont", 8), foreground="#555",
                 anchor="w", wraplength=200).grid(
            row=r, column=0, columnspan=3, sticky="ew"); r += 1

        return self._sep(r)

    def _build_advanced_section(self, r):
        hdr = tk.Frame(self._ctrl_frame)
        hdr.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self._adv_var = tk.IntVar(value=0)
        tk.Checkbutton(hdr, text="── Advanced Config ──",
                       variable=self._adv_var, command=self._toggle_advanced,
                       foreground="#555").pack(side="left")
        self._adv_frame = ttk.Frame(self._ctrl_frame)
        self._adv_frame.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1

        params = [
            ("Stiffness", "K", _DEF_STIFFNESS, "_stiff_vars"),
            ("Damping",   "D", _DEF_DAMPING,   "_damp_vars"),
            ("Inertia",   "M", _DEF_INERTIA,   "_iner_vars"),
        ]
        for name, sym, defs, attr in params:
            lbl = ttk.Label(self._adv_frame,
                            text=f"{name} ({sym}) [Fx,Fy,Fz,Tx,Ty,Tz]:")
            lbl.grid(sticky="w")
            row_frm = ttk.Frame(self._adv_frame)
            row_frm.grid(sticky="ew")
            vs = []
            for i, d in enumerate(defs):
                v = tk.DoubleVar(value=d)
                tk.Entry(row_frm, textvariable=v, width=7).pack(side="left", padx=1)
                vs.append(v)
            setattr(self, attr, vs)

        frm_sf = ttk.Frame(self._adv_frame)
        frm_sf.grid(sticky="ew")
        ttk.Label(frm_sf, text="Max spring F (N):").pack(side="left")
        self._var_spring_f = tk.DoubleVar(value=50.0)
        ttk.Entry(frm_sf, textvariable=self._var_spring_f, width=6).pack(side="left", padx=2)
        ttk.Label(frm_sf, text="T (N·m):").pack(side="left")
        self._var_spring_t = tk.DoubleVar(value=4.0)
        ttk.Entry(frm_sf, textvariable=self._var_spring_t, width=6).pack(side="left", padx=2)

        # Send stiffness/damping live while running
        tk.Button(self._adv_frame, text="Apply Stiffness & Damping (live)",
                  command=self._send_stiff_damp).grid(sticky="ew", pady=2)

        # Hide advanced frame initially
        self._adv_frame.grid_remove()

        return self._sep(r)

    def _build_logging_section(self, r):
        ttk.Label(self._ctrl_frame, text="── Logging ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        frm = ttk.Frame(self._ctrl_frame)
        frm.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self._log_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Enable CSV log", variable=self._log_var,
                       command=self._toggle_log).pack(side="left")
        self._log_path_var = tk.StringVar(value="(not set)")
        tk.Label(frm, textvariable=self._log_path_var, foreground="#555",
                 font=("TkDefaultFont", 7), wraplength=160).pack(side="left", padx=4)
        tk.Button(self._ctrl_frame, text="Choose log path…",
                  command=self._choose_log_path).grid(
            row=r, column=0, columnspan=3, sticky="ew"); r += 1

        return self._sep(r)

    def _build_view_toggles(self, r):
        ttk.Label(self._ctrl_frame, text="── Visualisation ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        frm = ttk.Frame(self._ctrl_frame)
        frm.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1

        self._des_var = tk.IntVar(value=1)
        tk.Checkbutton(frm, text="Show desired", variable=self._des_var).pack(side="left")
        self._six_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="6-plot", variable=self._six_var,
                       command=self._toggle_6plot).pack(side="left")
        self._tmp_var = tk.IntVar(value=0)
        tk.Checkbutton(frm, text="Temps", variable=self._tmp_var,
                       command=self._toggle_temp).pack(side="left")

        return self._sep(r)

    def _build_wrench_display(self, r):
        ttk.Label(self._ctrl_frame, text="── Live Wrench ──",
                  foreground="#555").grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        # Desired (static, colored header)
        self._des_display = tk.Text(self._ctrl_frame, height=3, width=28,
                                    font=("Courier", 8), state="disabled",
                                    bg="#f0f0f0", relief="flat")
        self._des_display.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        # Configure color tags
        for tag, color in [("green", "#1a7a1a"), ("red", "#c0392b"), ("black", "#111")]:
            self._des_display.tag_configure(tag, foreground=color)

        ttk.Label(self._ctrl_frame, text="Sensed:",
                  font=("TkDefaultFont", 8)).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        self._sensed_display = scrolledtext.ScrolledText(
            self._ctrl_frame, height=6, width=28,
            font=("Courier", 8), state="disabled", wrap="word")
        self._sensed_display.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1

        # Color tags for sensed display
        self._sensed_display.tag_configure("grn", foreground="#1a7a1a",
                                            font=("Courier", 8, "bold"))
        self._sensed_display.tag_configure("red", foreground="#c0392b",
                                            font=("Courier", 8, "bold"))
        self._sensed_display.tag_configure("blk", foreground="#111",
                                            font=("Courier", 8))

        return r

    # ------------------------------------------------------------------
    # Widget callbacks
    # ------------------------------------------------------------------

    def _browse_binary(self):
        path = tkfiledialog.askopenfilename(
            title="Select force_control_demo binary",
            filetypes=[("Executable", "*"), ("All", "*.*")])
        if path:
            self._var_binary.set(path)

    def _on_pose_mode(self):
        mode = self._pose_mode_var.get()
        if mode == "delta":
            self._pose_mode_label.config(text="MODE: DELTA (relative)",
                                          bg="#2980b9")
            for i, lbl in enumerate(("dx (m)", "dy (m)", "dz (m)")):
                # update labels — find by scanning children (simpler: just note mode)
                pass
        else:
            self._pose_mode_label.config(text="MODE: ABSOLUTE (world)",
                                          bg="#8e44ad")

    def _toggle_advanced(self):
        if self._adv_var.get():
            self._adv_frame.grid()
        else:
            self._adv_frame.grid_remove()

    def _toggle_log(self):
        if self._log_var.get() and self._log_file is None:
            # Auto-set a default log path
            self._log_file = str(pathlib.Path.home() /
                                  f"fc_log_{int(time.time())}.csv")
            self._log_path_var.set(self._log_file)
        self._log_enabled = bool(self._log_var.get())
        if self._log_enabled and self._log_fh is None and self._log_file:
            self._open_log()
        elif not self._log_enabled and self._log_fh is not None:
            self._close_log()

    def _choose_log_path(self):
        path = tkfiledialog.asksaveasfilename(
            title="Save log CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self._log_file = path
            self._log_path_var.set(path)
            if self._log_enabled:
                self._close_log()
                self._open_log()

    def _toggle_6plot(self):
        # Destroy current canvas and rebuild
        self._canvas_ts.get_tk_widget().destroy()
        plt.close(self._fig_ts)
        if self._six_var.get():
            self._build_timeseries_6plot()
        else:
            self._build_timeseries_2plot()

    def _toggle_temp(self):
        self._show_temp = bool(self._tmp_var.get())

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

    # ------------------------------------------------------------------
    # Commands → stdin
    # ------------------------------------------------------------------

    def _send_cmd(self, line: str):
        """Write a command line to the subprocess stdin (thread-safe)."""
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def _send_wrench(self):
        raw = [v.get() for v in self._wrench_vars]
        clamped = self._clamp_wrench(raw)
        # Warn user if any value was clamped
        if any(abs(c - r) > 1e-9 for c, r in zip(clamped, raw)):
            self._status_var.set(
                "⚠ Wrench clamped to safety limits before sending")
            for i, (v, c) in enumerate(zip(self._wrench_vars, clamped)):
                v.set(round(c, 4))
        self._send_cmd("WRENCH " + " ".join(f"{x:.4f}" for x in clamped))

    def _send_axis(self):
        mask = " ".join(str(v.get()) for v in self._axis_vars)
        self._send_cmd(f"AXIS {mask}")

    def _send_pose_ref(self):
        vals = [v.get() for v in self._pose_vars]
        mode = self._pose_mode_var.get()
        cmd = "RELD" if mode == "delta" else "REFA"
        self._send_cmd(f"{cmd} {vals[0]:.4f} {vals[1]:.4f} {vals[2]:.4f}")

    def _send_stiff_damp(self):
        stiff = " ".join(f"{v.get():.4f}" for v in self._stiff_vars)
        damp  = " ".join(f"{v.get():.4f}" for v in self._damp_vars)
        self._send_cmd(f"STIFF {stiff}")
        self._send_cmd(f"DAMP {damp}")

    # ------------------------------------------------------------------
    # Controller subprocess lifecycle
    # ------------------------------------------------------------------

    def _start_controller(self):
        if self._proc is not None:
            self._status_var.set("Already running — stop first.")
            return

        binary = self._var_binary.get()
        if not os.path.isfile(binary):
            self._status_var.set(f"Binary not found:\n{binary}")
            return

        # Write temp config
        rate = self._var_rate.get()
        stiff = [v.get() for v in self._stiff_vars] if hasattr(self, "_stiff_vars") else _DEF_STIFFNESS
        damp  = [v.get() for v in self._damp_vars]  if hasattr(self, "_damp_vars")  else _DEF_DAMPING
        iner  = [v.get() for v in self._iner_vars]  if hasattr(self, "_iner_vars")  else _DEF_INERTIA
        sf    = self._var_spring_f.get() if hasattr(self, "_var_spring_f") else 50.0
        st    = self._var_spring_t.get() if hasattr(self, "_var_spring_t") else 4.0
        _write_config_yaml(self._tmp_config.name, rate,
                           stiff, damp, iner, _DEF_STICTION, sf, st)

        # Get initial wrench from UI
        initial_wrench = self._clamp_wrench([v.get() for v in self._wrench_vars])

        # Build command
        duration = self._var_duration.get()
        robot_ip = self._var_robot_ip.get()
        ft_ip    = self._var_ft_ip.get()

        cmd = [
            binary,
            "--robot-ip", robot_ip,
            "--ft-ip",    ft_ip,
            "--rate",     str(rate),
            "--time",     str(int(duration)),
            "--config",   self._tmp_config.name,
            "--wrench",
        ] + [f"{x:.4f}" for x in initial_wrench]

        # Pose delta from UI (delta only at start)
        pose_vals = [v.get() for v in self._pose_vars]
        if any(abs(v) > 1e-9 for v in pose_vals) and \
                self._pose_mode_var.get() == "delta":
            # Pass as pose delta: x y z qw qx qy qz (quat = identity for no rotation)
            cmd += ["--pose",
                    f"{pose_vals[0]:.4f}", f"{pose_vals[1]:.4f}", f"{pose_vals[2]:.4f}",
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
            self._status_var.set(f"Failed to start: {e}")
            self._proc = None
            return

        self._running_proc = True
        self._reader_thread = threading.Thread(
            target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        self._btn_start.config(state="disabled")
        self._btn_stop.config(state="normal")
        self._status_var.set(f"Running  rate={rate}Hz  t={duration}s")

        # Open log if enabled
        if self._log_enabled and self._log_fh is None and self._log_file:
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
    # Stdout reader thread
    # ------------------------------------------------------------------

    def _reader_loop(self):
        """Reads stdout from the subprocess, parses TELEM lines."""
        while self._running_proc and self._proc:
            line = self._proc.stdout.readline()
            if not line:
                break
            line = line.rstrip()
            if line.startswith("TELEM:"):
                sample = _parse_telem(line)
                if sample is not None:
                    with self._buf_lock:
                        self._buf.append(sample)
                        self._latest = sample
                    if self._log_enabled and self._log_writer:
                        self._write_log(sample)
            else:
                # Non-telemetry lines go to status (errors, startup msgs)
                self._telem_queue.put(("log", line))

        self._running_proc = False
        self._telem_queue.put(("done", ""))

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _open_log(self):
        if self._log_file is None:
            return
        try:
            self._log_fh = open(self._log_file, "w", newline="")
            self._log_writer = csv.writer(self._log_fh)
            header = ["epoch_s", "t_s"] + \
                     [f"tcp_{x}" for x in ("x","y","z","rx","ry","rz")] + \
                     [f"ref_{x}" for x in ("x","y","z","rx","ry","rz")] + \
                     [f"ws_{l}" for l in _W_LABELS] + \
                     [f"wd_{l}" for l in _W_LABELS] + \
                     [f"q{i}" for i in range(6)] + \
                     [f"T{i}" for i in range(6)]
            self._log_writer.writerow(header)
            self._log_path_var.set(self._log_file)
        except Exception as e:
            self._status_var.set(f"Log error: {e}")

    def _write_log(self, s: np.ndarray):
        try:
            row = [f"{time.time():.4f}", f"{s[0]:.4f}"] + \
                  [f"{x:.5f}" for x in s[1:]]
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
    # Plot update (scheduled on Tk main thread)
    # ------------------------------------------------------------------

    def _schedule_plot_update(self):
        if self._root.winfo_exists():
            self._update_plots()
            self._root.after(_PLOT_MS, self._schedule_plot_update)

    def _schedule_status_update(self):
        if self._root.winfo_exists():
            self._update_status()
            self._root.after(_STATUS_MS, self._schedule_status_update)

    def _update_plots(self):
        with self._buf_lock:
            buf = list(self._buf)
            latest = self._latest

        if not buf:
            return

        arr = np.array(buf)          # (N, 37)
        ts   = arr[:, 0]
        tcp  = arr[:, 1:7]           # N×6
        ws   = arr[:, 13:19]         # sensed wrench
        wd   = arr[:, 19:25]         # desired wrench
        q    = arr[:, 25:31]
        T    = arr[:, 31:37]

        show_des = bool(self._des_var.get())

        self._draw_3d(latest, show_des)
        if self._six_var.get():
            self._draw_6plot(ts, ws, wd, show_des)
        else:
            self._draw_2plot(ts, ws, wd, show_des)

        if self._show_temp and hasattr(self, "_ax_temp"):
            self._draw_temp(ts, T)

        try:
            self._canvas3d.draw_idle()
            self._canvas_ts.draw_idle()
        except Exception:
            pass

    def _draw_3d(self, s: Optional[np.ndarray], show_des: bool):
        ax = self._ax3d
        ax.cla()
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
        ax.set_title("EEF pose  (arrows in EEF frame)", fontsize=8)

        if s is None:
            return

        tcp = s[1:7]   # [x,y,z,rx,ry,rz] axis-angle
        ws  = s[13:19] # sensed
        wd  = s[19:25] # desired

        pos = tcp[:3]

        # Reconstruct EEF rotation matrix from axis-angle
        aa = tcp[3:6]
        angle = np.linalg.norm(aa)
        if angle < 1e-9:
            R = np.eye(3)
        else:
            ax_vec = aa / angle
            c, s_ = np.cos(angle), np.sin(angle)
            K = np.array([[0, -ax_vec[2], ax_vec[1]],
                          [ax_vec[2], 0, -ax_vec[0]],
                          [-ax_vec[1], ax_vec[0], 0]])
            R = c * np.eye(3) + s_ * K + (1 - c) * np.outer(ax_vec, ax_vec)

        frame_len = 0.05  # 5 cm frame axes
        for i, col in enumerate(["r", "g", "b"]):
            ax.quiver(pos[0], pos[1], pos[2],
                      R[0, i] * frame_len,
                      R[1, i] * frame_len,
                      R[2, i] * frame_len,
                      color=col, arrow_length_ratio=0.3, linewidth=1.2)

        # Force/torque arrows in EEF frame (transform to world)
        def draw_force_arrows(wrench, alpha, lw, ls):
            for i in range(3):
                Fvec = R[:, i] * wrench[i] * _FORCE_SCALE
                ax.quiver(pos[0], pos[1], pos[2],
                          Fvec[0], Fvec[1], Fvec[2],
                          color=_F_COLORS[i], alpha=alpha, linewidth=lw,
                          linestyle=ls, arrow_length_ratio=0.25)

        def draw_torque_arcs(wrench, alpha, lw):
            # Draw small circular arc around each EEF axis
            for i in range(3):
                tau = wrench[3 + i]
                if abs(tau) < 1e-4:
                    continue
                axis_w = R[:, i] * np.sign(tau)
                r_arc  = max(0.01, min(0.08, abs(tau) * _TORQUE_SCALE))
                # Perpendicular vector
                perp = np.array([0, 0, 1]) if abs(axis_w[2]) < 0.9 else np.array([1, 0, 0])
                u = np.cross(axis_w, perp); u /= np.linalg.norm(u)
                v = np.cross(axis_w, u)
                th = np.linspace(0, 1.5 * np.pi, 30)
                arc = np.outer(np.cos(th), u) + np.outer(np.sin(th), v)
                arc = pos + r_arc * arc
                ax.plot(arc[:, 0], arc[:, 1], arc[:, 2],
                        color=_T_COLORS[i], alpha=alpha, linewidth=lw)

        # Sensed: thin, fully opaque, solid
        draw_force_arrows(ws, 1.0, 1.5, "solid")
        draw_torque_arcs(ws, 1.0, 1.5)

        # Desired: thick, semi-transparent, dashed
        if show_des:
            draw_force_arrows(wd, 0.45, 2.5, "dashed")
            draw_torque_arcs(wd, 0.45, 2.5)

        # EEF trajectory trail
        with self._buf_lock:
            trail = np.array([b[1:4] for b in list(self._buf)])
        if len(trail) > 2:
            ax.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                    "k--", alpha=0.3, linewidth=0.8)

        # Center view around EEF
        pad = 0.12
        ax.set_xlim(pos[0] - pad, pos[0] + pad)
        ax.set_ylim(pos[1] - pad, pos[1] + pad)
        ax.set_zlim(pos[2] - pad, pos[2] + pad)

    def _draw_2plot(self, ts, ws, wd, show_des):
        ax_f, ax_t = self._ax_f, self._ax_t
        ax_f.cla(); ax_t.cla()
        ax_f.set_title("Forces (N)", fontsize=8)
        ax_t.set_title("Torques (N·m)", fontsize=8)

        for i in range(3):
            ax_f.plot(ts, ws[:, i], color=_F_COLORS[i],
                      linewidth=1, alpha=0.9, label=_F_LABELS[i])
            ax_t.plot(ts, ws[:, 3+i], color=_T_COLORS[i],
                      linewidth=1, alpha=0.9, label=_T_LABELS[i])
            if show_des:
                ax_f.plot(ts, wd[:, i], color=_F_COLORS[i],
                          linewidth=2, alpha=0.35,
                          linestyle="--")
                ax_t.plot(ts, wd[:, 3+i], color=_T_COLORS[i],
                          linewidth=2, alpha=0.35,
                          linestyle="--")

        ax_f.axhline(0, color="#aaa", linewidth=0.5)
        ax_t.axhline(0, color="#aaa", linewidth=0.5)
        ax_f.legend(fontsize=6, loc="upper left", ncol=3)
        ax_t.legend(fontsize=6, loc="upper left", ncol=3)
        for ax in (ax_f, ax_t):
            ax.set_xlabel("t (s)", fontsize=7)
            ax.tick_params(labelsize=7)

    def _draw_6plot(self, ts, ws, wd, show_des):
        for i, ax in enumerate(self._axes6):
            ax.cla()
            ax.set_title(_W_LABELS[i], fontsize=8, color=_W_COLORS[i])
            ax.tick_params(labelsize=6)
            ax.axhline(0, color="#bbb", linewidth=0.5)
            unit = "N" if i < 3 else "N·m"
            ax.plot(ts, ws[:, i], color=_W_COLORS[i], linewidth=1)
            if show_des:
                ax.plot(ts, wd[:, i], color=_W_COLORS[i],
                        linewidth=2, alpha=0.35, linestyle="--")
            ax.set_ylabel(unit, fontsize=6)

    def _draw_temp(self, ts, T):
        if not hasattr(self, "_ax_temp"):
            return
        ax = self._ax_temp
        ax.cla()
        ax.set_title("Joint temps (°C)", fontsize=8)
        for j in range(6):
            ax.plot(ts, T[:, j], linewidth=1, label=f"J{j+1}")
        ax.legend(fontsize=6, loc="upper left", ncol=3)
        ax.set_xlabel("t (s)", fontsize=7)
        ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # Status/wrench text update
    # ------------------------------------------------------------------

    def _update_status(self):
        # Drain log queue (non-telemetry lines from subprocess)
        try:
            while True:
                kind, msg = self._telem_queue.get_nowait()
                if kind == "done":
                    self._cleanup_proc()
                elif kind == "log":
                    self._status_var.set(msg[:120])
        except queue.Empty:
            pass

        latest = self._latest
        if latest is None:
            return

        ws = latest[13:19]  # sensed
        wd = latest[19:25]  # desired

        # --- Desired wrench display (static above sensed stream) ---
        self._des_display.config(state="normal")
        self._des_display.delete("1.0", tk.END)
        self._des_display.insert(tk.END, "Desired:\n")
        for i, lbl in enumerate(_W_LABELS):
            unit = "N" if i < 3 else "Nm"
            self._des_display.insert(tk.END, f"  {lbl}: {wd[i]:+7.3f} {unit}\n")
        self._des_display.config(state="disabled")

        # --- Sensed wrench stream (colored per axis) ---
        self._sensed_display.config(state="normal")
        self._sensed_display.insert(tk.END, f"t={latest[0]:.2f}s  ")
        for i, lbl in enumerate(_W_LABELS):
            sv = ws[i]
            dv = wd[i]
            unit = "N" if i < 3 else "Nm"
            txt = f"{lbl}:{sv:+6.2f}{unit} "
            # Colour: compare abs(sensed) to abs(desired)
            adv = abs(dv)
            asv = abs(sv)
            if adv < 0.05:
                tag = "blk"   # desired near zero — don't colour
            elif asv >= 0.9 * adv and asv <= 1.1 * adv:
                tag = "grn"
            elif asv > 1.1 * adv:
                tag = "red"
            else:
                tag = "blk"
            self._sensed_display.insert(tk.END, txt, tag)
        self._sensed_display.insert(tk.END, "\n")
        # Keep last 60 lines
        lines = int(self._sensed_display.index("end-1c").split(".")[0])
        if lines > 60:
            self._sensed_display.delete("1.0", "10.0")
        self._sensed_display.see(tk.END)
        self._sensed_display.config(state="disabled")

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
        self._root.destroy()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Force control debug UI")
    p.add_argument("--robot-ip", default="192.168.0.4")
    p.add_argument("--ft-ip",    default="192.168.0.3")
    p.add_argument("--binary",   default=_DEFAULT_BINARY)
    return p.parse_args()


def main():
    args = _parse_args()
    ui = ForceControlUI(robot_ip=args.robot_ip, ft_ip=args.ft_ip,
                        binary=args.binary)
    ui.run()


if __name__ == "__main__":
    main()
