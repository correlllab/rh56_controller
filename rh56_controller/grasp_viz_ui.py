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
        z_max = 400.0 if self._robot_mode else 200.0
        z_min = 0.0   if self._robot_mode else -200.0
        r = self._add_slider_row(outer, r, "Grasp Z (mm):",
                                 z_min, z_max, self._grasp_z * 1000, 5.0,
                                 "_var_z", "_sl_z", "_ent_z", self._on_z)
        if self._robot_mode:
            btn_up = tk.Button(outer, text="+10cm", command=self._on_move_up)
            btn_up.grid(row=r - 1, column=3, padx=2)

        # X, Y (robot mode only)
        if self._robot_mode:
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
        ent.bind("<Return>",   lambda e, s=sl: s.set(
            float(ent.get() or var.get())))
        ent.bind("<FocusOut>", lambda e, s=sl: s.set(
            float(ent.get() or var.get())))
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

        # Send to Real checkbox
        if self._hand is not None:
            self._send_real_var = tk.BooleanVar(value=self._send_real)
            ttk.Checkbutton(outer, text="Send to Real",
                            variable=self._send_real_var,
                            command=self._on_send_real).grid(
                row=r, column=0, columnspan=2, sticky="w"); r += 1

        # ── Real robot panel ──
        if self._real_robot_mode:
            self._build_real_robot_panel(outer, r)
            return   # real robot panel manages its own rows

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
        tk.Button(parent, text="Set Pose",
                  command=self._on_set_pose_from_robot,
                  state="normal" if _arm_ok else "disabled").grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

        self._btn_sendarm = tk.Button(parent, text="Send Arm",
                                      command=self._on_send_arm,
                                      state="normal" if _arm_ok else "disabled")
        self._btn_sendarm.grid(row=r, column=0, sticky="ew", padx=2, pady=1)
        tk.Button(parent, text="Sim Traj",
                  command=self._on_simulate_trajectory,
                  state="normal" if _arm_ok else "disabled").grid(
            row=r, column=1, sticky="ew", padx=2, pady=1); r += 1

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
                                    bg="#2ecc71", fg="white",
                                    command=self._on_grasp,
                                    state="normal" if _arm_ok else "disabled")
        self._btn_grasp.grid(row=r, column=0, columnspan=2, sticky="ew",
                             padx=2, pady=4); r += 1

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
        self._update_plot()

    # ------------------------------------------------------------------
    # Slider / control callbacks
    # ------------------------------------------------------------------
    def _on_mode_radio(self):
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
        z_max_mm = 400.0 if self._robot_mode else 200.0
        new_z_mm = min(z_max_mm, self._grasp_z * 1000.0 + 100.0)
        self._grasp_z = new_z_mm / 1000.0
        self._sl_z.set(new_z_mm)
        self._ent_z.delete(0, "end")
        self._ent_z.insert(0, f"{new_z_mm:.1f}")
        self._push_viewer_ctrl()
        self._schedule_plot_only()  # arm-pose only — do not re-send fingers

    def _on_send_real(self):
        self._send_real = self._send_real_var.get()
        self._grasp_hand_locked = False  # explicit toggle always clears the post-grasp lock
        if self._send_real:
            self._send_real_hand()

    def _on_strategy_radio(self):
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
            self._btn_teach.config(text="TEACH MODE ACTIVE", bg="#ff4444", fg="white")

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
        self._real_tracking.value = 0
        self._sim_grasp_t.value   = 0.0
        self._launch_robot_viewer_ours()
        self._push_viewer_ctrl()
        self._update_status("Sim: arm moving to pose, fingers will close...")

        self._sim_grasp_gen += 1
        gen = self._sim_grasp_gen

        def _animate():
            time.sleep(2.0)
            n_steps = 40
            for i in range(n_steps + 1):
                if self._sim_grasp_gen != gen:
                    return
                self._sim_grasp_t.value = i / n_steps
                time.sleep(0.05)
            self._status_queue.put("Sim grasp complete.")

        threading.Thread(target=_animate, daemon=True, name="sim-grasp").start()

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
        logger = GraspLogger(log_name)
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
        self._hand.speed_set([1000] * 6)
        self._hand.force_set([1000] * 6)
        time.sleep(0.25)

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
