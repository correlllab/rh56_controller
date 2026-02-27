#!/usr/bin/env python3
"""tools/hand_mirror.py  –  RH56 real-to-sim mirror and limit debugger.

Maps real hand angles into MuJoCo in real-time and shows a live bar-chart
comparison so that limit offsets and mapping errors are immediately visible.

Modes
-----
  default         – real → sim: real hand input drives sim joints
  --sim-only      – no hardware; sine-wave sweep of all joints for demo
  --sim-to-real   – sim → real + toggle: start with matplotlib sliders driving
                    the sim (and mirroring to real hand), with an in-window
                    toggle button to flip into Real→Sim mode (real hand drives
                    the sim for comparison / calibration)

Usage
-----
  python tools/hand_mirror.py --sim-only
  python tools/hand_mirror.py --port /dev/ttyUSB0
  python tools/hand_mirror.py --port /dev/ttyUSB0 --sim-to-real

DOF order (matches angle_set / angle_read / force_act):
  [0] pinky  [1] ring  [2] middle  [3] index  [4] thumb_bend  [5] thumb_yaw

Sign convention (INVERTED for all DOFs):
  real = 1000 − round(ctrl_rad / ctrl_max_rad × 1000)
  → real=0   (closed / adducted) ↔ ctrl=ctrl_max
  → real=1000 (open  / abducted) ↔ ctrl=0
"""

import argparse
import sys
import threading
import time
import pathlib

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

# ── Paths / constants ──────────────────────────────────────────────────────────

ROOT = pathlib.Path(__file__).parent.parent
SCENE_XML = str(ROOT / "h1_mujoco" / "inspire" / "inspire_scene.xml")

# DOF labels — same order as angle_set / angle_read and sim actuators
FINGER_NAMES = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_yaw"]
# Actuator names in the model (index-aligned with FINGER_NAMES)
_ACT_NAMES   = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

N = len(FINGER_NAMES)


def _load_ctrl_ranges(xml_path: str):
    """Read ctrl [min, max] arrays from a MuJoCo model file."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    mins, maxs = [], []
    for aname in _ACT_NAMES:
        aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
        if aid < 0:
            raise ValueError(f"Actuator '{aname}' not found in {xml_path}")
        mins.append(float(m.actuator_ctrlrange[aid, 0]))
        maxs.append(float(m.actuator_ctrlrange[aid, 1]))
    return np.array(mins), np.array(maxs)


# Initialise from default scene XML; overwritten in main() if --xml is passed.
SIM_CTRL_MIN, SIM_CTRL_MAX = _load_ctrl_ranges(SCENE_XML)

# Toggle state labels / colours
# True  = matplotlib sliders drive data.ctrl
# False = MuJoCo viewer sliders drive data.ctrl (we don't override it)
_TOGGLE_LABELS = {
    True:  "▶  Matplotlib Sliders → Real   (click to use MuJoCo Viewer Sliders)",
    False: "▶  MuJoCo Viewer Sliders → Real (click to use Matplotlib Sliders)",
}
_TOGGLE_COLORS = {True: "#f4a0a0", False: "#a0d4a0"}  # red-ish / green-ish


# ── Unit-conversion helpers ────────────────────────────────────────────────────

def real_to_ctrl(real: np.ndarray) -> np.ndarray:
    """Real hand [0–1000] → sim ctrl [rad].  INVERTED convention.
    real=1000 (open/abducted) → ctrl=ctrl_min; real=0 (closed) → ctrl=ctrl_max.
    """
    frac = 1.0 - np.clip(real / 1000.0, 0.0, 1.0)
    return SIM_CTRL_MIN + frac * (SIM_CTRL_MAX - SIM_CTRL_MIN)


def ctrl_to_real(ctrl: np.ndarray) -> np.ndarray:
    """Sim ctrl [rad] → real hand [0–1000]."""
    rng = SIM_CTRL_MAX - SIM_CTRL_MIN
    frac = np.where(rng > 0, np.clip((ctrl - SIM_CTRL_MIN) / rng, 0.0, 1.0), 0.0)
    return np.round((1.0 - frac) * 1000).astype(int)


def qpos_to_real(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Actual sim joint q-positions → real-hand 0–1000 scale."""
    out = np.zeros(N, dtype=int)
    for i in range(N):
        jnt_id = model.actuator_trnid[i, 0]
        q = data.qpos[model.jnt_qposadr[jnt_id]]
        rng = SIM_CTRL_MAX[i] - SIM_CTRL_MIN[i]
        frac = float(np.clip((q - SIM_CTRL_MIN[i]) / rng, 0.0, 1.0)) if rng > 0 else 0.0
        out[i] = int((1.0 - frac) * 1000)
    return out


# ── Thread-safe shared state ───────────────────────────────────────────────────

class SharedAngles:
    """Holds the latest real-hand angles (0–1000), updated by a background thread."""

    def __init__(self):
        self._lock = threading.Lock()
        self._angles = np.full(N, 500, dtype=int)

    def set(self, angles: np.ndarray):
        with self._lock:
            self._angles[:] = angles

    def get(self) -> np.ndarray:
        with self._lock:
            return self._angles.copy()


# ── Background threads ─────────────────────────────────────────────────────────

def reader_thread_fn(shared: SharedAngles, hand, stop: threading.Event):
    """Poll real hand at ~30 Hz."""
    while not stop.is_set():
        try:
            angles = hand.angle_read()
            if angles is not None and len(angles) == N:
                shared.set(np.array(angles, dtype=int))
        except Exception as exc:
            print(f"[reader] {exc}")
        time.sleep(1 / 30)


def sine_demo_fn(shared: SharedAngles, stop: threading.Event):
    """Sine-wave sweep of all joints (no hardware needed)."""
    t0 = time.time()
    while not stop.is_set():
        t = time.time() - t0
        angles = np.array([
            int(500 + 450 * np.sin(2 * np.pi * t / (5.0 + 0.7 * i)))
            for i in range(N)
        ], dtype=int)
        shared.set(angles)
        time.sleep(1 / 60)


# ── Matplotlib figure ──────────────────────────────────────────────────────────

_MODE_TITLES = {
    "real_to_sim": (
        "Real → Sim  |  blue = real hand input  |  orange = sim joint qpos\n"
        "Degree/rad values shown below bars"
    ),
    "sim_only": (
        "Sim-only demo  |  blue = sine-wave target  |  orange = sim joint qpos\n"
        "Degree/rad values shown below bars"
    ),
    "sim_to_real": (
        "Sim ↔ Real  |  blue = commanded target  |  orange = actual read-back\n"
        "Use toggle button to switch control direction"
    ),
}


def build_figure(mode: str):
    """Build grouped bar chart with optional sliders + toggle (sim-to-real only).

    Layout (sim-to-real, figure 10×8):
      [0.08, 0.55, 0.89, 0.40]  ← bar chart
      [0.15, 0.425, 0.72, 0.050] ← toggle button
      [0.15, 0.035+i*0.065, 0.72, 0.040] ← sliders (i=0..5, pinky→thumb_yaw)

    Returns
    -------
    fig, artists, sliders, slider_target, toggle_btn, mode_flag
      artists      = (bars_l, bars_r, txts_l, txts_r, deg_texts)
      sliders      = list[Slider] (length N in sim-to-real, else [])
      slider_target= np.ndarray (N,) float, real units [0–1000]
      toggle_btn   = Button | None
      mode_flag    = [bool]  True → Slider→Real,  False → Real→Sim
    """
    show_sliders = (mode == "sim_to_real")

    if show_sliders:
        fig = plt.figure(figsize=(10, 8))
        fig.suptitle(_MODE_TITLES[mode], fontsize=9)
        ax = fig.add_axes([0.08, 0.55, 0.89, 0.40])
    else:
        fig = plt.figure(figsize=(10, 4))
        fig.suptitle(_MODE_TITLES[mode], fontsize=9)
        ax = fig.add_axes([0.08, 0.22, 0.89, 0.65])

    x = np.arange(N)
    w = 0.35

    bars_l = ax.bar(x - w / 2, np.full(N, 500), w,
                    color="steelblue", alpha=0.85)
    bars_r = ax.bar(x + w / 2, np.full(N, 500), w,
                    color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(FINGER_NAMES, rotation=12, ha="right", fontsize=9)
    ax.set_ylim(-90, 1130)
    ax.set_ylabel("Angle  (0 = closed / adducted,  1000 = open / abducted)")
    ax.axhline(1000, color="red", lw=1.2, ls="--", alpha=0.6)
    ax.axhline(0,    color="gray", lw=1.0, ls="--", alpha=0.4)

    # Legend labels set dynamically; placeholders here
    bars_l[0].set_label("commanded (blue)")
    bars_r[0].set_label("actual (orange)")
    ax.legend(fontsize=8, loc="upper right")

    txts_l = [
        ax.text(b.get_x() + w / 2, 0, "", ha="center", va="bottom",
                fontsize=7, color="royalblue", fontweight="bold")
        for b in bars_l
    ]
    txts_r = [
        ax.text(b.get_x() + w / 2, 0, "", ha="center", va="bottom",
                fontsize=7, color="saddlebrown", fontweight="bold")
        for b in bars_r
    ]
    deg_texts = [
        ax.text(xi, -65, "", ha="center", va="top", fontsize=7, color="gray")
        for xi in x
    ]

    # ── Toggle button + sliders (sim-to-real only) ───────────────────────────
    sliders: list = []
    slider_target = np.full(N, 500.0)
    toggle_btn = None
    mode_flag = [True]   # True = Slider→Real

    clear_btn = None

    if show_sliders:
        # Button row between sliders and bar chart — toggle on the left,
        # "Clear Errors" on the right.
        ax_toggle = fig.add_axes([0.15, 0.425, 0.52, 0.050])
        toggle_btn = Button(
            ax_toggle,
            _TOGGLE_LABELS[True],
            color=_TOGGLE_COLORS[True],
            hovercolor="#e08080",
        )
        toggle_btn.label.set_fontsize(9)

        ax_clear = fig.add_axes([0.69, 0.425, 0.18, 0.050])
        clear_btn = Button(ax_clear, "Clear Errors",
                           color="#d0d0d0", hovercolor="#b0b0b0")
        clear_btn.label.set_fontsize(9)

        # Per-DOF sliders: pinky (i=0) at bottom, thumb_yaw (i=5) at top
        for i in range(N):
            ax_s = fig.add_axes([0.15, 0.035 + i * 0.065, 0.72, 0.040])
            s = Slider(
                ax_s, FINGER_NAMES[i], 0, 1000,
                valinit=500, valstep=1, color="steelblue",
            )
            s.label.set_fontsize(8)
            s.valtext.set_fontsize(8)

            def _cb(val, idx=i):
                slider_target[idx] = val

            s.on_changed(_cb)
            sliders.append(s)

    return fig, (bars_l, bars_r, txts_l, txts_r, deg_texts), sliders, slider_target, toggle_btn, clear_btn, mode_flag


def refresh_bars(artists, left_vals: np.ndarray, right_vals: np.ndarray,
                 ctrl_rad: np.ndarray):
    bars_l, bars_r, txts_l, txts_r, deg_texts = artists
    for i in range(N):
        lv, rv = int(left_vals[i]), int(right_vals[i])
        bars_l[i].set_height(lv)
        bars_r[i].set_height(rv)
        txts_l[i].set_y(lv + 12);  txts_l[i].set_text(str(lv))
        txts_r[i].set_y(rv + 12);  txts_r[i].set_text(str(rv))
        deg_texts[i].set_text(
            f"{np.degrees(ctrl_rad[i]):.1f}°\n{ctrl_rad[i]:.3f} rad")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="RH56 real-to-sim mirror and limit debugger")
    ap.add_argument("--port", default="/dev/ttyUSB0",
                    help="Serial port for real hand (default: /dev/ttyUSB0)")
    ap.add_argument("--sim-only", action="store_true",
                    help="No real hand — sine-wave demo")
    ap.add_argument("--sim-to-real", action="store_true",
                    help="Start in Slider→Real mode with in-window toggle")
    ap.add_argument("--xml", default=SCENE_XML,
                    help="Path to MuJoCo scene XML")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT))

    if args.sim_to_real:
        mode = "sim_to_real"
    elif args.sim_only:
        mode = "sim_only"
    else:
        mode = "real_to_sim"

    shared = SharedAngles()
    stop = threading.Event()

    # ── Real hand ──────────────────────────────────────────────────────────────
    hand = None
    if not args.sim_only:
        try:
            from rh56_controller.rh56_hand import RH56Hand
            hand = RH56Hand(args.port)
            print(f"[hand_mirror] Connected on {args.port}")
        except Exception as exc:
            print(f"[hand_mirror] Could not connect: {exc}")
            if mode == "real_to_sim":
                print("[hand_mirror] Falling back to --sim-only mode.")
                mode = "sim_only"
                args.sim_only = True

    # ── Background reader thread ───────────────────────────────────────────────
    if mode == "real_to_sim":
        bg = threading.Thread(
            target=reader_thread_fn, args=(shared, hand, stop), daemon=True)
        bg.start()
    elif mode == "sim_only":
        bg = threading.Thread(
            target=sine_demo_fn, args=(shared, stop), daemon=True)
        bg.start()
    elif mode == "sim_to_real" and hand is not None:
        bg = threading.Thread(
            target=reader_thread_fn, args=(shared, hand, stop), daemon=True)
        bg.start()

    # ── MuJoCo ────────────────────────────────────────────────────────────────
    print(f"[hand_mirror] Loading: {args.xml}")
    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    try:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    except Exception:
        pass

    # Refresh ctrl ranges from the actually-loaded model (may differ from default)
    global SIM_CTRL_MIN, SIM_CTRL_MAX
    SIM_CTRL_MIN, SIM_CTRL_MAX = _load_ctrl_ranges(args.xml)

    print("\n[hand_mirror] Actuator → joint mapping:")
    for i, name in enumerate(FINGER_NAMES):
        jnt_id = model.actuator_trnid[i, 0]
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
        jnt_range = model.jnt_range[jnt_id]
        print(f"  [{i}] {name:12s}  ctrl=[{SIM_CTRL_MIN[i]:.3f}, {SIM_CTRL_MAX[i]:.3f}] rad"
              f"  joint='{jnt_name}'  range=[{jnt_range[0]:.3f}, {jnt_range[1]:.3f}]")

    # ── Matplotlib ─────────────────────────────────────────────────────────────
    plt.ion()
    fig, artists, sliders, slider_target, toggle_btn, clear_btn, mode_flag = build_figure(mode)

    # Wire up the toggle callback (needs access to shared, sliders, mode_flag)
    if toggle_btn is not None:
        def _on_toggle(_):
            mode_flag[0] = not mode_flag[0]
            s2r = mode_flag[0]
            toggle_btn.label.set_text(_TOGGLE_LABELS[s2r])
            toggle_btn.ax.set_facecolor(_TOGGLE_COLORS[s2r])
            toggle_btn.hovercolor = "#e08080" if s2r else "#70c070"
            if s2r:
                # Switching back to matplotlib sliders: snap them to whatever
                # the viewer was last commanding so the hand doesn't jump.
                cur = ctrl_to_real(data.ctrl)
                for i, s in enumerate(sliders):
                    s.set_val(float(cur[i]))  # fires _cb → updates slider_target
                print("[hand_mirror] Matplotlib sliders active.")
            else:
                print("[hand_mirror] MuJoCo viewer sliders active — "
                      "open Control panel in viewer (press Ctrl).")
            fig.canvas.draw_idle()

        toggle_btn.on_clicked(_on_toggle)

    if clear_btn is not None:
        def _on_clear(_):
            if hand is not None:
                try:
                    hand.clear_errors()
                    print("[hand_mirror] clear_errors() called.")
                except Exception as exc:
                    print(f"[hand_mirror] clear_errors() failed: {exc}")

        clear_btn.on_clicked(_on_clear)

    plt.show(block=False)

    # ── Main sim loop ──────────────────────────────────────────────────────────
    SIM_DT = model.opt.timestep
    STEPS_PER_FRAME = max(1, round((1 / 60) / SIM_DT))
    step = 0
    SEND_EVERY = 3
    DRAW_EVERY = 6

    print(f"\n[hand_mirror] timestep={SIM_DT*1000:.1f} ms  "
          f"steps_per_frame={STEPS_PER_FRAME}")
    if mode == "sim_to_real":
        print("[hand_mirror] Matplotlib sliders active — drag sliders to command.")
        print("[hand_mirror] Click the toggle button to switch to MuJoCo viewer sliders.")
        print("  (MuJoCo viewer Control panel: press Ctrl inside the viewer window)")
    print("[hand_mirror] Running — close the MuJoCo viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            if mode == "sim_to_real":
                s2r = mode_flag[0]

                if s2r:
                    # ── Matplotlib sliders → Real ────────────────────────────
                    # Overwrite data.ctrl from slider values before stepping.
                    data.ctrl[:] = real_to_ctrl(slider_target)
                    for _ in range(STEPS_PER_FRAME):
                        mujoco.mj_step(model, data)
                    viewer.sync()

                    if hand is not None and step % SEND_EVERY == 0:
                        hand.angle_set(ctrl_to_real(data.ctrl).tolist())

                    left      = slider_target.astype(int)   # what we commanded
                    right     = shared.get()                # real hand read-back
                    ctrl_rad  = data.ctrl.copy()

                else:
                    # ── MuJoCo viewer sliders → Real ─────────────────────────
                    # Don't touch data.ctrl — the passive viewer already wrote
                    # it from the Control panel sliders.  Just step and send.
                    for _ in range(STEPS_PER_FRAME):
                        mujoco.mj_step(model, data)
                    viewer.sync()

                    if hand is not None and step % SEND_EVERY == 0:
                        hand.angle_set(ctrl_to_real(data.ctrl).tolist())

                    left      = ctrl_to_real(data.ctrl)     # viewer commanded
                    right     = shared.get()                # real hand read-back
                    ctrl_rad  = data.ctrl.copy()

            elif mode == "real_to_sim":
                target_real = shared.get()
                data.ctrl[:] = real_to_ctrl(target_real)
                for _ in range(STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                viewer.sync()

                left     = target_real
                right    = qpos_to_real(model, data)
                ctrl_rad = data.ctrl.copy()

            else:
                # sim_only: sine wave
                target_real = shared.get()
                data.ctrl[:] = real_to_ctrl(target_real)
                for _ in range(STEPS_PER_FRAME):
                    mujoco.mj_step(model, data)
                viewer.sync()

                left     = target_real
                right    = qpos_to_real(model, data)
                ctrl_rad = data.ctrl.copy()

            if step % DRAW_EVERY == 0:
                refresh_bars(artists, left, right, ctrl_rad)
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            step += 1
            time.sleep(1 / 60)

    stop.set()
    plt.close("all")
    print("[hand_mirror] Done.")


if __name__ == "__main__":
    main()
