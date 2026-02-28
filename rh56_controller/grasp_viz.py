"""
grasp_viz.py — Interactive antipodal grasp geometry visualizer for Inspire RH56.

Usage:
    python -m rh56_controller.grasp_viz [--xml path/to/inspire_right.xml] [--rebuild]
    python -m rh56_controller.grasp_viz --port /dev/ttyUSB0       # sim2real
    python -m rh56_controller.grasp_viz --robot                   # UR5+hand viewer (IK)
    python -m rh56_controller.grasp_viz --robot --port /dev/ttyUSB0

    Run via uv to get mink IK support:
        uv run python -m rh56_controller.grasp_viz --robot

Controls:
    Radio buttons : select grasp mode (2-finger line / 3/4/5-finger plane / cylinder)
    Width slider  : target object width or diameter (mm)
    Z slider      : world-frame height of the grasp midplane (mm)
    X slider      : grasp X position in UR5 world frame (mm) [robot mode only]
    Y slider      : grasp Y position in UR5 world frame (mm) [robot mode only]
    MuJoCo button : open floating-hand viewer synced to current settings
    Send to Real  : checkbox — mirrors computed grasp pose to real hand

World frame convention (matplotlib):
    +Z up  (world gravity down)
    Hand hangs fingers-down above the grasp plane.
    The hand base origin appears above the fingertips.

Robot mode (--robot):
    Uses mink differential IK to drive a simulated UR5e arm.
    X/Y sliders position the grasp centroid in the UR5 world XY plane.
    UR5e workspace radius ≈ 850 mm; avoid XY near origin (base cylinder).
    The wrist_3_link TCP transform is stored in _wrist3_pos/_wrist3_mat
    for future real-robot deployment.
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
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mujoco
import mujoco.viewer

from .grasp_geometry import (
    InspireHandFK, ClosureGeometry, ClosureResult,
    CTRL_MAX, NON_THUMB_FINGERS, _DEFAULT_XML,
)

_HERE = pathlib.Path(__file__).parent.parent
_GRASP_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml")
_ROBOT_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "ur5_inspire.xml")
_RIGHT_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "inspire_right.xml")

# Actuator names in DOF order (same as real hand angle_set / angle_read)
_ACTUATOR_ORDER = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

# eeff site local position in hand base body frame (from inspire_right_ur5.xml)
_EEFF_LOCAL = np.array([0.070, 0.016, 0.155])

# TCP transform: wrist_3_link frame → hand base body
#   from ur5_inspire.xml: <body name="gripper_attachment" pos="0 0.156 0"
#                                quat="-0.707108 0.707108 0 0">
# Rotation: Rx(-90°)  Translation: [0, 0.156, 0] in wrist_3_link frame
_WRIST3_TO_HAND_POS  = np.array([0.0, 0.156, 0.0])       # metres
_WRIST3_TO_HAND_QUAT = np.array([-0.707108, 0.707108, 0.0, 0.0])  # [w,x,y,z]

# Default grasp XY position in robot mode (UR5 world frame, metres)
_DEFAULT_ROBOT_X = 0.40   # 400 mm forward from base
_DEFAULT_ROBOT_Y = 0.00   # centred laterally

# mink IK settings for robot viewer
_IK_DT        = 0.05   # IK integration step [s]
_IK_MAX_ITERS = 5      # IK iterations per visual frame
_IK_POS_THR   = 5e-3   # position convergence threshold [m]
_IK_ORI_THR   = 0.05   # orientation convergence threshold [rad]

# Colour palette per finger
FINGER_COLORS = {
    "thumb":  "#e74c3c",   # red
    "index":  "#e67e22",   # orange
    "middle": "#2ecc71",   # green
    "ring":   "#3498db",   # blue
    "pinky":  "#9b59b6",   # purple
}

# Viewer geom RGBA per finger (float32)
_FINGER_RGBA = {
    "thumb":  np.array([0.9, 0.3, 0.2, 0.9], dtype=np.float32),
    "index":  np.array([0.9, 0.55, 0.1, 0.9], dtype=np.float32),
    "middle": np.array([0.2, 0.8, 0.4, 0.9], dtype=np.float32),
    "ring":   np.array([0.2, 0.5, 0.9, 0.9], dtype=np.float32),
    "pinky":  np.array([0.6, 0.3, 0.9, 0.9], dtype=np.float32),
}

MODES = [
    "2-finger line",
    "3-finger plane",
    "4-finger plane",
    "5-finger plane",
    "cylinder",
]


# ---------------------------------------------------------------------------
# Small rotation matrix helpers
# ---------------------------------------------------------------------------
def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


# ---------------------------------------------------------------------------
# GraspViz
# ---------------------------------------------------------------------------
class GraspViz:
    def __init__(
        self,
        xml_path: str = _DEFAULT_XML,
        rebuild: bool = False,
        port: Optional[str] = None,
        robot_mode: bool = False,
        send_real: bool = False,
        mink_viz: bool = False,
    ):
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

        # Robot mode flag
        self._robot_mode = robot_mode

        # Grasp XY position in robot world frame (metres); Z is _grasp_z
        self._grasp_x = _DEFAULT_ROBOT_X if robot_mode else 0.0
        self._grasp_y = _DEFAULT_ROBOT_Y if robot_mode else 0.0

        # Wrist-3 TCP pose (updated by robot viewer thread for future real deployment)
        self._wrist3_pos: Optional[np.ndarray] = None
        self._wrist3_mat: Optional[np.ndarray] = None

        # Real hand connection
        self._hand = None
        self._send_real = send_real
        if port is not None:
            try:
                from .rh56_hand import RH56Hand
                self._hand = RH56Hand(port=port)
                print(f"[GraspViz] Connected to real hand on {port}")
            except Exception as e:
                print(f"[GraspViz] Could not connect to real hand: {e}")

        # Mink comparison viewer state
        self._mink_enabled  = mink_viz
        self._mink_planner  = None
        self._mink_result   = None
        self._mink_viewer_ctrl: Optional[np.ndarray] = None
        self._mink_viewer_thread: Optional[threading.Thread] = None
        self._mink_lock     = threading.Lock()
        self._mink_solve_event = threading.Event()
        self._mink_solve_thread: Optional[threading.Thread] = None

        if mink_viz:
            try:
                from .mink_grasp_planner import MinkGraspPlanner
                print("[GraspViz] Loading mink comparison planner...")
                self._mink_planner = MinkGraspPlanner(
                    _RIGHT_SCENE, dt=0.005, max_iters=150, conv_thr=5e-3,
                )
                self._mink_solve_thread = threading.Thread(
                    target=self._mink_solve_loop, daemon=True, name="mink-solve",
                )
                self._mink_solve_thread.start()
                print("[GraspViz] Mink planner ready.")
            except Exception as e:
                print(f"[GraspViz] Could not load mink planner: {e}")
                self._mink_enabled = False

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
            self._send_real_hand()
            if self._mink_enabled:
                self._mink_solve_event.set()

    # ------------------------------------------------------------------
    # Sim2Real: send computed grasp pose to real hand
    # ------------------------------------------------------------------
    def _send_real_hand(self):
        """Convert closure ctrl_values to real [0-1000] and call angle_set()."""
        if self._hand is None or not self._send_real or self._result is None:
            return
        r = self._result
        finger_ctrl = np.array([
            r.ctrl_values.get("pinky",           0.0),
            r.ctrl_values.get("ring",            0.0),
            r.ctrl_values.get("middle",          0.0),
            r.ctrl_values.get("index",           0.0),
            r.ctrl_values.get("thumb_proximal",  0.0),
            r.ctrl_values.get("thumb_yaw",       0.0),
        ])
        # Inverted convention: real = 1000 - round((ctrl-min)/(max-min) * 1000)
        ctrl_min = np.array([self.fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        ctrl_max = np.array([self.fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
        rng = ctrl_max - ctrl_min
        real_cmd = np.round(
            (1.0 - np.clip((finger_ctrl - ctrl_min) / np.where(rng > 0, rng, 1.0), 0.0, 1.0)) * 1000
        ).astype(int)
        try:
            self._hand.angle_set(real_cmd.tolist())
        except Exception as e:
            print(f"[GraspViz] angle_set failed: {e}")

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
        wbase = r.world_base(gz)
        if self._robot_mode:
            # Translate grasp to configured XY position in UR5 world frame
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])

        # Build 12-element ctrl vector:
        # [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
        #  pinky, ring, middle, index, thumb_prox, thumb_yaw]
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
        """Launch MuJoCo viewer in a background thread (floating or robot)."""
        if self._viewer_thread is not None and self._viewer_thread.is_alive():
            print("[GraspViz] Viewer already open.")
            return

        self._viewer_stop.clear()
        target = self._robot_viewer_loop if self._robot_mode else self._viewer_loop
        self._viewer_thread = threading.Thread(
            target=target, daemon=True, name="mujoco-viewer")
        self._viewer_thread.start()

    # ------------------------------------------------------------------
    # Floating-hand viewer (inspire_grasp_scene.xml)
    # ------------------------------------------------------------------
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

        # Non-thumb fingers — coupling from inspire_grasp_scene.xml <equality> polycoefs
        data.qpos[jm["pinky"]]        = pinky
        data.qpos[jm["pinky_inter"]]  = -0.15 + 1.1169 * pinky   # polycoef="-0.15 1.1169"
        data.qpos[jm["ring"]]         = ring
        data.qpos[jm["ring_inter"]]   = -0.15 + 1.1169 * ring
        data.qpos[jm["middle"]]       = middle
        data.qpos[jm["middle_inter"]] = -0.15 + 1.1169 * middle
        data.qpos[jm["index"]]        = index
        data.qpos[jm["index_inter"]]  = -0.05 + 1.1169 * index   # polycoef="-0.05 1.1169"

        # Thumb — coupling from inspire_grasp_scene.xml <equality> polycoefs
        data.qpos[jm["thumb_yaw"]]    = yaw
        data.qpos[jm["thumb_pitch"]]  = pitch
        data.qpos[jm["thumb_inter"]]  = 0.15 + 1.33 * pitch      # polycoef="0.15 1.33"
        data.qpos[jm["thumb_distal"]] = 0.15 + 0.66 * pitch      # polycoef="0.15 0.66"

        mujoco.mj_kinematics(model, data)

    def _viewer_loop(self):
        """Background thread: floating-hand passive viewer."""
        try:
            model = mujoco.MjModel.from_xml_path(_GRASP_SCENE)
            data  = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            jm = self._setup_jnt_map(model)

            self._push_viewer_ctrl()
            with self._viewer_lock:
                if self._viewer_ctrl is not None:
                    self._apply_qpos(jm, data, self._viewer_ctrl, model)

            with mujoco.viewer.launch_passive(model, data) as v:
                while v.is_running() and not self._viewer_stop.is_set():
                    with self._viewer_lock:
                        ctrl   = (self._viewer_ctrl.copy()
                                  if self._viewer_ctrl is not None else None)
                        result = self._result
                        gz     = self._grasp_z
                    if ctrl is not None:
                        self._apply_qpos(jm, data, ctrl, model)
                    self._add_viewer_geoms(v, result, gz)
                    v.sync()
                    time.sleep(0.033)
        except Exception as e:
            print(f"[GraspViz] Viewer error: {e}")

    # ------------------------------------------------------------------
    # Robot viewer (ur5_inspire.xml — mink differential IK)
    # ------------------------------------------------------------------
    def _robot_viewer_loop(self):
        """Background thread: UR5+hand viewer driven by mink differential IK.

        mink solves differential IK each visual frame to track the target eeff
        SE3 pose derived from the grasp geometry.  Joint position servos in the
        MuJoCo model then track the IK-computed joint angles.

        Only arm joints [shoulder_pan … wrist_3] are moved by IK.  Finger joints
        are set directly from the grasp geometry ctrl values.
        """
        try:
            import mink  # available via uv environment (pyproject.toml)
        except ImportError:
            print("[GraspViz] mink not found — run with 'uv run python -m rh56_controller.grasp_viz --robot'")
            return
        try:
            model = mujoco.MjModel.from_xml_path(_ROBOT_SCENE)
            data  = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)

            # --- mink Configuration (IK state, separate from sim data) ---
            configuration = mink.Configuration(model)

            eeff_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eeff")
            if eeff_id < 0:
                print("[GraspViz] WARNING: 'eeff' site not found in robot scene.")

            # IK tasks
            eeff_task = mink.FrameTask(
                frame_name="eeff",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=0.5,
                lm_damping=1e-6,
            )
            posture_task = mink.PostureTask(model, cost=1e-4)
            tasks = [eeff_task, posture_task]

            # Arm actuator / joint names
            _ARM_JNT = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
            _ARM_ACT = ["shoulder_pan", "shoulder_lift", "elbow",
                        "wrist_1", "wrist_2", "wrist_3"]
            _FNG_ACT = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

            arm_jnt_qposadr = [model.jnt_qposadr[model.joint(n).id] for n in _ARM_JNT]
            arm_ctrl_ids    = [model.actuator(n).id for n in _ARM_ACT]
            finger_ctrl_ids = [model.actuator(n).id for n in _FNG_ACT]

            # IK limits
            limits = [
                mink.ConfigurationLimit(model),
                mink.VelocityLimit(model, {n: np.pi for n in _ARM_JNT}),
            ]

            # Initialise arm to a sensible home pose (elbow-up reaching forward)
            data.qpos[model.jnt_qposadr[model.joint("shoulder_lift_joint").id]] = -np.pi / 2
            data.qpos[model.jnt_qposadr[model.joint("elbow_joint").id]]         =  np.pi / 2
            data.qpos[model.jnt_qposadr[model.joint("wrist_1_joint").id]]       = -np.pi / 2
            mujoco.mj_forward(model, data)

            # Set arm ctrl to match initial qpos (no startup jerk)
            for ctrl_id, qadr in zip(arm_ctrl_ids, arm_jnt_qposadr):
                data.ctrl[ctrl_id] = data.qpos[qadr]

            # Sync mink configuration from simulation state
            configuration.update(data.qpos)
            posture_task.set_target_from_configuration(configuration)

            # Wrist-3 force/torque site for TCP tracking (future real-robot use)
            wrist3_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, "wrist_ft")

            dt      = model.opt.timestep
            n_steps = max(1, round(0.033 / dt))  # physics steps per visual frame

            with mujoco.viewer.launch_passive(model, data) as v:
                while v.is_running() and not self._viewer_stop.is_set():
                    with self._viewer_lock:
                        ctrl   = (self._viewer_ctrl.copy()
                                  if self._viewer_ctrl is not None else None)
                        result = self._result
                        gz     = self._grasp_z

                    if ctrl is not None and eeff_id >= 0:
                        # --- Target SE3 for eeff site ---
                        # ctrl[0:3] = hand base world pos (already includes XY offset)
                        # ctrl[3:6] = [rot_x=π, rot_y=-tilt, rot_z=0]
                        R_hand     = _Rx(ctrl[3]) @ _Ry(ctrl[4])
                        target_pos = ctrl[0:3] + R_hand @ _EEFF_LOCAL
                        T_target   = np.eye(4)
                        T_target[:3, :3] = R_hand
                        T_target[:3,  3] = target_pos
                        eeff_task.set_target(mink.SE3.from_matrix(T_target))

                        # --- Differential IK iterations ---
                        for _ in range(_IK_MAX_ITERS):
                            try:
                                vel = mink.solve_ik(
                                    configuration, tasks, _IK_DT, "daqp",
                                    limits=limits)
                            except mink.NoSolutionFound:
                                break
                            configuration.integrate_inplace(vel, _IK_DT)
                            err = eeff_task.compute_error(configuration)
                            if (np.linalg.norm(err[:3]) < _IK_POS_THR and
                                    np.linalg.norm(err[3:]) < _IK_ORI_THR):
                                break

                        # --- Apply IK arm joint angles as position servo targets ---
                        for ctrl_id, qadr in zip(arm_ctrl_ids, arm_jnt_qposadr):
                            data.ctrl[ctrl_id] = configuration.q[qadr]

                        # --- Finger position targets (direct from grasp geometry) ---
                        for ctrl_id, val in zip(finger_ctrl_ids, ctrl[6:12]):
                            data.ctrl[ctrl_id] = val

                    # Step physics simulation
                    for _ in range(n_steps):
                        mujoco.mj_step(model, data)

                    # Sync mink IK state to actual simulation qpos
                    configuration.update(data.qpos)

                    # Store wrist-3 TCP pose for future real-robot deployment
                    if wrist3_site_id >= 0:
                        with self._viewer_lock:
                            self._wrist3_pos = data.site_xpos[wrist3_site_id].copy()
                            self._wrist3_mat = data.site_xmat[wrist3_site_id].reshape(3, 3).copy()

                    self._add_viewer_geoms(v, result, gz)
                    v.sync()
                    time.sleep(0.033)
        except Exception as e:
            print(f"[GraspViz] Robot viewer error: {e}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Mink comparison viewer
    # ------------------------------------------------------------------

    def _mink_solve_loop(self):
        """Background thread: re-solve with mink whenever mode/width changes."""
        while not self._viewer_stop.is_set():
            triggered = self._mink_solve_event.wait(timeout=0.1)
            if not triggered:
                continue
            self._mink_solve_event.clear()
            with self._viewer_lock:
                result = self._result
            if result is None or self._mink_planner is None:
                continue
            try:
                m_res = self._run_mink_for_result(result)
                with self._mink_lock:
                    self._mink_result = m_res
                self._push_mink_viewer_ctrl()
            except Exception as e:
                print(f"[Mink] Solve error: {e}")

    def _run_mink_for_result(self, result):
        """Run mink planner targeting the same tip positions as the custom result."""
        from .grasp_geometry import GRASP_FINGER_SETS, NON_THUMB_FINGERS
        mode = result.mode
        tips = result.tip_positions
        if mode == "2-finger line":
            return self._mink_planner.solve_line(
                result.width,
                thumb_target=tips["thumb"],
                index_target=tips["index"],
            )
        elif "plane" in mode:
            n = int(mode[0])
            fingers = GRASP_FINGER_SETS[n] + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_plane(result.width, targets)
        elif mode == "cylinder":
            fingers = NON_THUMB_FINGERS + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_cylinder(result.width, targets)
        raise ValueError(f"Unknown mode: {mode}")

    def _push_mink_viewer_ctrl(self):
        """Package mink finger ctrls + custom base pose for the mink viewer thread."""
        with self._viewer_lock:
            result = self._result
            gz     = self._grasp_z
        with self._mink_lock:
            m_res = self._mink_result
        if result is None or m_res is None:
            return
        wbase = result.world_base(gz)
        if self._robot_mode:
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])
        ctrl = np.array([
            wbase[0], wbase[1], wbase[2],
            np.pi, -result.base_tilt_y, 0.0,
            m_res.ctrl[0],   # pinky
            m_res.ctrl[1],   # ring
            m_res.ctrl[2],   # middle
            m_res.ctrl[3],   # index
            m_res.ctrl[4],   # thumb_proximal
            m_res.ctrl[5],   # thumb_yaw
        ], dtype=float)
        with self._mink_lock:
            self._mink_viewer_ctrl = ctrl

    def _launch_mink_viewer(self):
        """Launch the mink comparison viewer in a background thread."""
        if self._mink_viewer_thread is not None and self._mink_viewer_thread.is_alive():
            print("[GraspViz] Mink viewer already open.")
            return
        self._push_mink_viewer_ctrl()
        t = threading.Thread(
            target=self._mink_viewer_loop, daemon=True, name="mink-viewer",
        )
        self._mink_viewer_thread = t
        t.start()

    def _mink_viewer_loop(self):
        """Background thread: mink comparison floating-hand viewer.

        Uses the same inspire_grasp_scene.xml and hand pose as the custom viewer,
        but with finger joint angles from the mink IK solution.

        Overlay shows:
          • Translucent finger-coloured spheres  — custom planner targets
          • Cyan spheres                         — mink achieved tip positions
          • Yellow capsule lines                 — per-finger tip error vectors
        """
        try:
            model = mujoco.MjModel.from_xml_path(_GRASP_SCENE)
            data  = mujoco.MjData(model)
            mujoco.mj_resetData(model, data)
            jm = self._setup_jnt_map(model)

            # Seed initial pose
            self._push_mink_viewer_ctrl()
            with self._mink_lock:
                seed_ctrl = self._mink_viewer_ctrl
            if seed_ctrl is not None:
                self._apply_qpos(jm, data, seed_ctrl, model)

            with mujoco.viewer.launch_passive(model, data) as v:
                while v.is_running() and not self._viewer_stop.is_set():
                    with self._viewer_lock:
                        result = self._result
                        gz     = self._grasp_z
                    with self._mink_lock:
                        ctrl  = (self._mink_viewer_ctrl.copy()
                                 if self._mink_viewer_ctrl is not None else None)
                        m_res = self._mink_result
                    if ctrl is not None:
                        self._apply_qpos(jm, data, ctrl, model)
                    # First: draw standard overlay (custom target tips + mode geometry)
                    self._add_viewer_geoms(v, result, gz)
                    # Then: draw mink achieved tips on top
                    self._add_mink_overlay(v, result, gz, m_res)
                    v.sync()
                    time.sleep(0.033)
        except Exception as e:
            print(f"[GraspViz/Mink] Viewer error: {e}")

    def _add_mink_overlay(self, viewer, result, gz, m_res):
        """Append mink achieved-tip markers to the already-populated user_scn.

        Call AFTER _add_viewer_geoms (which resets ngeom).  Adds to the existing
        geom list without clearing it.

        Cyan spheres   = mink achieved tip positions (in world frame)
        Yellow lines   = error vectors from achieved → custom target (only if > 2 mm)
        """
        if result is None or m_res is None:
            return
        scn = viewer.user_scn

        def add_sphere(pos, radius, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                np.asarray(pos, dtype=np.float64),
                np.eye(3).flatten(),
                np.asarray(rgba, dtype=np.float32),
            )
            scn.ngeom += 1

        def add_line(p0, p1, rgba, width=0.0015):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9),
                np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                np.asarray(p0, dtype=np.float64),
                np.asarray(p1, dtype=np.float64),
            )
            scn.ngeom += 1

        # World transform: same tilt as custom planner, same base position.
        # Mink tip_positions are in hand base frame (same coord system as FK tables).
        from .grasp_geometry import ClosureResult as _CR
        R     = _CR._rot_matrix(result.base_tilt_y)
        mid_w = R @ result.midpoint
        base_w = np.array([-mid_w[0], -mid_w[1], gz - mid_w[2]])
        if self._robot_mode:
            base_w += np.array([self._grasp_x, self._grasp_y, 0.0])

        # Custom target world tips (same transform used by _add_viewer_geoms)
        wtips_custom = result.world_tips(gz)
        if self._robot_mode:
            xy_off = np.array([self._grasp_x, self._grasp_y, 0.0])
            wtips_custom = {f: p + xy_off for f, p in wtips_custom.items()}

        _CYAN = np.array([0.0, 0.9, 0.9, 1.0], dtype=np.float32)
        _YELLOW = np.array([1.0, 1.0, 0.0, 0.8], dtype=np.float32)

        for fname, pos_base in m_res.tip_positions.items():
            world_pos = R @ pos_base + base_w
            add_sphere(world_pos, radius=0.004, rgba=_CYAN)
            # Error line to custom target
            if fname in wtips_custom:
                err = float(np.linalg.norm(world_pos - wtips_custom[fname]))
                if err > 0.002:
                    add_line(world_pos, wtips_custom[fname], rgba=_YELLOW)

    # ------------------------------------------------------------------
    # MuJoCo viewer geometry overlay
    # ------------------------------------------------------------------
    def _add_viewer_geoms(
        self,
        viewer,
        result: Optional[ClosureResult],
        gz: float,
    ):
        """Populate viewer.user_scn with grasp geometry markers."""
        scn = viewer.user_scn
        scn.ngeom = 0
        if result is None:
            return

        wtips = result.world_tips(gz)

        # Helper: add line segment via mjv_connector
        def add_line(from_pt, to_pt, rgba, width=0.003):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3), np.zeros(3), np.zeros(9),
                np.asarray(rgba, dtype=np.float32))
            mujoco.mjv_connector(
                g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                np.asarray(from_pt, dtype=np.float64),
                np.asarray(to_pt,   dtype=np.float64))
            scn.ngeom += 1

        # Helper: add sphere
        def add_sphere(pos, radius=0.003, rgba=(0.9, 0.9, 0.2, 0.9)):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, radius, radius]),
                np.asarray(pos, dtype=np.float64),
                np.eye(3).flatten(),
                np.asarray(rgba, dtype=np.float32))
            scn.ngeom += 1

        # Fingertip spheres
        for fname, pos in wtips.items():
            add_sphere(pos, radius=0.002,
                       rgba=_FINGER_RGBA.get(fname, np.array([0.8, 0.8, 0.2, 0.9])))

        mode = result.mode

        if mode == "2-finger line":
            th = wtips.get("thumb")
            idx = wtips.get("index")
            if th is not None and idx is not None:
                add_line(th, idx, rgba=(1.0, 0.2, 0.2, 0.8), width=0.004)
                mid = (th + idx) / 2.0
                add_sphere(mid, radius=0.004, rgba=(1.0, 0.9, 0.1, 0.9))

        elif "plane" in mode:
            from .grasp_geometry import GRASP_FINGER_SETS
            n_fing = int(mode[0])
            fnames = GRASP_FINGER_SETS[n_fing]
            # Lines between adjacent non-thumb fingers
            for i in range(len(fnames) - 1):
                add_line(wtips[fnames[i]], wtips[fnames[i + 1]],
                         rgba=(0.2, 0.5, 0.9, 0.8), width=0.003)
            # Thumb connector to each finger
            th = wtips["thumb"]
            for f in fnames:
                add_line(th, wtips[f], rgba=(0.9, 0.3, 0.2, 0.5), width=0.002)
            # Midpoint sphere between thumb and non-thumb centroid
            nf_pts = np.array([wtips[f] for f in fnames])
            nf_cen = nf_pts.mean(axis=0)
            mid = (th + nf_cen) / 2.0
            add_sphere(mid, radius=0.004, rgba=(1.0, 0.9, 0.1, 0.9))

        elif mode == "cylinder":
            # Lines between all non-thumb fingers
            for i in range(len(NON_THUMB_FINGERS) - 1):
                add_line(wtips[NON_THUMB_FINGERS[i]], wtips[NON_THUMB_FINGERS[i + 1]],
                         rgba=(0.2, 0.5, 0.9, 0.8), width=0.003)
            # Thumb to centroid of non-thumb fingers
            nf_pts  = np.array([wtips[f] for f in NON_THUMB_FINGERS])
            nf_cen  = nf_pts.mean(axis=0)
            th      = wtips["thumb"]
            add_line(th, nf_cen, rgba=(0.9, 0.3, 0.2, 0.6), width=0.003)
            # Cylinder axis (centre of finger spread)
            add_sphere(nf_cen, radius=0.005, rgba=(0.2, 0.9, 0.5, 0.7))
            # Small cylinder arcs (8-segment half-circle at min/max Y of fingers)
            if result.cylinder_radius > 0:
                r_cyl = result.cylinder_radius
                cen_x = float(nf_cen[0])
                cen_z = float(nf_cen[2])
                y_vals = [float(nf_pts[:, 1].min()), float(nf_pts[:, 1].max())]
                angles = np.linspace(0, np.pi, 9)
                for y in y_vals:
                    for j in range(len(angles) - 1):
                        a0, a1 = angles[j], angles[j + 1]
                        p0 = np.array([cen_x + r_cyl * np.cos(a0),
                                       y,
                                       cen_z + r_cyl * np.sin(a0)])
                        p1 = np.array([cen_x + r_cyl * np.cos(a1),
                                       y,
                                       cen_z + r_cyl * np.sin(a1)])
                        add_line(p0, p1, rgba=(0.2, 0.7, 0.9, 0.5), width=0.002)

    # ------------------------------------------------------------------
    # matplotlib setup
    # ------------------------------------------------------------------
    def run(self):
        """Build and show the interactive matplotlib figure."""
        from matplotlib.widgets import TextBox  # local import to avoid polluting top-level

        fig = plt.figure(figsize=(14, 8))
        fig.suptitle("Inspire RH56 — Antipodal Grasp Geometry Planner", fontsize=12)

        # 3D axes: left 65%
        ax3d: Axes3D = fig.add_axes([0.02, 0.05, 0.60, 0.88], projection="3d")
        ax3d.set_xlabel("X  (closure direction)")
        ax3d.set_ylabel("Y  (finger spread)")
        ax3d.set_zlabel("Z  (world up)")

        self._ax3d = ax3d

        if self._robot_mode:
            # ---- Robot mode: compact layout with 4 sliders + textboxes ----
            # Slider axes: [left, bottom, width, height]
            # TextBox axes: immediately right of each slider
            _SL = 0.65   # left edge of slider
            _SW = 0.24   # slider width
            _TL = 0.90   # textbox left edge
            _TW = 0.07   # textbox width
            _SH = 0.025  # slider height

            # ---- Radio buttons ----
            ax_radio = fig.add_axes([0.65, 0.68, 0.14, 0.23])
            self._radio = RadioButtons(ax_radio, MODES, active=MODES.index(self._mode))
            self._radio.on_clicked(self._on_mode)

            def _make_slider_row(label, y_lbl, y_sl,
                                 vmin, vmax, vinit, vstep):
                ax_lbl = fig.add_axes([_SL, y_lbl, 0.30, 0.03])
                ax_lbl.axis("off")
                ax_lbl.text(0.0, 0.5, label, fontsize=9)
                ax_sl  = fig.add_axes([_SL, y_sl,  _SW,  _SH])
                sl     = Slider(ax_sl, "", vmin, vmax,
                                valinit=vinit, valstep=vstep)
                ax_tb  = fig.add_axes([_TL, y_sl, _TW, _SH])
                tb     = TextBox(ax_tb, "", initial=f"{vinit:.0f}")
                return sl, tb

            wmin_mm, wmax_mm = (x * 1000 for x in self._width_range)
            self._slider_w, self._tb_w = _make_slider_row(
                "Width / Diameter (mm):", 0.64, 0.60,
                wmin_mm, wmax_mm, self._width_m * 1000, 1.0)
            self._slider_z, self._tb_z = _make_slider_row(
                "Grasp Z (mm):", 0.54, 0.50,
                -200.0, 200.0, self._grasp_z * 1000, 5.0)
            self._slider_x, self._tb_x = _make_slider_row(
                "Grasp X (mm, UR5 world):", 0.44, 0.40,
                -850.0, 850.0, self._grasp_x * 1000, 5.0)
            self._slider_y, self._tb_y = _make_slider_row(
                "Grasp Y (mm, UR5 world):", 0.34, 0.30,
                -850.0, 850.0, self._grasp_y * 1000, 5.0)

            # Wire slider ↔ textbox for all four sliders
            def _wire(slider, tb, on_val_fn):
                slider.on_changed(lambda v: (on_val_fn(v),
                                             tb.set_val(f"{v:.0f}")))
                tb.on_submit(lambda s: slider.set_val(
                    float(s) if s.strip() else slider.val))

            _wire(self._slider_w, self._tb_w, self._on_width)
            _wire(self._slider_z, self._tb_z, self._on_z)
            _wire(self._slider_x, self._tb_x, self._on_x)
            _wire(self._slider_y, self._tb_y, self._on_y)

            info_bottom = 0.12
            btn_y       = 0.03

        else:
            # ---- Standard mode: original layout ----
            ax_radio = fig.add_axes([0.65, 0.60, 0.16, 0.30])
            self._radio = RadioButtons(ax_radio, MODES, active=MODES.index(self._mode))
            self._radio.on_clicked(self._on_mode)

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

            ax_ztxt = fig.add_axes([0.65, 0.37, 0.16, 0.04])
            ax_ztxt.axis("off")
            ax_ztxt.text(0.0, 0.5, "Grasp plane Z (mm):", fontsize=9)
            ax_zslide = fig.add_axes([0.65, 0.32, 0.28, 0.03])
            self._slider_z = Slider(
                ax_zslide, "", -200.0, 200.0,
                valinit=self._grasp_z * 1000, valstep=5.0,
            )
            self._slider_z.on_changed(self._on_z)

            info_bottom = 0.14
            btn_y       = 0.04

        # ---- Info text ----
        self._ax_info = fig.add_axes([0.65, info_bottom, 0.30, 0.16])
        self._ax_info.axis("off")
        self._info_text = self._ax_info.text(
            0.0, 1.0, "", fontsize=8,
            verticalalignment="top", family="monospace",
        )

        # ---- MuJoCo viewer button ----
        viewer_label = "Open Robot Viewer" if self._robot_mode else "Open MuJoCo Viewer"
        ax_btn = fig.add_axes([0.72, btn_y, 0.20, 0.05])
        self._btn_mujoco = Button(ax_btn, viewer_label)
        self._btn_mujoco.on_clicked(self._on_open_viewer)

        # ---- Mink comparison viewer button ----
        if self._mink_enabled:
            ax_btn_mink = fig.add_axes([0.72, btn_y + 0.07, 0.20, 0.05])
            self._btn_mink = Button(ax_btn_mink, "Open Mink Viewer")
            self._btn_mink.on_clicked(lambda _e: self._launch_mink_viewer())

        # ---- Send-to-real checkbox (only if hand is connected) ----
        if self._hand is not None:
            ax_real = fig.add_axes([0.65, btn_y, 0.06, 0.06])
            self._check_real = CheckButtons(
                ax_real, ["Send\nto Real"], [self._send_real])
            self._check_real.on_clicked(self._on_send_real)

        # Initial draw
        self._update_plot()
        plt.show()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    def _on_mode(self, label: str):
        self._mode = label
        n = int(label[0]) if label[0].isdigit() else 4
        wrange = self.closure.width_range(label, n_fingers=n)
        self._width_range = wrange
        wmin_mm, wmax_mm = wrange[0] * 1000, wrange[1] * 1000
        self._width_m = float(np.clip(self._width_m, wrange[0], wrange[1]))
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

    def _on_x(self, val: float):
        """Robot mode: set grasp X position in UR5 world frame."""
        self._grasp_x = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_y(self, val: float):
        """Robot mode: set grasp Y position in UR5 world frame."""
        self._grasp_y = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_open_viewer(self, event):
        self._launch_viewer()

    def _on_send_real(self, label: str):
        self._send_real = not self._send_real
        if self._send_real:
            self._send_real_hand()

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
        wtips = r.world_tips(gz)
        wbase = r.world_base(gz)

        # ---------- Fingertip dots ----------
        for fname, pos in wtips.items():
            col = FINGER_COLORS.get(fname, "gray")
            ax.scatter(*pos, color=col, s=60, zorder=5)
            ax.text(pos[0]+0.003, pos[1], pos[2]+0.003, fname[:3],
                    fontsize=7, color=col)

        # ---------- Mink comparison dots (cyan) ----------
        if self._mink_enabled:
            with self._mink_lock:
                m_res = self._mink_result
            if m_res is not None:
                from .grasp_geometry import ClosureResult as _CR
                R      = _CR._rot_matrix(r.base_tilt_y)
                mid_w  = R @ r.midpoint
                base_w = np.array([-mid_w[0], -mid_w[1], gz - mid_w[2]])
                for fname, pos_base in m_res.tip_positions.items():
                    wpos = R @ pos_base + base_w
                    ax.scatter(*wpos, color="cyan", s=30, marker="D",
                               zorder=6, alpha=0.85)
                    # Dashed error line to custom target
                    if fname in wtips:
                        err = float(np.linalg.norm(wpos - wtips[fname]))
                        if err > 0.002:
                            ax.plot(
                                [wpos[0], wtips[fname][0]],
                                [wpos[1], wtips[fname][1]],
                                [wpos[2], wtips[fname][2]],
                                "--", color="gold", lw=0.8, alpha=0.7,
                            )

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
        if self._robot_mode:
            lines.append(f"[ROBOT] X={self._grasp_x*1000:.0f} Y={self._grasp_y*1000:.0f} mm")
        if self._hand is not None:
            status = "ON" if self._send_real else "off"
            lines.append(f"Real hand: {status}")
        lines.append("")
        lines.append("Ctrl values (rad):")
        for k in ["index", "middle", "ring", "pinky",
                  "thumb_proximal", "thumb_yaw"]:
            v = r.ctrl_values.get(k, 0.0)
            if v > 0.001:
                lines.append(f"  {k[:12]:12s}: {v:.3f}")

        if self._mink_enabled:
            with self._mink_lock:
                m_res = self._mink_result
            if m_res is None:
                lines.append("\nMink: solving…")
            else:
                status = "✓" if m_res.converged else "✗"
                mean_err = float(np.mean(list(m_res.position_errors_m.values()))) * 1000
                lines.append(f"\nMink: {status} {m_res.n_iters} iters"
                             f" | err {mean_err:.1f} mm")

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

        fpts = np.array([wtips[f] for f in fnames])
        ax.plot(fpts[:, 0], fpts[:, 1], fpts[:, 2],
                "b-o", lw=2, ms=5, label="non-thumb row")

        th = wtips["thumb"]
        for f in fnames:
            fp = wtips[f]
            ax.plot([th[0], fp[0]], [th[1], fp[1]], [th[2], fp[2]],
                    "--", color="gray", lw=0.8, alpha=0.6)

        y_min = fpts[:, 1].min()
        y_max = fpts[:, 1].max()
        z_ref = gz
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

        mid_x = (x_nf + x_th) / 2
        ax.plot([mid_x, mid_x], [y_min - 0.01, y_max + 0.01], [z_ref, z_ref],
                "g:", lw=1.5, label="midplane")

    def _draw_cylinder_closure(self, ax, wtips, gz, r: ClosureResult):
        """Draw the cylinder closure as a wireframe cylinder."""
        from .grasp_geometry import NON_THUMB_FINGERS
        fpts = np.array([wtips[f] for f in NON_THUMB_FINGERS])

        centroid_x = fpts[:, 0].mean()
        centroid_z = fpts[:, 2].mean()
        radius = r.cylinder_radius

        y_min = fpts[:, 1].min() - 0.005
        y_max = fpts[:, 1].max() + 0.005
        theta = np.linspace(0, np.pi, 60)

        for y in [y_min, y_max]:
            xc = centroid_x + radius * np.cos(theta)
            zc = gz + (centroid_z - gz) + radius * np.sin(theta)
            ax.plot(xc, np.full_like(xc, y), zc,
                    "b-", lw=1.5, alpha=0.7)

        for ang in [0, np.pi]:
            xp = centroid_x + radius * np.cos(ang)
            zp = gz + (centroid_z - gz) + radius * np.sin(ang)
            ax.plot([xp, xp], [y_min, y_max], [zp, zp], "b-", lw=1.0, alpha=0.5)

        ax.plot([centroid_x, centroid_x], [y_min, y_max],
                [centroid_z, centroid_z], "g:", lw=1.5, label="cyl axis")

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
    parser.add_argument("--port", default=None,
                        help="Serial port for real hand (e.g. /dev/ttyUSB0). "
                             "Enables sim2real control.")
    parser.add_argument("--robot", action="store_true",
                        help="Use UR5+hand robot viewer instead of floating hand")
    parser.add_argument("--send-real", action="store_true",
                        help="Start with Send-to-Real enabled (requires --port)")
    parser.add_argument("--mink", action="store_true",
                        help="Enable mink IK comparison viewer (second MuJoCo window)")
    args = parser.parse_args()

    viz = GraspViz(
        xml_path=args.xml,
        rebuild=args.rebuild,
        port=args.port,
        robot_mode=args.robot,
        send_real=args.send_real,
        mink_viz=args.mink,
    )
    viz.run()


if __name__ == "__main__":
    main()
