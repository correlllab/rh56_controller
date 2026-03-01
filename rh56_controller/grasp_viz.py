"""
grasp_viz.py — Interactive antipodal grasp geometry visualizer for Inspire RH56.

Usage:
    python -m rh56_controller.grasp_viz [--xml path/to/inspire_right.xml] [--rebuild]
    python -m rh56_controller.grasp_viz --port /dev/ttyUSB0            # sim2real (hand only)
    python -m rh56_controller.grasp_viz --robot                         # UR5+hand sim viewer (IK)
    python -m rh56_controller.grasp_viz --robot --port /dev/ttyUSB0    # sim + real hand
    python -m rh56_controller.grasp_viz --real-robot --ur5-ip 192.168.0.4 \\
                                         --port /dev/ttyUSB0            # full real robot control

    Run via uv to get mink IK support:
        uv run python -m rh56_controller.grasp_viz
        uv run python -m rh56_controller.grasp_viz --robot
        uv run python -m rh56_controller.grasp_viz --real-robot --ur5-ip 192.168.0.4

Controls (common):
    Radio buttons : select grasp mode (2-finger line / 3/4/5-finger plane / cylinder)
    Width slider  : target object width or diameter (mm)
    Z slider      : world-frame height of the grasp midplane (mm)
    Plane Rx/Ry/Rz: extra plane orientation (degrees)
    Hand: Ours    : open floating-hand viewer (custom planner)
    Hand: Mink    : open floating-hand viewer (mink IK planner)
    Send to Real  : checkbox — mirrors computed grasp pose to real hand

Robot sim mode (--robot):
    Uses mink differential IK to drive a simulated UR5e arm.
    X/Y sliders position the grasp centroid in the UR5 world XY plane.
    Robot: Ours / Robot: Mink — open UR5+hand sim viewer.
    UR5e workspace radius ≈ 850 mm; avoid XY near origin (base cylinder).

Real robot mode (--real-robot):
    Implies --robot.  Adds real UR5 arm control panel below sim viewer buttons.
    Requires magpie_control (installed separately; see README_REAL.md).

    Real Robot Control panel:
        Teach Mode          — toggle teach mode (robot guidable by hand);
                              red indicator; all arm commands blocked in teach mode.
        Set Pose from Robot — read current TCP pose → update X/Y/Z/plane sliders.
        Send to Robot (Arm) — moveL to current planned TCP pose.
        Simulate Trajectory — (re)launch the robot sim viewer showing planned pose.
        Grasp Strategy      — Naive / Plan / Thumb Reflex (radio selector).
        Force (N)           — target grasp force (0 = disabled).
        Step (mm)           — chunk size for synchronized arm+finger trajectory.
        GRASP!              — execute the selected grasp strategy.

    Cylinder mode < 71 mm diameter: power grasp disabled (large discontinuity
    at the thumb-yaw switch point); GRASP! button greyed out with warning.

Viewer architecture:
    Each MuJoCo viewer (hand or robot) runs in its own subprocess via
    multiprocessing.Process, avoiding GLFW multi-thread crashes.  State is
    communicated through shared multiprocessing.Array objects updated from
    the main process each time the grasp parameters change.

    In real-robot mode, a separate _real_q_arr (6-element shared array) holds
    the last-known UR5 joint angles.  After each arm move the executor calls
    arm.snapshot_joints(_real_q_arr) to update it, and the robot viewer
    subprocess reads it (when _real_tracking=1) to show the real robot state.
"""

import argparse
import multiprocessing
import pathlib
import queue
import threading
import time
from typing import Optional, Dict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RadioButtons, Slider, Button, CheckButtons
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mujoco
import mujoco.viewer

from .grasp_geometry import (
    InspireHandFK, ClosureGeometry, ClosureResult,
    CTRL_MAX, NON_THUMB_FINGERS, _DEFAULT_XML, GRASP_FINGER_SETS,
)

_HERE = pathlib.Path(__file__).parent.parent
_GRASP_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml")
_ROBOT_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "ur5_inspire.xml")
_RIGHT_SCENE  = str(_HERE / "h1_mujoco" / "inspire" / "inspire_right.xml")

# Actuator names in DOF order (same as real hand angle_set / angle_read)
_ACTUATOR_ORDER = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

# eeff site local position in hand base body frame (from ur5_inspire.xml)
_EEFF_LOCAL = np.array([0.070, 0.016, 0.155])

# TCP transform: wrist_3_link frame → hand base body
_WRIST3_TO_HAND_POS  = np.array([0.0, 0.156, 0.0])
_WRIST3_TO_HAND_QUAT = np.array([-0.707108, 0.707108, 0.0, 0.0])

# Default grasp XY position in robot mode (UR5 world frame, metres)
_DEFAULT_ROBOT_X = 0.40
_DEFAULT_ROBOT_Y = 0.00

# mink IK settings for robot viewer
_IK_DT        = 0.05
_IK_MAX_ITERS = 5
_IK_POS_THR   = 5e-3
_IK_ORI_THR   = 0.05

# Colour palette per finger
FINGER_COLORS = {
    "thumb":  "#e74c3c",
    "index":  "#e67e22",
    "middle": "#2ecc71",
    "ring":   "#3498db",
    "pinky":  "#9b59b6",
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


# ===========================================================================
# Viewer subprocess shared-memory layout and worker functions
#
# These module-level functions run inside separate multiprocessing.Process
# instances — one per open MuJoCo viewer window.  Using separate processes
# avoids GLFW multi-thread crashes that occur when two passive viewers are
# launched from different threads of the same process.
#
# Shared arrays (multiprocessing.Array('d', ...)):
#
#   ctrl_arr  [12]  :  [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
#                       pinky, ring, middle, index, thumb_pitch, thumb_yaw]
#
#   state_arr [20]  :  [gz, mode_idx, cyl_radius, robot_x, robot_y,
#                       tip_thumb(3), tip_index(3), tip_middle(3),
#                       tip_ring(3),  tip_pinky(3)]
#                       (tip positions are world-frame; NaN if not active)
# ===========================================================================

_VIEWER_CTRL_LEN  = 12
_VIEWER_STATE_LEN = 20
_VIEWER_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Active fingers per mode index (matches MODES list order)
_VIEWER_MODE_ACTIVE = {
    0: ["thumb", "index"],
    1: ["thumb", "index", "middle"],
    2: ["thumb", "index", "middle", "ring"],
    3: ["thumb", "index", "middle", "ring", "pinky"],
    4: ["thumb", "index", "middle", "ring", "pinky"],
}


def _worker_jnt_map(model: mujoco.MjModel) -> dict:
    """Build qpos address map for grasp-scene joints (worker-process version)."""
    def jadr(name):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Joint '{name}' not found in model")
        return int(model.jnt_qposadr[jid])
    return {
        "pos_x": jadr("right_pos_x"), "pos_y": jadr("right_pos_y"),
        "pos_z": jadr("right_pos_z"), "rot_x": jadr("right_rot_x"),
        "rot_y": jadr("right_rot_y"), "rot_z": jadr("right_rot_z"),
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


def _worker_apply_qpos(jm: dict, data: mujoco.MjData, ctrl: np.ndarray,
                        model: mujoco.MjModel) -> None:
    """Apply 12-element ctrl vector to qpos with joint coupling (worker version)."""
    pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = ctrl[0:6]
    pinky, ring, middle, index, pitch, yaw     = ctrl[6:12]

    data.qpos[jm["pos_x"]] = pos_x;  data.qpos[jm["pos_y"]] = pos_y
    data.qpos[jm["pos_z"]] = pos_z;  data.qpos[jm["rot_x"]] = rot_x
    data.qpos[jm["rot_y"]] = rot_y;  data.qpos[jm["rot_z"]] = rot_z

    data.qpos[jm["pinky"]]        = pinky
    data.qpos[jm["pinky_inter"]]  = -0.15 + 1.1169 * pinky
    data.qpos[jm["ring"]]         = ring
    data.qpos[jm["ring_inter"]]   = -0.15 + 1.1169 * ring
    data.qpos[jm["middle"]]       = middle
    data.qpos[jm["middle_inter"]] = -0.15 + 1.1169 * middle
    data.qpos[jm["index"]]        = index
    data.qpos[jm["index_inter"]]  = -0.05 + 1.1169 * index
    data.qpos[jm["thumb_yaw"]]    = yaw
    data.qpos[jm["thumb_pitch"]]  = pitch
    data.qpos[jm["thumb_inter"]]  = 0.15 + 1.33 * pitch
    data.qpos[jm["thumb_distal"]] = 0.15 + 0.66 * pitch

    mujoco.mj_kinematics(model, data)


def _worker_add_geoms(viewer, state: np.ndarray) -> None:
    """Draw grasp geometry overlay using pre-computed world-frame tip positions.

    state layout:
        [0]    gz
        [1]    mode_idx
        [2]    cyl_radius
        [3]    robot_x  (not used for geoms — already baked into tip positions)
        [4]    robot_y
        [5:8]  tip_world[thumb]
        [8:11] tip_world[index]
        [11:14] tip_world[middle]
        [14:17] tip_world[ring]
        [17:20] tip_world[pinky]
    """
    mode_idx   = int(round(state[1]))
    cyl_radius = float(state[2])
    mode       = MODES[mode_idx] if 0 <= mode_idx < len(MODES) else ""

    # Decode world-frame tip positions
    wtips: Dict[str, np.ndarray] = {}
    for i, fname in enumerate(_VIEWER_FINGER_ORDER):
        p = state[5 + i * 3: 5 + i * 3 + 3]
        if not np.any(np.isnan(p)):
            wtips[fname] = p.copy()

    scn = viewer.user_scn
    scn.ngeom = 0

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

    # Fingertip spheres
    for fname, pos in wtips.items():
        rgba = _FINGER_RGBA.get(fname, np.array([0.8, 0.8, 0.2, 0.9]))
        add_sphere(pos, radius=0.002, rgba=rgba)

    if mode == "2-finger line":
        th  = wtips.get("thumb")
        idx = wtips.get("index")
        if th is not None and idx is not None:
            add_line(th, idx, rgba=(1.0, 0.2, 0.2, 0.8), width=0.004)
            add_sphere((th + idx) / 2, radius=0.004, rgba=(1.0, 0.9, 0.1, 0.9))

    elif "plane" in mode:
        non_thumb = [f for f in ["index", "middle", "ring", "pinky"] if f in wtips]
        for i in range(len(non_thumb) - 1):
            add_line(wtips[non_thumb[i]], wtips[non_thumb[i + 1]],
                     rgba=(0.2, 0.5, 0.9, 0.8), width=0.003)
        th = wtips.get("thumb")
        if th is not None:
            for f in non_thumb:
                add_line(th, wtips[f], rgba=(0.9, 0.3, 0.2, 0.5), width=0.002)
            cen = np.mean([wtips[f] for f in non_thumb], axis=0)
            add_sphere((th + cen) / 2, radius=0.004, rgba=(1.0, 0.9, 0.1, 0.9))

    elif mode == "cylinder":
        non_thumb = [f for f in ["index", "middle", "ring", "pinky"] if f in wtips]
        for i in range(len(non_thumb) - 1):
            add_line(wtips[non_thumb[i]], wtips[non_thumb[i + 1]],
                     rgba=(0.2, 0.5, 0.9, 0.8), width=0.003)
        if non_thumb and "thumb" in wtips:
            nf_cen = np.mean([wtips[f] for f in non_thumb], axis=0)
            add_line(wtips["thumb"], nf_cen, rgba=(0.9, 0.3, 0.2, 0.6), width=0.003)
            add_sphere(nf_cen, radius=0.005, rgba=(0.2, 0.9, 0.5, 0.7))
            # Cylinder arc outlines
            if cyl_radius > 0:
                cen_x = float(nf_cen[0])
                cen_z = float(nf_cen[2])
                nf_pts = np.array([wtips[f] for f in non_thumb])
                y_vals = [float(nf_pts[:, 1].min()), float(nf_pts[:, 1].max())]
                angles = np.linspace(0, np.pi, 9)
                for y in y_vals:
                    for j in range(len(angles) - 1):
                        a0, a1 = angles[j], angles[j + 1]
                        p0 = np.array([cen_x + cyl_radius * np.cos(a0), y,
                                       cen_z + cyl_radius * np.sin(a0)])
                        p1 = np.array([cen_x + cyl_radius * np.cos(a1), y,
                                       cen_z + cyl_radius * np.sin(a1)])
                        add_line(p0, p1, rgba=(0.2, 0.7, 0.9, 0.5), width=0.002)


def _hand_viewer_worker(xml_path: str, ctrl_arr, state_arr,
                         stop_event, title: str = "") -> None:
    """Subprocess entry: passive floating-hand MuJoCo viewer.

    Reads ctrl and state from shared multiprocessing.Array objects and
    updates the viewer at ~30 fps.  Runs in a separate process to avoid
    GLFW crashes when multiple viewers are open simultaneously.
    """
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        jm    = _worker_jnt_map(model)

        ctrl  = np.array(ctrl_arr[:])
        _worker_apply_qpos(jm, data, ctrl, model)

        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running() and not stop_event.is_set():
                ctrl  = np.array(ctrl_arr[:])
                state = np.array(state_arr[:])
                _worker_apply_qpos(jm, data, ctrl, model)
                _worker_add_geoms(v, state)
                v.sync()
                time.sleep(0.033)
    except Exception as exc:
        print(f"[HandViewerWorker{' ' + title if title else ''}] {exc}")
        import traceback; traceback.print_exc()


def _robot_viewer_worker(xml_path: str, ctrl_arr, state_arr,
                          stop_event,
                          ik_dt: float = 0.05,
                          ik_max_iters: int = 5,
                          ik_pos_thr: float = 5e-3,
                          ik_ori_thr: float = 0.05,
                          eeff_local: tuple = (0.070, 0.016, 0.155),
                          real_q_arr=None,
                          real_tracking=None) -> None:
    """Subprocess entry: UR5+hand robot viewer with mink differential arm IK.

    ctrl_arr[0:6]  = world position + rotation of hand base (updated by main process)
    ctrl_arr[6:12] = finger ctrl values (set directly, not by IK)

    The arm joints are solved online via mink FrameTask targeting the eeff site.
    The finger joints are set directly from ctrl_arr[6:12], so 'ours' and 'mink'
    robot viewers differ only in which finger ctrl values are written to ctrl_arr.

    real_q_arr    : multiprocessing.Array('d', 6) — real UR5 joint angles snapshot.
    real_tracking : multiprocessing.Value('b')    — when 1, viewer displays real
                    joint angles instead of running mink IK (shows actual robot state).
    """
    try:
        import mink
    except ImportError:
        print("[RobotViewerWorker] mink not available — install via 'uv run'")
        return
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        eeff_local_arr = np.array(eeff_local)
        configuration  = mink.Configuration(model)
        eeff_id        = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eeff")

        eeff_task    = mink.FrameTask("eeff", "site",
                                      position_cost=1.0, orientation_cost=0.5,
                                      lm_damping=1e-6)
        posture_task = mink.PostureTask(model, cost=1e-4)
        tasks        = [eeff_task, posture_task]

        _ARM_JNT = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        _ARM_ACT = ["shoulder_pan", "shoulder_lift", "elbow",
                    "wrist_1", "wrist_2", "wrist_3"]
        _FNG_ACT = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

        arm_jnt_qposadr = [model.jnt_qposadr[model.joint(n).id] for n in _ARM_JNT]
        arm_ctrl_ids    = [model.actuator(n).id for n in _ARM_ACT]
        finger_ctrl_ids = [model.actuator(n).id for n in _FNG_ACT]

        limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(model, {n: np.pi for n in _ARM_JNT}),
        ]

        # Home pose
        data.qpos[model.jnt_qposadr[model.joint("shoulder_lift_joint").id]] = -np.pi / 2
        data.qpos[model.jnt_qposadr[model.joint("elbow_joint").id]]         =  np.pi / 2
        data.qpos[model.jnt_qposadr[model.joint("wrist_1_joint").id]]       = -np.pi / 2
        mujoco.mj_forward(model, data)
        for ctrl_id, qadr in zip(arm_ctrl_ids, arm_jnt_qposadr):
            data.ctrl[ctrl_id] = data.qpos[qadr]
        configuration.update(data.qpos)
        posture_task.set_target_from_configuration(configuration)

        dt      = model.opt.timestep
        n_steps = max(1, round(0.033 / dt))

        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running() and not stop_event.is_set():
                ctrl = np.array(ctrl_arr[:])

                # Real-robot tracking: display actual joint angles instead of IK
                use_real = (
                    real_tracking is not None and real_q_arr is not None
                    and real_tracking.value
                )
                if use_real:
                    q_real = np.array(real_q_arr[:])
                    if not np.all(q_real == 0):
                        for ctrl_id, qadr in zip(arm_ctrl_ids, arm_jnt_qposadr):
                            i = arm_ctrl_ids.index(ctrl_id)
                            data.ctrl[ctrl_id] = q_real[i]
                            data.qpos[qadr]    = q_real[i]
                        # Still update finger ctrl from shared array
                        for ctrl_id, val in zip(finger_ctrl_ids, ctrl[6:12]):
                            data.ctrl[ctrl_id] = val
                        mujoco.mj_forward(model, data)
                        configuration.update(data.qpos)
                elif eeff_id >= 0:
                    R_hand     = _Rx(ctrl[3]) @ _Ry(ctrl[4])
                    target_pos = ctrl[0:3] + R_hand @ eeff_local_arr
                    T_target   = np.eye(4)
                    T_target[:3, :3] = R_hand
                    T_target[:3,  3] = target_pos
                    eeff_task.set_target(mink.SE3.from_matrix(T_target))

                    for _ in range(ik_max_iters):
                        try:
                            vel = mink.solve_ik(configuration, tasks, ik_dt, "daqp",
                                                limits=limits)
                        except mink.NoSolutionFound:
                            break
                        configuration.integrate_inplace(vel, ik_dt)
                        err = eeff_task.compute_error(configuration)
                        if (np.linalg.norm(err[:3]) < ik_pos_thr and
                                np.linalg.norm(err[3:]) < ik_ori_thr):
                            break

                    for ctrl_id, qadr in zip(arm_ctrl_ids, arm_jnt_qposadr):
                        data.ctrl[ctrl_id] = configuration.q[qadr]
                    for ctrl_id, val in zip(finger_ctrl_ids, ctrl[6:12]):
                        data.ctrl[ctrl_id] = val

                if not use_real:
                    for _ in range(n_steps):
                        mujoco.mj_step(model, data)
                    configuration.update(data.qpos)

                state = np.array(state_arr[:])
                _worker_add_geoms(v, state)
                v.sync()
                time.sleep(0.033)
    except Exception as exc:
        print(f"[RobotViewerWorker] {exc}")
        import traceback; traceback.print_exc()


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
        mink_viz: bool = True,
        # Real-robot options (requires magpie_control + ur_rtde)
        real_robot: bool = False,
        ur5_ip: Optional[str] = None,
        ur5_speed: float = 0.10,
    ):
        print("[GraspViz] Initialising FK model...")
        self.fk      = InspireHandFK(xml_path=xml_path, rebuild=rebuild)
        self.closure = ClosureGeometry(self.fk)

        # Grasp state
        self._mode    = "4-finger plane"
        self._width_m = 0.040
        self._grasp_z = 0.0
        self._result: Optional[ClosureResult] = None
        self._width_range = (0.015, 0.090)

        # Real-robot mode implies robot_mode
        self._real_robot_mode = real_robot
        if real_robot:
            robot_mode = True

        # Robot mode flags
        self._robot_mode = robot_mode
        self._grasp_x = _DEFAULT_ROBOT_X if robot_mode else 0.0
        self._grasp_y = _DEFAULT_ROBOT_Y if robot_mode else 0.0

        # Plane orientation (radians); applied on top of auto-computed tilt
        self._plane_rx = 0.0
        self._plane_ry = 0.0
        self._plane_rz = 0.0

        # Wrist-3 TCP pose (stored by robot viewer for future real deployment)
        self._wrist3_pos: Optional[np.ndarray] = None
        self._wrist3_mat: Optional[np.ndarray] = None

        # Real hand connection
        self._hand      = None
        self._send_real = send_real
        if port is not None:
            try:
                from .rh56_hand import RH56Hand
                self._hand = RH56Hand(port=port)
                print(f"[GraspViz] Connected to real hand on {port}")
            except Exception as exc:
                print(f"[GraspViz] Could not connect to real hand: {exc}")

        # ---- Multiprocessing viewer state ----
        # Use 'fork' start method (Linux default) for fast subprocess creation.
        # GLFW is NOT initialized in the parent at fork time, so child processes
        # each get their own clean GLFW context — no multi-thread crashes.
        _mp = multiprocessing.get_context("fork")
        self._mp_ctx = _mp

        # Shared ctrl arrays (one per planner type)
        self._custom_ctrl_arr = _mp.Array("d", _VIEWER_CTRL_LEN)
        self._mink_ctrl_arr   = _mp.Array("d", _VIEWER_CTRL_LEN)
        # Shared state (world-frame tip positions, mode, etc.)
        self._viewer_state_arr = _mp.Array("d", _VIEWER_STATE_LEN)

        # Per-viewer stop events
        self._hand_ours_stop  = _mp.Event()
        self._hand_mink_stop  = _mp.Event()
        self._robot_ours_stop = _mp.Event()
        self._robot_mink_stop = _mp.Event()

        # Viewer process handles
        self._hand_ours_proc:  Optional[multiprocessing.Process] = None
        self._hand_mink_proc:  Optional[multiprocessing.Process] = None
        self._robot_ours_proc: Optional[multiprocessing.Process] = None
        self._robot_mink_proc: Optional[multiprocessing.Process] = None

        # ---- Mink grasp planner (background thread, not subprocess) ----
        # Runs in the main process; only the MuJoCo *display* needs a subprocess.
        self._state_lock        = threading.Lock()   # protects _result, _grasp_z/x/y
        self._mink_enabled      = False
        self._mink_planner      = None
        self._mink_result       = None
        self._mink_lock         = threading.Lock()
        self._mink_solve_event  = threading.Event()
        self._mink_solve_stop   = threading.Event()
        self._mink_solve_thread: Optional[threading.Thread] = None

        if mink_viz:
            try:
                from .mink_grasp_planner import MinkGraspPlanner
                print("[GraspViz] Loading mink comparison planner...")
                self._mink_planner = MinkGraspPlanner(
                    _RIGHT_SCENE, dt=0.005, max_iters=150, conv_thr=5e-3,
                )
                self._mink_enabled = True
                self._mink_solve_thread = threading.Thread(
                    target=self._mink_solve_loop, daemon=True, name="mink-solve",
                )
                self._mink_solve_thread.start()
                print("[GraspViz] Mink planner ready.")
            except Exception as exc:
                print(f"[GraspViz] Could not load mink planner: {exc}")

        # ---- Real robot arm (UR5) state ----
        self._arm      = None   # UR5Bridge (set below if real_robot)
        self._executor = None   # GraspExecutor (set below if arm + hand)
        self._teach_mode   = False
        self._grasp_strategy = "Plan"   # Naive / Plan / Thumb Reflex
        self._grasp_force_N  = 0.0
        self._grasp_step_mm  = 10.0

        # Shared memory for real UR5 joint snapshot (robot viewer subprocess)
        _mp2 = multiprocessing.get_context("fork")
        self._real_q_arr    = _mp2.Array("d", 6)    # last known real UR5 q
        self._real_tracking = _mp2.Value("b", 0)    # 1 = viewer shows real q

        # Thread-safe status queue (executor→UI main thread via timer)
        self._status_queue: queue.Queue = queue.Queue()
        self._status_lines: list = []       # last N status lines for display
        self._status_timer: Optional[threading.Timer] = None
        self._ws_status_text = None         # matplotlib Text widget (set in run())

        if real_robot and ur5_ip is not None:
            try:
                from .ur5_bridge import UR5Bridge
                self._arm = UR5Bridge(ip=ur5_ip, speed=ur5_speed)
                if self._arm.connect():
                    self._arm.snapshot_joints(self._real_q_arr)
                    self._real_tracking.value = 1
                    print(f"[GraspViz] UR5 arm connected at {ur5_ip}")
                else:
                    self._arm = None
            except Exception as exc:
                print(f"[GraspViz] UR5 arm setup failed: {exc}")
                self._arm = None

        if self._arm is not None and self._hand is not None:
            try:
                from .grasp_executor import GraspExecutor
                self._executor = GraspExecutor(
                    arm=self._arm,
                    hand=self._hand,
                    fk=self.fk,
                    status_cb=lambda s: self._status_queue.put(s),
                    force_cb=lambda f: self._status_queue.put(f"forces: {f}"),
                )
                print("[GraspViz] GraspExecutor ready.")
            except Exception as exc:
                print(f"[GraspViz] GraspExecutor setup failed: {exc}")

        # Initial computation
        self._recompute()

    # ------------------------------------------------------------------
    # Compute closure
    # ------------------------------------------------------------------
    def _recompute(self):
        """Recompute ClosureResult for current mode + width."""
        try:
            result = self.closure.solve(self._mode, self._width_m)
        except Exception as exc:
            print(f"[GraspViz] solve failed: {exc}")
            result = None

        with self._state_lock:
            self._result = result

        if result is not None:
            self._push_viewer_ctrl()
            self._send_real_hand()
            if self._mink_enabled:
                self._mink_solve_event.set()

    # ------------------------------------------------------------------
    # Sim2Real
    # ------------------------------------------------------------------
    def _send_real_hand(self):
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
        ctrl_min = np.array([self.fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        ctrl_max = np.array([self.fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
        rng = ctrl_max - ctrl_min
        real_cmd = np.round(
            (1.0 - np.clip((finger_ctrl - ctrl_min) / np.where(rng > 0, rng, 1.0),
                           0.0, 1.0)) * 1000
        ).astype(int).tolist()

        # In Thumb Reflex strategy, slider adjustments only actuate the thumb
        if (self._real_robot_mode
                and self._grasp_strategy == "Thumb Reflex"
                and not (self._executor is not None and self._executor.is_running())):
            # Keep non-thumb fingers open (1000 = open, INVERTED convention)
            real_cmd[0] = 1000  # pinky
            real_cmd[1] = 1000  # ring
            real_cmd[2] = 1000  # middle
            real_cmd[3] = 1000  # index

        try:
            self._hand.angle_set(real_cmd)
        except Exception as exc:
            print(f"[GraspViz] angle_set failed: {exc}")

    # ------------------------------------------------------------------
    # Plane orientation helpers
    # ------------------------------------------------------------------
    def _plane_R_matrix(self) -> np.ndarray:
        """Extra world-frame rotation from the plane Rx/Ry/Rz sliders."""
        return ClosureResult._plane_rot(self._plane_rx, self._plane_ry, self._plane_rz)

    @staticmethod
    def _mat_to_xyz_euler(R: np.ndarray):
        """Extract (a, b, c) s.t. Rx(a) @ Ry(b) @ Rz(c) = R (XYZ intrinsic Euler)."""
        b = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
        cb = np.cos(b)
        if abs(cb) > 1e-6:
            a = float(np.arctan2(-R[1, 2], R[2, 2]))
            c = float(np.arctan2(-R[0, 1], R[0, 0]))
        else:
            # Gimbal lock: b = ±π/2
            a = float(np.arctan2(R[2, 1], R[1, 1]))
            c = 0.0
        return a, b, c

    # ------------------------------------------------------------------
    # Real-robot helpers
    # ------------------------------------------------------------------
    def _build_world_T_hand(self, r: ClosureResult) -> np.ndarray:
        """Build 4×4 world_T_hand from current closure result + slider state.

        This is the arm-pose equivalent of _build_ctrl_array — it produces
        the continuous-frame pose used by UR5Bridge.move_to_hand_pose().
        """
        gz    = self._grasp_z
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode:
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])
        R_full = self._plane_R_matrix() @ ClosureResult._rot_matrix(r.base_tilt_y)
        T = np.eye(4)
        T[:3, :3] = R_full
        T[:3, 3]  = wbase
        return T

    def _update_status(self, msg: str):
        """Append msg to the on-screen status area (main-thread safe)."""
        self._status_lines.append(msg)
        self._status_lines = self._status_lines[-6:]  # keep last 6 lines
        if self._ws_status_text is not None:
            self._ws_status_text.set_text("\n".join(self._status_lines))
            try:
                import matplotlib.pyplot as plt
                plt.gcf().canvas.draw_idle()
            except Exception:
                pass

    def _poll_status_queue(self):
        """Drain the executor→UI message queue (called by repeating timer)."""
        try:
            while True:
                msg = self._status_queue.get_nowait()
                if msg == "__reset_grasp_btn__":
                    # Restore GRASP! button appearance after execution
                    if hasattr(self, "_btn_grasp"):
                        self._btn_grasp.ax.set_facecolor("#2ecc71")
                        self._btn_grasp.label.set_text("GRASP!")
                        try:
                            import matplotlib.pyplot as plt
                            plt.draw()
                        except Exception:
                            pass
                else:
                    self._update_status(msg)
        except queue.Empty:
            pass
        # Re-schedule
        self._status_timer = threading.Timer(0.15, self._poll_status_queue)
        self._status_timer.daemon = True
        self._status_timer.start()

    def _is_cylinder_bad(self) -> bool:
        """True when cylinder mode diameter < 71 mm (power grasp disabled)."""
        return self._mode == "cylinder" and self._width_m * 2 < 0.071

    # ------------------------------------------------------------------
    # Shared-memory state update helpers
    # ------------------------------------------------------------------
    def _build_ctrl_array(self, r: ClosureResult,
                           finger_ctrl: Optional[Dict] = None) -> np.ndarray:
        """Build 12-element ctrl array from a ClosureResult.

        finger_ctrl overrides the finger angles (used for mink planner output).
        Plane orientation (rx/ry/rz) is incorporated via XYZ Euler decomposition.
        """
        gz    = self._grasp_z
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode:
            wbase = wbase + np.array([self._grasp_x, self._grasp_y, 0.0])
        fc = finger_ctrl if finger_ctrl is not None else r.ctrl_values
        # Combine auto-tilt with user plane rotation, then extract MuJoCo XYZ Euler.
        # MuJoCo joint hierarchy applies Rx(rot_x) @ Ry(rot_y) @ Rz(rot_z).
        R_full = self._plane_R_matrix() @ ClosureResult._rot_matrix(r.base_tilt_y)
        rot_x, rot_y, rot_z = self._mat_to_xyz_euler(R_full)
        return np.array([
            wbase[0], wbase[1], wbase[2],
            rot_x, rot_y, rot_z,
            fc.get("pinky",          0.0),
            fc.get("ring",           0.0),
            fc.get("middle",         0.0),
            fc.get("index",          0.0),
            fc.get("thumb_proximal", 0.0),
            fc.get("thumb_yaw",      0.0),
        ], dtype=float)

    def _build_state_array(self, r: ClosureResult) -> np.ndarray:
        """Build 20-element state array with world-frame tip positions."""
        gz   = self._grasp_z
        mode_idx = MODES.index(r.mode) if r.mode in MODES else 0
        wtips = r.world_tips(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        if self._robot_mode:
            xy_off = np.array([self._grasp_x, self._grasp_y, 0.0])
            wtips  = {f: p + xy_off for f, p in wtips.items()}
        state = np.full(_VIEWER_STATE_LEN, np.nan)
        state[0] = gz
        state[1] = float(mode_idx)
        state[2] = r.cylinder_radius
        state[3] = self._grasp_x if self._robot_mode else 0.0
        state[4] = self._grasp_y if self._robot_mode else 0.0
        for i, fname in enumerate(_VIEWER_FINGER_ORDER):
            if fname in wtips:
                state[5 + i * 3: 5 + i * 3 + 3] = wtips[fname]
        return state

    def _push_viewer_ctrl(self):
        """Push current custom-planner ctrl to shared arrays."""
        with self._state_lock:
            r  = self._result
            gz = self._grasp_z
        if r is None:
            return
        ctrl  = self._build_ctrl_array(r)
        state = self._build_state_array(r)
        self._custom_ctrl_arr[:]   = ctrl
        self._viewer_state_arr[:]  = state

    def _push_mink_viewer_ctrl(self):
        """Push mink-planner finger ctrl (same base pose) to shared arrays."""
        with self._state_lock:
            r  = self._result
        with self._mink_lock:
            m_res = self._mink_result
        if r is None or m_res is None:
            return
        mink_fc = {
            "pinky":          float(m_res.ctrl[0]),
            "ring":           float(m_res.ctrl[1]),
            "middle":         float(m_res.ctrl[2]),
            "index":          float(m_res.ctrl[3]),
            "thumb_proximal": float(m_res.ctrl[4]),
            "thumb_yaw":      float(m_res.ctrl[5]),
        }
        ctrl = self._build_ctrl_array(r, finger_ctrl=mink_fc)
        self._mink_ctrl_arr[:] = ctrl
        # State (tip positions) are the same as for custom — we use the custom
        # planner's world-frame tips as the reference overlay.

    # ------------------------------------------------------------------
    # Viewer process launch helpers
    # ------------------------------------------------------------------
    def _launch_hand_viewer_ours(self):
        """Open floating-hand viewer with custom-planner finger angles."""
        if self._hand_ours_proc is not None and self._hand_ours_proc.is_alive():
            print("[GraspViz] Hand viewer (Ours) already open.")
            return
        self._hand_ours_stop.clear()
        self._push_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_hand_viewer_worker,
            args=(_GRASP_SCENE, self._custom_ctrl_arr,
                  self._viewer_state_arr, self._hand_ours_stop, "Ours"),
            daemon=True,
        )
        proc.start()
        self._hand_ours_proc = proc
        print("[GraspViz] Hand viewer (Ours) launched.")

    def _launch_hand_viewer_mink(self):
        """Open floating-hand viewer with mink-planner finger angles."""
        if not self._mink_enabled or self._mink_planner is None:
            print("[GraspViz] Mink planner not available.")
            return
        if self._hand_mink_proc is not None and self._hand_mink_proc.is_alive():
            print("[GraspViz] Hand viewer (Mink) already open.")
            return
        self._hand_mink_stop.clear()
        self._push_mink_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_hand_viewer_worker,
            args=(_GRASP_SCENE, self._mink_ctrl_arr,
                  self._viewer_state_arr, self._hand_mink_stop, "Mink"),
            daemon=True,
        )
        proc.start()
        self._hand_mink_proc = proc
        print("[GraspViz] Hand viewer (Mink) launched.")

    def _launch_robot_viewer_ours(self):
        """Open UR5+hand robot viewer with custom-planner finger angles."""
        if self._robot_ours_proc is not None and self._robot_ours_proc.is_alive():
            print("[GraspViz] Robot viewer (Ours) already open.")
            return
        self._robot_ours_stop.clear()
        self._push_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_robot_viewer_worker,
            args=(_ROBOT_SCENE, self._custom_ctrl_arr,
                  self._viewer_state_arr, self._robot_ours_stop),
            kwargs=dict(ik_dt=_IK_DT, ik_max_iters=_IK_MAX_ITERS,
                        ik_pos_thr=_IK_POS_THR, ik_ori_thr=_IK_ORI_THR,
                        eeff_local=tuple(_EEFF_LOCAL),
                        real_q_arr=self._real_q_arr,
                        real_tracking=self._real_tracking),
            daemon=True,
        )
        proc.start()
        self._robot_ours_proc = proc
        print("[GraspViz] Robot viewer (Ours) launched.")

    def _launch_robot_viewer_mink(self):
        """Open UR5+hand robot viewer with mink-planner finger angles."""
        if not self._mink_enabled or self._mink_planner is None:
            print("[GraspViz] Mink planner not available.")
            return
        if self._robot_mink_proc is not None and self._robot_mink_proc.is_alive():
            print("[GraspViz] Robot viewer (Mink) already open.")
            return
        self._robot_mink_stop.clear()
        self._push_mink_viewer_ctrl()
        proc = self._mp_ctx.Process(
            target=_robot_viewer_worker,
            args=(_ROBOT_SCENE, self._mink_ctrl_arr,
                  self._viewer_state_arr, self._robot_mink_stop),
            kwargs=dict(ik_dt=_IK_DT, ik_max_iters=_IK_MAX_ITERS,
                        ik_pos_thr=_IK_POS_THR, ik_ori_thr=_IK_ORI_THR,
                        eeff_local=tuple(_EEFF_LOCAL),
                        real_q_arr=self._real_q_arr,
                        real_tracking=self._real_tracking),
            daemon=True,
        )
        proc.start()
        self._robot_mink_proc = proc
        print("[GraspViz] Robot viewer (Mink) launched.")

    # Keep backward-compatible alias used by old button callback
    def _launch_viewer(self):
        if self._robot_mode:
            self._launch_robot_viewer_ours()
        else:
            self._launch_hand_viewer_ours()

    # ------------------------------------------------------------------
    # Mink solve loop (background thread — NOT a subprocess)
    # ------------------------------------------------------------------
    def _mink_solve_loop(self):
        """Background thread: re-solve with mink whenever mode/width changes."""
        while not self._mink_solve_stop.is_set():
            triggered = self._mink_solve_event.wait(timeout=0.1)
            if not triggered:
                continue
            self._mink_solve_event.clear()
            with self._state_lock:
                result = self._result
            if result is None or self._mink_planner is None:
                continue
            try:
                m_res = self._run_mink_for_result(result)
                with self._mink_lock:
                    self._mink_result = m_res
                self._push_mink_viewer_ctrl()
            except Exception as exc:
                print(f"[Mink] Solve error: {exc}")

    def _run_mink_for_result(self, result: ClosureResult):
        """Run mink planner targeting the same tip positions as the custom result."""
        mode = result.mode
        tips = result.tip_positions
        if mode == "2-finger line":
            return self._mink_planner.solve_line(
                result.width,
                thumb_target=tips["thumb"],
                index_target=tips["index"],
            )
        elif "plane" in mode:
            n       = int(mode[0])
            fingers = GRASP_FINGER_SETS[n] + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_plane(result.width, targets)
        elif mode == "cylinder":
            fingers = NON_THUMB_FINGERS + ["thumb"]
            targets = {f: tips[f] for f in fingers if f in tips}
            return self._mink_planner.solve_cylinder(result.width, targets)
        raise ValueError(f"Unknown mode: {mode}")

    # ------------------------------------------------------------------
    # MuJoCo viewer geometry overlay  (used for in-process geom helpers;
    # the actual rendering uses the module-level worker functions above)
    # ------------------------------------------------------------------
    @staticmethod
    def _setup_jnt_map(model: mujoco.MjModel) -> dict:
        """Cache qpos addresses for the 12 joints in the grasp scene."""
        return _worker_jnt_map(model)

    @staticmethod
    def _apply_qpos(jm: dict, data: mujoco.MjData, ctrl: np.ndarray,
                    model: mujoco.MjModel):
        _worker_apply_qpos(jm, data, ctrl, model)

    def _add_viewer_geoms(self, viewer, result: Optional[ClosureResult], gz: float):
        """Used only by the deprecated single-process viewer path (kept for reference)."""
        if result is None:
            return
        state = self._build_state_array(result)
        _worker_add_geoms(viewer, state)

    # ------------------------------------------------------------------
    # matplotlib setup
    # ------------------------------------------------------------------
    def run(self):
        """Build and show the interactive matplotlib figure."""
        matplotlib.use("TkAgg")

        from matplotlib.widgets import TextBox

        # Expanded figure for real-robot mode to accommodate extra panel
        _figsize = (20, 11) if self._real_robot_mode else (14, 8)
        fig = plt.figure(figsize=_figsize)
        fig.suptitle("Inspire RH56 — Antipodal Grasp Geometry Planner", fontsize=12)

        # 3D axes: left 65%
        ax3d: Axes3D = fig.add_axes([0.02, 0.05, 0.60, 0.88], projection="3d")
        ax3d.set_xlabel("X  (closure direction)")
        ax3d.set_ylabel("Y  (finger spread)")
        ax3d.set_zlabel("Z  (world up)")
        self._ax3d = ax3d

        _mink_ready = self._mink_enabled and self._mink_planner is not None

        # ---- Shared slider/textbox factory (used by both modes) ----
        _SL  = 0.65   # left edge of label + slider
        _SW  = 0.22   # slider width
        _TL  = 0.88   # textbox left edge
        _TW  = 0.09   # textbox width
        _SH  = 0.025  # slider height
        _LH  = 0.030  # label height

        def _make_slider_row(label, y_lbl, y_sl, vmin, vmax, vinit, vstep,
                             fmt="{:.0f}"):
            ax_lbl = fig.add_axes([_SL, y_lbl, 0.32, _LH])
            ax_lbl.axis("off")
            ax_lbl.text(0.0, 0.5, label, fontsize=8.5)
            ax_sl = fig.add_axes([_SL, y_sl, _SW, _SH])
            sl    = Slider(ax_sl, "", vmin, vmax, valinit=vinit, valstep=vstep)
            ax_tb = fig.add_axes([_TL, y_sl, _TW, _SH])
            tb    = TextBox(ax_tb, "", initial=fmt.format(vinit))
            return sl, tb

        def _wire(slider, tb, on_val_fn, fmt="{:.0f}"):
            slider.on_changed(lambda v: (on_val_fn(v), tb.set_val(fmt.format(v))))
            tb.on_submit(lambda s: slider.set_val(
                float(s) if s.strip() else slider.val))

        # ---- Plane orientation separator label ----
        def _section_label(y, text):
            ax = fig.add_axes([_SL, y, 0.32, _LH])
            ax.axis("off")
            ax.text(0.0, 0.5, text, fontsize=7.5, color="#555555",
                    style="italic")

        # ---- Button dimensions ----
        _BH  = 0.032
        _BW  = 0.135
        _GAP = 0.015

        wmin_mm, wmax_mm = (x * 1000 for x in self._width_range)

        if self._robot_mode:
            # ---- Robot mode: Width + Z + X + Y + Rx + Ry + Rz (7 rows) ----
            # In real-robot mode the layout is shifted up by 0.25 to make
            # room for the real arm control panel at the bottom.
            _Y = 0.25 if self._real_robot_mode else 0.0   # vertical offset

            ax_radio = fig.add_axes([0.65, 0.71 + _Y, 0.14, 0.21])
            self._radio = RadioButtons(ax_radio, MODES, active=MODES.index(self._mode))
            self._radio.on_clicked(self._on_mode)

            self._slider_w, self._tb_w = _make_slider_row(
                "Width / Diameter (mm):", 0.665 + _Y, 0.630 + _Y, wmin_mm, wmax_mm,
                self._width_m * 1000, 1.0)
            self._slider_z, self._tb_z = _make_slider_row(
                "Grasp Z (mm):", 0.575 + _Y, 0.540 + _Y, -200.0, 200.0,
                self._grasp_z * 1000, 5.0)
            self._slider_x, self._tb_x = _make_slider_row(
                "Grasp X (mm, UR5 world):", 0.485 + _Y, 0.450 + _Y, -850.0, 850.0,
                self._grasp_x * 1000, 5.0)
            self._slider_y, self._tb_y = _make_slider_row(
                "Grasp Y (mm, UR5 world):", 0.395 + _Y, 0.360 + _Y, -850.0, 850.0,
                self._grasp_y * 1000, 5.0)

            _section_label(0.325 + _Y, "── Plane orientation ──")

            self._slider_rx, self._tb_rx = _make_slider_row(
                "Plane Rx (°):", 0.295 + _Y, 0.260 + _Y, -180.0, 180.0, 0.0, 1.0)
            self._slider_ry, self._tb_ry = _make_slider_row(
                "Plane Ry (°):", 0.205 + _Y, 0.170 + _Y, -180.0, 180.0, 0.0, 1.0)
            self._slider_rz, self._tb_rz = _make_slider_row(
                "Plane Rz (°):", 0.115 + _Y, 0.080 + _Y, -180.0, 180.0, 0.0, 1.0)

            _wire(self._slider_w,  self._tb_w,  self._on_width)
            _wire(self._slider_z,  self._tb_z,  self._on_z)
            _wire(self._slider_x,  self._tb_x,  self._on_x)
            _wire(self._slider_y,  self._tb_y,  self._on_y)
            _wire(self._slider_rx, self._tb_rx, self._on_plane_rx)
            _wire(self._slider_ry, self._tb_ry, self._on_plane_ry)
            _wire(self._slider_rz, self._tb_rz, self._on_plane_rz)

            btn_y = 0.010 + _Y
            # Top row: Hand viewers
            ax_h_ours = fig.add_axes([0.65,               btn_y + _BH + 0.006, _BW, _BH])
            ax_h_mink = fig.add_axes([0.65 + _BW + _GAP,  btn_y + _BH + 0.006, _BW, _BH])
            self._btn_h_ours = Button(ax_h_ours, "Hand: Ours")
            self._btn_h_mink = Button(ax_h_mink,
                                      "Hand: Mink" if _mink_ready else "Hand: Mink (N/A)")
            self._btn_h_ours.on_clicked(lambda _e: self._launch_hand_viewer_ours())
            self._btn_h_mink.on_clicked(lambda _e: self._launch_hand_viewer_mink())
            # Bottom row: Robot viewers
            ax_r_ours = fig.add_axes([0.65,               btn_y, _BW, _BH])
            ax_r_mink = fig.add_axes([0.65 + _BW + _GAP,  btn_y, _BW, _BH])
            self._btn_r_ours = Button(ax_r_ours, "Robot: Ours")
            self._btn_r_mink = Button(ax_r_mink,
                                      "Robot: Mink" if _mink_ready else "Robot: Mink (N/A)")
            self._btn_r_ours.on_clicked(lambda _e: self._launch_robot_viewer_ours())
            self._btn_r_mink.on_clicked(lambda _e: self._launch_robot_viewer_mink())

            # ---- Real Robot Control panel (only in --real-robot mode) ----
            if self._real_robot_mode:
                _arm_connected = self._arm is not None and self._arm.connected

                _section_label(0.222, "── Real Robot Control ──")

                # Row 1: Teach Mode + Set Pose from Robot
                ax_teach   = fig.add_axes([0.65,              0.182, _BW, _BH])
                ax_setpose = fig.add_axes([0.65 + _BW + _GAP, 0.182, _BW, _BH])
                self._btn_teach   = Button(ax_teach,   "Teach Mode")
                self._btn_setpose = Button(ax_setpose, "Set Pose from Robot")
                self._btn_teach.on_clicked(self._on_teach_mode)
                self._btn_setpose.on_clicked(self._on_set_pose_from_robot)

                # Row 2: Send to Robot (Arm) + Simulate Trajectory
                ax_sendarm = fig.add_axes([0.65,              0.144, _BW, _BH])
                ax_simtraj = fig.add_axes([0.65 + _BW + _GAP, 0.144, _BW, _BH])
                self._btn_sendarm = Button(ax_sendarm, "Send to Robot (Arm)")
                self._btn_simtraj = Button(ax_simtraj, "Simulate Trajectory")
                self._btn_sendarm.on_clicked(self._on_send_arm)
                self._btn_simtraj.on_clicked(self._on_simulate_trajectory)

                # Grasp controls label
                _section_label(0.114, "── Grasp Controls ──")

                # Grasp Strategy radio (3 items, horizontal layout via fig width)
                ax_strategy = fig.add_axes([0.65, 0.053, 0.155, 0.058])
                self._radio_strategy = RadioButtons(
                    ax_strategy,
                    ["Naive", "Plan", "Thumb Reflex"],
                    active=["Naive", "Plan", "Thumb Reflex"].index(self._grasp_strategy),
                )
                self._radio_strategy.on_clicked(self._on_strategy)

                # Force (N) label + textbox
                ax_f_lbl = fig.add_axes([0.812, 0.090, 0.08, 0.022])
                ax_f_lbl.axis("off")
                ax_f_lbl.text(0.0, 0.5, "Force (N):", fontsize=7.5)
                ax_f_tb  = fig.add_axes([0.812, 0.065, 0.08, 0.022])
                self._tb_grasp_force = TextBox(ax_f_tb, "", initial="0")

                # Step (mm) label + textbox
                ax_s_lbl = fig.add_axes([0.812, 0.042, 0.08, 0.022])
                ax_s_lbl.axis("off")
                ax_s_lbl.text(0.0, 0.5, "Step (mm):", fontsize=7.5)
                ax_s_tb  = fig.add_axes([0.812, 0.018, 0.08, 0.022])
                self._tb_grasp_step = TextBox(ax_s_tb, "", initial="10")

                # GRASP! button (full width)
                ax_grasp = fig.add_axes([0.65, 0.005, 0.32, 0.042])
                self._btn_grasp = Button(ax_grasp, "GRASP!")
                self._btn_grasp.ax.set_facecolor("#2ecc71")
                self._btn_grasp.on_clicked(self._on_grasp)

                # Status text area
                ax_status = fig.add_axes([0.65, 0.050, 0.155, 0.055])
                ax_status.axis("off")
                self._ws_status_text = ax_status.text(
                    0.0, 1.0, "",
                    fontsize=6.5, family="monospace",
                    verticalalignment="top", wrap=True,
                    transform=ax_status.transAxes,
                )

                # Grey out arm buttons if not connected
                if not _arm_connected:
                    for btn in [self._btn_teach, self._btn_setpose,
                                self._btn_sendarm, self._btn_grasp]:
                        btn.ax.set_facecolor("#cccccc")
                    self._update_status(
                        "No UR5 connection. Use --ur5-ip to connect.")

                # Start status queue polling timer
                self._poll_status_queue()

        else:
            # ---- Standard mode: Width + Z + Rx + Ry + Rz (5 rows) ----
            ax_radio = fig.add_axes([0.65, 0.61, 0.16, 0.29])
            self._radio = RadioButtons(ax_radio, MODES, active=MODES.index(self._mode))
            self._radio.on_clicked(self._on_mode)

            self._slider_w, self._tb_w = _make_slider_row(
                "Width / Diameter (mm):", 0.560, 0.525, wmin_mm, wmax_mm,
                self._width_m * 1000, 1.0)
            self._slider_z, self._tb_z = _make_slider_row(
                "Grasp Z (mm):", 0.470, 0.435, -200.0, 200.0,
                self._grasp_z * 1000, 5.0)

            _section_label(0.400, "── Plane orientation ──")

            self._slider_rx, self._tb_rx = _make_slider_row(
                "Plane Rx (°):", 0.370, 0.335, -180.0, 180.0, 0.0, 1.0)
            self._slider_ry, self._tb_ry = _make_slider_row(
                "Plane Ry (°):", 0.280, 0.245, -180.0, 180.0, 0.0, 1.0)
            self._slider_rz, self._tb_rz = _make_slider_row(
                "Plane Rz (°):", 0.190, 0.155, -180.0, 180.0, 0.0, 1.0)

            _wire(self._slider_w,  self._tb_w,  self._on_width)
            _wire(self._slider_z,  self._tb_z,  self._on_z)
            _wire(self._slider_rx, self._tb_rx, self._on_plane_rx)
            _wire(self._slider_ry, self._tb_ry, self._on_plane_ry)
            _wire(self._slider_rz, self._tb_rz, self._on_plane_rz)

            btn_y = 0.080
            # Single row: Hand viewers
            ax_h_ours = fig.add_axes([0.65,               btn_y, _BW, _BH])
            ax_h_mink = fig.add_axes([0.65 + _BW + _GAP,  btn_y, _BW, _BH])
            self._btn_h_ours = Button(ax_h_ours, "Hand: Ours")
            self._btn_h_mink = Button(ax_h_mink,
                                      "Hand: Mink" if _mink_ready else "Hand: Mink (N/A)")
            self._btn_h_ours.on_clicked(lambda _e: self._launch_hand_viewer_ours())
            self._btn_h_mink.on_clicked(lambda _e: self._launch_hand_viewer_mink())

        # ---- Send-to-real checkbox (only if hand is connected) ----
        if self._hand is not None:
            # In robot modes btn_y is set inside the if block above; in hand-only
            # mode it is set further below.  Use a safe fallback if not yet defined.
            _check_y = locals().get("btn_y", 0.080) + 2 * (_BH + 0.008)
            ax_real = fig.add_axes([0.65, _check_y, 0.06, 0.050])
            self._check_real = CheckButtons(
                ax_real, ["Send\nto Real"], [self._send_real])
            self._check_real.on_clicked(self._on_send_real)

        # Info text lives inside the 3D plot (redrawn each _update_plot call).
        self._ax3d_info_text = None   # set in _update_plot via ax3d.text2D

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
        self._update_cylinder_guard()

    def _on_width(self, val: float):
        self._width_m = float(val) / 1000.0
        self._recompute()
        self._update_plot()
        self._update_cylinder_guard()

    def _update_cylinder_guard(self):
        """Grey out GRASP! and Send-to-Robot when cylinder < 71 mm."""
        if not self._real_robot_mode or not hasattr(self, "_btn_grasp"):
            return
        bad = self._is_cylinder_bad()
        color = "#cccccc" if bad else "#2ecc71"
        self._btn_grasp.ax.set_facecolor(color)
        color_arm = "#cccccc" if bad else "white"
        if hasattr(self, "_btn_sendarm"):
            self._btn_sendarm.ax.set_facecolor(color_arm)
        if bad:
            self._update_status(
                f"WARN: cylinder {self._width_m*2000:.0f}mm < 71mm — "
                "power grasp disabled"
            )
        try:
            import matplotlib.pyplot as plt
            plt.draw()
        except Exception:
            pass

    def _on_z(self, val: float):
        self._grasp_z = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_x(self, val: float):
        self._grasp_x = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_y(self, val: float):
        self._grasp_y = float(val) / 1000.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_plane_rx(self, val: float):
        self._plane_rx = float(val) * np.pi / 180.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_plane_ry(self, val: float):
        self._plane_ry = float(val) * np.pi / 180.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_plane_rz(self, val: float):
        self._plane_rz = float(val) * np.pi / 180.0
        self._push_viewer_ctrl()
        self._update_plot()

    def _on_send_real(self, label: str):
        self._send_real = not self._send_real
        if self._send_real:
            self._send_real_hand()

    def _on_strategy(self, label: str):
        """Grasp strategy radio callback."""
        self._grasp_strategy = label

    def _on_teach_mode(self, _event):
        """Toggle teach mode on the real UR5."""
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._arm.disable_teach_mode()
            self._teach_mode = False
            self._btn_teach.label.set_text("Teach Mode")
            self._btn_teach.ax.set_facecolor("white")
        else:
            self._arm.enable_teach_mode()
            self._teach_mode = True
            self._btn_teach.label.set_text("TEACH MODE ACTIVE")
            self._btn_teach.ax.set_facecolor("#ff4444")
        import matplotlib.pyplot as plt
        plt.draw()

    def _on_set_pose_from_robot(self, _event):
        """Read current UR5 TCP pose and update grasp sliders."""
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        # Works even in teach mode (read-only)
        with self._state_lock:
            r = self._result
        params = self._arm.decode_tcp_to_grasp_params(current_result=r)
        if not params:
            self._update_status("Failed to read arm pose.")
            return
        self._grasp_x   = params["grasp_x"]
        self._grasp_y   = params["grasp_y"]
        self._grasp_z   = params["grasp_z"]
        self._plane_rx  = params["plane_rx"]
        self._plane_ry  = params["plane_ry"]
        self._plane_rz  = params["plane_rz"]
        # Sync sliders
        self._slider_x.set_val(self._grasp_x * 1000)
        self._slider_y.set_val(self._grasp_y * 1000)
        self._slider_z.set_val(self._grasp_z * 1000)
        self._slider_rx.set_val(np.degrees(self._plane_rx))
        self._slider_ry.set_val(np.degrees(self._plane_ry))
        self._slider_rz.set_val(np.degrees(self._plane_rz))
        self._push_viewer_ctrl()
        self._update_plot()
        self._update_status(
            f"Pose set from robot: "
            f"X={self._grasp_x*1000:.0f} Y={self._grasp_y*1000:.0f} "
            f"Z={self._grasp_z*1000:.0f} mm"
        )

    def _on_send_arm(self, _event):
        """Move the real UR5 arm to the current planned grasp pose."""
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._update_status("BLOCKED: teach mode is active.")
            return
        if self._is_cylinder_bad():
            self._update_status(
                "BLOCKED: cylinder < 71 mm — power grasp disabled.")
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
        # Move arm only (no fingers)
        self._update_status("Moving arm to planned pose...")
        self._real_tracking.value = 0   # viewer shows planned pose
        def _do_move():
            warns2 = self._arm.move_to_hand_pose(world_T_hand, blocking=True)
            for w in warns2:
                self._status_queue.put(w)
            self._arm.snapshot_joints(self._real_q_arr)
            self._real_tracking.value = 1
            self._status_queue.put("Arm move complete.")
        t = threading.Thread(target=_do_move, daemon=True, name="arm-move")
        t.start()

    def _on_simulate_trajectory(self, _event):
        """Show planned grasp pose in the robot sim viewer (mink IK mode)."""
        self._real_tracking.value = 0   # viewer uses mink IK (planned pose)
        self._launch_robot_viewer_ours()
        self._push_viewer_ctrl()
        self._update_status("Sim viewer open — showing planned trajectory.")

    def _on_grasp(self, _event):
        """Execute the selected grasp strategy on the real robot."""
        if self._arm is None:
            self._update_status("No UR5 connected.")
            return
        if self._teach_mode:
            self._update_status("BLOCKED: teach mode is active.")
            return
        if self._is_cylinder_bad():
            self._update_status(
                "BLOCKED: cylinder < 71 mm — power grasp disabled.")
            return
        if self._executor is None:
            self._update_status("GraspExecutor not available (hand connected?)")
            return
        if self._executor.is_running():
            self._update_status("Executor busy — abort first.")
            return
        with self._state_lock:
            r = self._result
        if r is None:
            self._update_status("No grasp result.")
            return

        try:
            force_N  = float(self._tb_grasp_force.text.strip() or "0")
            step_mm  = float(self._tb_grasp_step.text.strip()  or "10")
        except ValueError:
            force_N, step_mm = 0.0, 10.0

        world_T_hand = self._build_world_T_hand(r)
        warns = self._arm.check_pose_workspace(world_T_hand)
        for w in warns:
            self._update_status(w)

        # Enable real hand send automatically during grasp
        self._send_real = True

        # Switch viewer to planned pose while executing
        self._real_tracking.value = 0
        strategy = self._grasp_strategy

        def _post_grasp():
            """Called from executor callbacks after completion."""
            self._arm.snapshot_joints(self._real_q_arr)
            self._real_tracking.value = 1

        # Update GRASP button appearance during execution
        self._btn_grasp.ax.set_facecolor("#e67e22")
        self._btn_grasp.label.set_text("RUNNING…")

        def _on_done(msg):
            self._status_queue.put(msg)
            if "complete" in msg.lower() or "aborted" in msg.lower():
                _post_grasp()
                self._status_queue.put("__reset_grasp_btn__")

        orig_status_cb = self._executor._status

        def _wrapped_status(msg):
            orig_status_cb(msg)
            _on_done(msg)

        self._executor._status = _wrapped_status

        if strategy == "Naive":
            self._executor.execute_naive(world_T_hand, force_N, move_arm=True)
        elif strategy == "Plan":
            self._executor.execute_plan(r, world_T_hand, force_N, step_mm, move_arm=True)
        else:  # Thumb Reflex
            self._executor.execute_thumb_reflex(r, world_T_hand, force_N, move_arm=True)

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
            plt.draw()
            return

        wtips = r.world_tips(gz, self._plane_rx, self._plane_ry, self._plane_rz)
        wbase = r.world_base(gz, self._plane_rx, self._plane_ry, self._plane_rz)

        # Fingertip dots
        for fname, pos in wtips.items():
            col = FINGER_COLORS.get(fname, "gray")
            ax.scatter(*pos, color=col, s=60, zorder=5)
            ax.text(pos[0] + 0.003, pos[1], pos[2] + 0.003, fname[:3],
                    fontsize=7, color=col)

        # Mink comparison dots (cyan)
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
                                "--", color="gold", lw=0.8, alpha=0.7,
                            )

        # Hand base
        ax.scatter(*wbase, color="black", s=80, marker="x", zorder=6)
        ax.text(wbase[0] + 0.003, wbase[1], wbase[2] + 0.003, "base", fontsize=7)

        # Grasp plane Z reference lines
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

        # Info text — rendered inside the 3D plot axes (bottom-left corner).
        prx_deg = np.degrees(self._plane_rx)
        pry_deg = np.degrees(self._plane_ry)
        prz_deg = np.degrees(self._plane_rz)
        lines = [
            f"Mode:   {r.mode}",
            f"Width:  {r.width * 1000:.1f} mm",
            f"Span:   {r.finger_span * 1000:.1f} mm",
        ]
        if r.cylinder_radius > 0:
            lines.append(f"Radius: {r.cylinder_radius * 1000:.1f} mm")
        lines.append(f"Tilt Y: {r.tilt_deg:.1f}°")
        lines.append(f"Base Z: {wbase[2] * 1000:.1f} mm")
        if any(abs(v) > 0.1 for v in (prx_deg, pry_deg, prz_deg)):
            lines.append(f"Plane:  Rx={prx_deg:.0f}° Ry={pry_deg:.0f}° Rz={prz_deg:.0f}°")
        if self._robot_mode:
            lines.append(f"[ROBOT] X={self._grasp_x*1000:.0f} Y={self._grasp_y*1000:.0f} mm")
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
                lines.append("Mink: solving\u2026")
            else:
                status   = "\u2713" if m_res.converged else "\u2717"
                mean_err = float(np.mean(list(m_res.position_errors_m.values()))) * 1000
                lines.append(f"Mink: {status} {m_res.n_iters} iters"
                             f" | err {mean_err:.1f} mm")

        ax.text2D(0.02, 0.02, "\n".join(lines), transform=ax.transAxes,
                  fontsize=7.0, family="monospace", verticalalignment="bottom",
                  bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75))

        all_pts = np.array(list(wtips.values()) + [wbase])
        margin  = 0.025
        ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
        ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
        ax.set_zlim(all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)
        plt.draw()

    # ------------------------------------------------------------------
    # Geometry overlays (matplotlib)
    # ------------------------------------------------------------------
    def _draw_line_closure(self, ax, wtips, gz):
        t = wtips["thumb"];  i = wtips["index"]
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
        y_min = fpts[:, 1].min();  y_max = fpts[:, 1].max()
        x_nf  = fpts[:, 0].mean(); x_th  = th[0]
        corners = np.array([
            [x_nf, y_min, gz], [x_nf, y_max, gz],
            [x_th, y_max, gz], [x_th, y_min, gz],
        ])
        poly = Poly3DCollection([corners], alpha=0.10, facecolor="cyan",
                                edgecolor="steelblue", linewidth=1.2)
        ax.add_collection3d(poly)
        mid_x = (x_nf + x_th) / 2
        ax.plot([mid_x, mid_x], [y_min - 0.01, y_max + 0.01], [gz, gz],
                "g:", lw=1.5)

    def _draw_cylinder_closure(self, ax, wtips, gz, r: ClosureResult):
        fpts      = np.array([wtips[f] for f in NON_THUMB_FINGERS])
        cx        = fpts[:, 0].mean();  cz = fpts[:, 2].mean()
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
                        help="Serial port for real hand (e.g. /dev/ttyUSB0)")
    parser.add_argument("--robot", action="store_true",
                        help="Enable UR5+hand robot viewer buttons")
    parser.add_argument("--send-real", action="store_true",
                        help="Start with Send-to-Real enabled (requires --port)")
    parser.add_argument("--no-mink", action="store_true",
                        help="Disable mink IK comparison planner (faster startup)")
    parser.add_argument("--real-robot", action="store_true",
                        help="Enable real UR5 arm control panel (requires --ur5-ip)")
    parser.add_argument("--ur5-ip", default=None,
                        help="UR5 robot IP address (e.g. 192.168.0.4)")
    parser.add_argument("--ur5-speed", type=float, default=0.10,
                        help="UR5 arm linear speed in m/s (default 0.10)")
    args = parser.parse_args()

    viz = GraspViz(
        xml_path=args.xml,
        rebuild=args.rebuild,
        port=args.port,
        robot_mode=args.robot,
        send_real=args.send_real,
        mink_viz=not args.no_mink,
        real_robot=args.real_robot,
        ur5_ip=args.ur5_ip,
        ur5_speed=args.ur5_speed,
    )
    viz.run()


if __name__ == "__main__":
    main()
