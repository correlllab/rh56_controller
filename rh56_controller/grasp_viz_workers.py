"""
grasp_viz_workers.py — Subprocess worker functions and shared constants for GraspViz.

These module-level functions run inside separate multiprocessing.Process instances
(one per open MuJoCo viewer window).  They are in a separate module so that:
  1. They can be forked cleanly without importing the UI or core stacks.
  2. Constants can be shared across grasp_viz_core.py and grasp_viz_ui.py.
"""

import time
import pathlib
from typing import Dict

import numpy as np
import mujoco
import mujoco.viewer

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent
_GRASP_SCENE = str(_HERE / "h1_mujoco" / "inspire" / "inspire_grasp_scene.xml")
_ROBOT_SCENE = str(_HERE / "h1_mujoco" / "inspire" / "ur5_inspire.xml")
_RIGHT_SCENE = str(_HERE / "h1_mujoco" / "inspire" / "inspire_right.xml")

# ---------------------------------------------------------------------------
# Actuator / viewer constants
# ---------------------------------------------------------------------------
_ACTUATOR_ORDER = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

_DEFAULT_ROBOT_Z = 0.17
_DEFAULT_ROBOT_X = -0.25
_DEFAULT_ROBOT_Y = -0.67

_IK_DT        = 0.05
_IK_MAX_ITERS = 5
_IK_POS_THR   = 5e-3
_IK_ORI_THR   = 0.05

# eeff site local position in hand base body frame (from ur5_inspire.xml)
_EEFF_LOCAL = np.array([0.070, 0.016, 0.155])

# Colour palette per finger (matplotlib / tkinter)
FINGER_COLORS = {
    "thumb":  "#e74c3c",
    "index":  "#e67e22",
    "middle": "#2ecc71",
    "ring":   "#3498db",
    "pinky":  "#9b59b6",
}

# Viewer geom RGBA per finger (float32, MuJoCo)
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
# Viewer shared-memory layout
#
# ctrl_arr  [12]: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z,
#                  pinky, ring, middle, index, thumb_pitch, thumb_yaw]
#
# state_arr [20]: [gz, mode_idx, cyl_radius, robot_x, robot_y,
#                  tip_thumb(3), tip_index(3), tip_middle(3),
#                  tip_ring(3),  tip_pinky(3)]
#                 (tip positions are world-frame; NaN if not active)
# ---------------------------------------------------------------------------
_VIEWER_CTRL_LEN  = 12
_VIEWER_STATE_LEN = 20
_VIEWER_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

_VIEWER_MODE_ACTIVE = {
    0: ["thumb", "index"],
    1: ["thumb", "index", "middle"],
    2: ["thumb", "index", "middle", "ring"],
    3: ["thumb", "index", "middle", "ring", "pinky"],
    4: ["thumb", "index", "middle", "ring", "pinky"],
}


# ---------------------------------------------------------------------------
# Rotation helpers (module-level so subprocess workers can use them directly)
# ---------------------------------------------------------------------------
def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def _mat_to_xyz_euler(R: np.ndarray):
    """Extract (a, b, c) s.t. Rx(a) @ Ry(b) @ Rz(c) = R (XYZ intrinsic Euler)."""
    b = float(np.arcsin(np.clip(R[0, 2], -1.0, 1.0)))
    cb = np.cos(b)
    if abs(cb) > 1e-6:
        a = float(np.arctan2(-R[1, 2], R[2, 2]))
        c = float(np.arctan2(-R[0, 1], R[0, 0]))
    else:
        a = float(np.arctan2(R[2, 1], R[1, 1]))
        c = 0.0
    return a, b, c


# ---------------------------------------------------------------------------
# Worker functions (run in subprocesses — no imports from grasp_viz_core/ui)
# ---------------------------------------------------------------------------

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
    """Apply 12-element ctrl vector to qpos with joint coupling."""
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
    """Draw grasp geometry overlay from pre-computed world-frame tip positions."""
    mode_idx   = int(round(state[1]))
    cyl_radius = float(state[2])
    mode       = MODES[mode_idx] if 0 <= mode_idx < len(MODES) else ""

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
    """Subprocess entry: passive floating-hand MuJoCo viewer."""
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
                          real_tracking=None,
                          sim_grasp_t=None,
                          ctrl_open_fingers=None) -> None:
    """Subprocess entry: UR5+hand robot viewer with mink differential arm IK."""
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

        _ctrl_open = (np.array(ctrl_open_fingers)
                      if ctrl_open_fingers is not None else None)

        with mujoco.viewer.launch_passive(model, data) as v:
            while v.is_running() and not stop_event.is_set():
                ctrl = np.array(ctrl_arr[:])

                t_grasp = sim_grasp_t.value if sim_grasp_t is not None else 1.0
                if _ctrl_open is not None and 0.0 <= t_grasp < 1.0:
                    eff_finger = _ctrl_open + t_grasp * (ctrl[6:12] - _ctrl_open)
                else:
                    eff_finger = ctrl[6:12]

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
                        for ctrl_id, val in zip(finger_ctrl_ids, eff_finger):
                            data.ctrl[ctrl_id] = val
                        mujoco.mj_forward(model, data)
                        configuration.update(data.qpos)
                elif eeff_id >= 0:
                    R_hand     = _Rx(ctrl[3]) @ _Ry(ctrl[4]) @ _Rz(ctrl[5])
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
                    for ctrl_id, val in zip(finger_ctrl_ids, eff_finger):
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
