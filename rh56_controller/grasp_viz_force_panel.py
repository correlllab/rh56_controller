"""
grasp_viz_force_panel.py — Force visualisation panel for GraspViz.

Two operating modes controlled by "Sim Geometry" toggle:

  Sim Geometry ON  — launches a MuJoCo subprocess with a snapping box,
                     uses physics contacts for GWS and Ferrari-Canny Q.
  Sim Geometry OFF — geometry-free, real-forces-only mode:
                     * GWS built from real hand forces + theoretical contact
                       normals (centroid direction from state_arr positions).
                     * Force closure decided by heuristic rules (no MuJoCo).
                     * External wrench check (can current GWS resist W?).

In either mode:
  - 6-bar force chart: 5 finger normals + thumb-yaw tangential (T_yaw).
  - Heuristic FC status always shown alongside formal Q.
  - External wrench check: Fx/Fy/Fz (world frame) + optional gravity.

Box snap (Sim mode only):
  Width auto-computed from fingertip spread; box oriented to hand frame.
  2-finger: centroid = thumb + index midpoint; N>2: + middle.
"""

import multiprocessing
import pathlib
import threading
import time
import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:
    from scipy.spatial import ConvexHull
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent
_FORCE_SCENE = str(_HERE / "h1_mujoco" / "inspire" / "inspire_force_scene.xml")

# ---------------------------------------------------------------------------
# Shared-memory layout
# ---------------------------------------------------------------------------
# box_params_arr [5]: [half_length, half_depth, friction, mass_kg, unused]
#   half_width is auto-computed from fingertip spread each frame.
_BOX_PARAMS_LEN = 5
_BOX_PARAMS_DEFAULT = [0.030, 0.020, 0.70, 0.100, 0.0]

# force_metrics_arr [26]:
#   [0:5]   per-finger normal force magnitude (N): thumb,index,middle,ring,pinky
#   [5]     Ferrari-Canny Q (< 0 → not force-closed)
#   [6]     is_force_closed (1.0 yes, 0.0 no)
#   [7]     n_contacts
#   [8:23]  5 finger × 3 actual tip positions (world frame, from sim FK)
#   [23:26] thumb tangential force direction unit vector (world frame)
_FORCE_METRICS_LEN = 26

_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Fingers for centroid per mode index
_CENTROID_FINGERS = {
    0: ["thumb", "index"],
    1: ["thumb", "index", "middle"],
    2: ["thumb", "index", "middle"],
    3: ["thumb", "index", "middle"],
    4: ["thumb", "index", "middle"],
}

_CONE_EDGES = 8

# Thumb yaw tangential calibration (mirrors rh56_hand.py)
_THUMB_YAW_CALIB     = (0.012547, -0.384)
_THUMB_YAW_REFF_POLY = (-0.029209, -0.063028, 0.091856)

# Bar chart labels / colours (5 normals + T_yaw)
_BAR_LABELS     = ["thmb", "idx", "mid", "ring", "pink", "T_yaw"]
_BAR_COLORS_5   = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db", "#9b59b6"]
_BAR_COLOR_TYAW = "#ff6b35"   # orange-red for thumb yaw tangential


# ---------------------------------------------------------------------------
# Rotation helpers — module-level so they are picklable for subprocesses
# ---------------------------------------------------------------------------

def _euler_xyz_to_mat(rx, ry, rz):
    """XYZ Euler angles → rotation matrix R = Rx @ Ry @ Rz."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0,  0 ], [0,  cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0,   1,   0 ], [-sy, 0, cy]])
    Rz = np.array([[cz,-sz, 0], [sz, cz,   0 ], [0,  0,  1]])
    return Rx @ Ry @ Rz


def _mat_to_quat_wxyz(R):
    """Rotation matrix → (w, x, y, z) quaternion (Shepperd's method)."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        return np.array([0.25 / s,
                         (R[2, 1] - R[1, 2]) * s,
                         (R[0, 2] - R[2, 0]) * s,
                         (R[1, 0] - R[0, 1]) * s])
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s,
                         (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s,
                         0.25 * s, (R[1, 2] + R[2, 1]) / s])
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s,
                         (R[1, 2] + R[2, 1]) / s, 0.25 * s])


# ---------------------------------------------------------------------------
# GWS helpers
# ---------------------------------------------------------------------------

def _compute_gws(contacts_pos, contacts_normal, forces_N, mu,
                 n_edges=_CONE_EDGES):
    """Build Grasp Wrench Space primitive wrenches (6D)."""
    if not contacts_pos:
        return np.zeros((0, 6)), np.zeros(3)
    centroid   = np.mean(contacts_pos, axis=0)
    primitives = []
    angles     = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)
    for pos, n, F in zip(contacts_pos, contacts_normal, forces_N):
        if F < 1e-4:
            continue
        n  = np.asarray(n, dtype=float)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n    = n / nn
        perp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        t1   = np.cross(n, perp); t1 /= np.linalg.norm(t1)
        t2   = np.cross(n, t1)
        r    = pos - centroid
        for theta in angles:
            f_vec = F * (n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2))
            tau   = np.cross(r, f_vec)
            primitives.append(np.concatenate([f_vec, tau]))
    return (np.array(primitives) if primitives else np.zeros((0, 6))), centroid


def _ferrari_canny_q(primitives):
    """Ferrari-Canny Q: min distance from origin to GWS hull."""
    if len(primitives) < 4 or not _HAS_SCIPY:
        return False, -1.0
    try:
        hull   = ConvexHull(primitives[:, :3])
        inside = np.all(
            hull.equations[:, :3] @ np.zeros(3) + hull.equations[:, 3] <= 1e-10)
        Q      = float(np.min(np.abs(hull.equations[:, 3])))
        return inside, Q if inside else -Q
    except Exception:
        return False, -1.0


def _gws_from_tips(tip_pts: Dict[str, np.ndarray], forces_5, mu,
                   mode_idx: int, extra_contact=None):
    """Build GWS from explicit tip positions + optional tangential contact."""
    if "thumb" not in tip_pts or "index" not in tip_pts:
        return np.zeros((0, 6)), np.zeros(3)
    cen_fnames = _CENTROID_FINGERS.get(mode_idx, ["thumb", "index", "middle"])
    cen_pts    = [tip_pts[f] for f in cen_fnames if f in tip_pts]
    if not cen_pts:
        return np.zeros((0, 6)), np.zeros(3)
    centroid = np.mean(cen_pts, axis=0)
    cp, cn, cf = [], [], []
    for i, fname in enumerate(_FINGER_ORDER):
        if fname not in tip_pts:
            continue
        F = float(forces_5[i]) if i < len(forces_5) else 0.0
        if F < 0.05:
            continue
        d  = centroid - tip_pts[fname]
        dn = np.linalg.norm(d)
        if dn < 1e-6:
            continue
        cp.append(tip_pts[fname]); cn.append(d / dn); cf.append(F)
    if extra_contact is not None:
        pos, direction, F = extra_contact
        dn = np.linalg.norm(direction)
        if abs(F) > 0.05 and dn > 1e-6:
            cp.append(np.asarray(pos))
            cn.append(np.asarray(direction) / dn)
            cf.append(abs(F))
    return _compute_gws(cp, cn, cf, mu)


def _gws_from_state(state_arr, forces_5, mu, extra_contact=None):
    """GWS from state_arr theoretical tip positions (geometry-free fallback)."""
    tip_pts: Dict[str, np.ndarray] = {}
    for i, fname in enumerate(_FINGER_ORDER):
        p = state_arr[5 + i * 3: 5 + i * 3 + 3]
        if not np.any(np.isnan(p)):
            tip_pts[fname] = p.copy()
    if "thumb" not in tip_pts or "index" not in tip_pts:
        return np.zeros((0, 6)), np.zeros(3)
    mode_idx = int(round(float(state_arr[1])))
    return _gws_from_tips(tip_pts, forces_5, mu, mode_idx,
                          extra_contact=extra_contact)


# ---------------------------------------------------------------------------
# Force-closure heuristics (geometry-free)
# ---------------------------------------------------------------------------

def _compute_heuristic_fc(forces_5):
    """
    Heuristic force closure from real sensor readings.

    Loss-of-contact conditions (any → not FC):
      1. thumb < 0.30 N  (thumb must always be in contact)
      2. index < 0.50 N  AND  middle < 0.30 N  (near-full pinch disengaged)

    Otherwise → assume FC (object being held by multiple contacts).

    Returns: (bool | None, str)  — None means no data available.
    """
    if forces_5 is None:
        return None, "No real data"
    thumb_N  = float(forces_5[0])
    index_N  = float(forces_5[1])
    middle_N = float(forces_5[2])
    if thumb_N < 0.30:
        return False, f"Thumb {thumb_N:.2f} N < 0.30 N"
    if index_N < 0.50 and middle_N < 0.30:
        return False, f"Index {index_N:.2f} N + Middle {middle_N:.2f} N both low"
    n_active = sum(1 for f in forces_5 if f > 0.10)
    return True, f"{n_active}/5 fingers > 0.10 N"


def _check_wrench_in_gws(primitives, wrench_f3):
    """
    Check if a 3D force wrench is inside the GWS force-subspace hull.

    Returns (inside: bool, margin: float).
    margin > 0: inside (N of safety); margin < 0: outside (|N| to resist it).
    """
    if len(primitives) < 4 or not _HAS_SCIPY:
        return False, 0.0
    try:
        hull   = ConvexHull(primitives[:, :3])
        vals   = hull.equations[:, :3] @ np.asarray(wrench_f3, float) + hull.equations[:, 3]
        inside = bool(np.all(vals <= 1e-10))
        margin = float(-np.max(vals))
        return inside, margin
    except Exception:
        return False, 0.0


# ---------------------------------------------------------------------------
# Subprocess worker (Sim Geometry mode only)
# ---------------------------------------------------------------------------

def _force_viewer_worker(xml_path: str, ctrl_arr, state_arr, box_params_arr,
                         force_metrics_arr, stop_event,
                         real_hand_arr=None) -> None:
    """
    Subprocess: floating-hand force scene viewer (sim-geometry mode).

    Finger pose: real_hand_arr[0:6] if non-zero, else ctrl_arr[6:12].
    Hand base:   ctrl_arr[0:6] always.
    Box snaps:   centroid + hand orientation + auto half-width each frame.
    Writes force_metrics_arr[0:26] each frame (see layout above).
    """
    import re
    import mujoco
    import mujoco.viewer

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        def _jadr(name):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            return int(model.jnt_qposadr[jid]) if jid >= 0 else -1

        jm = {
            "pos_x": _jadr("right_pos_x"), "pos_y": _jadr("right_pos_y"),
            "pos_z": _jadr("right_pos_z"), "rot_x": _jadr("right_rot_x"),
            "rot_y": _jadr("right_rot_y"), "rot_z": _jadr("right_rot_z"),
            "pinky":        _jadr("pinky_proximal_joint"),
            "pinky_inter":  _jadr("pinky_intermediate_joint"),
            "ring":         _jadr("ring_proximal_joint"),
            "ring_inter":   _jadr("ring_intermediate_joint"),
            "middle":       _jadr("middle_proximal_joint"),
            "middle_inter": _jadr("middle_intermediate_joint"),
            "index":        _jadr("index_proximal_joint"),
            "index_inter":  _jadr("index_intermediate_joint"),
            "thumb_yaw":    _jadr("thumb_proximal_yaw_joint"),
            "thumb_pitch":  _jadr("thumb_proximal_pitch_joint"),
            "thumb_inter":  _jadr("thumb_intermediate_joint"),
            "thumb_distal": _jadr("thumb_distal_joint"),
        }

        box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM,  "box")
        yaw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                         "thumb_proximal_base")

        _TIP_SITES = {
            "thumb":  "right_thumb_tip",  "index":  "right_index_tip",
            "middle": "right_middle_tip", "ring":   "right_ring_tip",
            "pinky":  "right_pinky_tip",
        }
        tip_site_ids = {}
        for fname, sname in _TIP_SITES.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                tip_site_ids[fname] = sid

        _FORCE_SENSORS = {
            "thumb":  "thumb_tip_force", "index":  "index_tip_force",
            "middle": "middle_tip_force", "ring":  "ring_tip_force",
            "pinky":  "pinky_tip_force",
        }
        sensor_addrs = {}
        for fname, sname in _FORCE_SENSORS.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid >= 0:
                sensor_addrs[fname] = int(model.sensor_adr[sid])

        _patterns = [
            (re.compile(r"^thumb_"),  "thumb"),
            (re.compile(r"^index_"),  "index"),
            (re.compile(r"^middle_"), "middle"),
            (re.compile(r"^ring_"),   "ring"),
            (re.compile(r"^pinky_"),  "pinky"),
        ]
        body_to_finger = {}
        for bid in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            for pat, fn in _patterns:
                if pat.match(bname):
                    body_to_finger[bid] = fn
                    break

        dt     = model.opt.timestep
        settle = max(1, round(0.010 / dt))

        _FINGER_RGBA = {
            "thumb":  np.array([0.9, 0.3, 0.2, 0.9], dtype=np.float32),
            "index":  np.array([0.9, 0.55, 0.1, 0.9], dtype=np.float32),
            "middle": np.array([0.2, 0.8, 0.4, 0.9], dtype=np.float32),
            "ring":   np.array([0.2, 0.5, 0.9, 0.9], dtype=np.float32),
            "pinky":  np.array([0.6, 0.3, 0.9, 0.9], dtype=np.float32),
        }

        with mujoco.viewer.launch_passive(model, data,
                                          show_left_ui=False,
                                          show_right_ui=False) as v:
            while v.is_running() and not stop_event.is_set():
                ctrl  = np.array(ctrl_arr[:])
                state = np.array(state_arr[:])
                bpar  = np.array(box_params_arr[:])

                real_fc  = (np.array(real_hand_arr[:])
                            if real_hand_arr is not None else None)
                use_real = (real_fc is not None
                            and np.any(np.abs(real_fc) > 1e-6))
                if use_real:
                    pinky, ring, middle, index, pitch, yaw = real_fc[0:6]
                else:
                    pinky, ring, middle, index, pitch, yaw = ctrl[6:12]

                pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = ctrl[0:6]
                data.qpos[jm["pos_x"]] = pos_x; data.qpos[jm["pos_y"]] = pos_y
                data.qpos[jm["pos_z"]] = pos_z; data.qpos[jm["rot_x"]] = rot_x
                data.qpos[jm["rot_y"]] = rot_y; data.qpos[jm["rot_z"]] = rot_z
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
                data.ctrl[0:12] = ctrl[0:12]

                mujoco.mj_kinematics(model, data)
                tip_positions = {}
                for fname, sid in tip_site_ids.items():
                    tip_positions[fname] = data.site_xpos[sid].copy()

                hl, hd, friction, mass = (float(bpar[i]) for i in range(4))
                hl = max(hl, 0.003); hd = max(hd, 0.003); mass = max(mass, 0.001)

                mode_idx  = int(round(float(state[1])))
                cen_names = _CENTROID_FINGERS.get(mode_idx, ["thumb", "index", "middle"])
                cen_pts   = [tip_positions[f] for f in cen_names if f in tip_positions]
                centroid  = np.mean(cen_pts, axis=0) if cen_pts else np.zeros(3)

                R      = _euler_xyz_to_mat(rot_x, rot_y, rot_z)
                hand_x = R[:, 0]
                hw     = 0.005
                for tip in tip_positions.values():
                    hw = max(hw, abs(float(np.dot(tip - centroid, hand_x))))

                if box_body_id >= 0 and box_geom_id >= 0:
                    model.body_pos[box_body_id]  = centroid
                    model.body_quat[box_body_id] = _mat_to_quat_wxyz(R)
                    model.geom_size[box_geom_id] = [hw, hl, hd]
                    model.geom_friction[box_geom_id, 0] = friction
                if box_body_id >= 0:
                    model.body_mass[box_body_id] = mass

                # Thumb tangential direction (z_yaw × r_hat)
                tang_dir = np.zeros(3)
                if yaw_body_id >= 0:
                    p_yaw  = data.xpos[yaw_body_id].copy()
                    R_base = data.xmat[yaw_body_id].reshape(3, 3)
                    z_yaw  = R_base @ np.array([0.0, 0.0, -1.0])
                    r_vec  = tip_positions.get("thumb", p_yaw) - p_yaw
                    r_perp = r_vec - np.dot(r_vec, z_yaw) * z_yaw
                    rp_n   = np.linalg.norm(r_perp)
                    if rp_n > 1e-4:
                        r_hat    = r_perp / rp_n
                        tang_dir = np.cross(z_yaw, r_hat)
                        tn = np.linalg.norm(tang_dir)
                        if tn > 1e-9:
                            tang_dir /= tn

                mujoco.mj_forward(model, data)
                for _ in range(settle):
                    mujoco.mj_step(model, data)

                fn_per_finger = {}
                for fname in _FINGER_ORDER:
                    adr = sensor_addrs.get(fname)
                    sid = tip_site_ids.get(fname)
                    if adr is not None and sid is not None:
                        f_local = data.sensordata[adr:adr + 3].copy()
                        xmat    = data.site_xmat[sid].reshape(3, 3)
                        fn_per_finger[fname] = float(np.linalg.norm(xmat @ f_local))
                    else:
                        fn_per_finger[fname] = 0.0

                contact_pos, contact_normals, contact_forces = [], [], []
                result_buf = np.zeros(6)
                for i in range(data.ncon):
                    c  = data.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    fname = None; normal_sign = 1.0
                    if g1 == box_geom_id:
                        fname = body_to_finger.get(int(model.geom_bodyid[g2]))
                        normal_sign = -1.0
                    elif g2 == box_geom_id:
                        fname = body_to_finger.get(int(model.geom_bodyid[g1]))
                    if fname is None:
                        continue
                    mujoco.mj_contactForce(model, data, i, result_buf)
                    if result_buf[0] < 1e-4:
                        continue
                    contact_pos.append(c.pos.copy())
                    contact_normals.append(c.frame.reshape(3, 3)[0] * normal_sign)
                    contact_forces.append(float(result_buf[0]))

                primitives, _ = _compute_gws(contact_pos, contact_normals,
                                             contact_forces, friction)
                is_fc, Q = _ferrari_canny_q(primitives)

                for i, fname in enumerate(_FINGER_ORDER):
                    force_metrics_arr[i] = fn_per_finger.get(fname, 0.0)
                force_metrics_arr[5] = Q
                force_metrics_arr[6] = 1.0 if is_fc else 0.0
                force_metrics_arr[7] = float(len(contact_pos))
                for i, fname in enumerate(_FINGER_ORDER):
                    force_metrics_arr[8 + i * 3: 8 + i * 3 + 3] = tip_positions.get(
                        fname, np.zeros(3))
                force_metrics_arr[23:26] = tang_dir

                # Draw overlays
                scn = v.user_scn
                scn.ngeom = 0

                def _sphere(pos, radius, rgba):
                    if scn.ngeom >= scn.maxgeom: return
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_SPHERE,
                                        np.array([radius] * 3),
                                        np.asarray(pos, float),
                                        np.eye(3).flatten(),
                                        np.asarray(rgba, np.float32))
                    scn.ngeom += 1

                def _line(p0, p1, rgba, width=0.002):
                    if scn.ngeom >= scn.maxgeom: return
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                                        np.zeros(3), np.zeros(3), np.zeros(9),
                                        np.asarray(rgba, np.float32))
                    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                         np.asarray(p0, float), np.asarray(p1, float))
                    scn.ngeom += 1

                for fname, sid in tip_site_ids.items():
                    _sphere(data.site_xpos[sid], 0.003,
                            _FINGER_RGBA.get(fname, np.array([0.8, 0.8, 0.2, 0.9])))

                if box_body_id >= 0 and box_geom_id >= 0:
                    bc   = model.body_pos[box_body_id].copy()
                    corners = []
                    for sx in [hw, -hw]:
                        for sy in [hl, -hl]:
                            for sz in [hd, -hd]:
                                corners.append(bc + sx * R[:, 0] + sy * R[:, 1] + sz * R[:, 2])
                    box_rgba = np.array([0.6, 0.85, 0.6, 0.6], np.float32)
                    for a, b in [(0,1),(2,3),(4,5),(6,7),
                                 (0,2),(1,3),(4,6),(5,7),
                                 (0,4),(1,5),(2,6),(3,7)]:
                        _line(corners[a], corners[b], box_rgba)

                v.sync()
                time.sleep(0.033)

    except Exception as exc:
        print(f"[ForceViewerWorker] {exc}")
        import traceback; traceback.print_exc()


# ---------------------------------------------------------------------------
# ForceVizPanel — Toplevel window
# ---------------------------------------------------------------------------

class ForceVizPanel:
    """
    Tkinter Toplevel window for force quality visualisation.

    Parameters
    ----------
    root      : tk.Tk parent
    ctrl_arr  : shared mp Array (12 doubles) — hand ctrl (pose + fingers)
    state_arr : shared mp Array — grasp state (mode, tips, ...)
    mp_ctx    : multiprocessing context ('fork')
    hand      : optional RH56Hand for real force readings
    fk        : optional InspireHandFK for ctrl range lookups
    """

    def __init__(self, root: tk.Tk, ctrl_arr, state_arr, mp_ctx,
                 hand=None, fk=None):
        self._root      = root
        self._ctrl_arr  = ctrl_arr
        self._state_arr = state_arr
        self._mp_ctx    = mp_ctx
        self._hand      = hand
        self._fk        = fk

        self._win = tk.Toplevel(root)
        self._win.title("Force Viz")
        self._win.resizable(True, True)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        self._box_params_arr    = mp_ctx.Array("d", _BOX_PARAMS_DEFAULT)
        self._force_metrics_arr = mp_ctx.Array("d", [0.0] * _FORCE_METRICS_LEN)
        self._real_hand_arr     = mp_ctx.Array("d", [0.0] * 6)
        self._stop_event        = mp_ctx.Event()
        self._viewer_proc: Optional[multiprocessing.Process] = None

        # Visualization toggles
        self._show_sim_geo    = tk.BooleanVar(value=True)
        self._show_sim_gws    = tk.BooleanVar(value=True)
        self._show_fc         = tk.BooleanVar(value=True)
        self._show_tangential = tk.BooleanVar(value=False)
        # Wrench check
        self._live_wrench     = tk.BooleanVar(value=False)
        self._wrench_gravity  = tk.BooleanVar(value=True)
        self._wrench_vars: Dict[str, tk.DoubleVar] = {}

        self._build_ui()
        self._update_running = True

        if self._hand is not None and self._fk is not None:
            threading.Thread(target=self._real_hand_poll_loop,
                             daemon=True, name="fv-real-poll").start()

        threading.Thread(target=self._metrics_update_loop,
                         daemon=True, name="force-metrics").start()

    # ------------------------------------------------------------------
    # Real hand position poll
    # ------------------------------------------------------------------

    def _real_hand_poll_loop(self):
        """Background: angle_read() → ctrl space → _real_hand_arr."""
        from .grasp_viz_workers import _ACTUATOR_ORDER
        ctrl_min = np.array([self._fk.ctrl_min[a] for a in _ACTUATOR_ORDER])
        ctrl_max = np.array([self._fk.ctrl_max[a] for a in _ACTUATOR_ORDER])
        rng      = ctrl_max - ctrl_min
        while self._update_running:
            try:
                angles    = self._hand.angle_read()
                real_ctrl = ctrl_min + (1.0 - np.clip(
                    np.array(angles, float) / 1000.0, 0.0, 1.0)) * rng
                self._real_hand_arr[:] = real_ctrl
            except Exception:
                pass
            time.sleep(0.08)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._win.columnconfigure(0, weight=1)
        self._win.columnconfigure(1, weight=3)
        self._win.rowconfigure(0, weight=1)

        cf = ttk.LabelFrame(self._win, text="Controls", padding=6)
        cf.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        cf.columnconfigure(1, weight=1)
        r = 0

        # ── Sim Geometry ──
        ttk.Label(cf, text="── Sim Geometry ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Use Sim Geometry",
                        variable=self._show_sim_geo,
                        command=self._on_sim_geo_toggle).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        self._btn_launch = tk.Button(cf, text="Launch Viewer",
                                     command=self._launch_force_viewer)
        self._btn_launch.grid(row=r, column=0, columnspan=2, sticky="ew", pady=1)
        tk.Button(cf, text="Close", command=self._close_force_viewer).grid(
            row=r, column=2, sticky="ew", pady=1); r += 1

        ttk.Separator(cf, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        # ── Object Properties ──
        ttk.Label(cf, text="── Object ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._box_vars = {}
        params = [
            ("Length (mm)", "hl",   5, 200, _BOX_PARAMS_DEFAULT[0] * 1000),
            ("Depth  (mm)", "hd",   5, 100, _BOX_PARAMS_DEFAULT[1] * 1000),
            ("Friction",    "mu",   0,   2, _BOX_PARAMS_DEFAULT[2]),
            ("Mass   (g)",  "mass", 1, 1000, _BOX_PARAMS_DEFAULT[3] * 1000),
        ]
        for label, key, vmin, vmax, vinit in params:
            ttk.Label(cf, text=label).grid(row=r, column=0, sticky="w")
            var = tk.DoubleVar(value=vinit)
            sl  = ttk.Scale(cf, from_=vmin, to=vmax, variable=var,
                            orient="horizontal", length=110,
                            command=lambda _v, k=key: self._on_param_change(k))
            sl.grid(row=r, column=1, sticky="ew")
            ent = tk.Entry(cf, width=6)
            ent.insert(0, f"{vinit:.1f}")
            ent.grid(row=r, column=2, padx=2)
            ent.bind("<Return>",   lambda e, s=sl, v=var, en=ent: s.set(
                float(en.get() or v.get())))
            ent.bind("<FocusOut>", lambda e, s=sl, v=var, en=ent: s.set(
                float(en.get() or v.get())))
            self._box_vars[key] = (var, ent, sl, vmin, vmax)
            r += 1

        ttk.Separator(cf, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        # ── Visualization ──
        ttk.Label(cf, text="── Visualization ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Sim GWS overlay",
                        variable=self._show_sim_gws).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Ferrari-Canny Q",
                        variable=self._show_fc).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Thumb Tangential",
                        variable=self._show_tangential,
                        state="normal" if self._hand is not None else "disabled").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Separator(cf, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        # ── External Wrench Check ──
        ttk.Label(cf, text="── Wrench Check ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Live check",
                        variable=self._live_wrench).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(cf, text="Include gravity (from mass)",
                        variable=self._wrench_gravity).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        for axis in ("Fx", "Fy", "Fz"):
            ttk.Label(cf, text=f"{axis} (N):").grid(row=r, column=0, sticky="w")
            var = tk.DoubleVar(value=0.0)
            ent = tk.Entry(cf, width=7, textvariable=var)
            ent.grid(row=r, column=1, columnspan=2, sticky="ew", padx=2)
            self._wrench_vars[axis.lower()] = var
            r += 1

        ttk.Separator(cf, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=3); r += 1

        ttk.Label(cf, text="── Status ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        self._status_lbl = ttk.Label(cf, text="Ready", foreground="#333",
                                     wraplength=200)
        self._status_lbl.grid(row=r, column=0, columnspan=3, sticky="nw"); r += 1

        # Right column: matplotlib
        pf = ttk.LabelFrame(self._win, text="Force Metrics", padding=4)
        pf.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        pf.rowconfigure(0, weight=1)
        pf.columnconfigure(0, weight=1)

        self._fig = plt.figure(figsize=(6, 6))
        gs = GridSpec(5, 1, figure=self._fig, hspace=0.4)
        self._ax_bar = self._fig.add_subplot(gs[0])
        self._ax_gws = self._fig.add_subplot(gs[1:], projection="3d")

        self._canvas_fp = FigureCanvasTkAgg(self._fig, master=pf)
        self._canvas_fp.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._draw_empty_plots()
        self._on_sim_geo_toggle()  # set initial button states

    def _on_sim_geo_toggle(self):
        """Enable/disable sim viewer buttons based on sim-geo checkbox."""
        state = "normal" if self._show_sim_geo.get() else "disabled"
        if hasattr(self, "_btn_launch"):
            self._btn_launch.config(state=state)

    # ------------------------------------------------------------------
    # Param → shared memory
    # ------------------------------------------------------------------

    def _on_param_change(self, key: str):
        var, ent, sl, vmin, vmax = self._box_vars[key]
        val = float(var.get())
        ent.delete(0, "end")
        ent.insert(0, f"{val:.2f}")
        idx_map = {"hl": 0, "hd": 1, "mu": 2, "mass": 3}
        i = idx_map[key]
        self._box_params_arr[i] = (val / 1000.0 if key in ("hl", "hd", "mass") else val)

    # ------------------------------------------------------------------
    # Viewer process
    # ------------------------------------------------------------------

    def _launch_force_viewer(self):
        if self._viewer_proc is not None and self._viewer_proc.is_alive():
            self._set_status("Force viewer already running.")
            return
        self._stop_event.clear()
        self._viewer_proc = self._mp_ctx.Process(
            target=_force_viewer_worker,
            args=(_FORCE_SCENE, self._ctrl_arr, self._state_arr,
                  self._box_params_arr, self._force_metrics_arr,
                  self._stop_event, self._real_hand_arr),
            daemon=True,
        )
        self._viewer_proc.start()
        self._set_status("Force viewer launched.")

    def _close_force_viewer(self):
        self._stop_event.set()
        if self._viewer_proc is not None:
            self._viewer_proc.terminate()
            self._viewer_proc = None
        self._set_status("Force viewer closed.")

    # ------------------------------------------------------------------
    # Metrics update loop
    # ------------------------------------------------------------------

    def _metrics_update_loop(self):
        while self._update_running:
            time.sleep(0.25)
            try:
                self._win.after(0, self._refresh_plots)
            except Exception:
                break

    def _refresh_plots(self):
        if not self._win.winfo_exists():
            return

        # --- Sim data (from MuJoCo subprocess) ---
        metrics    = np.array(self._force_metrics_arr[:])
        sim_f5     = metrics[0:5]
        Q_sim      = float(metrics[5])
        is_fc_sim  = bool(metrics[6] > 0.5)
        n_con      = int(metrics[7])
        sim_tips: Dict[str, np.ndarray] = {}
        for i, fname in enumerate(_FINGER_ORDER):
            tp = metrics[8 + i * 3: 8 + i * 3 + 3]
            if np.all(np.isfinite(tp)) and np.any(tp != 0):
                sim_tips[fname] = tp
        tang_dir_world = metrics[23:26].copy()

        state    = np.array(self._state_arr[:])
        mode_idx = int(round(float(state[1])))
        mu       = float(self._box_params_arr[2])
        use_sim  = self._show_sim_geo.get()

        # --- Real forces ---
        real_f5  = None
        tang_N   = 0.0
        tang_info = None
        if self._hand is not None:
            try:
                raw = self._hand.force_act()
                c   = 0.012547
                real_f5 = np.array([
                    abs(float(raw[4])) * c,
                    abs(float(raw[3])) * c,
                    abs(float(raw[2])) * c,
                    abs(float(raw[1])) * c,
                    abs(float(raw[0])) * c,
                ])
            except Exception:
                real_f5 = None

            if self._show_tangential.get() and real_f5 is not None:
                try:
                    raw_f = self._hand.force_act()
                    raw_a = self._hand.angle_read()
                    tr    = self._hand.thumb_yaw_tangential_N(
                        float(raw_f[5]), int(raw_a[4]))
                    tang_N = tr["signed_N"]
                    thumb_tip = sim_tips.get("thumb")
                    tang_dn   = np.linalg.norm(tang_dir_world)
                    if thumb_tip is not None and abs(tang_N) > 0.05 and tang_dn > 0.5:
                        tang_info = (thumb_tip, tang_dir_world / tang_dn, tang_N)
                except Exception:
                    pass

        # --- Heuristic FC (always computed from real forces) ---
        h_fc, h_reason = _compute_heuristic_fc(real_f5)

        # --- GWS primitives (from real forces + geometric normals) ---
        # Used for: Real GWS visualization, wrench check, FC overlay
        gws_prims = np.zeros((0, 6))
        if real_f5 is not None:
            if sim_tips:
                gws_prims, _ = _gws_from_tips(
                    sim_tips, real_f5, mu, mode_idx, extra_contact=tang_info)
            else:
                gws_prims, _ = _gws_from_state(
                    state, real_f5, mu, extra_contact=tang_info)

        # GWS-based FC (from real forces geometry, not sim contacts)
        is_fc_real, Q_real = _ferrari_canny_q(gws_prims)

        # Sim GWS primitives (only when sim is on and viewer running)
        sim_gws_prims = np.zeros((0, 6))
        if use_sim and n_con > 0:
            # Build sim GWS from sim contact forces + sim tip positions
            if sim_tips:
                sim_gws_prims, _ = _gws_from_tips(
                    sim_tips, sim_f5, mu, mode_idx)

        # --- Wrench check ---
        wrench_result = None
        if self._live_wrench.get():
            W = self._get_ext_wrench()
            if W is not None and len(gws_prims) >= 4:
                inside, margin = _check_wrench_in_gws(gws_prims, W)
                wrench_result = (W, inside, margin)

        # --- Build 6-element force vectors for bar chart ---
        sim_f6  = np.append(sim_f5,  0.0)            # no sim tangential
        real_f6 = (np.append(real_f5, abs(tang_N))
                   if real_f5 is not None else None)

        # Bar chart: show sim if sim_geo is on (and viewer running), always show real
        self._draw_bar_chart(
            sim_f6 if (use_sim and n_con > 0) else None,
            real_f6)

        self._draw_gws_3d(gws_prims, sim_gws_prims if self._show_sim_gws.get() else None,
                          is_fc_real, Q_real,
                          h_fc, h_reason,
                          is_fc_sim if use_sim else None, Q_sim,
                          wrench_result)
        self._canvas_fp.draw_idle()

        # Status bar
        h_label = "✓" if h_fc is True else ("✗" if h_fc is False else "?")
        fc_label = "✓" if is_fc_real else "✗"
        msg = f"Heuristic: {h_label} ({h_reason})   GWS FC: {fc_label} Q={Q_real:.3f}"
        if use_sim and n_con > 0:
            msg += f"  Sim: {'✓' if is_fc_sim else '✗'} ({n_con} contacts)"
        self._set_status(msg)

    def _get_ext_wrench(self):
        """Assemble external wrench [Fx, Fy, Fz] in world frame."""
        try:
            fx = float(self._wrench_vars["fx"].get())
            fy = float(self._wrench_vars["fy"].get())
            fz = float(self._wrench_vars["fz"].get())
        except Exception:
            fx = fy = fz = 0.0
        if self._wrench_gravity.get():
            try:
                mass_kg = float(self._box_params_arr[3])
            except Exception:
                mass_kg = 0.0
            fz -= mass_kg * 9.81
        W = np.array([fx, fy, fz])
        return W if np.linalg.norm(W) > 1e-3 else None

    # ------------------------------------------------------------------
    # Plot helpers
    # ------------------------------------------------------------------

    def _draw_empty_plots(self):
        ax = self._ax_bar
        ax.cla()
        ax.bar(np.arange(6), [0] * 6,
               color=_BAR_COLORS_5 + [_BAR_COLOR_TYAW], alpha=0.8)
        ax.set_xticks(np.arange(6))
        ax.set_xticklabels(_BAR_LABELS, fontsize=7)
        ax.set_ylabel("F (N)", fontsize=7); ax.set_ylim(0, 5)
        ax.tick_params(labelsize=7)
        ax.set_title("Fingertip forces", fontsize=8, pad=2)

        ax3 = self._ax_gws
        ax3.cla()
        ax3.set_title("GWS — Force Subspace", fontsize=9)
        ax3.text2D(0.5, 0.5, "Waiting for real forces…",
                   transform=ax3.transAxes, ha="center", va="center",
                   color="gray", fontsize=10)
        self._canvas_fp.draw_idle()

    def _draw_bar_chart(self, sim_f6, real_f6):
        ax = self._ax_bar
        ax.cla()
        x      = np.arange(6)
        colors = _BAR_COLORS_5 + [_BAR_COLOR_TYAW]
        have_s = sim_f6 is not None
        have_r = real_f6 is not None

        if have_s:
            ax.bar(x - (0.2 if have_r else 0), sim_f6,
                   0.35 if have_r else 0.6,
                   label="Sim", color=colors, alpha=0.85)
        if have_r:
            ax.bar(x + (0.2 if have_s else 0), real_f6,
                   0.35 if have_s else 0.6,
                   label="Real", color=colors, alpha=0.55, hatch="//")
        if have_s and have_r:
            ax.legend(fontsize=6, loc="upper right")

        ax.set_xticks(x)
        ax.set_xticklabels(_BAR_LABELS, fontsize=7)
        ax.set_ylabel("F (N)", fontsize=7)
        ax.set_title("Fingertip forces", fontsize=8, pad=2)
        all_vals = []
        if have_s: all_vals.extend(sim_f6)
        if have_r: all_vals.extend(real_f6)
        peak = max(max(all_vals) if all_vals else 0, 0.1)
        ax.set_ylim(0, peak * 1.4)
        ax.tick_params(labelsize=7)

    def _draw_gws_3d(self, real_prims, sim_prims,
                     is_fc_real, Q_real,
                     h_fc, h_reason,
                     is_fc_sim, Q_sim,
                     wrench_result):
        ax = self._ax_gws
        ax.cla()
        ax.set_title("GWS — Force Subspace", fontsize=9)

        lim = 0.5  # minimum axis range

        def _add_hull(prims, color, alpha):
            nonlocal lim
            if len(prims) < 4 or not _HAS_SCIPY:
                return False
            try:
                hull  = ConvexHull(prims[:, :3])
                verts = [prims[s, :3] for s in hull.simplices]
                ax.add_collection3d(Poly3DCollection(
                    verts, alpha=alpha, facecolor=color,
                    edgecolor="steelblue", linewidth=0.3))
                lim = max(lim, float(np.max(np.abs(prims[:, :3]))) * 1.3 + 0.01)
                return True
            except Exception:
                return False

        drawn = _add_hull(real_prims, "#3498db", 0.30)   # real GWS (primary)
        if sim_prims is not None:
            _add_hull(sim_prims, "#e67e22", 0.20)         # sim GWS (overlay)

        if not drawn:
            ax.text2D(0.5, 0.5,
                      "No real forces detected.\nConnect hand to see GWS.",
                      transform=ax.transAxes, ha="center", va="center",
                      color="gray", fontsize=9)

        ax.set_xlim3d(-lim, lim); ax.set_ylim3d(-lim, lim); ax.set_zlim3d(-lim, lim)
        ax.scatter([0], [0], [0], color="red", s=60, marker="+", zorder=10)

        # Wrench arrow
        if wrench_result is not None:
            W, inside, margin = wrench_result
            wn = np.linalg.norm(W)
            if wn > 1e-6:
                scale   = lim * 0.75 / wn
                w_s     = W * scale
                w_color = "#2ecc71" if inside else "#e74c3c"
                ax.quiver(0, 0, 0, w_s[0], w_s[1], w_s[2],
                          color=w_color, arrow_length_ratio=0.15, linewidth=2)
                m_str = (f"+{margin:.3f} N margin" if inside
                         else f"{margin:.3f} N outside")
                ax.text2D(0.05, 0.85, f"Wrench: {m_str}",
                          transform=ax.transAxes, fontsize=7, color=w_color,
                          bbox=dict(fc="white", alpha=0.8, boxstyle="round,pad=0.2"))

        # FC annotations — heuristic (primary) + formal Q
        if h_fc is None:
            h_color, h_sym = "gray", "?"
        elif h_fc:
            h_color, h_sym = "#2ecc71", "✓"
        else:
            h_color, h_sym = "#e74c3c", "✗"

        ax.text2D(0.05, 0.97,
                  f"Heuristic: {h_sym} FC",
                  transform=ax.transAxes, ha="left", va="top",
                  fontsize=9, color=h_color,
                  bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.2"))
        ax.text2D(0.05, 0.90,
                  h_reason,
                  transform=ax.transAxes, ha="left", va="top",
                  fontsize=7, color="#444")

        if self._show_fc.get():
            fc_color = "#2ecc71" if is_fc_real else "#e74c3c"
            q_str    = f"Q={Q_real:.4f}" if not np.isnan(Q_real) else ""
            ax.text2D(0.55, 0.97,
                      f"GWS: {'✓' if is_fc_real else '✗'} {q_str}",
                      transform=ax.transAxes, ha="left", va="top",
                      fontsize=8, color=fc_color,
                      bbox=dict(fc="white", alpha=0.85, boxstyle="round,pad=0.2"))

            if is_fc_sim is not None:
                sim_color = "#2ecc71" if is_fc_sim else "#e74c3c"
                ax.text2D(0.55, 0.89,
                          f"Sim: {'✓' if is_fc_sim else '✗'} Q={Q_sim:.4f}",
                          transform=ax.transAxes, ha="left", va="top",
                          fontsize=7, color=sim_color)

        ax.set_xlabel("Fx (N)", fontsize=7, labelpad=1)
        ax.set_ylabel("Fy (N)", fontsize=7, labelpad=1)
        ax.set_zlabel("Fz (N)", fontsize=7, labelpad=1)
        ax.tick_params(labelsize=6)

    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        try:
            self._status_lbl.config(text=msg)
        except Exception:
            pass

    def _on_close(self):
        self._update_running = False
        self._close_force_viewer()
        plt.close(self._fig)
        self._win.destroy()
