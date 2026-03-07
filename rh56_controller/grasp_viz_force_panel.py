"""
grasp_viz_force_panel.py — Force visualisation panel for GraspViz.

Opens a Toplevel window with:
  - Object (box) parameter controls: width, length, depth, friction, mass
  - MuJoCo force viewer: floating hand + box with physics, force sensors
  - Matplotlib force plots: per-finger bar chart and optional wrench cone
  - Checkboxes: Sim wrench cone, Real wrench cone, Ferrari-Canny metric

The force viewer is a separate subprocess (like the hand/robot viewers).
Force metrics are written back via shared multiprocessing arrays and read
by a background thread that updates the matplotlib figure every 300 ms.

Box placement:
  2-finger line  → midpoint of thumb and index tips
  N > 2 fingers  → centroid of thumb + index + middle tips (user's spec)
  cylinder       → centroid of thumb + index + middle tips
"""

import multiprocessing
import pathlib
import threading
import time
import tkinter as tk
import tkinter.scrolledtext as scrolledtext
from tkinter import ttk
from typing import Optional, Dict

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
# box_params_arr [5]: [half_width, half_length, half_depth, friction, mass_kg]
_BOX_PARAMS_LEN = 5
_BOX_PARAMS_DEFAULT = [0.020, 0.030, 0.020, 0.70, 0.100]

# force_metrics_arr [8]:
#   [0:5] = per-finger normal force magnitude (N), order: thumb,index,middle,ring,pinky
#   [5]   = Ferrari-Canny Q (< 0 → not force-closed; >= 0 → force closure)
#   [6]   = is_force_closed (1.0 = yes, 0.0 = no)
#   [7]   = n_contacts (float, integer value)
_FORCE_METRICS_LEN = 8

_FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]

# Friction cone edges for GWS primitive forces
_CONE_EDGES = 8

# ---------------------------------------------------------------------------
# Rotation helpers (duplicated from workers to keep subprocess clean)
# ---------------------------------------------------------------------------

def _Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def _Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def _Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])


# ---------------------------------------------------------------------------
# Wrench / force closure helpers
# ---------------------------------------------------------------------------

def _compute_gws(contacts_pos, contacts_normal, forces_N, mu, n_edges=_CONE_EDGES):
    """
    Build Grasp Wrench Space primitive forces.

    contacts_pos:    list of (3,) world-frame contact positions
    contacts_normal: list of (3,) inward normals (pointing into object surface)
    forces_N:        list of scalar magnitudes (N) per contact
    mu:              friction coefficient
    Returns (primitives, centroid) where primitives is (K, 6) array of wrenches.
    """
    if not contacts_pos:
        return np.zeros((0, 6)), np.zeros(3)

    centroid = np.mean(contacts_pos, axis=0)
    primitives = []

    angles = np.linspace(0, 2 * np.pi, n_edges, endpoint=False)

    for pos, n, F in zip(contacts_pos, contacts_normal, forces_N):
        if F < 1e-4:
            continue
        n = np.asarray(n, dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-9:
            continue
        n = n / n_norm

        # Build two tangent vectors
        perp = np.array([1, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1, 0])
        t1 = np.cross(n, perp); t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)

        r = pos - centroid  # moment arm

        for theta in angles:
            f_vec = F * (n + mu * (np.cos(theta) * t1 + np.sin(theta) * t2))
            tau   = np.cross(r, f_vec)
            primitives.append(np.concatenate([f_vec, tau]))

    return np.array(primitives) if primitives else np.zeros((0, 6)), centroid


def _ferrari_canny_q(primitives):
    """
    Compute Ferrari-Canny Q metric: min distance from origin to convex hull of
    primitives projected to force subspace (first 3 columns).

    Returns (is_fc, Q) where is_fc is True if force-closed.
    """
    if len(primitives) < 4 or not _HAS_SCIPY:
        return False, -1.0
    try:
        hull = ConvexHull(primitives[:, :3])
        # Check if origin is inside hull via linear programming approach:
        # sign of (A @ 0 + b) for all facets
        inside = np.all(hull.equations[:, :3] @ np.zeros(3) + hull.equations[:, 3] <= 1e-10)
        if not inside:
            return False, -float(np.min(np.abs(hull.equations[:, 3])))
        Q = float(np.min(np.abs(hull.equations[:, 3])))
        return True, Q
    except Exception:
        return False, -1.0


# ---------------------------------------------------------------------------
# Subprocess worker
# ---------------------------------------------------------------------------

def _force_viewer_worker(xml_path: str, ctrl_arr, box_params_arr,
                         force_metrics_arr, stop_event) -> None:
    """
    Subprocess: floating-hand force scene viewer.

    - Sets hand pose from ctrl_arr (same layout as hand viewer)
    - Repositions and resizes the box from box_params_arr
    - Runs physics steps to generate contact forces
    - Reads force sensors → force_metrics_arr
    - Draws fingertip spheres and box outline as user_scn geoms
    """
    import mujoco
    import mujoco.viewer

    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
        data  = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Joint address map (same coupling as grasp scene)
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

        # Object IDs
        box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "object")
        box_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "box")

        # Fingertip site IDs
        _TIP_SITES = {
            "thumb":  "right_thumb_tip",
            "index":  "right_index_tip",
            "middle": "right_middle_tip",
            "ring":   "right_ring_tip",
            "pinky":  "right_pinky_tip",
        }
        tip_site_ids = {}
        for fname, sname in _TIP_SITES.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid >= 0:
                tip_site_ids[fname] = sid

        # Sensor addresses for force sensors
        _FORCE_SENSORS = {
            "thumb":  "thumb_tip_force",
            "index":  "index_tip_force",
            "middle": "middle_tip_force",
            "ring":   "ring_tip_force",
            "pinky":  "pinky_tip_force",
        }
        sensor_addrs = {}
        for fname, sname in _FORCE_SENSORS.items():
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sname)
            if sid >= 0:
                sensor_addrs[fname] = int(model.sensor_adr[sid])

        # Body-to-finger map for contact detection
        import re
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

        dt       = model.opt.timestep
        n_steps  = max(1, round(0.033 / dt))
        settle   = max(1, round(0.010 / dt))  # mini-settle steps per frame

        _FINGER_RGBA_LOCAL = {
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
                ctrl = np.array(ctrl_arr[:])
                bpar = np.array(box_params_arr[:])

                # Apply hand pose via qpos + joint coupling
                pos_x, pos_y, pos_z, rot_x, rot_y, rot_z = ctrl[0:6]
                pinky, ring, middle, index, pitch, yaw     = ctrl[6:12]

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

                # Apply ctrl to actuators so mj_step uses position control
                data.ctrl[0:12] = ctrl[0:12]

                # Update box size and friction from params
                hw, hl, hd, friction, mass = (float(bpar[i]) for i in range(5))
                hw  = max(hw,  0.003)
                hl  = max(hl,  0.003)
                hd  = max(hd,  0.003)
                mass = max(mass, 0.001)
                if box_geom_id >= 0:
                    model.geom_size[box_geom_id] = [hw, hl, hd]
                    model.geom_friction[box_geom_id, 0] = friction
                if box_body_id >= 0:
                    model.body_mass[box_body_id] = mass

                # Compute box centroid from fingertip positions (after kinematics)
                mujoco.mj_kinematics(model, data)
                tip_pts = []
                for fname in ["thumb", "index", "middle"]:
                    sid = tip_site_ids.get(fname)
                    if sid is not None:
                        tip_pts.append(data.site_xpos[sid].copy())
                if tip_pts and box_body_id >= 0:
                    centroid = np.mean(tip_pts, axis=0)
                    model.body_pos[box_body_id] = centroid

                # Run settle steps to compute contact forces
                mujoco.mj_forward(model, data)
                for _ in range(settle):
                    mujoco.mj_step(model, data)

                # Read force sensors
                fn_per_finger = {}
                for fname in _FINGER_ORDER:
                    adr = sensor_addrs.get(fname)
                    sid = tip_site_ids.get(fname)
                    if adr is not None and sid is not None:
                        f_local = data.sensordata[adr:adr+3].copy()
                        xmat    = data.site_xmat[sid].reshape(3, 3)
                        fn_per_finger[fname] = float(np.linalg.norm(xmat @ f_local))
                    else:
                        fn_per_finger[fname] = 0.0

                # Contact detection for GWS
                contact_pos, contact_normals, contact_forces = [], [], []
                result_buf = np.zeros(6)
                for i in range(data.ncon):
                    c = data.contact[i]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    fname = None
                    normal_sign = 1.0
                    if g1 == box_geom_id:
                        fname = body_to_finger.get(int(model.geom_bodyid[g2]))
                        normal_sign = -1.0  # normal points from box into finger → invert for object
                    elif g2 == box_geom_id:
                        fname = body_to_finger.get(int(model.geom_bodyid[g1]))
                    if fname is None:
                        continue
                    mujoco.mj_contactForce(model, data, i, result_buf)
                    if result_buf[0] < 1e-4:
                        continue
                    frame = c.frame.reshape(3, 3)
                    # Row 0 of frame is the contact normal in world frame
                    normal = frame[0] * normal_sign
                    contact_pos.append(c.pos.copy())
                    contact_normals.append(normal)
                    contact_forces.append(float(result_buf[0]))

                # Compute GWS and Ferrari-Canny Q
                primitives, _ = _compute_gws(contact_pos, contact_normals,
                                             contact_forces, friction)
                is_fc, Q = _ferrari_canny_q(primitives)

                # Write metrics to shared memory
                for i, fname in enumerate(_FINGER_ORDER):
                    force_metrics_arr[i] = fn_per_finger.get(fname, 0.0)
                force_metrics_arr[5] = Q
                force_metrics_arr[6] = 1.0 if is_fc else 0.0
                force_metrics_arr[7] = float(len(contact_pos))

                # Draw geoms: fingertip spheres + box wireframe
                scn = v.user_scn
                scn.ngeom = 0

                def _sphere(pos, radius, rgba):
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

                def _line(p0, p1, rgba, width=0.002):
                    if scn.ngeom >= scn.maxgeom:
                        return
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(
                        g, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9),
                        np.asarray(rgba, dtype=np.float32))
                    mujoco.mjv_connector(
                        g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                        np.asarray(p0, dtype=np.float64),
                        np.asarray(p1, dtype=np.float64))
                    scn.ngeom += 1

                for fname, sid in tip_site_ids.items():
                    pos  = data.site_xpos[sid]
                    rgba = _FINGER_RGBA_LOCAL.get(fname, np.array([0.8,0.8,0.2,0.9]))
                    _sphere(pos, 0.003, rgba)

                # Box wireframe (12 edges)
                if box_body_id >= 0:
                    bc = model.body_pos[box_body_id].copy()
                    xs = np.array([hw, -hw, -hw,  hw,  hw, -hw, -hw,  hw])
                    ys = np.array([hl,  hl, -hl, -hl,  hl,  hl, -hl, -hl])
                    zs = np.array([hd,  hd,  hd,  hd, -hd, -hd, -hd, -hd])
                    corners = bc + np.column_stack([xs, ys, zs])
                    box_rgba = np.array([0.6, 0.85, 0.6, 0.6], dtype=np.float32)
                    # 4 top, 4 bottom, 4 vertical edges
                    for i in range(4):
                        _line(corners[i], corners[(i+1) % 4], box_rgba)
                        _line(corners[i+4], corners[(i+4+1) % 4 + 4], box_rgba)
                        _line(corners[i], corners[i+4], box_rgba)

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
    root         : tk.Tk parent
    ctrl_arr     : shared multiprocessing Array (12 doubles) — the main ctrl array
    state_arr    : shared multiprocessing Array (20 doubles) — grasp state
    mp_ctx       : multiprocessing context ('fork')
    hand         : optional RH56Hand for real force readings
    """

    def __init__(self, root: tk.Tk, ctrl_arr, state_arr, mp_ctx, hand=None):
        self._root      = root
        self._ctrl_arr  = ctrl_arr
        self._state_arr = state_arr
        self._mp_ctx    = mp_ctx
        self._hand      = hand

        self._win = tk.Toplevel(root)
        self._win.title("Force Viz")
        self._win.resizable(True, True)
        self._win.protocol("WM_DELETE_WINDOW", self._on_close)

        # Shared memory for force scene
        self._box_params_arr   = mp_ctx.Array("d", _BOX_PARAMS_DEFAULT)
        self._force_metrics_arr = mp_ctx.Array("d", [0.0] * _FORCE_METRICS_LEN)
        self._stop_event       = mp_ctx.Event()
        self._viewer_proc: Optional[multiprocessing.Process] = None

        self._show_sim_cone  = tk.BooleanVar(value=True)
        self._show_real_cone = tk.BooleanVar(value=False)
        self._show_fc        = tk.BooleanVar(value=True)

        self._build_ui()
        self._update_running = True
        self._update_thread = threading.Thread(
            target=self._metrics_update_loop, daemon=True, name="force-metrics")
        self._update_thread.start()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._win.columnconfigure(0, weight=1)
        self._win.columnconfigure(1, weight=2)
        self._win.rowconfigure(0, weight=1)

        # Left column: controls
        ctrl_frame = ttk.LabelFrame(self._win, text="Object & Viewer", padding=6)
        ctrl_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        ctrl_frame.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(ctrl_frame, text="── Object ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        self._box_vars = {}
        params = [
            ("Width  (mm)",  "hw",  0,  200, _BOX_PARAMS_DEFAULT[0]*1000),
            ("Length (mm)",  "hl",  0,  200, _BOX_PARAMS_DEFAULT[1]*1000),
            ("Depth  (mm)",  "hd",  0,  100, _BOX_PARAMS_DEFAULT[2]*1000),
            ("Friction",     "mu",  0,    2, _BOX_PARAMS_DEFAULT[3]),
            ("Mass   (g)",   "mass",1, 1000, _BOX_PARAMS_DEFAULT[4]*1000),
        ]
        for label, key, vmin, vmax, vinit in params:
            ttk.Label(ctrl_frame, text=label).grid(row=r, column=0, sticky="w")
            var = tk.DoubleVar(value=vinit)
            sl  = ttk.Scale(ctrl_frame, from_=vmin, to=vmax,
                            variable=var, orient="horizontal", length=120,
                            command=lambda v, k=key: self._on_param_change(k))
            sl.grid(row=r, column=1, sticky="ew")
            ent = tk.Entry(ctrl_frame, width=7)
            ent.insert(0, f"{vinit:.1f}")
            ent.grid(row=r, column=2, padx=2)
            ent.bind("<Return>",   lambda e, s=sl, v=var, en=ent: s.set(
                float(en.get() or v.get())))
            ent.bind("<FocusOut>", lambda e, s=sl, v=var, en=ent: s.set(
                float(en.get() or v.get())))
            self._box_vars[key] = (var, ent, sl, vmin, vmax)
            r += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1

        tk.Button(ctrl_frame, text="Launch Force Viewer",
                  command=self._launch_force_viewer).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1
        tk.Button(ctrl_frame, text="Close Viewer",
                  command=self._close_force_viewer).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=2); r += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(ctrl_frame, text="── Visualization ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(ctrl_frame, text="Sim wrench cone",
                        variable=self._show_sim_cone).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(ctrl_frame, text="Real wrench cone (hand sensor)",
                        variable=self._show_real_cone,
                        state="normal" if self._hand is not None else "disabled").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Checkbutton(ctrl_frame, text="Ferrari-Canny Q",
                        variable=self._show_fc).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Separator(ctrl_frame, orient="horizontal").grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=4); r += 1
        ttk.Label(ctrl_frame, text="── Status ──", foreground="#555").grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        self._status_lbl = ttk.Label(ctrl_frame, text="Ready", foreground="#333",
                                     wraplength=200)
        self._status_lbl.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        # Right column: matplotlib force plot
        plot_frame = ttk.LabelFrame(self._win, text="Force Metrics", padding=4)
        plot_frame.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self._fig, (self._ax_bar, self._ax_cone) = plt.subplots(
            2, 1, figsize=(5, 6), gridspec_kw={"height_ratios": [1, 1.5]})
        self._fig.tight_layout(pad=2.0)

        self._canvas_fp = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._canvas_fp.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self._draw_empty_plots()

    # ------------------------------------------------------------------
    # Param change → update shared memory
    # ------------------------------------------------------------------

    def _on_param_change(self, key: str):
        var, ent, sl, vmin, vmax = self._box_vars[key]
        val = float(var.get())
        ent.delete(0, "end")
        ent.insert(0, f"{val:.2f}")
        # Update box_params_arr
        idx_map = {"hw": 0, "hl": 1, "hd": 2, "mu": 3, "mass": 4}
        i = idx_map[key]
        if key in ("hw", "hl", "hd"):
            self._box_params_arr[i] = val / 1000.0  # mm → m
        elif key == "mass":
            self._box_params_arr[i] = val / 1000.0  # g → kg
        else:
            self._box_params_arr[i] = val            # friction, unitless

    # ------------------------------------------------------------------
    # Viewer process management
    # ------------------------------------------------------------------

    def _launch_force_viewer(self):
        if self._viewer_proc is not None and self._viewer_proc.is_alive():
            self._set_status("Force viewer already running.")
            return
        self._stop_event.clear()
        self._viewer_proc = self._mp_ctx.Process(
            target=_force_viewer_worker,
            args=(_FORCE_SCENE, self._ctrl_arr, self._box_params_arr,
                  self._force_metrics_arr, self._stop_event),
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
    # Metrics update loop (background thread → schedule plot in main thread)
    # ------------------------------------------------------------------

    def _metrics_update_loop(self):
        while self._update_running:
            time.sleep(0.3)
            try:
                self._win.after(0, self._refresh_plots)
            except Exception:
                break

    def _refresh_plots(self):
        if not self._win.winfo_exists():
            return
        metrics = np.array(self._force_metrics_arr[:])
        forces  = metrics[0:5]   # per-finger normal forces (N)
        Q       = float(metrics[5])
        is_fc   = bool(metrics[6] > 0.5)
        n_con   = int(metrics[7])

        # Optionally blend real forces
        real_forces = None
        if self._show_real_cone.get() and self._hand is not None:
            try:
                raw = self._hand.force_act()
                # real order: [pinky=0, ring=1, middle=2, index=3, thumb_bend=4]
                # bar order:  [thumb, index, middle, ring, pinky]
                calib = 0.012547
                real_forces = np.array([
                    abs(float(raw[4])) * calib,  # thumb
                    abs(float(raw[3])) * calib,  # index
                    abs(float(raw[2])) * calib,  # middle
                    abs(float(raw[1])) * calib,  # ring
                    abs(float(raw[0])) * calib,  # pinky
                ])
            except Exception:
                real_forces = None

        self._draw_bar_chart(forces, real_forces)

        if self._show_sim_cone.get() or self._show_fc.get():
            self._draw_wrench_info(forces, Q, is_fc, n_con)

        self._canvas_fp.draw_idle()

        # Status label
        fc_str = f"FC ✓  Q={Q:.4f}" if is_fc else f"FC ✗  Q={Q:.4f}"
        self._set_status(f"{fc_str}\n{n_con} contacts")

    def _draw_empty_plots(self):
        ax = self._ax_bar
        ax.cla()
        ax.bar(_FINGER_ORDER, [0]*5, color=["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"])
        ax.set_ylabel("Force (N)")
        ax.set_title("Fingertip forces")
        ax.set_ylim(0, 10)

        ax2 = self._ax_cone
        ax2.cla()
        ax2.set_aspect("equal")
        ax2.set_title("GWS (Fx–Fy slice)")
        ax2.axhline(0, color="gray", lw=0.5)
        ax2.axvline(0, color="gray", lw=0.5)
        ax2.text(0.5, 0.5, "Run force viewer\nto compute GWS",
                 transform=ax2.transAxes, ha="center", va="center", color="gray")
        self._fig.tight_layout(pad=1.5)
        self._canvas_fp.draw_idle()

    def _draw_bar_chart(self, sim_forces, real_forces):
        ax = self._ax_bar
        ax.cla()
        x      = np.arange(5)
        colors = ["#e74c3c","#e67e22","#2ecc71","#3498db","#9b59b6"]
        bars   = ax.bar(x - 0.2, sim_forces, 0.35, label="Sim", color=colors, alpha=0.8)
        if real_forces is not None:
            ax.bar(x + 0.2, real_forces, 0.35, label="Real",
                   color=colors, alpha=0.4, hatch="//")
            ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(_FINGER_ORDER, fontsize=8)
        ax.set_ylabel("Force (N)", fontsize=8)
        ax.set_title("Fingertip forces", fontsize=9)
        ax.set_ylim(0, max(10, float(np.max(sim_forces)) * 1.3 + 0.1))

    def _draw_wrench_info(self, forces, Q, is_fc, n_con):
        ax = self._ax_cone
        ax.cla()
        ax.set_aspect("equal")
        ax.set_title("GWS Fx–Fy slice", fontsize=9)
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

        if not _HAS_SCIPY or not self._show_sim_cone.get() or n_con == 0:
            fc_str = f"FC {'✓' if is_fc else '✗'}  Q={Q:.4f}"
            ax.text(0.5, 0.5, fc_str if self._show_fc.get() else
                    "No contacts\n(adjust box size)",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#2ecc71" if is_fc else "#e74c3c", fontsize=10)
            return

        # Re-compute GWS using latest state_arr tip positions (approximate normals)
        state  = np.array(self._state_arr[:])
        finger_order_state = ["thumb", "index", "middle", "ring", "pinky"]
        tip_pts = {}
        for i, fname in enumerate(finger_order_state):
            p = state[5 + i*3: 5 + i*3 + 3]
            if not np.any(np.isnan(p)):
                tip_pts[fname] = p.copy()

        active = [f for f in finger_order_state if f in tip_pts]
        if len(active) < 2:
            return

        # Centroid for moment arm + normal estimation
        thumb_pt = tip_pts.get("thumb")
        idx_pt   = tip_pts.get("index")
        mid_pt   = tip_pts.get("middle")
        if thumb_pt is None or idx_pt is None:
            return
        cpts = [p for p in [thumb_pt, idx_pt, mid_pt] if p is not None]
        centroid = np.mean(cpts, axis=0)

        contact_pos, contact_normals, contact_forces = [], [], []
        for i, fname in enumerate(finger_order_state):
            if fname not in tip_pts:
                continue
            F = float(forces[i]) if i < len(forces) else 0.0
            if F < 0.05:
                continue
            d = centroid - tip_pts[fname]
            dn = np.linalg.norm(d)
            if dn < 1e-6:
                continue
            contact_pos.append(tip_pts[fname])
            contact_normals.append(d / dn)
            contact_forces.append(F)

        mu = float(self._box_params_arr[3])
        primitives, _ = _compute_gws(contact_pos, contact_normals, contact_forces, mu)

        if len(primitives) < 4:
            ax.text(0.5, 0.5, "Not enough contact\nforces for GWS",
                    transform=ax.transAxes, ha="center", va="center", color="gray")
            return

        pts = primitives[:, :2]  # Fx–Fy slice
        try:
            hull = ConvexHull(pts)
            hpts = pts[hull.vertices]
            hpts = np.vstack([hpts, hpts[0]])
            ax.fill(hpts[:, 0], hpts[:, 1], alpha=0.25, color="#3498db")
            ax.plot(hpts[:, 0], hpts[:, 1], "b-", lw=1.5)
        except Exception:
            ax.scatter(pts[:, 0], pts[:, 1], s=10, c="b", alpha=0.5)

        ax.scatter([0], [0], color="red", s=60, zorder=5, marker="+")
        ax.set_xlabel("Fx (N)", fontsize=7)
        ax.set_ylabel("Fy (N)", fontsize=7)

        if self._show_fc.get():
            fc_str = f"Q = {Q:.4f}  {'✓' if is_fc else '✗'}"
            color  = "#2ecc71" if is_fc else "#e74c3c"
            ax.text(0.05, 0.95, fc_str, transform=ax.transAxes,
                    ha="left", va="top", fontsize=9, color=color,
                    bbox=dict(fc="white", alpha=0.7, boxstyle="round"))

        self._fig.tight_layout(pad=1.5)

    # ------------------------------------------------------------------
    # Helpers
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
