"""
grasp_geometry.py — Offline kinematic model + antipodal closure geometry
for the Inspire RH56 dexterous hand.

Provides:
  InspireHandFK     — MuJoCo FK sweep → interpolation tables (base frame)
  ClosureResult     — typed result of a closure computation
  ClosureGeometry   — line / plane / cylinder closure from FK tables

Coordinate conventions (HAND BASE FRAME):
  +Z  : finger extension direction (wrist → fingertip when fully open)
  +Y  : finger spread direction (index at +0.032 m, pinky at −0.026 m)
  +X  : palm closure direction (fingers curl from +Z toward +X when flexed)

For world-frame (top-down approach) visualization, apply Rx(π):
  base +Z → world −Z (fingers hang down)
  base +X → world +X (closure direction stays horizontal)
"""

import os
import pathlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).parent.parent   # repo root
_DEFAULT_XML = str(_HERE / "h1_mujoco" / "inspire" / "inspire_right.xml")
_CACHE_PATH  = str(_HERE / "h1_mujoco" / "inspire" / ".fk_cache.npz")


# ---------------------------------------------------------------------------
# Constants (mirror SimAnalyzer for compatibility)
# ---------------------------------------------------------------------------
TIP_SITES: Dict[str, str] = {
    "thumb":  "right_thumb_tip",
    "index":  "right_index_tip",
    "middle": "right_middle_tip",
    "ring":   "right_ring_tip",
    "pinky":  "right_pinky_tip",
}

# Actuator order in inspire_right.xml
ACTUATOR_NAMES = ["pinky", "ring", "middle", "index", "thumb_proximal", "thumb_yaw"]

CTRL_MAX: Dict[str, float] = {
    "pinky":         1.57,
    "ring":          1.57,
    "middle":        1.50,
    "index":         1.50,
    "thumb_proximal": 0.60,
    "thumb_yaw":     1.308,
}

# Non-thumb fingers in medial-to-lateral order (index = most radial)
NON_THUMB_FINGERS = ["index", "middle", "ring", "pinky"]

# Grasp finger sets per "n_fingers" parameter
GRASP_FINGER_SETS: Dict[int, List[str]] = {
    2: ["index"],
    3: ["index", "middle"],
    4: ["index", "middle", "ring"],
    5: ["index", "middle", "ring", "pinky"],
}


# ---------------------------------------------------------------------------
# InspireHandFK
# ---------------------------------------------------------------------------
class InspireHandFK:
    """
    Fingertip positions as a function of actuator ctrl values, computed via
    MuJoCo forward kinematics on inspire_right.xml.

    Tables are built once (FK sweep) then cached to .npz for fast reload.
    All positions are in the HAND BASE FRAME (the worldbody of inspire_right.xml).

    Non-thumb fingers: 1-D table ctrl → tip_pos[3]
    Thumb:             2-D table (ctrl_pitch, ctrl_yaw) → tip_pos[3]
    """

    N_SAMPLES_1D = 200   # samples per non-thumb finger
    N_SAMPLES_2D = 50    # samples per thumb DOF (50×50 = 2500 pts)

    def __init__(self, xml_path: str = _DEFAULT_XML, rebuild: bool = False):
        self.xml_path = xml_path
        self._model = mujoco.MjModel.from_xml_path(xml_path)
        self._data  = mujoco.MjData(self._model)

        # Cache site IDs
        self._site_ids: Dict[str, int] = {}
        for fname, sname in TIP_SITES.items():
            sid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                raise ValueError(f"Site '{sname}' not found in model")
            self._site_ids[fname] = sid

        # Cache actuator indices
        self._act_ids: Dict[str, int] = {}
        for aname in ACTUATOR_NAMES:
            aid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise ValueError(f"Actuator '{aname}' not found in model")
            self._act_ids[aname] = aid

        # Cache joint qpos addresses (needed for sweep and for _load_tables path)
        self._cache_joint_addrs()

        # Build or load FK tables
        if not rebuild and os.path.exists(_CACHE_PATH):
            self._load_tables(_CACHE_PATH)
        else:
            print("[InspireHandFK] Building FK tables (first run, ~2s)...")
            self._build_tables()
            self._save_tables(_CACHE_PATH)
            print(f"[InspireHandFK] Saved FK cache to {_CACHE_PATH}")

        # Build interpolators
        self._build_interpolators()
        print("[InspireHandFK] Ready.")

    # ------------------------------------------------------------------
    # FK sweep helpers
    # ------------------------------------------------------------------
    def _cache_joint_addrs(self):
        """Cache qpos addresses for each joint by name."""
        joint_names = [
            "thumb_proximal_yaw_joint",
            "thumb_proximal_pitch_joint",
            "thumb_intermediate_joint",
            "thumb_distal_joint",
            "index_proximal_joint",
            "index_intermediate_joint",
            "middle_proximal_joint",
            "middle_intermediate_joint",
            "ring_proximal_joint",
            "ring_intermediate_joint",
            "pinky_proximal_joint",
            "pinky_intermediate_joint",
        ]
        self._jnt_adr: Dict[str, int] = {}
        for jname in joint_names:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not found in model")
            self._jnt_adr[jname] = int(self._model.jnt_qposadr[jid])

    def _set_finger_qpos(self, fname: str, ctrl: float):
        """
        Set qpos for a single non-thumb finger (proximal + coupled intermediate)
        using the coupling equations from inspire_right.xml.
        """
        d = self._data
        a = self._jnt_adr
        ctrl = float(ctrl)
        if fname == "index":
            d.qpos[a["index_proximal_joint"]]      = ctrl
            # polycoef="0.15 1 0 0 0": index_intermediate = 0.15 + 1.0 * index_proximal
            inter = ctrl + 0.15
            d.qpos[a["index_intermediate_joint"]]  = inter
        elif fname == "middle":
            d.qpos[a["middle_proximal_joint"]]     = ctrl
            d.qpos[a["middle_intermediate_joint"]] = ctrl   # 1:1 coupling
        elif fname == "ring":
            d.qpos[a["ring_proximal_joint"]]       = ctrl
            d.qpos[a["ring_intermediate_joint"]]   = ctrl
        elif fname == "pinky":
            d.qpos[a["pinky_proximal_joint"]]      = ctrl
            d.qpos[a["pinky_intermediate_joint"]]  = ctrl

    def _set_thumb_qpos(self, ctrl_pitch: float, ctrl_yaw: float):
        """Set thumb qpos including all coupled joints."""
        d = self._data
        a = self._jnt_adr
        p = float(ctrl_pitch)
        y = float(ctrl_yaw)
        d.qpos[a["thumb_proximal_yaw_joint"]]   = y
        d.qpos[a["thumb_proximal_pitch_joint"]] = p
        # polycoef="0.15 1.25 0 0": thumb_intermediate = 0.15 + 1.25 * thumb_proximal_pitch
        inter = 0.15 + 1.25 * p
        d.qpos[a["thumb_intermediate_joint"]]   = inter
        # polycoef="0.15 0.75 0 0 0": thumb_distal = 0.15 + 0.75 * thumb_proximal_pitch
        distal = 0.15 + 0.75 * p
        d.qpos[a["thumb_distal_joint"]]         = distal

    def _reset_qpos(self):
        """Zero all qpos and update FK."""
        self._data.qpos[:] = 0.0
        mujoco.mj_kinematics(self._model, self._data)

    def _tip_pos(self, finger_name: str) -> np.ndarray:
        """Read current tip site position in base frame."""
        return self._data.site_xpos[self._site_ids[finger_name]].copy()

    def _build_tables(self):
        """Sweep each finger's ctrl directly via qpos and record fingertip positions."""
        self._cache_joint_addrs()

        self._finger_ctrl  = {}
        self._finger_tips  = {}

        for fname in NON_THUMB_FINGERS:
            c_max = CTRL_MAX[fname]
            ctrl_vals = np.linspace(0.0, c_max, self.N_SAMPLES_1D)
            tips = np.zeros((self.N_SAMPLES_1D, 3))
            for i, cv in enumerate(ctrl_vals):
                self._reset_qpos()
                self._set_finger_qpos(fname, cv)
                mujoco.mj_kinematics(self._model, self._data)
                tips[i] = self._tip_pos(fname)
            self._finger_ctrl[fname] = ctrl_vals
            self._finger_tips[fname] = tips

        # Thumb: 2-D sweep (pitch × yaw)
        pitch_vals = np.linspace(0.0, CTRL_MAX["thumb_proximal"], self.N_SAMPLES_2D)
        yaw_vals   = np.linspace(0.0, CTRL_MAX["thumb_yaw"],      self.N_SAMPLES_2D)
        thumb_tips = np.zeros((self.N_SAMPLES_2D, self.N_SAMPLES_2D, 3))
        for i, pv in enumerate(pitch_vals):
            for j, yv in enumerate(yaw_vals):
                self._reset_qpos()
                self._set_thumb_qpos(pv, yv)
                mujoco.mj_kinematics(self._model, self._data)
                thumb_tips[i, j] = self._tip_pos("thumb")
        self._thumb_pitch_vals = pitch_vals
        self._thumb_yaw_vals   = yaw_vals
        self._thumb_tips       = thumb_tips

    def _save_tables(self, path: str):
        np.savez(
            path,
            **{f"fc_{n}":  self._finger_ctrl[n] for n in NON_THUMB_FINGERS},
            **{f"ft_{n}":  self._finger_tips[n] for n in NON_THUMB_FINGERS},
            thumb_pitch=self._thumb_pitch_vals,
            thumb_yaw=self._thumb_yaw_vals,
            thumb_tips=self._thumb_tips,
        )

    def _load_tables(self, path: str):
        d = np.load(path)
        self._finger_ctrl = {n: d[f"fc_{n}"] for n in NON_THUMB_FINGERS}
        self._finger_tips  = {n: d[f"ft_{n}"] for n in NON_THUMB_FINGERS}
        self._thumb_pitch_vals = d["thumb_pitch"]
        self._thumb_yaw_vals   = d["thumb_yaw"]
        self._thumb_tips       = d["thumb_tips"]

    def _build_interpolators(self):
        """Build scipy interpolators from the raw tables."""
        self._finger_interp: Dict[str, interp1d] = {}
        for fname in NON_THUMB_FINGERS:
            # Interpolate each xyz component
            self._finger_interp[fname] = interp1d(
                self._finger_ctrl[fname],
                self._finger_tips[fname],
                axis=0,
                kind="linear",
                bounds_error=False,
                fill_value=(self._finger_tips[fname][0],
                            self._finger_tips[fname][-1]),
            )
        self._thumb_interp = RegularGridInterpolator(
            (self._thumb_pitch_vals, self._thumb_yaw_vals),
            self._thumb_tips,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def finger_tip(self, name: str, ctrl: float) -> np.ndarray:
        """Return (3,) fingertip position in hand base frame."""
        ctrl = float(np.clip(ctrl, 0.0, CTRL_MAX[name]))
        return self._finger_interp[name](ctrl)

    def thumb_tip(self, ctrl_pitch: float, ctrl_yaw: float) -> np.ndarray:
        """Return (3,) thumb tip position in hand base frame."""
        cp = float(np.clip(ctrl_pitch, 0.0, CTRL_MAX["thumb_proximal"]))
        cy = float(np.clip(ctrl_yaw,   0.0, CTRL_MAX["thumb_yaw"]))
        return self._thumb_interp([[cp, cy]])[0]

    def all_tips(self, ctrl_dict: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Return all fingertip positions for given ctrl dict.
        ctrl_dict keys: 'index', 'middle', 'ring', 'pinky', 'thumb_proximal', 'thumb_yaw'
        """
        tips = {}
        for fname in NON_THUMB_FINGERS:
            tips[fname] = self.finger_tip(fname, ctrl_dict.get(fname, 0.0))
        tips["thumb"] = self.thumb_tip(
            ctrl_dict.get("thumb_proximal", 0.0),
            ctrl_dict.get("thumb_yaw", 0.0),
        )
        return tips

    def z_range_for_finger(self, name: str) -> Tuple[float, float]:
        """Min and max Z of this finger's tip in base frame over its full ctrl range."""
        if name in NON_THUMB_FINGERS:
            zvals = self._finger_tips[name][:, 2]
        else:
            zvals = self._thumb_tips[:, :, 2].ravel()
        return float(zvals.min()), float(zvals.max())

    def coplanar_ctrls(self, target_z_base: float,
                       fingers: Optional[List[str]] = None) -> Dict[str, Optional[float]]:
        """
        For each requested non-thumb finger, find the ctrl value that places
        the tip at target_z_base (base frame Z).  Returns None for a finger
        if the target Z is out of range.
        """
        if fingers is None:
            fingers = NON_THUMB_FINGERS
        result = {}
        for fname in fingers:
            zvals = self._finger_tips[fname][:, 2]
            z_min, z_max = float(zvals.min()), float(zvals.max())
            if not (z_min <= target_z_base <= z_max):
                result[fname] = None
                continue
            # z(ctrl) is monotone (decreasing) → use interp1d inverse
            # Build inverse: z → ctrl
            # Sort by z ascending for brentq
            ctrl_vals = self._finger_ctrl[fname]
            # z decreases as ctrl increases → reverse
            def _z_err(c, tgt=target_z_base):
                return float(self._finger_interp[fname](c)[2]) - tgt
            try:
                c_sol = brentq(_z_err, 0.0, CTRL_MAX[fname], xtol=1e-6)
                result[fname] = float(c_sol)
            except ValueError:
                result[fname] = None
        return result

    def thumb_tip_at_z(self, target_z_base: float,
                        ctrl_yaw: float) -> Optional[float]:
        """
        Find thumb ctrl_pitch such that thumb tip Z == target_z_base,
        at fixed ctrl_yaw.  Returns None if out of range.
        """
        cy = float(np.clip(ctrl_yaw, 0.0, CTRL_MAX["thumb_yaw"]))
        # Sample z over pitch range at fixed yaw
        pitches = self._thumb_pitch_vals
        zvals = self._thumb_interp(
            np.column_stack([pitches, np.full_like(pitches, cy)])
        )[:, 2]
        z_min, z_max = float(zvals.min()), float(zvals.max())
        if not (z_min <= target_z_base <= z_max):
            return None
        def _z_err(p):
            return float(self._thumb_interp([[p, cy]])[0, 2]) - target_z_base
        try:
            return float(brentq(_z_err, 0.0, CTRL_MAX["thumb_proximal"], xtol=1e-6))
        except ValueError:
            return None

    # ------------------------------------------------------------------
    # Proximal-to-intermediate joint positions (= end of proximal link)
    # ------------------------------------------------------------------
    def _cache_prox_body_ids(self):
        """Cache body IDs for the intermediate bodies (= proximal-intermediate joint location)."""
        names = {
            "index":  "index_intermediate",
            "middle": "middle_intermediate",
            "ring":   "ring_intermediate",
            "pinky":  "pinky_intermediate",
            "thumb":  "thumb_intermediate",
        }
        self._prox_body_ids: Dict[str, int] = {}
        for fname, bname in names.items():
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                raise ValueError(f"Body '{bname}' not found in model")
            self._prox_body_ids[fname] = bid

    def prox_joint_pos(self, fname: str, ctrl: float,
                        ctrl_yaw: float = 0.0) -> np.ndarray:
        """
        Return the proximal-intermediate joint position (= intermediate body origin)
        in hand base frame.  Performs a single live FK evaluation.
        NOT thread-safe — call from the main thread only.
        """
        if not hasattr(self, "_prox_body_ids"):
            self._cache_prox_body_ids()
        self._reset_qpos()
        if fname == "thumb":
            self._set_thumb_qpos(float(ctrl), float(ctrl_yaw))
        else:
            self._set_finger_qpos(fname, float(ctrl))
        mujoco.mj_kinematics(self._model, self._data)
        return self._data.xpos[self._prox_body_ids[fname]].copy()

    def get_raw_sweep(self) -> Dict:
        """Return raw sweep data for analysis/plotting."""
        return {
            "finger_ctrl": self._finger_ctrl,
            "finger_tips": self._finger_tips,
            "thumb_pitch": self._thumb_pitch_vals,
            "thumb_yaw":   self._thumb_yaw_vals,
            "thumb_tips":  self._thumb_tips,
        }


# ---------------------------------------------------------------------------
# ClosureResult dataclass
# ---------------------------------------------------------------------------
@dataclass
class ClosureResult:
    """
    Typed result of a single closure computation.

    All positions are in the HAND BASE FRAME.
    To convert to world frame (top-down, Z-up):
        R = Rx(π)  →  world_pos = R @ base_pos + [0, 0, world_grasp_z - midpoint[2]*(-1)]
    The helper method world_tips(world_grasp_z) does this conversion.
    """
    mode: str                             # '2-finger line', '3-finger plane', etc.
    midpoint: np.ndarray                  # (3,) centroid of active tips, base frame
    width: float                          # 3D distance thumb-to-finger(s) (metres)
    finger_span: float                    # Y-span of non-thumb active tips (metres)
    cylinder_radius: float                # approximated radius (metres), cylinder only
    tip_positions: Dict[str, np.ndarray]  # {fname: (3,) in base frame}
    ctrl_values: Dict[str, float]         # {actuator_name: ctrl_rad}
    base_z_offset: float = 0.0           # unused legacy field
    base_tilt_y: float = 0.0             # Y-axis tilt (rad) to make grasp plane horizontal
    tilt_deg: float = 0.0                # human-readable tilt magnitude in degrees

    @staticmethod
    def _rot_matrix(tilt_y: float) -> np.ndarray:
        """Combined Ry(tilt_y) @ Rx(π): maps base → world for top-down approach."""
        cy, sy = np.cos(tilt_y), np.sin(tilt_y)
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
        Rx_pi = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)
        return Ry @ Rx_pi

    def world_tips(self, world_grasp_z: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Convert tip positions to world frame with tilt correction.
        The midpoint is placed at (0, 0, world_grasp_z).
        """
        R = self._rot_matrix(self.base_tilt_y)
        mid_w = R @ self.midpoint
        # base offset: shift so that rotated midpoint lands at world_grasp_z on Z
        base_w = np.array([-mid_w[0], -mid_w[1], world_grasp_z - mid_w[2]])
        return {fname: R @ pos + base_w for fname, pos in self.tip_positions.items()}

    def world_base(self, world_grasp_z: float = 0.0) -> np.ndarray:
        """World-frame position of hand base origin (= [0,0,0] in base frame)."""
        R = self._rot_matrix(self.base_tilt_y)
        mid_w = R @ self.midpoint
        return np.array([-mid_w[0], -mid_w[1], world_grasp_z - mid_w[2]])


# ---------------------------------------------------------------------------
# ClosureGeometry
# ---------------------------------------------------------------------------
class ClosureGeometry:
    """
    Computes line / plane / cylinder closure geometry from FK tables.

    Solves:
      - ctrl values for each finger given a target width (or radius)
      - coplanarity corrections (per-finger ctrl adjustments)
      - thumb yaw for power vs precision grasps
      - hand base Z offset to maintain midpoint at a given world Z
    """

    # Practical minimum graspable width (XZ distance).  XZ → 0 at the
    # configuration where thumb and finger are at the same X,Z position;
    # that's geometrically degenerate, so we cap at 5 mm for the UI.
    PRACTICAL_MIN_WIDTH = 0.005   # metres

    def __init__(self, fk: InspireHandFK):
        self.fk = fk
        self._thumb_yaw_threshold: Optional[float] = None
        self._compute_thumb_yaw_threshold()
        # Ensure prox body IDs are cached (used by cylinder solver)
        if not hasattr(self.fk, "_prox_body_ids"):
            self.fk._cache_prox_body_ids()
        # Cylinder thumb-yaw transition: below this prox-joint diameter the
        # thumb tip would collide with the non-thumb finger tips at max yaw.
        self._cyl_transition_ctrl: float = CTRL_MAX["index"]
        self._cyl_transition_diameter: float = 0.0
        self._compute_cyl_transition()

    # ------------------------------------------------------------------
    # Thumb yaw threshold for power grasp
    # ------------------------------------------------------------------
    def _compute_thumb_yaw_threshold(self):
        """
        Find the cylinder diameter below which thumb yaw should be 0
        (because at max yaw the thumb would overlap with the curled fingers).

        Strategy: sweep ctrl_fingers (uniform for index).  For each ctrl,
        check if thumb_tip_x(max_yaw, optimal_pitch) > index_tip_x(ctrl).
        If yes → thumb can oppose without overlap → use max yaw.
        If no  → overlap → use 0 yaw.
        The threshold is the crossover diameter.
        """
        n = 80
        ctrls = np.linspace(0.0, CTRL_MAX["index"], n)
        crossover = None
        for c in ctrls:
            idx_x = float(self.fk.finger_tip("index", c)[0])
            # thumb pitch that gives same Z as index at ctrl c
            idx_z = float(self.fk.finger_tip("index", c)[2])
            cp = self.fk.thumb_tip_at_z(idx_z, ctrl_yaw=CTRL_MAX["thumb_yaw"])
            if cp is None:
                continue
            th_x = float(self.fk.thumb_tip(cp, CTRL_MAX["thumb_yaw"])[0])
            # width at this config (thumb opposes index)
            w = abs(th_x - idx_x)
            if th_x > idx_x:
                # thumb cleared the finger → max yaw is fine
                crossover = w
                break
        self._thumb_yaw_threshold = crossover if crossover is not None else 0.030

    def thumb_yaw_for_power(self, cylinder_diameter: float) -> float:
        """Return optimal thumb yaw ctrl (rad) for given cylinder diameter."""
        if self._thumb_yaw_threshold is None or \
                cylinder_diameter >= self._thumb_yaw_threshold:
            return CTRL_MAX["thumb_yaw"]
        return 0.0

    # ------------------------------------------------------------------
    # Joint-closure solver: jointly optimize thumb pitch + finger ctrl
    # ------------------------------------------------------------------
    _N_S = 120   # samples of closure parameter s for sweeps

    def _joint_dist(self, ref_finger: str, s: float) -> float:
        """
        XZ-plane distance between thumb tip and reference finger tip at uniform
        closure parameter s.  Both scale proportionally with s; thumb yaw is max.

        WHY XZ DISTANCE (not 3D):
          world_X_separation = R @ d, component 0 = hypot(d[0], d[2])  (algebraic identity
          with R = Ry(tilt)@Rx(π)).  The Y component of d is the anatomical thumb offset
          and is NOT compressible by rotating the hand, so it does not contribute to the
          effective graspable width.  Using 3D distance causes the minimum to be reached
          when the two tips are side-by-side in Y (d[2] sign flip → tilt discontinuity).
        """
        ctrl_pitch = s * CTRL_MAX["thumb_proximal"]
        ctrl_ref   = s * CTRL_MAX[ref_finger]
        T = self.fk.thumb_tip(ctrl_pitch, CTRL_MAX["thumb_yaw"])
        I = self.fk.finger_tip(ref_finger, ctrl_ref)
        d = T - I
        return float(np.hypot(d[0], d[2]))

    def _joint_closure_range(self, ref_finger: str) -> Tuple[float, float, float]:
        """
        Sweep s∈[0,1] and return (s_at_min, D_min, D_open).
        D is the XZ-distance between thumb and ref_finger at closure s.
        """
        s_vals = np.linspace(0.0, 1.0, self._N_S)
        d_vals = np.array([self._joint_dist(ref_finger, float(s)) for s in s_vals])
        idx_min = int(np.argmin(d_vals))
        return float(s_vals[idx_min]), float(d_vals[idx_min]), float(d_vals[0])

    def _solve_joint_closure(self, ref_finger: str,
                              target_width: float) -> Tuple[float, float, float]:
        """
        Find closure parameter s such that |T(s) - finger_tip(s)| = target_width,
        where both thumb pitch and finger ctrl scale linearly with s.

        Returns (s, ctrl_pitch, ctrl_ref).
        Clamps to achievable range silently.
        """
        s_min, d_min, d_open = self._joint_closure_range(ref_finger)

        if target_width >= d_open:
            return 0.0, 0.0, 0.0
        if target_width <= d_min:
            s = s_min
        else:
            def _err(s):
                return self._joint_dist(ref_finger, s) - target_width
            try:
                s = float(brentq(_err, 0.0, s_min, xtol=1e-5))
            except ValueError:
                s = s_min

        return s, s * CTRL_MAX["thumb_proximal"], s * CTRL_MAX[ref_finger]

    # ------------------------------------------------------------------
    # Cylinder helpers: proximal-intermediate joint distance model
    # ------------------------------------------------------------------
    def _prox_joint_diameter_at_ctrl(self, ctrl: float) -> float:
        """
        Distance between thumb proximal-intermediate joint and the centroid of
        non-thumb proximal-intermediate joints at uniform ctrl for all fingers.
        Thumb is at max_yaw, pitch = ctrl * (max_pitch / max_index_ctrl) for proportional motion.
        """
        # Scale thumb pitch proportionally to finger ctrl
        thumb_pitch = ctrl * (CTRL_MAX["thumb_proximal"] / CTRL_MAX["index"])
        T_prox = self.fk.prox_joint_pos("thumb", thumb_pitch, CTRL_MAX["thumb_yaw"])
        prox_pts = np.array([
            self.fk.prox_joint_pos(f, ctrl) for f in NON_THUMB_FINGERS
        ])
        centroid = prox_pts.mean(axis=0)
        return float(np.linalg.norm(T_prox - centroid))

    def _find_cyl_transition_ctrl(self) -> float:
        """
        Find the uniform ctrl at which the thumb TIP (at max_yaw, scaled pitch)
        first approaches within COLLISION_DIST of any non-thumb finger TIP.
        Sweeping from open (ctrl=0) toward closed; returns the ctrl at transition,
        or CTRL_MAX["index"] if no collision threshold is crossed.
        """
        COLLISION_DIST = 0.020   # 20 mm — comfortable clearance at max yaw
        ctrls = np.linspace(0.0, CTRL_MAX["index"], 100)
        for c in ctrls:
            thumb_pitch = c * (CTRL_MAX["thumb_proximal"] / CTRL_MAX["index"])
            th_tip = self.fk.thumb_tip(thumb_pitch, CTRL_MAX["thumb_yaw"])
            finger_tips = np.array([self.fk.finger_tip(f, c) for f in NON_THUMB_FINGERS])
            min_dist = float(np.min(np.linalg.norm(finger_tips - th_tip, axis=1)))
            if min_dist < COLLISION_DIST:
                return float(c)
        return float(CTRL_MAX["index"])

    def _compute_cyl_transition(self):
        """Pre-compute the cylinder diameter at the thumb-yaw transition point."""
        c_trans = self._find_cyl_transition_ctrl()
        d_trans = self._prox_joint_diameter_at_ctrl(c_trans)
        self._cyl_transition_ctrl      = c_trans
        self._cyl_transition_diameter  = d_trans
        print(f"[ClosureGeometry] Cylinder thumb-yaw transition: "
              f"ctrl={c_trans:.3f}  prox_diam={d_trans*1000:.1f} mm")

    # ------------------------------------------------------------------
    # Width/radius range query
    # ------------------------------------------------------------------
    def width_range(self, mode: str, n_fingers: int = 4) -> Tuple[float, float]:
        """Return (min_width, max_width) achievable for the given mode.

        For precision modes the width is the XZ-plane distance (effective graspable
        width after tilt correction).  The minimum is capped at PRACTICAL_MIN_WIDTH
        (5 mm) because the XZ minimum is near-zero (tips touching in Y only).
        """
        if "cylinder" in mode:
            ctrls = np.linspace(0.0, CTRL_MAX["index"], 60)
            diams = np.array([self._prox_joint_diameter_at_ctrl(float(c)) for c in ctrls])
            return (max(float(diams.min()), self.PRACTICAL_MIN_WIDTH), float(diams.max()))
        else:
            ref = "middle" if n_fingers >= 3 else "index"
            _, d_min, d_open = self._joint_closure_range(ref)
            return (max(d_min, self.PRACTICAL_MIN_WIDTH), d_open)

    # ------------------------------------------------------------------
    # Closure computations
    # ------------------------------------------------------------------
    def line(self, target_width: float) -> ClosureResult:
        """
        2-finger index-thumb line closure.

        Both index ctrl and thumb pitch scale proportionally with a single closure
        parameter s ∈ [0, 1].  Thumb yaw is fixed at maximum.  The base_tilt_y
        angle makes the thumb→index direction horizontal in world frame.
        """
        _, ctrl_pitch, c_idx = self._solve_joint_closure("index", target_width)
        I = self.fk.finger_tip("index", c_idx)
        T = self.fk.thumb_tip(ctrl_pitch, CTRL_MAX["thumb_yaw"])

        d = T - I
        # With R = Ry(θ)@Rx(π): (R@d)[2] = -sin(θ)*d[0]-cos(θ)*d[2] = 0
        # → θ = atan2(-d[2], d[0]), normalised to (-π/2, π/2)
        # Clip to ±π/2.  For d[0] > 0 the result is already in (−π/2, π/2).
        # For d[0] ≤ 0 (thumb has passed the finger in X after full closure), capping
        # at ±π/2 keeps the hand in a "side-approach" orientation and avoids the
        # 155° discontinuity that the old wrap (+= π) produced.
        tilt_y = float(np.clip(np.arctan2(-d[2], d[0]), -np.pi / 2, np.pi / 2))

        midpoint = (T + I) / 2.0
        # width = XZ distance = world-frame X separation after tilt correction
        # (algebraic identity: (R@d)[0] = hypot(d[0], d[2]))
        width    = float(np.hypot(d[0], d[2]))

        tips: Dict[str, np.ndarray] = {"index": I, "thumb": T}
        ctrl: Dict[str, float] = {
            "index":           float(c_idx),
            "middle": 0.0, "ring": 0.0, "pinky": 0.0,
            "thumb_proximal":  float(ctrl_pitch),
            "thumb_yaw":       CTRL_MAX["thumb_yaw"],
        }
        return ClosureResult(
            mode="2-finger line",
            midpoint=midpoint, width=width,
            finger_span=0.0, cylinder_radius=0.0,
            tip_positions=tips, ctrl_values=ctrl,
            base_tilt_y=tilt_y, tilt_deg=float(np.degrees(abs(tilt_y))),
        )

    def plane(self, target_width: float, n_fingers: int = 4) -> ClosureResult:
        """
        n-finger box plane closure (n ∈ {2, 3, 4, 5}).

        The reference finger (middle for n≥3, index for n=2) and the thumb pitch
        are jointly optimised via a uniform closure parameter s.  All other active
        fingers are coplanarity-corrected to match the reference finger's base-frame Z.
        """
        fingers    = GRASP_FINGER_SETS[n_fingers]
        ref_finger = "middle" if "middle" in fingers else fingers[0]

        _, ctrl_pitch, c_ref = self._solve_joint_closure(ref_finger, target_width)
        I_ref = self.fk.finger_tip(ref_finger, c_ref)
        T     = self.fk.thumb_tip(ctrl_pitch, CTRL_MAX["thumb_yaw"])

        # Coplanar corrections: bring all active fingers to reference finger's base-frame Z.
        # Direction-aware fallback: Z decreases as ctrl increases (finger curls).
        #   target_z > z_max → finger can't reach (too extended) → use ctrl=0 (highest Z)
        #   target_z < z_min → finger can't reach (too curled)   → use CTRL_MAX (lowest Z)
        target_z = float(I_ref[2])
        coplanar = self.fk.coplanar_ctrls(target_z, fingers=fingers)
        for f in fingers:
            if coplanar.get(f) is None:
                _, z_max = self.fk.z_range_for_finger(f)
                coplanar[f] = 0.0 if target_z >= z_max else float(CTRL_MAX[f])

        tips: Dict[str, np.ndarray] = {
            f: self.fk.finger_tip(f, coplanar[f]) for f in fingers
        }
        tips["thumb"] = T

        # Tilt computed AFTER coplanar correction, using centroid of non-thumb tips.
        # This is more robust than using a single reference finger tip, because
        # coplanar corrections shift finger positions slightly.
        all_nonthumb = np.array([tips[f] for f in fingers])
        centroid = all_nonthumb.mean(axis=0)
        d = T - centroid
        # Clip to ±π/2.  For d[0] > 0 the result is already in (−π/2, π/2).
        # For d[0] ≤ 0 (thumb has passed the finger in X after full closure), capping
        # at ±π/2 keeps the hand in a "side-approach" orientation and avoids the
        # 155° discontinuity that the old wrap (+= π) produced.
        tilt_y = float(np.clip(np.arctan2(-d[2], d[0]), -np.pi / 2, np.pi / 2))

        midpoint = np.vstack([all_nonthumb, T]).mean(axis=0)
        # width = XZ distance = world-frame separation after tilt (hypot identity)
        width    = float(np.hypot(d[0], d[2]))
        span     = float(all_nonthumb[:, 1].max() - all_nonthumb[:, 1].min()) \
                   if len(fingers) > 1 else 0.0

        ctrl: Dict[str, float] = {f: 0.0 for f in NON_THUMB_FINGERS}
        for f in fingers:
            ctrl[f] = float(coplanar[f])
        ctrl["thumb_proximal"] = float(ctrl_pitch)
        ctrl["thumb_yaw"]      = CTRL_MAX["thumb_yaw"]

        return ClosureResult(
            mode=f"{n_fingers}-finger plane",
            midpoint=midpoint, width=width,
            finger_span=span, cylinder_radius=0.0,
            tip_positions=tips, ctrl_values=ctrl,
            base_tilt_y=tilt_y, tilt_deg=float(np.degrees(abs(tilt_y))),
        )

    def cylinder(self, target_diameter: float) -> ClosureResult:
        """
        5-finger power (cylinder) grasp.

        The cylinder diameter is modelled as the distance between the thumb
        proximal-intermediate joint and the centroid of the non-thumb
        proximal-intermediate joints (= ends of the proximal links).  The object
        sits between the proximal links, not at the fingertips — which is the
        physically correct model for power grasps.

        Both finger ctrl and thumb pitch scale proportionally with a single closure
        parameter (ctrl_finger → thumb_pitch = ctrl * max_pitch / max_index_ctrl).
        Thumb yaw is fixed at maximum (opposing the finger side).
        """
        n_sweep     = 60
        ctrls_sweep = np.linspace(0.0, CTRL_MAX["index"], n_sweep)
        prox_diams  = np.array([self._prox_joint_diameter_at_ctrl(float(c))
                                 for c in ctrls_sweep])
        idx_min     = int(np.argmin(prox_diams))
        c_min       = float(ctrls_sweep[idx_min])
        d_min       = float(prox_diams[idx_min])
        d_open      = float(prox_diams[0])

        if target_diameter >= d_open:
            c_uni = 0.0
        elif target_diameter <= d_min:
            c_uni = c_min
        else:
            # Monotone-decreasing region [0, c_min]
            def _prox_err(c):
                return self._prox_joint_diameter_at_ctrl(c) - target_diameter
            try:
                c_uni = float(brentq(_prox_err, 0.0, c_min, xtol=1e-5))
            except ValueError:
                c_uni = c_min

        # Thumb yaw: at max for opposition (large cylinder), 0 for palm grasp (small).
        # The transition diameter was pre-computed from the tip-clearance check.
        thumb_yaw   = (CTRL_MAX["thumb_yaw"]
                       if target_diameter >= self._cyl_transition_diameter
                       else 0.0)
        thumb_pitch = c_uni * (CTRL_MAX["thumb_proximal"] / CTRL_MAX["index"])
        T = self.fk.thumb_tip(thumb_pitch, thumb_yaw)

        # Coplanar corrections from index Z at c_uni.
        # Direction-aware fallback (same logic as plane()):
        #   target_z > z_max → ctrl=0 (finger can't extend that far)
        #   target_z < z_min → CTRL_MAX (finger can't curl that far)
        ref_tip  = self.fk.finger_tip("index", c_uni)
        target_z = float(ref_tip[2])
        coplanar = self.fk.coplanar_ctrls(target_z, fingers=NON_THUMB_FINGERS)
        for f in NON_THUMB_FINGERS:
            if coplanar.get(f) is None:
                _, z_max = self.fk.z_range_for_finger(f)
                coplanar[f] = 0.0 if target_z >= z_max else float(CTRL_MAX[f])

        tips: Dict[str, np.ndarray] = {
            f: self.fk.finger_tip(f, coplanar[f]) for f in NON_THUMB_FINGERS
        }
        tips["thumb"] = T

        all_nonthumb = np.array([tips[f] for f in NON_THUMB_FINGERS])
        centroid = all_nonthumb.mean(axis=0)

        if thumb_yaw > 0.0:
            # Opposition mode: tilt to make thumb→centroid direction horizontal.
            d = T - centroid
            tilt_y = float(np.arctan2(-d[2], d[0]))
            if tilt_y < -np.pi / 2:
                tilt_y += np.pi
            elif tilt_y > np.pi / 2:
                tilt_y -= np.pi
        else:
            # Palm grasp mode: thumb alongside fingers, no tilt needed.
            tilt_y = 0.0

        midpoint = np.vstack([all_nonthumb, T]).mean(axis=0)
        d_for_width = T - centroid
        width    = float(np.hypot(d_for_width[0], d_for_width[2]))
        span     = float(all_nonthumb[:, 1].max() - all_nonthumb[:, 1].min())

        # Cylinder radius: half the proximal-joint centroid-to-thumb distance
        prox_pts = np.array([self.fk.prox_joint_pos(f, coplanar[f])
                              for f in NON_THUMB_FINGERS])
        T_prox   = self.fk.prox_joint_pos("thumb", thumb_pitch, thumb_yaw)
        radius   = float(np.linalg.norm(T_prox - prox_pts.mean(axis=0))) / 2.0

        ctrl: Dict[str, float] = {f: float(coplanar[f]) for f in NON_THUMB_FINGERS}
        ctrl["thumb_proximal"] = float(thumb_pitch)
        ctrl["thumb_yaw"]      = float(thumb_yaw)

        return ClosureResult(
            mode="cylinder",
            midpoint=midpoint, width=width,
            finger_span=span, cylinder_radius=radius,
            tip_positions=tips, ctrl_values=ctrl,
            base_tilt_y=tilt_y, tilt_deg=float(np.degrees(abs(tilt_y))),
        )

    def solve(self, mode: str, target_width_or_diameter: float,
              n_fingers: int = 4) -> ClosureResult:
        """
        Unified entry point.
        mode: '2-finger line' | '3-finger plane' | '4-finger plane' |
              '5-finger plane' | 'cylinder'
        target_width_or_diameter: metres
        n_fingers: only used for plane modes
        """
        if mode == "2-finger line":
            return self.line(target_width_or_diameter)
        elif "plane" in mode:
            # parse n from mode string OR use n_fingers arg
            n = int(mode[0]) if mode[0].isdigit() else n_fingers
            return self.plane(target_width_or_diameter, n_fingers=n)
        elif mode == "cylinder":
            return self.cylinder(target_width_or_diameter)
        else:
            raise ValueError(f"Unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    rebuild = "--rebuild" in sys.argv
    xml_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    xml = xml_args[0] if xml_args else _DEFAULT_XML
    print(f"Loading model: {xml}")
    fk = InspireHandFK(xml_path=xml, rebuild=rebuild)

    print("\n=== FK range summary (base frame) ===")
    for fname in NON_THUMB_FINGERS:
        z0 = fk.finger_tip(fname, 0.0)
        z1 = fk.finger_tip(fname, CTRL_MAX[fname])
        print(f"  {fname:8s}: open={z0}, closed={z1}")
    th0 = fk.thumb_tip(0.0, 0.0)
    th1 = fk.thumb_tip(CTRL_MAX["thumb_proximal"], CTRL_MAX["thumb_yaw"])
    print(f"  {'thumb':8s}: open={th0}, closed={th1}")

    print("\n=== Closure geometry ===")
    cg = ClosureGeometry(fk)
    print(f"  Thumb yaw threshold diameter: {cg._thumb_yaw_threshold*1000:.1f} mm")

    for mode, w in [("2-finger line", 0.040),
                    ("4-finger plane", 0.040),
                    ("cylinder", 0.060)]:
        r = cg.solve(mode, w)
        print(f"\n  {r.mode} @ {w*1000:.0f}mm:")
        print(f"    width={r.width*1000:.1f}mm  span={r.finger_span*1000:.1f}mm"
              f"  tilt_y={r.tilt_deg:.1f}°")
        wt = r.world_tips(world_grasp_z=0.0)
        wb = r.world_base(world_grasp_z=0.0)
        print(f"    world_base={np.round(wb*1000, 1)} mm")
        for f, pos in r.tip_positions.items():
            wp = wt[f]
            print(f"    {f:10s} base={np.round(pos*1000,1)}  world={np.round(wp*1000,1)}")