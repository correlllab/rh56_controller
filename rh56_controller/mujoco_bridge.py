"""MuJoCo analysis bridge for Inspire hand grasp force evaluation.

Wraps `inspire_scene.xml` and provides:
  - Contact detection (box-finger pairs)
  - Site force sensor reading
  - Wrench cone computation:
      - Sim:  full linearized friction cone  (k primitive forces per contact)
      - Real: normal-only                    (1 primitive force per contact)
  - Ferrari-Canny force closure metric

No hardware dependency — import freely for sim-only analysis or as a
component of real2sim_viz.py.
"""

import re
import numpy as np
import mujoco
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not available — force closure analysis disabled")


# ─── Data classes ────────────────────────────────────────────────────────────

@dataclass
class ContactInfo:
    """One active contact between a finger geom and the box."""
    finger_name: str
    position: np.ndarray     # (3,) contact point, world frame
    frame: np.ndarray        # (3,3) rows: [normal, tangent1, tangent2]
    normal_force: float      # N, from mj_contactForce (positive = compression)
    friction_force: np.ndarray  # (2,) N, tangential (sim only)
    box_is_geom1: bool       # True → box is geom1; negate wrench to get force on finger


# ─── SimAnalyzer ─────────────────────────────────────────────────────────────

class SimAnalyzer:
    """Core MuJoCo analysis wrapper for Inspire hand grasp.

    Load inspire_scene.xml, step the sim, read contacts and sensors, and
    compute wrench cones + force closure for both sim (full cone) and a
    real-hardware proxy (normal-only cone).

    Typical use:
        sim = SimAnalyzer()
        sim.reset()
        while True:
            sim.step()
            contacts = sim.get_contacts()
            obj_pos  = sim.get_object_pos()
            pw_sim   = sim.compute_sim_wrench_cone(contacts, obj_pos)
            pw_real  = sim.compute_real_wrench_cone(contacts, real_forces_N, obj_pos)
            fc_sim, q_sim  = sim.evaluate_force_closure(pw_sim)
            fc_real, q_real = sim.evaluate_force_closure(pw_real)
    """

    FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

    TIP_SITES = {
        "thumb":  "right_thumb_tip",
        "index":  "right_index_tip",
        "middle": "right_middle_tip",
        "ring":   "right_ring_tip",
        "pinky":  "right_pinky_tip",
    }

    FORCE_SENSORS = {
        "thumb":  "thumb_tip_force",
        "index":  "index_tip_force",
        "middle": "middle_tip_force",
        "ring":   "ring_tip_force",
        "pinky":  "pinky_tip_force",
    }

    TORQUE_SENSORS = {
        "thumb":  "thumb_tip_torque",
        "index":  "index_tip_torque",
        "middle": "middle_tip_torque",
        "ring":   "ring_tip_torque",
        "pinky":  "pinky_tip_torque",
    }

    # Actuator ctrlrange maxima (rad) — same DOF order as real hand angle_set()
    # [pinky, ring, middle, index, thumb_proximal, thumb_yaw]
    SIM_CTRL_RANGES = [1.57, 1.57, 1.57, 1.57, 0.60, 1.308]

    # Default XML path relative to this file's grandparent directory
    _DEFAULT_XML_RELPATH = "h1_mujoco/inspire/inspire_scene.xml"

    def __init__(
        self,
        xml_path: Optional[str] = None,
        friction_cone_edges: int = 8,
    ):
        if xml_path is None:
            xml_path = str(
                Path(__file__).parent.parent / self._DEFAULT_XML_RELPATH
            )
        self.xml_path = str(xml_path)
        self.friction_cone_edges = friction_cone_edges

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Pre-compute IDs for speed
        self.box_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "box")
        self.object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object")
        self.body_to_finger = self._build_body_to_finger_map()
        self.tip_site_ids = self._cache_site_ids()
        self.sensor_addrs = self._cache_sensor_addrs()
        self.mu = float(self.model.geom_friction[self.box_geom_id, 0])

        print(f"[SimAnalyzer] Loaded: {self.xml_path}")
        print(f"[SimAnalyzer] friction μ={self.mu:.3f}, cone edges={friction_cone_edges}")
        print(f"[SimAnalyzer] Tip sites: {self.tip_site_ids}")

    # ── ID caching ───────────────────────────────────────────────────────────

    def _build_body_to_finger_map(self) -> Dict[int, str]:
        patterns = [
            (re.compile(r"^thumb_"),  "thumb"),
            (re.compile(r"^index_"),  "index"),
            (re.compile(r"^middle_"), "middle"),
            (re.compile(r"^ring_"),   "ring"),
            (re.compile(r"^pinky_"),  "pinky"),
            (re.compile(r"^base$"),   "palm"),
        ]
        body_to_finger: Dict[int, str] = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name is None:
                continue
            for pattern, finger in patterns:
                if pattern.match(name):
                    body_to_finger[body_id] = finger
                    break
        return body_to_finger

    def _cache_site_ids(self) -> Dict[str, int]:
        ids: Dict[str, int] = {}
        for finger, site_name in self.TIP_SITES.items():
            try:
                ids[finger] = self.model.site(site_name).id
            except KeyError:
                print(f"[SimAnalyzer] Warning: site '{site_name}' not found")
        return ids

    def _cache_sensor_addrs(self) -> Dict[str, Dict[str, int]]:
        addrs: Dict[str, Dict[str, int]] = {}
        for finger in self.FINGER_NAMES:
            try:
                f_id = self.model.sensor(self.FORCE_SENSORS[finger]).id
                t_id = self.model.sensor(self.TORQUE_SENSORS[finger]).id
                addrs[finger] = {
                    "force_adr":  int(self.model.sensor_adr[f_id]),
                    "torque_adr": int(self.model.sensor_adr[t_id]),
                }
            except KeyError as e:
                print(f"[SimAnalyzer] Warning: sensor not found for {finger}: {e}")
        return addrs

    # ── Simulation control ────────────────────────────────────────────────────

    def reset(self):
        """Reset to the 'home' keyframe and run forward kinematics."""
        key_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
            print("[SimAnalyzer] Reset to 'home' keyframe")
        else:
            print("[SimAnalyzer] Warning: 'home' keyframe not found")
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        """Advance the simulation by one timestep."""
        mujoco.mj_step(self.model, self.data)

    def get_time(self) -> float:
        return float(self.data.time)

    # ── Ctrl / angle bridge ───────────────────────────────────────────────────

    def get_ctrl(self) -> np.ndarray:
        """Current actuator ctrl values (rad)."""
        return self.data.ctrl.copy()

    def get_ctrl_as_real_angles(self) -> List[int]:
        """Convert sim ctrl (rad) to real hand angle units [0, 1000].

        Actuator order matches real hand angle_set() order:
          [pinky, ring, middle, index, thumb_bend, thumb_yaw]

        Inversion: MuJoCo ctrl=0 means joint at 0 rad (open/extended for flexors,
        spread-out for thumb yaw), but the real hand uses the opposite convention —
        0 = closed/curled for flexors, 0 = yawed-in for thumb yaw.
        So the mapping is inverted for all DOFs:
          real = 1000 - round(ctrl / ctrl_max * 1000)
        """
        ctrl = self.data.ctrl
        cmd = [
            1000 - int(np.clip(c / r, 0.0, 1.0) * 1000)
            for c, r in zip(ctrl, self.SIM_CTRL_RANGES)
        ]
        print(cmd)
        return cmd

    # ── Contact detection ─────────────────────────────────────────────────────

    def get_contacts(self) -> List[ContactInfo]:
        """Scan active contacts and return box-finger pairs."""
        contacts: List[ContactInfo] = []
        result = np.zeros(6)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2

            finger_name: Optional[str] = None
            box_is_geom1 = False

            if geom1 == self.box_geom_id:
                body_id = self.model.geom_bodyid[geom2]
                finger_name = self.body_to_finger.get(body_id)
                box_is_geom1 = True
            elif geom2 == self.box_geom_id:
                body_id = self.model.geom_bodyid[geom1]
                finger_name = self.body_to_finger.get(body_id)
                box_is_geom1 = False

            if finger_name is None or finger_name == "palm":
                continue

            mujoco.mj_contactForce(self.model, self.data, i, result)
            frame = contact.frame.reshape(3, 3).copy()

            contacts.append(ContactInfo(
                finger_name=finger_name,
                position=contact.pos.copy(),
                frame=frame,
                normal_force=float(result[0]),
                friction_force=result[1:3].copy(),
                box_is_geom1=box_is_geom1,
            ))

        return contacts

    # ── Sensor reading ────────────────────────────────────────────────────────

    def get_sensor_forces_N(self) -> Dict[str, float]:
        """Site force sensor magnitude in Newtons, keyed by finger name."""
        forces: Dict[str, float] = {}
        for finger in self.FINGER_NAMES:
            if finger not in self.sensor_addrs or finger not in self.tip_site_ids:
                continue
            adr = self.sensor_addrs[finger]["force_adr"]
            site_id = self.tip_site_ids[finger]
            force_site = self.data.sensordata[adr:adr + 3].copy()
            site_xmat = self.data.site_xmat[site_id].reshape(3, 3)
            force_world = site_xmat @ force_site
            forces[finger] = float(np.linalg.norm(force_world))
        return forces

    def get_sensor_force_vectors_N(self) -> Dict[str, np.ndarray]:
        """Site force sensor vectors in world frame (N), keyed by finger name."""
        forces: Dict[str, np.ndarray] = {}
        for finger in self.FINGER_NAMES:
            if finger not in self.sensor_addrs or finger not in self.tip_site_ids:
                continue
            adr = self.sensor_addrs[finger]["force_adr"]
            site_id = self.tip_site_ids[finger]
            force_site = self.data.sensordata[adr:adr + 3].copy()
            site_xmat = self.data.site_xmat[site_id].reshape(3, 3)
            forces[finger] = site_xmat @ force_site
        return forces

    def get_tip_site_positions(self) -> Dict[str, np.ndarray]:
        """World-frame positions of all fingertip sites."""
        positions: Dict[str, np.ndarray] = {}
        for finger, site_id in self.tip_site_ids.items():
            positions[finger] = self.data.site_xpos[site_id].copy()
        return positions

    def get_object_pos(self) -> np.ndarray:
        """World-frame position of the grasped object (box body)."""
        return self.data.xpos[self.object_body_id].copy()

    # ── Wrench cone computation ───────────────────────────────────────────────

    def compute_sim_wrench_cone(
        self,
        contacts: List[ContactInfo],
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """Full linearized friction cone: k primitive 6D wrenches per contact.

        Each primitive is the wrench [f, r×f] applied AT the object CoM,
        where f = normalize(n + μ*(cos θ * t1 + sin θ * t2)).

        Primitives are unit vectors (Fn is a free variable — represents the
        full scalable cone).  Ferrari-Canny Q is dimensionless / geometric.

        Returns (k * n_contacts, 6) array, or zeros (0, 6) if no contacts.
        """
        if not contacts:
            return np.zeros((0, 6))

        k = self.friction_cone_edges
        all_wrenches: List[np.ndarray] = []

        for c in contacts:
            normal = c.frame[0]
            t1 = c.frame[1]
            t2 = c.frame[2]
            r = c.position - object_pos  # moment arm to object CoM

            for j in range(k):
                theta = 2.0 * np.pi * j / k
                f = normal + self.mu * (np.cos(theta) * t1 + np.sin(theta) * t2)
                norm = np.linalg.norm(f)
                if norm > 1e-12:
                    f /= norm

                # Force direction ON object (from finger).
                # MuJoCo: mj_contactForce gives force on geom1.
                # If box is geom1 → normal points away from box → that IS the
                # direction the box feels from the finger.
                # If box is geom2 → negate to flip from finger-on-box to box-on-box.
                f_on_obj = f if c.box_is_geom1 else -f

                torque = np.cross(r, f_on_obj)
                all_wrenches.append(np.concatenate([f_on_obj, torque]))

        return np.array(all_wrenches) if all_wrenches else np.zeros((0, 6))

    def compute_sim_actual_wrenches(
        self,
        contacts: List[ContactInfo],
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """Actual sim contact wrenches (one per contact, in Newtons).

        Uses the full 3D contact force [Fn, Ft1, Ft2] from mj_contactForce —
        the exact force vector currently applied at each contact.

        This is in the same unit space (Newtons) as compute_real_wrench_cone,
        so Q_sim and Q_real from these two methods ARE directly comparable:

          Q_sim  = quality using the known Ft components (exact disk position)
          Q_real = quality from friction-disk boundary (Ft direction unknown)

        Returns (n_contacts, 6) array, or zeros (0, 6) if no contacts.
        """
        if not contacts:
            return np.zeros((0, 6))

        # Aggregate per finger (dominant contact)
        dominant: Dict[str, ContactInfo] = {}
        for c in contacts:
            if (c.finger_name not in dominant
                    or c.normal_force > dominant[c.finger_name].normal_force):
                dominant[c.finger_name] = c

        all_wrenches: List[np.ndarray] = []
        for c in dominant.values():
            # Reconstruct full contact force vector in world frame
            f = (c.normal_force * c.frame[0]
                 + c.friction_force[0] * c.frame[1]
                 + c.friction_force[1] * c.frame[2])
            # Force ON the object
            f_on_obj = f if c.box_is_geom1 else -f
            r = c.position - object_pos
            torque = np.cross(r, f_on_obj)
            all_wrenches.append(np.concatenate([f_on_obj, torque]))

        return np.array(all_wrenches) if all_wrenches else np.zeros((0, 6))

    def compute_real_wrench_cone(
        self,
        contacts: List[ContactInfo],
        real_forces_N: Dict[str, float],
        object_pos: np.ndarray,
    ) -> np.ndarray:
        """Friction-disk wrench cone: k primitive 6D wrenches per finger in contact.

        The real hand gives us the normal force magnitude Fn per finger but NOT
        the tangential components.  The feasible contact forces therefore form a
        disk in 3D force space:

            center: Fn * n
            radius: μ * Fn  (in the t1-t2 tangential plane)

        i.e. any force f = Fn*n + Ft1*t1 + Ft2*t2 with |Ft| ≤ μ*Fn is reachable.

        We linearize the boundary of that disk into k primitive wrenches — the
        same structure as compute_sim_wrench_cone but with ACTUAL force magnitudes
        rather than normalized unit vectors:

            f_j = Fn * n + Fn * μ * (cos θ_j * t1 + sin θ_j * t2)

        The convex hull of these primitives (across all finger contacts) represents
        the achievable wrench set given the measured normal forces.  Checking
        whether the origin lies inside that hull is a valid static-equilibrium test.

        Real2sim gap: the sim uses normalized primitives (Fn is a free variable →
        full scalable cone), so Q_sim captures arbitrary-wrench resistance.  The
        real case uses fixed Fn → the hull is smaller, and Q_real ≤ Q_sim.

        Uses contact positions / normals / tangents borrowed from the simulation
        (shared geometry assumption).  For each finger, the dominant contact
        (highest sim normal force) is used as the representative geometry.

        Returns (k * M_fingers, 6) array, or zeros (0, 6) if no valid forces.
        """
        if not contacts:
            return np.zeros((0, 6))

        k = self.friction_cone_edges

        # Dominant contact per finger (highest sim normal force)
        dominant: Dict[str, ContactInfo] = {}
        for c in contacts:
            if (c.finger_name not in dominant
                    or c.normal_force > dominant[c.finger_name].normal_force):
                dominant[c.finger_name] = c

        all_wrenches: List[np.ndarray] = []
        for finger, c in dominant.items():
            fn = real_forces_N.get(finger, 0.0)
            if fn < 1e-3:  # skip negligible real forces
                continue

            normal = c.frame[0]
            t1 = c.frame[1]
            t2 = c.frame[2]
            r = c.position - object_pos

            for j in range(k):
                theta = 2.0 * np.pi * j / k
                # Point on the boundary of the friction disk at the measured Fn
                f = fn * normal + fn * self.mu * (
                    np.cos(theta) * t1 + np.sin(theta) * t2)

                # Force on the object (sign convention matches compute_sim_wrench_cone)
                f_on_obj = f if c.box_is_geom1 else -f

                torque = np.cross(r, f_on_obj)
                all_wrenches.append(np.concatenate([f_on_obj, torque]))

        return np.array(all_wrenches) if all_wrenches else np.zeros((0, 6))

    # ── Force closure ─────────────────────────────────────────────────────────

    def evaluate_force_closure(
        self,
        primitive_wrenches: np.ndarray,
    ) -> Tuple[bool, float]:
        """Ferrari-Canny metric via convex hull of primitive wrenches.

        Returns (is_force_closure, ferrari_canny_metric).
        Positive metric → origin inside hull → force closure.

        Falls back to 3D force-only analysis when rank < 6.
        """
        if not HAS_SCIPY:
            return False, 0.0

        n = len(primitive_wrenches)
        if n < 7:
            if n >= 4:
                return self._evaluate_3d(primitive_wrenches[:, :3])
            return False, 0.0

        rank = np.linalg.matrix_rank(primitive_wrenches, tol=1e-8)
        if rank < 6:
            return self._evaluate_3d(primitive_wrenches[:, :3])

        try:
            hull = ConvexHull(primitive_wrenches)
        except Exception:
            return False, 0.0

        offsets = hull.equations[:, -1]  # signed distances; ≤0 → inside
        if np.all(offsets <= 1e-10):
            return True, float(np.min(np.abs(offsets)))
        return False, 0.0

    def _evaluate_3d(self, forces: np.ndarray) -> Tuple[bool, float]:
        """3D force-only fallback for Ferrari-Canny."""
        if len(forces) < 4:
            return False, 0.0
        try:
            hull = ConvexHull(forces)
        except Exception:
            return False, 0.0
        offsets = hull.equations[:, -1]
        if np.all(offsets <= 1e-10):
            return True, float(np.min(np.abs(offsets)))
        return False, 0.0

    # ── Viewer geometry helpers ───────────────────────────────────────────────

    def fill_viewer_geoms(
        self,
        scn,
        contacts: List[ContactInfo],
        sensor_forces_world: Dict[str, np.ndarray],
        tip_positions: Dict[str, np.ndarray],
        object_pos: np.ndarray,
        force_closure_sim: bool,
        ferrari_canny_sim: float,
        force_scale: float = 0.025,
    ):
        """Populate mujoco viewer user_scn with force arrows and status sphere.

        Arrows: contact forces on fingers (solid), sensor forces (semi-transparent).
        Sphere at object CoM: green = FC, red = no FC; radius ∝ Q.
        """
        FINGER_COLORS = {
            "thumb":  np.array([1.0, 0.2, 0.2, 0.9], dtype=np.float32),
            "index":  np.array([0.2, 0.4, 1.0, 0.9], dtype=np.float32),
            "middle": np.array([0.2, 0.8, 0.2, 0.9], dtype=np.float32),
            "ring":   np.array([0.8, 0.2, 0.8, 0.9], dtype=np.float32),
            "pinky":  np.array([1.0, 0.6, 0.1, 0.9], dtype=np.float32),
        }

        scn.ngeom = 0

        # Contact force arrows (force on finger, reconstructed in world frame)
        for c in contacts:
            if scn.ngeom >= scn.maxgeom:
                break
            f_world = (c.normal_force * c.frame[0]
                       + c.friction_force[0] * c.frame[1]
                       + c.friction_force[1] * c.frame[2])
            if c.box_is_geom1:
                f_world = -f_world  # flip: force on finger
            f_mag = np.linalg.norm(f_world)
            if f_mag < 1e-6:
                continue

            color = FINGER_COLORS.get(c.finger_name,
                                      np.array([0.5, 0.5, 0.5, 0.9], dtype=np.float32))
            end = c.position + f_world * force_scale
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9), color)
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.002,
                np.asarray(c.position, dtype=np.float64),
                np.asarray(end, dtype=np.float64))
            scn.geoms[scn.ngeom].rgba[:] = color
            scn.ngeom += 1

        # Sensor force arrows at fingertip sites (semi-transparent)
        for finger, f_world in sensor_forces_world.items():
            if scn.ngeom >= scn.maxgeom:
                break
            f_mag = np.linalg.norm(f_world)
            if f_mag < 1e-6 or finger not in tip_positions:
                continue
            site_pos = tip_positions[finger]
            end = site_pos + f_world * force_scale
            color = FINGER_COLORS.get(finger,
                                      np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)).copy()
            color[3] = 0.45  # semi-transparent
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW,
                np.zeros(3), np.zeros(3), np.zeros(9), color)
            mujoco.mjv_connector(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_ARROW, 0.001,
                np.asarray(site_pos, dtype=np.float64),
                np.asarray(end, dtype=np.float64))
            scn.geoms[scn.ngeom].rgba[:] = color
            scn.ngeom += 1

        # Force-closure indicator sphere at object CoM
        if scn.ngeom < scn.maxgeom:
            fc_color = (np.array([0.1, 0.9, 0.1, 0.6], dtype=np.float32)
                        if force_closure_sim
                        else np.array([0.9, 0.1, 0.1, 0.6], dtype=np.float32))
            radius = 0.005 + min(ferrari_canny_sim * 0.5, 0.015)
            mujoco.mjv_initGeom(
                scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0.0, 0.0]),
                object_pos.astype(np.float64),
                np.eye(3).flatten(),
                fc_color)
            scn.ngeom += 1
