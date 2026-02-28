"""
mink_grasp_planner.py — Mink IK-based grasp synthesis for the Inspire RH56 hand.

Baseline comparison against the analytical ClosureGeometry approach.

Pipeline:
    1. Receive target tip positions (derived from same geometric spec as
       ClosureGeometry: width / grasp type)
    2. Set FrameTask targets per active fingertip
    3. EqualityConstraintTask enforces XML joint coupling automatically —
       no manual polycoef implementation needed
    4. Iterate mink solve_ik until convergence
    5. Return MinkGraspResult with achieved ctrl, tips, and convergence metrics

Model: inspire_right.xml  (fixed hand, 12 DOF: 6 proximal + 6 coupled intermediate)
Coordinate frame: same as InspireHandFK base frame (hand base at origin).

Key comparison with ClosureGeometry:
  ClosureGeometry: analytical FK tables + brentq root-finding → zero width error,
                   guaranteed smooth, instant computation.
  MinkGraspPlanner: IK iterations → may find different finger poses, handles
                    coupling constraints automatically, flexible multi-finger
                    simultaneous optimisation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import mink
import mujoco
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Tip site names matching inspire_right.xml
_SITE: Dict[str, str] = {
    "thumb":  "right_thumb_tip",
    "index":  "right_index_tip",
    "middle": "right_middle_tip",
    "ring":   "right_ring_tip",
    "pinky":  "right_pinky_tip",
}

# Actuator indices in inspire_right.xml (order matches actuator list)
_ACT_IDX: Dict[str, int] = {
    "pinky": 0, "ring": 1, "middle": 2, "index": 3,
    "thumb_proximal": 4, "thumb_yaw": 5,
}

# Actuator index → actuated joint name (for extracting ctrl from qpos)
_ACT_TO_JOINT: List[str] = [
    "pinky_proximal_joint",
    "ring_proximal_joint",
    "middle_proximal_joint",
    "index_proximal_joint",
    "thumb_proximal_pitch_joint",
    "thumb_proximal_yaw_joint",
]

# Fixed thumb yaw for 2-finger line grasps (matches ClosureGeometry)
_THUMB_YAW_LINE: float = 1.16  # rad

# Polycoef coupling from inspire_right.xml equality constraints.
# (intermediate_joint = offset + slope * proximal_joint)
_COUPLING: Dict[str, tuple] = {
    "index_intermediate_joint":  (-0.05,  1.1169, "index_proximal_joint"),
    "middle_intermediate_joint": (-0.15,  1.1169, "middle_proximal_joint"),
    "ring_intermediate_joint":   (-0.15,  1.1169, "ring_proximal_joint"),
    "pinky_intermediate_joint":  (-0.15,  1.1169, "pinky_proximal_joint"),
    "thumb_intermediate_joint":  ( 0.15,  1.33,   "thumb_proximal_pitch_joint"),
    "thumb_distal_joint":        ( 0.15,  0.66,   "thumb_proximal_pitch_joint"),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MinkGraspResult:
    """Result from a single mink IK grasp solve."""
    mode: str
    target_width_m: float
    achieved_width_m: float       # XZ distance: thumb ↔ centroid of non-thumb tips
    coplanarity_err_m: float      # std of active fingertip Z values (0 = perfect coplanar)
    ctrl: np.ndarray              # (6,) [pinky, ring, middle, index, thumb_bend, thumb_yaw]
    tip_positions: Dict[str, np.ndarray]   # achieved world positions per finger
    target_positions: Dict[str, np.ndarray]  # input target positions per finger
    position_errors_m: Dict[str, float]    # per-finger Euclidean error vs target
    n_iters: int
    wall_time_s: float
    converged: bool
    error_history: List[float] = field(default_factory=list)  # total error per iteration


# ---------------------------------------------------------------------------
# MinkGraspPlanner
# ---------------------------------------------------------------------------

class MinkGraspPlanner:
    """
    Differential IK grasp planner for the Inspire RH56 hand via mink.

    Uses inspire_right.xml (fixed hand, 12 DOF).  Finger tip positions from
    target specification are tracked with FrameTask; joint coupling is
    maintained by EqualityConstraintTask (automatic polycoef enforcement).

    Args:
        xml_path:  Path to inspire_right.xml (fixed-hand model, no floating base).
        dt:        IK integration timestep [s].  Smaller → smoother but slower.
        max_iters: Maximum IK iterations per solve.
        conv_thr:  Per-finger convergence threshold [m].
        solver:    qpsolvers backend ('daqp', 'proxqp', ...).
        eq_cost:   EqualityConstraintTask weight (higher → stricter coupling).
        damp_cost: DampingTask base cost (higher → prefer smaller joint moves).
    """

    def __init__(
        self,
        xml_path: str | Path,
        dt: float = 0.005,
        max_iters: int = 500,
        conv_thr: float = 3e-3,
        solver: str = "daqp",
        eq_cost: float = 50.0,
        damp_cost: float = 1e-2,
    ):
        self.dt = dt
        self.max_iters = max_iters
        self.conv_thr = conv_thr
        self.solver = solver

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.config = mink.Configuration(self.model)

        # ctrl ranges for the 6 finger actuators (no base in inspire_right.xml)
        self.ctrl_min = self.model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_max = self.model.actuator_ctrlrange[:, 1].copy()

        # Cache site IDs
        self._site_id: Dict[str, int] = {}
        for fname, sname in _SITE.items():
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                raise ValueError(f"Site '{sname}' not in model")
            self._site_id[fname] = sid

        # Cache joint → qpos address for fast access
        self._qadr: Dict[str, int] = {}
        all_joints = list(_COUPLING.keys()) + _ACT_TO_JOINT
        for jname in set(all_joints):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise ValueError(f"Joint '{jname}' not in model")
            self._qadr[jname] = int(self.model.jnt_qposadr[jid])

        # DOF address for thumb yaw (needed for per-DOF damping)
        yaw_jid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "thumb_proximal_yaw_joint"
        )
        self._yaw_dofadr = int(self.model.jnt_dofadr[yaw_jid])

        # Shared tasks / limits
        self.eq_task = mink.EqualityConstraintTask(self.model, cost=eq_cost)
        self.cfg_limit = mink.ConfigurationLimit(self.model, gain=0.9)

        # Default uniform damping task
        self._damp_cost_val = damp_cost
        self.damping = mink.DampingTask(self.model, cost=damp_cost)

        # Line-grasp damping: strong yaw freezing
        line_damp = np.full(self.model.nv, damp_cost)
        line_damp[self._yaw_dofadr] = 1.0  # 100× stronger on yaw
        self.line_damping = mink.DampingTask(self.model, cost=line_damp)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _open_q(self, thumb_yaw: Optional[float] = None) -> np.ndarray:
        """Build open-hand qpos: all proximal joints at ctrl_min."""
        q = np.zeros(self.model.nq)
        for i, jname in enumerate(_ACT_TO_JOINT):
            q[self._qadr[jname]] = self.ctrl_min[i]
        if thumb_yaw is not None:
            q[self._qadr["thumb_proximal_yaw_joint"]] = thumb_yaw
        # Seed intermediate joints via coupling (for valid initial FK)
        for int_jname, (offset, slope, prox_jname) in _COUPLING.items():
            q[self._qadr[int_jname]] = offset + slope * q[self._qadr[prox_jname]]
        return q

    def _reset(self, thumb_yaw: Optional[float] = None) -> None:
        self.config.update(q=self._open_q(thumb_yaw))

    def _tip_pos(self, fname: str) -> np.ndarray:
        return self.config.data.site_xpos[self._site_id[fname]].copy()

    def _extract_ctrl(self) -> np.ndarray:
        """Read proximal joint qpos as ctrl values (position-actuated model)."""
        ctrl = np.array(
            [self.config.data.qpos[self._qadr[jname]] for jname in _ACT_TO_JOINT]
        )
        return np.clip(ctrl, self.ctrl_min, self.ctrl_max)

    def _achieved_width(self, active_fingers: List[str]) -> float:
        """XZ distance from thumb to centroid of non-thumb tips."""
        non_thumb = [f for f in active_fingers if f != "thumb"]
        if not non_thumb:
            return 0.0
        thumb = self._tip_pos("thumb")
        others = np.mean([self._tip_pos(f) for f in non_thumb], axis=0)
        d = thumb - others
        return float(np.hypot(d[0], d[2]))

    def _coplanarity_err(self, active_fingers: List[str]) -> float:
        """Std-dev of active fingertip Z values (0 = perfectly coplanar)."""
        zvals = [self._tip_pos(f)[2] for f in active_fingers]
        return float(np.std(zvals)) if len(zvals) > 1 else 0.0

    # -----------------------------------------------------------------------
    # Core IK loop
    # -----------------------------------------------------------------------

    def _run_ik(
        self,
        frame_tasks: List,
        damping_task,
        active_fingers: List[str],
        target_tips: Dict[str, np.ndarray],
        width_m: float,
        mode: str,
    ) -> MinkGraspResult:
        """
        Iterate mink solve_ik until convergence or max_iters.

        Tasks stack:
            FrameTask × active fingers  (position_cost=1.0, orientation=0)
            EqualityConstraintTask      (coupling enforcement)
            DampingTask                 (velocity regularisation)
        Limits:
            ConfigurationLimit          (joint bounds)
        """
        tasks = frame_tasks + [self.eq_task, damping_task]
        limits = [self.cfg_limit]
        error_history: List[float] = []
        t0 = time.time()
        converged = False
        n_iters = self.max_iters

        for i in range(self.max_iters):
            vel = mink.solve_ik(
                self.config,
                tasks,
                limits=limits,
                dt=self.dt,
                solver=self.solver,
            )
            self.config.integrate_inplace(vel, self.dt)

            # Total per-finger tip error
            total_err = sum(
                np.linalg.norm(self._tip_pos(f) - target_tips[f])
                for f in active_fingers
            )
            error_history.append(total_err)

            if total_err < self.conv_thr * len(active_fingers):
                converged = True
                n_iters = i + 1
                break

        wall = time.time() - t0

        achieved_tips = {f: self._tip_pos(f) for f in active_fingers}
        pos_errors = {
            f: float(np.linalg.norm(achieved_tips[f] - target_tips[f]))
            for f in active_fingers
        }
        return MinkGraspResult(
            mode=mode,
            target_width_m=width_m,
            achieved_width_m=self._achieved_width(active_fingers),
            coplanarity_err_m=self._coplanarity_err(active_fingers),
            ctrl=self._extract_ctrl(),
            tip_positions=achieved_tips,
            target_positions=target_tips,
            position_errors_m=pos_errors,
            n_iters=n_iters,
            wall_time_s=wall,
            converged=converged,
            error_history=error_history,
        )

    # -----------------------------------------------------------------------
    # Public API — grasp types
    # -----------------------------------------------------------------------

    def solve_line(
        self,
        width_m: float,
        thumb_target: np.ndarray,
        index_target: np.ndarray,
    ) -> MinkGraspResult:
        """
        2-finger pinch: index opposes thumb.

        Thumb yaw is seeded at _THUMB_YAW_LINE and frozen via heavy damping,
        matching the ClosureGeometry convention.

        Args:
            width_m:       Target XZ separation [m].
            thumb_target:  Desired thumb tip position in base frame.
            index_target:  Desired index tip position in base frame.
        """
        self._reset(thumb_yaw=_THUMB_YAW_LINE)

        thumb_task = mink.FrameTask(
            "right_thumb_tip", "site",
            position_cost=1.0, orientation_cost=0.0,
        )
        index_task = mink.FrameTask(
            "right_index_tip", "site",
            position_cost=1.0, orientation_cost=0.0,
        )
        thumb_task.set_target(mink.SE3.from_translation(thumb_target))
        index_task.set_target(mink.SE3.from_translation(index_target))

        return self._run_ik(
            [thumb_task, index_task],
            self.line_damping,
            ["thumb", "index"],
            {"thumb": thumb_target, "index": index_target},
            width_m,
            "2-finger line",
        )

    def solve_plane(
        self,
        width_m: float,
        target_positions: Dict[str, np.ndarray],
        active_fingers: Optional[List[str]] = None,
    ) -> MinkGraspResult:
        """
        Plane/antipodal grasp: n fingers on one side, thumb opposing.

        Unlike ClosureGeometry's sequential coplanarity correction via brentq,
        mink optimises ALL fingertip positions simultaneously.  Coplanarity
        emerges naturally from the shared Z targets.

        Args:
            width_m:          Target XZ separation [m].
            target_positions: Dict {finger_name: (3,) target in base frame}.
            active_fingers:   List of finger names (default: all keys).
        """
        if active_fingers is None:
            active_fingers = list(target_positions.keys())
        self._reset(thumb_yaw=self.ctrl_max[_ACT_IDX["thumb_yaw"]])

        frame_tasks = []
        for fname, tgt in target_positions.items():
            task = mink.FrameTask(
                _SITE[fname], "site",
                position_cost=1.0, orientation_cost=0.0,
            )
            task.set_target(mink.SE3.from_translation(tgt))
            frame_tasks.append(task)

        return self._run_ik(
            frame_tasks,
            self.damping,
            active_fingers,
            target_positions,
            width_m,
            f"{len(active_fingers)}-finger plane",
        )

    def solve_cylinder(
        self,
        diameter_m: float,
        target_positions: Dict[str, np.ndarray],
        active_fingers: Optional[List[str]] = None,
    ) -> MinkGraspResult:
        """
        Cylinder / power grasp: all fingers surround the object.

        Targets are proximal-link intermediate body positions (end of proximal
        links), matching ClosureGeometry's cylinder model.

        Args:
            diameter_m:       Target cylinder diameter [m].
            target_positions: Dict {finger_name: (3,) target in base frame}.
            active_fingers:   List of finger names.
        """
        if active_fingers is None:
            active_fingers = list(target_positions.keys())
        self._reset(thumb_yaw=self.ctrl_max[_ACT_IDX["thumb_yaw"]])

        frame_tasks = []
        for fname, tgt in target_positions.items():
            task = mink.FrameTask(
                _SITE[fname], "site",
                position_cost=1.0, orientation_cost=0.0,
            )
            task.set_target(mink.SE3.from_translation(tgt))
            frame_tasks.append(task)

        return self._run_ik(
            frame_tasks,
            self.damping,
            active_fingers,
            target_positions,
            diameter_m,
            "cylinder",
        )
